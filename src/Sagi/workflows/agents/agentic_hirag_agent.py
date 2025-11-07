import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, Optional, Set

from api.ui.utils import chunks_to_reference_chunks
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from configs.functions import get_config_manager, get_llm_config
from hirag_prod.tracing import traced, traced_async_gen
from resources.functions import get_chat_service
from resources.remote_function_executor import execute_remote_function

from Sagi.utils.chat_template import format_memory_to_string
from Sagi.utils.prompt import (
    get_file_selection_prompt,
    get_header_selection_prompt,
    get_judge_whether_need_memory_prompt,
    get_memory_augmented_user_query_prompt,
    get_rag_summary_plus_markdown_prompt,
    get_rag_summary_plus_prompt,
)
from Sagi.vercel import (
    FileToolCallItem,
    FilterChunkData,
    RagFileListToolCallInput,
    RagFileListToolCallOutput,
    RagFileSelectToolCallInput,
    RagFileSelectToolCallOutput,
    RagFilterToolCallInput,
    RagFilterToolCallOutput,
    RagHeaderSelectToolCallInput,
    RagHeaderSelectToolCallOutput,
    RagRetrievalToolCallInput,
    RagRetrievalToolCallOutput,
    ToolInputAvailable,
    ToolInputStart,
    ToolOutputAvailable,
)
from Sagi.workflows.sagi_memory import SagiMemory

# Expected ToC format:
# {
#     "summary": "Summary of toc",
#     "hierarchy": {
#         "content": "Concatened markdown of headers",
#         "blocks": [
#             {
#                 "type": "title" or "section_header",
#                 "hierarchyLevel": 0 - 6,
#                 "id": "item-xxx",
#                 "sourceBoundingBox": {
#                     "x0": int,
#                     "y0": int,
#                     "x1": int,
#                     "y1": int
#                 },
#                 "markdown": "text of the header",
#                 "pageIndex": int,
#                 "fileUrl": "url of the file"
#             },
#             ...
#         ]
#     }
# }


class RagAgenticAgent:
    agent: AssistantAgent
    language: str
    memory: SagiMemory
    gdb_path: str
    model_client_stream: bool
    retrieval_tool_name: str = "agenticRagRetrieval"
    filter_tool_name: str = "agenticRagFilter"
    list_files_tool_name: str = "agenticRagListFiles"
    select_files_tool_name: str = "agenticSelectFiles"
    select_headers_tool_name: str = "agenticSelectHeaders"

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        gdb_path: str,
        model_client_stream: bool = True,
        markdown_output: bool = False,
    ):
        self.memory = memory
        self.language = language
        self.system_prompt = None
        self.rag_summary_agent = None
        self.model_client = model_client
        self.model_client_stream = model_client_stream
        self.vdb_path = get_config_manager().postgres_url_async
        self.gdb_path = gdb_path
        self.ret: Optional[Dict[str, Any]] = None
        self.raw_chunks = None
        self.markdown_output = markdown_output
        self.augmented_user_input = None
        self.memory_context = None
        # Intermediate state for agentic workflow
        self.candidate_files: list[dict] | None = None
        self.selected_files: list[str] | None = None
        self.file_tocs: Dict[str, Any] | None = None
        self.selected_headers: list[dict] | None = None
        self.header_chunks: Dict[str, list[Dict[str, Any]]] | None = None

    @classmethod
    @traced(record_args=[])
    async def create(
        cls,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        gdb_path: str,
        model_client_stream: bool = True,
        markdown_output: bool = False,
    ):
        self = cls(
            model_client=model_client,
            memory=memory,
            language=language,
            gdb_path=gdb_path,
            model_client_stream=model_client_stream,
            markdown_output=markdown_output,
        )
        return self

    def _init_rag_summary_agent(self):
        self.rag_summary_agent = AssistantAgent(
            name="rag_agentic_summary_agent",
            model_client=self.model_client,
            model_client_stream=self.model_client_stream,
            memory=None,  # Disable default memory injection for this agent
            system_message=self.system_prompt,
        )

    def set_system_prompt(self, chunks: str, memory_context: Optional[str] = None):

        system_prompt = get_rag_summary_plus_prompt(
            chunks_data=chunks, memory_context=memory_context, language=self.language
        )
        if self.markdown_output:
            system_prompt = get_rag_summary_plus_markdown_prompt(
                chunks_data=chunks,
                memory_context=memory_context,
                language=self.language,
            )
        self.system_prompt = system_prompt

    # -----------------------------
    # Agentic HI-RAG helper prompts
    # -----------------------------
    def _normalize_title(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = "".join(c for c in s if c.isalnum() or c.isspace())
        s = " ".join(s.split())
        s = s.lower()

        return s

    def _make_names_unique(
        self, list_to_process: list[dict], key: str, normalize: bool = False
    ) -> list[dict]:
        used_names: set[str] = set()
        base_counts: dict[str, int] = {}

        for f in list_to_process:
            # Get the original name and normalize to a non-empty string
            raw_name = f.get(key)
            if normalize:
                name = (
                    self._normalize_title(raw_name)
                    if raw_name is not None
                    else "unknown"
                )
            else:
                name = str(raw_name).strip() if raw_name is not None else "unknown"
            if not name:
                name = "unknown"

            # If unused, accept as-is
            if name not in used_names:
                used_names.add(name)
                base_counts.setdefault(name, 0)
                f[key] = name
                continue

            # Otherwise, generate a unique variant using increasing counters
            base, ext = os.path.splitext(name)
            count = base_counts.get(name, 0) + 1
            while True:
                candidate = f"{base}({count}){ext}"
                if candidate not in used_names:
                    used_names.add(candidate)
                    base_counts[name] = count
                    f[key] = candidate
                    break
                count += 1

        return list_to_process

    def _prepare_file_headers(self, files: list[dict]) -> list[dict]:
        cleaned_files = self._make_names_unique(files, "fileName")
        for f in cleaned_files:
            toc: dict = f.get("toc", {})
            hierarchy: dict = toc.get("hierarchy", {})
            blocks: list[dict] = hierarchy.get("blocks", [])
            cleaned_blocks = self._make_names_unique(blocks, "markdown", normalize=True)
            hierarchy["blocks"] = cleaned_blocks
            toc["hierarchy"] = hierarchy
            f["toc"] = toc
        return cleaned_files

    def _build_choose_files_prompt(self, query: str, files: list[dict]) -> str:
        file_str = ""
        for f in files:
            # Normalize a readable file name
            file_name: str = (
                f.get("fileName")
                or f.get("name")
                or f.get("filename")
                or str(
                    f.get("uri") or f.get("url") or f.get("fileUrl") or "unknown"
                ).rsplit("/", 1)[-1]
            )
            toc: dict = f.get("toc", {})
            summary: str = toc.get("summary") or "Summary unavailable."
            # Match the template format: "fileName | Summary"
            file_str += f"{file_name} | {summary}\n"

        return get_file_selection_prompt(
            user_query=query, candidate_files=file_str, language=self.language
        )

    def _build_choose_headers_prompt(self, query: str, files: Dict[str, Any]) -> str:
        """Build the choose-headers prompt using the shared prompt builder.

        The table_of_contents string is grouped by file with indented headers.
        """
        toc_lines: list[str] = []
        for file_name, toc in files.items():
            hierarchy: dict = (toc or {}).get("hierarchy", {})
            blocks: list[dict] = hierarchy.get("blocks", [])
            if not blocks:
                continue
            toc_lines.append(f"File: {file_name}")
            for b in blocks:
                title: str = b.get("markdown", "unknown")
                level: int = int(b.get("hierarchyLevel", 0))
                toc_lines.append(f"{'  ' * level}- \"{title}\"")

        table_of_contents = "\n".join(toc_lines)
        return get_header_selection_prompt(
            user_query=query,
            table_of_contents=table_of_contents,
            language=self.language,
        )

    def _parse_simple_list_response(self, response_text: str) -> list[str]:
        lines = response_text.strip().splitlines()
        results = []
        for line in lines:
            line = line.strip()
            if line.startswith("- "):
                line = line[2:].strip()
            # Remove wrapping quotes if present
            if (line.startswith('"') and line.endswith('"')) or (
                line.startswith("'") and line.endswith("'")
            ):
                line = line[1:-1].strip()
            results.append(line)
        return results

    def _parse_file_header_in_response(
        self,
        response_text: str,
        toc_map: Dict[str, dict],
    ) -> list[dict]:

        def _strip_quotes(s: str) -> str:
            s = s.strip()
            if (s.startswith('"') and s.endswith('"')) or (
                s.startswith("'") and s.endswith("'")
            ):
                return s[1:-1].strip()
            return s

        header_index: Dict[str, Dict[str, dict]] = {}
        for file_name, toc in toc_map.items():
            blocks = (toc.get("hierarchy") or {}).get("blocks", []) or []
            header_index[file_name] = {
                b.get("markdown").strip(): b for b in blocks if b.get("markdown")
            }

        headers: list[dict] = []
        seen: set[tuple[str, str]] = set()

        lines = response_text.strip().splitlines()
        for line in lines:
            line = line.strip()
            if "::" not in line:
                continue
            left_part, right_part = line.split("::", 1)
            file_name = _strip_quotes(left_part.strip())
            header_title = _strip_quotes(right_part.strip())

            if file_name not in header_index:
                continue

            block = header_index[file_name].get(header_title)
            if not block:
                continue

            key = (file_name, header_title)
            if key in seen:
                continue
            seen.add(key)
            headers.append(
                {
                    "fileName": file_name,
                    "title": block.get("markdown", "").strip(),
                    "id": block.get("id", ""),
                }
            )

        return headers

    @traced_async_gen(record_return=True)
    async def run_list_files(
        self,
        user_input: str,
        workspace_id: str,
        knowledge_base_id: str,
        file_ids: Optional[Set[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        """Run a query through the RAG system and prepare the summary agent.

        Args:
            user_input (str): The user's input query.
            workspace_id (str): The ID of the workspace.
            knowledge_base_id (str): The ID of the knowledge base.
            file_ids (Optional[Set[str]]): Set of file/folder IDs to filter the search.
            cancellation_token (Optional[CancellationToken]): Token for cancellation support.

        Yields:
            Tool input and output events for the query process.
        """
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()

        self.memory_context = None
        self.augmented_user_input = None
        try:
            # Skip memory augmentation if memory is None (e.g., in batch jobs)
            if self.memory is not None:
                start_time = time.time()
                memory_query_result = await self.memory.query(
                    query=user_input, type="rag"
                )
                memory_results = memory_query_result.results
                if len(memory_results) == 0:
                    self.augmented_user_input = user_input
                else:
                    self.memory_context = format_memory_to_string(memory_results)
                    memory_augmented_user_input_prompt = (
                        get_memory_augmented_user_query_prompt(
                            user_input=user_input,
                            memory=self.memory_context,
                            language=self.language,
                        )
                    )
                    self.augmented_user_input = await get_chat_service().complete(
                        prompt=memory_augmented_user_input_prompt,
                        model=get_llm_config().model_name,
                    )
                    duration_seconds = time.time() - start_time
                    changed = self.augmented_user_input.strip() != user_input.strip()
                    logging.info(
                        f"RAG memory augmented used={changed} duration_seconds={duration_seconds:.2f}"
                    )
            else:
                # No memory available, use user input directly
                logging.info("No memory available, skipping augmentation.")
                self.augmented_user_input = user_input

            yield ToolInputStart(toolName=self.list_files_tool_name)

            yield ToolInputAvailable(
                input=RagFileListToolCallInput().to_dict(),
            )

            # Build query parameters
            query_params = {
                "workspace_id": workspace_id,
                "knowledge_base_id": knowledge_base_id,
            }

            function_call_info: Dict[str, Any] = {
                "function_id": "hi_rag_list_kb_files",
            }
            function_call_info.update(query_params)

            rag_task = asyncio.create_task(
                execute_remote_function("HI_RAG", function_call_info)
            )
            if cancellation_token is not None:
                cancellation_token.link_future(rag_task)
            try:
                ret_raw = await rag_task
            except asyncio.CancelledError:
                rag_task.cancel()
                raise
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            filtered_files = []
            # Filter by file_ids if provided, used for @file or @folder mentions
            for file in ret_raw:
                file_id = file.get("id")
                if not file_ids or file_id in file_ids:
                    filtered_files.append(file)

            tool_call_output = self._make_names_unique(filtered_files, "fileName")
            self.candidate_files = tool_call_output
            self.candidate_files_simple = [
                FileToolCallItem(
                    fileName=file.get("fileName", "unknown"),
                    fileUrl=file.get("uri", ""),
                    type=file.get("type", "unknown"),
                )
                for file in self.candidate_files
            ]

            yield ToolOutputAvailable(
                output=RagFileListToolCallOutput(
                    data=self.candidate_files_simple
                ).to_dict(),
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"Query failed: {str(e)}")

    @traced_async_gen(record_return=True)
    async def run_llm_choose_files(
        self,
        user_input: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        """Ask LLM to select the most relevant files out of candidates."""
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()
        try:
            yield ToolInputStart(toolName=self.select_files_tool_name)
            yield ToolInputAvailable(
                input=RagFileSelectToolCallInput(
                    query=user_input,
                    data=[f.fileName for f in (self.candidate_files_simple or [])],
                ).to_dict(),
            )
            if not self.candidate_files:
                self.selected_files = []
                yield ToolOutputAvailable(output={"data": []})
                return

            prompt = self._build_choose_files_prompt(user_input, self.candidate_files)
            result_text = await get_chat_service().complete(
                prompt=prompt, model=get_llm_config().model_name
            )

            # Fallback: if the LLM returns an empty / whitespace-only response, select all candidates.
            if not result_text or not result_text.strip():
                logging.warning(
                    "LLM returned empty file selection; falling back to all candidate files."
                )
                # Mark the response so downstream observers know a fallback occurred.
                result_text = "<EMPTY_RESPONSE_FALLBACK_ALL>"
                chosen = [
                    f.get("fileName", "unknown") for f in (self.candidate_files or [])
                ]
            else:
                chosen = self._parse_simple_list_response(result_text)

            # Keep only those that exist in candidates
            candidate_names = {f.get("fileName") for f in (self.candidate_files or [])}
            selected_files_raw = [f for f in chosen if f in candidate_names]
            # Prepare cleaned ToCs from already-available candidate file data
            selected_file_objs = [
                f
                for f in (self.candidate_files or [])
                if f.get("fileName") in selected_files_raw
            ]
            cleaned_files = self._prepare_file_headers(selected_file_objs)
            # Map file name -> toc for quick lookups in header selection
            self.file_tocs = {
                f.get("fileName"): (f.get("tableOfContents") or {})
                for f in cleaned_files
            }
            # Keep selection as list of file names
            self.selected_files = list(self.file_tocs.keys())

            yield ToolOutputAvailable(
                output=RagFileSelectToolCallOutput(
                    response_text=result_text,
                    data=[
                        FileToolCallItem(
                            fileName=file.get("fileName", "unknown"),
                            fileUrl=file.get("uri", ""),
                            type=file.get("type", "unknown"),
                        )
                        for file in cleaned_files
                    ],
                ).to_dict(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"LLM choose files failed: {str(e)}")

    @traced_async_gen(record_return=True)
    async def run_llm_choose_headers(
        self,
        user_input: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        """Ask LLM to select headers from ToCs relevant to the question."""
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()
        try:
            yield ToolInputStart(toolName=self.select_headers_tool_name)
            # Convert file_tocs mapping to a list of dicts for tool input schema
            toc_list = [
                {"fileName": name, "toc": toc}
                for name, toc in (self.file_tocs or {}).items()
            ]
            yield ToolInputAvailable(
                input=RagHeaderSelectToolCallInput(
                    table_of_contents=toc_list,
                ).to_dict(),
            )
            if not self.file_tocs:
                self.selected_headers = {}
                yield ToolOutputAvailable(
                    RagHeaderSelectToolCallOutput(data={}).to_dict()
                )
                return

            prompt = self._build_choose_headers_prompt(user_input, self.file_tocs)
            result_text = await get_chat_service().complete(
                prompt=prompt, model=get_llm_config().model_name
            )
            # Parse selections into list[{fileName,title,id}]
            header_selections = self._parse_file_header_in_response(
                result_text, self.file_tocs
            )

            self.selected_headers = header_selections

            yield ToolOutputAvailable(
                output=RagHeaderSelectToolCallOutput(
                    response_text=result_text,
                    headers=header_selections,
                ).to_dict()
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"LLM choose headers failed: {str(e)}")

    @traced_async_gen(record_return=True)
    async def run_get_chunks_by_headers(
        self,
        workspace_id: str,
        knowledge_base_id: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        """Get chunks for the selected headers per file via RAG tool."""
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()
        try:
            yield ToolInputStart(toolName=self.retrieval_tool_name)
            yield ToolInputAvailable(
                input=RagRetrievalToolCallInput(
                    data=self.selected_headers or []
                ).to_dict()
            )
            self.header_chunks = {}
            collected_chunks: list[Dict[str, Any]] = []
            if not self.selected_headers:
                # No selections; nothing to retrieve
                yield ToolOutputAvailable(
                    output=RagRetrievalToolCallOutput(data=[]).to_dict()
                )
                # Ensure downstream filter sees no chunks
                self.raw_chunks = {"chunks": []}
                return

            # Build list of header block IDs by matching selected header titles to ToC blocks
            headers = []  # list of strings with header ids
            for header_info in self.selected_headers:
                header_id = header_info.get("id", "")
                if header_id:
                    headers.append(header_id)

            task = asyncio.create_task(
                execute_remote_function(
                    "HI_RAG",
                    {
                        "function_id": "hi_rag_query_by_headers",
                        "workspace_id": workspace_id,
                        "knowledge_base_id": knowledge_base_id,
                        "headers": headers,
                    },
                )
            )
            if cancellation_token is not None:
                cancellation_token.link_future(task)
            try:
                chunks = await task
            except asyncio.CancelledError:
                task.cancel()
                raise
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            # Aggregate into the same format expected by run_filter
            self.raw_chunks = {"chunks": chunks}
            ref_chunks, _, _ = chunks_to_reference_chunks(chunks, from_ofnil=False)
            yield ToolOutputAvailable(
                output=RagRetrievalToolCallOutput(data=ref_chunks).to_dict()
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"Get chunks by headers failed: {str(e)}")

    @traced_async_gen(record_return=True)
    async def run_filter(
        self,
        user_input: str,
        workspace_id: str,
        knowledge_base_id: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        """Run the filtering step on raw chunks."""
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()

        try:
            num_chunks = 0
            if self.raw_chunks:
                self.raw_chunks["query"] = user_input
                if "chunks" in self.raw_chunks:
                    num_chunks = len(self.raw_chunks["chunks"])

            yield ToolInputStart(toolName=self.filter_tool_name)

            yield ToolInputAvailable(
                input=RagFilterToolCallInput(num_chunks=num_chunks).to_dict(),
            )

            if num_chunks == 0:
                logging.warning("No chunks available for filtering.")
                yield ToolOutputAvailable(
                    output=RagFilterToolCallOutput(
                        data=FilterChunkData(
                            included=[],
                            excluded=[],
                        )
                    ).to_dict(),
                )
                self.set_system_prompt(chunks=[])
                self._init_rag_summary_agent()
                self.ret = {"chunks": []}
                return

            # Apply hybrid strategy to get final chunks
            rag_task = asyncio.create_task(
                execute_remote_function(
                    "HI_RAG",
                    {
                        "function_id": "hi_rag_apply_strategy_to_chunks",
                        "language": self.language,
                        "chunks_dict": self.raw_chunks,
                        "strategy": "hybrid",
                        "workspace_id": workspace_id,
                        "knowledge_base_id": knowledge_base_id,
                        "filter_by_clustering": True,
                    },
                )
            )
            if cancellation_token is not None:
                cancellation_token.link_future(rag_task)
            try:
                ret = await rag_task
            except asyncio.CancelledError:
                rag_task.cancel()
                raise
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            included_chunks, _, _ = chunks_to_reference_chunks(
                ret["chunks"], from_ofnil=False
            )

            excluded_chunks, _, _ = chunks_to_reference_chunks(
                ret["outliers"], from_ofnil=False
            )

            yield ToolOutputAvailable(
                output=RagFilterToolCallOutput(
                    data=FilterChunkData(
                        included=included_chunks,
                        excluded=excluded_chunks,
                    )
                ).to_dict(),
            )
            chunks_string = "\n" + "\n".join(
                f"    [{i}] {' '.join((c.get('text', '') or '').split())}"
                for i, c in enumerate(ret["chunks"], start=1)
            )

            need_memory = False
            if self.memory_context:
                try:
                    judge_prompt = get_judge_whether_need_memory_prompt(
                        user_query=self.augmented_user_input or "",
                        chunks_data=chunks_string,
                    )
                    judge_res = await get_chat_service().complete(
                        prompt=judge_prompt,
                        model=get_llm_config().model_name,
                    )
                    need_memory = (judge_res or "").strip().lower().startswith("y")
                except Exception as e:
                    logging.warning(f"Memory need judge failed: {e}")
                    need_memory = False

            self.set_system_prompt(
                chunks=chunks_string,
                memory_context=self.memory_context if need_memory else None,
            )
            self._init_rag_summary_agent()
            self.ret = ret

        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"Filtering failed: {str(e)}")

    @traced_async_gen(record_return=True)
    async def run_agentic_pipeline(
        self,
        user_input: str,
        workspace_id: str,
        knowledge_base_id: str,
        file_ids: Optional[Set[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        """Execute the full Agentic HI-RAG pipeline end-to-end up to generation init.

        Steps: Query augmentation -> [RAG] List Files -> [LLM] Select Files ->
            [LLM] Select Headers -> [RAG] Retrieve Chunks by Headers -> [RAG] Filter Chunks ->
                  initialize summary agent for final [LLM] generation.

           After this completes, call run_workflow() to stream the final answer.
        """
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()

        # Reset state
        self.memory_context = None
        self.augmented_user_input = None
        self.candidate_files = None
        self.selected_files = None
        self.file_tocs = None
        self.selected_headers = None
        self.header_chunks = None
        self.ret = None
        self.raw_chunks = None

        # 0) Memory augmentation (optional)
        if self.memory is not None:
            try:
                memory_query_result = await self.memory.query(
                    query=user_input, type="rag"
                )
                memory_results = memory_query_result.results
                if memory_results:
                    self.memory_context = format_memory_to_string(memory_results)
                    memory_augmented_user_input_prompt = (
                        get_memory_augmented_user_query_prompt(
                            user_input=user_input,
                            memory=self.memory_context,
                            language=self.language,
                        )
                    )
                    self.augmented_user_input = await get_chat_service().complete(
                        prompt=memory_augmented_user_input_prompt,
                        model=get_llm_config().model_name,
                    )
                else:
                    self.augmented_user_input = user_input
            except Exception:
                # If memory augmentation fails, proceed with original query
                self.augmented_user_input = user_input
        else:
            self.augmented_user_input = user_input

        # 1) [RAG] List Files (via raw search + dedupe)
        async for ev in self.run_list_files(
            user_input=self.augmented_user_input,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            file_ids=file_ids,
            cancellation_token=cancellation_token,
        ):
            yield ev

        # 2) [LLM] Choose Files
        async for ev in self.run_llm_choose_files(
            user_input=self.augmented_user_input,
            cancellation_token=cancellation_token,
        ):
            yield ev

        if not self.selected_files:
            # No files selected -> finish with empty setup
            self.set_system_prompt(chunks=[])
            self._init_rag_summary_agent()
            self.ret = {"chunks": []}
            return

        # 3) [LLM] Choose Headers
        async for ev in self.run_llm_choose_headers(
            user_input=self.augmented_user_input,
            cancellation_token=cancellation_token,
        ):
            yield ev

        # 4) [RAG] Get Chunks under headers
        async for ev in self.run_get_chunks_by_headers(
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            cancellation_token=cancellation_token,
        ):
            yield ev

        # 5) [RAG] Filter Chunks (re-use existing implementation)
        async for ev in self.run_filter(
            user_input=self.augmented_user_input,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            cancellation_token=cancellation_token,
        ):
            yield ev

    @traced(record_return=True)
    def run_workflow(
        self,
        user_input: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> tuple[Optional[Dict[str, Any]], Any]:
        """Run the full workflow for processing a user query.

        Args:
            user_input (str): The user's input query.
            cancellation_token (Optional[CancellationToken]): Token for cancellation support.

        Returns:
            tuple: The query results and the agentic agent's streaming output.
        """
        if not self.rag_summary_agent:
            raise RuntimeError("RAG agentic agent not initialized.")
        if not self.ret:
            raise RuntimeError("Query results are not available.")

        kwargs = {}
        if cancellation_token is not None:
            kwargs["cancellation_token"] = cancellation_token
        return self.ret, self.rag_summary_agent.run_stream(
            task=self.augmented_user_input, **kwargs
        )

    async def cleanup(self):
        pass
