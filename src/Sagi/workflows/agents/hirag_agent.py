import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional, Set

from api.ui.utils import chunks_to_reference_chunks
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from hirag_prod import HiRAG
from hirag_prod.configs.functions import get_llm_config
from hirag_prod.resources.functions import get_chat_service
from resources.functions import get_hi_rag_client, get_settings

from Sagi.utils.chat_template import format_memory_to_string
from Sagi.utils.prompt import (
    get_judge_whether_need_memory_prompt,
    get_memory_augmented_user_query_prompt,
    get_rag_summary_plus_markdown_prompt,
    get_rag_summary_plus_prompt,
)
from Sagi.vercel import (
    FilterChunkData,
    RagFilterToolCallInput,
    RagFilterToolCallOutput,
    RagSearchToolCallInput,
    RagSearchToolCallOutput,
    RagSearchToolCallOutputItem,
    ToolInputAvailable,
    ToolInputStart,
    ToolOutputAvailable,
)
from Sagi.workflows.sagi_memory import SagiMemory


class RagSummaryAgent:
    agent: AssistantAgent
    language: str
    memory: SagiMemory
    gdb_path: str
    model_client_stream: bool
    rag_instance: HiRAG
    search_tool_name: str = "ragSearch"
    filter_tool_name: str = "ragFilter"

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
        self.rag_instance = None
        self.system_prompt = None
        self.rag_summary_agent = None
        self.model_client = model_client
        self.model_client_stream = model_client_stream
        self.vdb_path = get_settings().postgres_url_async
        self.gdb_path = gdb_path
        self.ret: Optional[Dict[str, Any]] = None
        self.raw_chunks = None
        self.search_tool_name = "ragSearch"
        self.filter_tool_name = "ragFilter"
        self.markdown_output = markdown_output
        self.augmented_user_input = None
        self.memory_context = None

    @classmethod
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
        self.rag_instance = get_hi_rag_client()
        await self.rag_instance.set_language(language)
        return self

    def _init_rag_summary_agent(self):
        self.rag_summary_agent = AssistantAgent(
            name="rag_summary_agent",
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

    async def run_query(
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
            start_time = time.time()
            memory_query_result = await self.memory.query(query=user_input, type="rag")
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

            yield ToolInputStart(toolName=self.search_tool_name)

            yield ToolInputAvailable(
                input=RagSearchToolCallInput(query=user_input).to_dict(),
            )

            # Build query parameters
            query_params = {
                "workspace_id": workspace_id,
                "knowledge_base_id": knowledge_base_id,
                "strategy": "raw",
                "translation": ["en", "zh-t-hk", "zh"],
                "summary": False,
                "filter_by_clustering": False,
                "threshold": 0.0,
            }

            # Add file_ids if provided
            if file_ids:
                query_params["file_list"] = list(file_ids)

            rag_task = asyncio.create_task(
                self.rag_instance.query(self.augmented_user_input, **query_params)
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

            tool_call_output = set(
                (chunk["fileName"], chunk["uri"]) for chunk in ret_raw["chunks"]
            )

            yield ToolOutputAvailable(
                output=RagSearchToolCallOutput(
                    data=[
                        RagSearchToolCallOutputItem(
                            fileName=chunk_tuple[0],
                            fileUrl=chunk_tuple[1],
                            type=chunk_tuple[1].split(".")[-1],
                        )
                        for chunk_tuple in tool_call_output
                    ]
                ).to_dict(),
            )

            self.raw_chunks = ret_raw

        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"Query failed: {str(e)}")

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
            if self.raw_chunks and "chunks" in self.raw_chunks:
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
                self.rag_instance.apply_strategy_to_chunks(
                    chunks_dict=self.raw_chunks,
                    strategy="hybrid",
                    workspace_id=workspace_id,
                    knowledge_base_id=knowledge_base_id,
                    filter_by_clustering=True,
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

            included_chunks, _ = chunks_to_reference_chunks(
                ret["chunks"], from_ofnil=False
            )

            excluded_chunks, _ = chunks_to_reference_chunks(
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
            tuple: The query results and the summary agent's streaming output.
        """
        if not self.rag_summary_agent:
            raise RuntimeError("RAG summary agent not initialized.")
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
