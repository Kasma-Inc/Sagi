import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, UserMessage
from resources.functions import get_web_search_service
from resources.remote_function_executor import execute_remote_function

from Sagi.utils.prompt import (
    get_template_based_generation_prompt,
    get_template_based_planning_prompt,
    get_web_search_query_rewrite_prompt,
)
from Sagi.vercel import ToolInputAvailable, ToolInputStart, ToolOutputAvailable
from Sagi.workflows.sagi_memory import SagiMemory
from Sagi.workflows.utils import (
    Plan,
    PlanStep,
    build_module_queries_block,
    build_plan_overview,
    decode_plan_from_json_like,
    dump_generation_messages,
    extract_template_and_instruction,
    join_text_messages,
)


class FinanceAgent:
    """
    Lightweight pipeline for finance tasks:
    1. Extract template and instruction from user messages;
    2. Plan per-module TODO list (emit Tool events);
    3. Run per-module RAG retrieval and web search concurrently (emit Tool events);
    4. Stream final markdown generation following template structure;
    """

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        model_client_stream: bool = True,
        markdown_output: bool = True,
    ):
        self.model_client = model_client
        self.memory = memory
        self.language = language
        self.model_client_stream = model_client_stream
        self.markdown_output = markdown_output

        self.template_text: str = ""
        self.user_instruction: str = ""
        self.plan: Optional[Plan] = None
        self.step_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.step_web_search_snippets: Dict[str, List[str]] = {}
        self.step_queries: Dict[str, str] = {}
        self.step_web_search_queries: Dict[str, str] = {}

    # ------------------------- helpers -------------------------
    def _build_generation_messages(self) -> List[UserMessage]:
        per_module_ctx: List[str] = []
        all_modules = set(self.step_chunks.keys()) | set(
            self.step_web_search_snippets.keys()
        )

        for module in sorted(all_modules):
            parts: List[str] = []

            # Chunk retrieval snippets
            chunks = self.step_chunks.get(module, [])
            chunk_texts: List[str] = []
            for c in chunks[:8]:
                t = (c.get("text") or "").strip()
                if t:
                    chunk_texts.append(t)
            if chunk_texts:
                parts.append("**Chunk Retrieval:**")
                parts.extend(f"- {s}" for s in chunk_texts)

            # Web search snippets
            web_snippets = self.step_web_search_snippets.get(module, [])
            web_texts = [s.strip() for s in web_snippets[:8] if s.strip()]
            if web_texts:
                parts.append("**Web Search:**")
                parts.extend(f"- {s}" for s in web_texts)

            # Build block
            if parts:
                block = f"## {module}\n" + "\n".join(parts)
            else:
                block = f"## {module}\n- (no retrieval)"
            per_module_ctx.append(block)

        ctx = "\n\n".join(per_module_ctx)

        # Build plan overview to guide headings and structure
        plan_block, plan_json_block = build_plan_overview(self.plan)
        module_queries_block = build_module_queries_block(self.step_queries)

        lang = (self.language or "en").lower()
        sys = UserMessage(
            source="system",
            content=get_template_based_generation_prompt(
                template=self.template_text,
                plan_json_block=plan_json_block,
                module_queries_block=module_queries_block,
                plan_block=plan_block,
                per_module_context=ctx,
                language=lang,
            ),
        )

        user = UserMessage(
            source="user",
            content=f"""
            <USER_ORIGINAL_QUERY>
            {self.user_instruction.strip()}
            </USER_ORIGINAL_QUERY>
            """.strip(),
        )
        return [sys, user]

    async def _prepare_inputs_from_messages(
        self, messages: List[TextMessage]
    ) -> Tuple[str, str]:
        full_text = join_text_messages(messages)
        template, instruction = extract_template_and_instruction(full_text)
        self.template_text = template
        self.user_instruction = instruction
        return template, instruction

    async def _rewrite_query_for_web_search(
        self,
        original_query: str,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> str:
        """
        Rewrite the query for web search using chat completion.
        """
        rewrite_prompt = get_web_search_query_rewrite_prompt(
            user_query=original_query,
            language=self.language,
        )

        res = await self.model_client.create(
            [UserMessage(content=rewrite_prompt, source="assistant")],
            cancellation_token=cancellation_token,
        )
        rewritten_query = (res.content or "").strip()
        return rewritten_query if rewritten_query else original_query

    # ------------------------- public streaming APIs -------------------------

    async def run_plan(
        self,
        messages: List[TextMessage],
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        template, instruction = await self._prepare_inputs_from_messages(messages)

        yield ToolInputStart(toolName="templatePlan")
        yield ToolInputAvailable(
            input={
                "type": "templatePlan-input",
                "instruction": self.user_instruction or "",
                "templatePreview": self.template_text or "",
            }
        )

        prompt = get_template_based_planning_prompt(
            template=template, user_input=instruction
        )
        res = await self.model_client.create(
            [UserMessage(content=prompt, source="user")],
            cancellation_token=cancellation_token,
        )
        content = (res.content or "").strip()
        self.plan = decode_plan_from_json_like(content)

        yield ToolOutputAvailable(
            output={
                "type": "templatePlan-output",
                "data": {
                    "steps": [
                        s.__dict__ for s in (self.plan.steps if self.plan else [])
                    ]
                },
            }
        )

    async def run_steps_retrieval(
        self,
        *,
        workspace_id: str,
        knowledge_base_id: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        if not self.plan:
            return

        self.step_chunks = {}
        self.step_queries = {}
        self.step_web_search_snippets = {}
        self.step_web_search_queries = {}

        queue: asyncio.Queue = asyncio.Queue()
        done_sentinel = object()
        total = len(self.plan.steps)

        # TODO: now directly use description as query, much space to optimize
        async def worker(step: PlanStep):
            query_text = step.description or step.module
            self.step_queries[step.module] = query_text

            await queue.put(ToolInputStart(toolName="ragSearch"))
            await queue.put(
                ToolInputAvailable(
                    input={
                        "type": "ragSearch-input",
                        "module": step.module,
                        "query": query_text,
                    }
                )
            )

            rag_task = asyncio.create_task(
                execute_remote_function(
                    "HI_RAG",
                    {
                        "function_id": "hi_rag_query",
                        "language": self.language,
                        "query": query_text,
                        "workspace_id": workspace_id,
                        "knowledge_base_id": knowledge_base_id,
                        "translation": ["en", "zh", "zh-t-hk"],
                        "summary": False,
                        "filter_by_clustering": False,
                    },
                )
            )
            if cancellation_token is not None:
                cancellation_token.link_future(rag_task)

            web_search_query_rewrite_task = asyncio.create_task(
                self._rewrite_query_for_web_search(
                    query_text, cancellation_token=cancellation_token
                )
            )
            if cancellation_token is not None:
                cancellation_token.link_future(web_search_query_rewrite_task)

            try:
                rewritten_query = await web_search_query_rewrite_task

                await queue.put(ToolInputStart(toolName="webSearch"))
                await queue.put(
                    ToolInputAvailable(
                        input={
                            "type": "webSearch-input",
                            "module": step.module,
                            "query": rewritten_query,
                        }
                    )
                )

                tavily_client = get_web_search_service()
                web_search_task = asyncio.create_task(
                    tavily_client.search(
                        query=rewritten_query,
                        search_depth="basic",
                        topic="general",
                        include_answer=True,
                        include_raw_content=False,
                        max_results=5,
                    )
                )
                if cancellation_token is not None:
                    cancellation_token.link_future(web_search_task)

                rag_done = False
                web_done = False
                while not (rag_done and web_done):
                    done, _ = await asyncio.wait(
                        {rag_task, web_search_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if rag_task in done and not rag_done:
                        ret = await rag_task
                        chunks = ret.get("chunks", []) or []
                        self.step_chunks[step.module] = chunks

                        items = list(
                            {
                                "fileName": c.get("fileName"),
                                "fileUrl": c.get("uri"),
                                "type": (c.get("uri") or "").split(".")[-1],
                            }
                            for c in chunks[:12]
                        )

                        await queue.put(
                            ToolOutputAvailable(
                                output={
                                    "type": "ragSearch-output",
                                    "module": step.module,
                                    "data": items,
                                }
                            )
                        )
                        rag_done = True

                    if web_search_task in done and not web_done:
                        try:
                            web_search_response = await web_search_task

                            web_search_answer = ""
                            if hasattr(web_search_response, "answer"):
                                web_search_answer = web_search_response.answer or ""
                            elif isinstance(web_search_response, dict):
                                web_search_answer = (
                                    web_search_response.get("answer", "") or ""
                                )

                            self.step_web_search_queries[step.module] = rewritten_query
                            if web_search_answer:
                                self.step_web_search_snippets[step.module] = [
                                    web_search_answer
                                ]

                            await queue.put(
                                ToolOutputAvailable(
                                    output={
                                        "type": "webSearch-output",
                                        "module": step.module,
                                        "data": {
                                            "query": rewritten_query,
                                            "answer": web_search_answer,
                                        },
                                    }
                                )
                            )
                        except Exception as e:
                            self.step_web_search_queries[step.module] = rewritten_query
                            self.step_web_search_snippets[step.module] = []
                            await queue.put(
                                ToolOutputAvailable(
                                    output={
                                        "type": "webSearch-output",
                                        "module": step.module,
                                        "data": {
                                            "query": rewritten_query,
                                            "answer": "",
                                            "error": str(e),
                                        },
                                    }
                                )
                            )
                        web_done = True

            except asyncio.CancelledError:
                rag_task.cancel()
                web_search_query_rewrite_task.cancel()
                try:
                    web_search_task.cancel()
                except Exception:
                    pass
                raise

            await queue.put(done_sentinel)

        tasks = [asyncio.create_task(worker(s)) for s in self.plan.steps]
        finished = 0
        try:
            while finished < total:
                evt = await queue.get()
                if evt is done_sentinel:
                    finished += 1
                else:
                    yield evt
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    def run_generate(
        self,
        *,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Any, None]:
        messages = self._build_generation_messages()

        # Optional debug dump (see Sagi/workflows/utils.py)
        try:
            dump_generation_messages(
                messages,
                chat_id=getattr(self.memory, "chat_id", None),
                agent=self.__class__.__name__,
            )
        except Exception:
            pass

        async def _stream():
            async for chunk in self.model_client.create_stream(
                messages, cancellation_token=cancellation_token
            ):
                if isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(
                        content=chunk, source="assistant"
                    )

        return _stream()
