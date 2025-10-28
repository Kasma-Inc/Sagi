import asyncio
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, UserMessage
from hirag_prod.entity.vanilla import json_repair
from resources.functions import get_hi_rag_client

from Sagi.utils.prompt import (
    get_template_based_generation_prompt,
    get_template_based_planning_prompt,
)
from Sagi.vercel import ToolInputAvailable, ToolInputStart, ToolOutputAvailable
from Sagi.workflows.sagi_memory import SagiMemory


@dataclass
class PlanStep:
    module: str
    description: str


@dataclass
class Plan:
    steps: List[PlanStep]


class TemplateAgent:
    """
    Lightweight pipeline (no AssistantAgent) for template-based writing:
      1) Extract template and instruction from user messages
      2) Plan per-module TODO list (emit Tool events)
      3) Run per-module RAG retrieval concurrently (emit Tool events)
      4) Stream final markdown generation following template structure
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
        self.step_queries: Dict[str, str] = {}

        self._rag = get_hi_rag_client()

    # ------------------------- helpers -------------------------
    @staticmethod
    def _extract_template_and_instruction(text: str) -> Tuple[str, str]:
        """
        Prefer extracting <template>...</template> and <user_input>...</user_input>.
        Fallback to fenced code block for template and use remaining text as instruction.
        """
        tag = lambda name: re.search(
            rf"<\s*{name}\s*>([\s\S]*?)<\s*/\s*{name}\s*>", text, flags=re.IGNORECASE
        )
        mt = tag("template")
        mi = tag("user_input")
        template = (mt.group(1) or "").strip()
        instruction = (mi.group(1) or "").strip()
        return template, instruction

    @staticmethod
    def _join_messages(messages: List[TextMessage]) -> str:
        return "\n\n".join(m.content for m in messages if getattr(m, "content", ""))

    def _build_generation_messages(self) -> List[UserMessage]:
        per_module_ctx: List[str] = []
        for module, chunks in self.step_chunks.items():
            snippet_texts: List[str] = []
            for c in chunks[:8]:
                t = (c.get("text") or "").strip()
                if t:
                    snippet_texts.append(t)
            block = f"## {module}\n" + (
                "\n".join(f"- {s}" for s in snippet_texts)
                if snippet_texts
                else "- (no retrieval)"
            )
            per_module_ctx.append(block)
        ctx = "\n\n".join(per_module_ctx)

        # Build plan overview to guide headings and structure
        plan_lines: List[str] = []
        plan_json_lines: List[str] = []
        if self.plan and self.plan.steps:
            for step in self.plan.steps:
                plan_lines.append(f"- {step.module}: {step.description}")
                # lightweight JSON-ish to guide model alignment
                plan_json_lines.append(
                    '  {"module": "'
                    + step.module.replace('"', "'")
                    + '", "required_h2_from": "'
                    + step.description.replace('"', "'")
                    + '"}'
                )
        plan_block = "\n".join(plan_lines) if plan_lines else "(none)"
        plan_json_block = (
            "[\n" + ",\n".join(plan_json_lines) + "\n]" if plan_json_lines else "[]"
        )
        module_queries_lines: List[str] = []
        for m, q in self.step_queries.items():
            module_queries_lines.append(f"- {m}: {q}")
        module_queries_block = "\n".join(module_queries_lines)

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
            content=f""""
            <USER_ORIGINAL_QUERY>
            {self.user_instruction.strip()}
            </USER_ORIGINAL_QUERY>
        """.strip(),
        )
        return [sys, user]

    async def _prepare_inputs_from_messages(
        self, messages: List[TextMessage]
    ) -> Tuple[str, str]:
        full_text = self._join_messages(messages)
        template, instruction = self._extract_template_and_instruction(full_text)
        self.template_text = template
        self.user_instruction = instruction
        return template, instruction

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

        plan_decoded_obj = json_repair.repair_json(content, return_objects=True) or {
            "steps": []
        }
        steps: List[PlanStep] = []
        for s in plan_decoded_obj.get("steps", []) or []:
            module = (s.get("module") or "").strip()
            desc = (s.get("description") or "").strip()
            if module:
                steps.append(PlanStep(module=module, description=desc))
        self.plan = Plan(steps=steps)

        yield ToolOutputAvailable(
            output={
                "type": "templatePlan-output",
                "data": {"steps": [s.__dict__ for s in steps]},
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

        await self._rag.set_language(self.language)
        self.step_chunks = {}
        self.step_queries = {}

        queue: asyncio.Queue = asyncio.Queue()
        done_sentinel = object()
        total = len(self.plan.steps)

        # TODO: now directly use description as query, much space to optimize
        async def worker(step: PlanStep):
            await queue.put(ToolInputStart(toolName="ragSearch"))
            await queue.put(
                ToolInputAvailable(
                    input={
                        "type": "ragSearch-input",
                        "module": step.module,
                        "query": step.description or step.module,
                    }
                )
            )

            query_text = step.description or step.module
            self.step_queries[step.module] = query_text
            rag_task = asyncio.create_task(
                self._rag.query(
                    query_text,
                    workspace_id=workspace_id,
                    knowledge_base_id=knowledge_base_id,
                    translation=["en", "zh", "zh-t-hk"],
                    summary=False,
                    filter_by_clustering=False,
                )
            )
            if cancellation_token is not None:
                cancellation_token.link_future(rag_task)
            try:
                ret = await rag_task
            except asyncio.CancelledError:
                rag_task.cancel()
                raise

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

        async def _stream():
            async for chunk in self.model_client.create_stream(
                messages, cancellation_token=cancellation_token
            ):
                if isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(
                        content=chunk, source="assistant"
                    )

        return _stream()
