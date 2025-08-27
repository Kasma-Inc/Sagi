from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from hirag_prod.parser import (
    DictParser,
)
from hirag_prod.prompt import PROMPTS
from resources.functions import get_hi_rag_client, get_settings

from Sagi.workflows.sagi_memory import SagiMemory


class RagSummaryAgent:
    agent: AssistantAgent
    language: str
    memory: SagiMemory
    gdb_path: str
    model_client_stream: bool

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        gdb_path: str,
        model_client_stream: bool = True,
    ):
        self.memory = memory
        self.language = language
        self.rag_instance = None
        self.system_prompt = None
        self.rag_summary_agent = None
        self.model_client = model_client
        self.model_client_stream = model_client_stream
        self.memory = memory
        self.vdb_path = get_settings().postgres_url_async
        self.gdb_path = gdb_path

    @classmethod
    async def create(
        cls,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        gdb_path: str,
        model_client_stream: bool = True,
    ):
        self = cls(
            model_client=model_client,
            memory=memory,
            language=language,
            gdb_path=gdb_path,
            model_client_stream=model_client_stream,
        )
        self.rag_instance = await get_hi_rag_client()
        await self.rag_instance.set_language(language)
        return self

    def _init_rag_summary_agent(self):
        self.rag_summary_agent = AssistantAgent(
            name="rag_summary_agent",
            model_client=self.model_client,
            model_client_stream=self.model_client_stream,
            memory=[self.memory],
            system_message=self.system_prompt,
        )

    def set_system_prompt(self, chunks: List[Dict[str, Any]]):
        raw_prompt = PROMPTS["summary_all_" + self.language]
        placeholder = PROMPTS["REFERENCE_PLACEHOLDER"]
        parser = DictParser()
        retrieved_chunks = (
            "Chunks:\n" + parser.parse_list_of_dicts(chunks, "table") + "\n\n"
        )
        system_prompt = raw_prompt.format(
            data=retrieved_chunks,
            max_report_length=5000,
            reference_placeholder=placeholder,
        )
        self.system_prompt = system_prompt

    async def run_workflow(
        self,
        user_input: str,
        workspace_id: str,
        knowledge_base_id: str,
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):
        ret = await self.rag_instance.query(
            user_input,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            summary=False,
        )
        self.set_system_prompt(ret["chunks"])
        self._init_rag_summary_agent()
        return ret, self.rag_summary_agent.run_stream(task=user_input)

    async def cleanup(self):
        pass
