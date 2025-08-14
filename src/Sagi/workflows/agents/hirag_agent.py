from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from hirag_prod import HiRAG
from hirag_prod.parser import (
    DictParser,
)
from hirag_prod.prompt import PROMPTS

from Sagi.workflows.sagi_memory import SagiMemory


class RagSummaryAgent:
    agent: AssistantAgent
    language: str
    memory: SagiMemory
    vdb_path: str
    gdb_path: str
    model_client_stream: bool

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        vdb_path: str,
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
        self.vdb_path = vdb_path
        self.gdb_path = gdb_path

    @classmethod
    async def create(
        cls,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        vdb_path: str,
        gdb_path: str,
        model_client_stream: bool = True,
    ):
        self = cls(
            model_client=model_client,
            memory=memory,
            language=language,
            vdb_path=vdb_path,
            gdb_path=gdb_path,
            model_client_stream=model_client_stream,
        )
        self.rag_instance = await HiRAG.create(
            vector_db_path=vdb_path, graph_db_path=gdb_path
        )
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
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):
        ret = await self.rag_instance.query(user_input, summary=False)
        self.set_system_prompt(ret["chunks"])
        self._init_rag_summary_agent()
        return self.rag_summary_agent.run_stream(task=user_input)

    async def cleanup(self):
        pass
