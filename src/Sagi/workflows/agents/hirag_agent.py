import uuid
from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from hirag_prod.parser import DictParser
from hirag_prod.prompt import PROMPTS
from resources.functions import get_hi_rag_client, get_settings

from Sagi.utils.prompt import get_multi_round_agent_system_prompt
from Sagi.vercel import (
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
        self.memory = memory
        self.vdb_path = get_settings().postgres_url_async
        self.gdb_path = gdb_path
        self.ret: Optional[Dict[str, Any]] = None
        self.tool_name = "ragSearch"
        self.markdown_output = markdown_output

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
            memory=[self.memory],
            system_message=self.system_prompt,
        )

    def set_system_prompt(self, user_query: str, chunks: List[Dict[str, Any]]):
        raw_prompt = PROMPTS["summary_all_" + self.language]
        placeholder = PROMPTS["REFERENCE_PLACEHOLDER"]
        parser = DictParser()
        clean_chunks = [
            {"id": i, "chunk": " ".join((c.get("text", "") or "").split())}
            for i, c in enumerate(chunks, start=1)
        ]
        data = "Chunks\n" + parser.parse_list_of_dicts(clean_chunks, "table") + "\n\n"
        system_prompt = raw_prompt.format(
            user_query=user_query,
            data=data,
            max_report_length=5000,
            reference_placeholder=placeholder,
        )
        if self.markdown_output:
            markdown_prompt = self._get_markdown_system_prompt()
            system_prompt = system_prompt + "\n" + markdown_prompt
        self.system_prompt = system_prompt

    def _get_markdown_system_prompt(self):
        markdown_prompt = get_multi_round_agent_system_prompt()
        return markdown_prompt.get(self.language, markdown_prompt["en"])

    async def run_query(
        self,
        user_input: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> None:
        """Run a query through the RAG system and prepare the summary agent.

        Args:
            user_input (str): The user's input query.
            workspace_id (str): The ID of the workspace.
            knowledge_base_id (str): The ID of the knowledge base.

        Yields:
            Tool input and output events for the query process.
        """
        tool_id = str(uuid.uuid4())
        try:
            yield ToolInputStart(toolName=self.tool_name)
            yield ToolInputAvailable(
                input=RagSearchToolCallInput(query=user_input).to_dict(),
            )

            ret = await self.rag_instance.query(
                user_input,
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                summary=False,
                translation=["en", "zh-TW", "zh"],
            )
            yield ToolOutputAvailable(
                output=RagSearchToolCallOutput(
                    data=[
                        RagSearchToolCallOutputItem(
                            fileName=chunk["fileName"],
                            fileUrl=chunk["uri"],
                            type=chunk["uri"].split(".")[-1],
                        )
                        for chunk in ret["chunks"]
                    ]
                ).to_dict(),
            )

            self.set_system_prompt(user_input, ret["chunks"])
            self._init_rag_summary_agent()
            self.ret = ret
        except Exception as e:
            yield ToolOutputAvailable(
                toolName=self.tool_name,
                toolCallId=tool_id,
                output={"error": f"Query failed: {str(e)}"},
            )

    def run_workflow(self, user_input: str) -> tuple[Optional[Dict[str, Any]], Any]:
        """Run the full workflow for processing a user query.

        Args:
            user_input (str): The user's input query.

        Returns:
            tuple: The query results and the summary agent's streaming output.
        """
        if not self.rag_summary_agent:
            raise RuntimeError("RAG summary agent not initialized.")
        if not self.ret:
            raise RuntimeError("Query results are not available.")
        return self.ret, self.rag_summary_agent.run_stream(task=user_input)

    async def cleanup(self):
        pass
