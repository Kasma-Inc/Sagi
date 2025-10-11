import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from api.ui.utils import chunks_to_reference_chunks
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from hirag_prod import HiRAG
from hirag_prod.prompt import PROMPTS
from resources.functions import get_hi_rag_client, get_settings

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
        self.memory = memory
        self.vdb_path = get_settings().postgres_url_async
        self.gdb_path = gdb_path
        self.ret: Optional[Dict[str, Any]] = None
        self.search_tool_name = "ragSearch"
        self.filter_tool_name = "ragFilter"
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
        raw_prompt = PROMPTS["summary_plus_" + self.language]
        data = "- Retrieved Chunks:\n" + "\n".join(
            f"    [{i}] {' '.join((c.get('text', '') or '').split())}"
            for i, c in enumerate(chunks, start=1)
        )
        system_prompt = raw_prompt.format(
            user_query=user_query,
            data=data,
        )
        if self.markdown_output:
            system_prompt = PROMPTS["summary_plus_markdown_" + self.language].format(
                user_query=user_query,
                data=data,
            )
        self.system_prompt = system_prompt

    async def run_query(
        self,
        user_input: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> AsyncGenerator[Any, None]:
        """Run a query through the RAG system and prepare the summary agent.

        Args:
            user_input (str): The user's input query.
            workspace_id (str): The ID of the workspace.
            knowledge_base_id (str): The ID of the knowledge base.

        Yields:
            Tool input and output events for the query process.
        """
        try:
            yield ToolInputStart(toolName=self.search_tool_name)
            yield ToolInputAvailable(
                input=RagSearchToolCallInput(query=user_input).to_dict(),
            )

            # Get raw chunks from HiRAG
            ret_raw = await self.rag_instance.query(
                user_input,
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                strategy="raw",
                translation=["en", "zh-TW", "zh"],
                summary=False,
                filter_by_clustering=False,
                threshold=0.0,
            )

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

        except Exception as e:
            raise RuntimeError(f"Query failed: {str(e)}")

    async def run_filter(
        self,
        user_input: str,
        workspace_id: str,
        knowledge_base_id: str,
    ) -> AsyncGenerator[Any, None]:
        """Run the filtering step on raw chunks."""
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
                self.set_system_prompt(user_input, [])
                self._init_rag_summary_agent()
                self.ret = {"chunks": []}
                return

            # Apply hybrid strategy to get final chunks
            ret = await self.rag_instance.apply_strategy_to_chunks(
                chunks_dict=self.raw_chunks,
                strategy="hybrid",
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                filter_by_clustering=True,
            )

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

            self.set_system_prompt(user_input, ret["chunks"])
            self._init_rag_summary_agent()
            self.ret = ret

        except Exception as e:
            raise RuntimeError(f"Filtering failed: {str(e)}")

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
