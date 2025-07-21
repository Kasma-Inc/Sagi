import json
from typing import Dict, List, Optional, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_core.models import ChatCompletionClient
from autogen_ext.tools.mcp._sse import SseMcpToolAdapter
from autogen_ext.tools.mcp._stdio import StdioMcpToolAdapter

from Sagi.workflows.sagi_memory import SagiMemory


class HiragAgent:
    agent: AssistantAgent
    mcp_tools: List[StdioMcpToolAdapter | SseMcpToolAdapter]
    language: str
    memory: SagiMemory

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        mcp_tools: List[StdioMcpToolAdapter | SseMcpToolAdapter],
        language: str,
        model_client_stream: bool = True,
    ):

        self.memory = memory
        self.language = language
        self.mcp_tools = mcp_tools

        system_prompt = self._get_system_prompt()

        self.agent = AssistantAgent(
            name="hirag_agent",
            model_client=model_client,
            model_client_stream=True,
            memory=[memory],
            system_message=system_prompt,
            tools=self.mcp_tools,
        )

    def _get_system_prompt(self):
        system_prompt = {
            "en": "You are a information retrieval agent that provides relevant information from the internal database.",
            "cn-s": "你是一个信息检索代理，从内部数据库中提供相关信息。",
            "cn-t": "你是一個信息檢索代理，從內部資料庫中提供相關信息。",
        }
        return system_prompt.get(self.language, system_prompt["en"])

    def run_workflow(
        self,
        user_input: str,
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):
        # TODO(klma): handle the case of experimental_attachments
        response = self.agent.run_stream(
            task=user_input,
        )
        return response

    async def cleanup(self):
        pass

    def message_to_memory_content(
        self,
        message: Union[
            TextMessage,
            ToolCallRequestEvent,
            ToolCallExecutionEvent,
            ToolCallSummaryMessage,
        ],
    ) -> str:
        if isinstance(message, TextMessage):
            # This is the message from the user
            return message.content
        elif isinstance(message, ToolCallRequestEvent):
            # function call name and arguments
            function_call_name = message.content[0].name
            function_call_args = message.content[0].arguments
            return json.dumps(
                {
                    "name": function_call_name,
                    "args": function_call_args,
                }
            )
        elif isinstance(message, ToolCallExecutionEvent):
            # function call name and arguments
            result = json.loads(json.loads(message.content[0].content)[0]["text"])
            entity_fields = ["text", "entity_type", "description", "_relevance_score"]
            entities = [
                {k: v for k, v in e.items() if k in entity_fields}
                for e in result["entities"]
            ]

            # chunks
            chunk_fields = ["text", "_relevance_score"]
            chunks = [
                {k: v for k, v in c.items() if k in chunk_fields}
                for c in result["chunks"]
            ]

            # summary
            summary = result["summary"]

            # relations
            relations = [
                r.get("properties", {}).get("description") for r in result["relations"]
            ]

            # neighbors
            neighbors = [
                {
                    "text": n.get("page_content"),
                    "entity_type": n.get("metadata", {}).get("entity_type"),
                    "_relevance_score": n.get("metadata", {}).get("description"),
                }
                for n in result["neighbors"]
            ]

            return json.dumps(
                {
                    "entities": entities,
                    "chunks": chunks,
                    "summary": summary,
                    "relations": relations,
                    "neighbors": neighbors,
                }
            )
        elif isinstance(message, ToolCallSummaryMessage):
            return message.content
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
