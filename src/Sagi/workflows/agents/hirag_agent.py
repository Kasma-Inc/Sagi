import json
from typing import Dict, List, Optional, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_ext.tools.mcp._sse import SseMcpToolAdapter
from autogen_ext.tools.mcp._stdio import StdioMcpToolAdapter

from Sagi.workflows.sagi_memory import SagiMemory


class HiragAgent:
    agent: AssistantAgent
    mcp_tools: List[StdioMcpToolAdapter | SseMcpToolAdapter]
    language: str
    memory: SagiMemory
    set_language_tool: Optional[StdioMcpToolAdapter | SseMcpToolAdapter] = None
    insert_chat_tool: Optional[StdioMcpToolAdapter | SseMcpToolAdapter] = None
    search_chat_tool: Optional[StdioMcpToolAdapter | SseMcpToolAdapter] = None

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        mcp_tools: List[StdioMcpToolAdapter | SseMcpToolAdapter],
        language: str,
        model_client_stream: bool = True,
        set_language_tool: Optional[StdioMcpToolAdapter | SseMcpToolAdapter] = None,
        insert_chat_tool: Optional[StdioMcpToolAdapter | SseMcpToolAdapter] = None,
        search_chat_tool: Optional[StdioMcpToolAdapter | SseMcpToolAdapter] = None,
    ):

        self.memory = memory
        self.language = language
        self.mcp_tools = mcp_tools
        self.set_language_tool = set_language_tool
        self.insert_chat_tool = insert_chat_tool
        self.search_chat_tool = search_chat_tool

        system_prompt = self._get_system_prompt()

        self.agent = AssistantAgent(
            name="hirag_agent",
            model_client=model_client,
            model_client_stream=True,
            memory=[memory],
            system_message=system_prompt,
            tools=self.mcp_tools,
        )

    async def set_language(self, language: str):
        """Set the language for HiRAG retrieval system."""
        if self.set_language_tool:
            try:
                # Use the MCP tool's run_json method for direct execution
                result = await self.set_language_tool.run_json(
                    {"language": language}, CancellationToken()
                )

                self.language = language
                return f"Language successfully set to {language}: {result}"

            except Exception as e:
                return f"Failed to set language: {e}"
        else:
            # Just update the local language setting
            self.language = language
            return f"Language set to {language} (local only)"

    async def insert_chat_message(self, chat_id: str, role: str, content: str):
        """
        Insert a User / Assistant / Tool message into the chat history.

        Args:
            chat_id: Unique identifier for the chat session
            role: Role of the message sender (user, assistant, tool)
            content: Content of the message

        Returns:
            Success or error message
        """
        # Validate inputs
        if (
            not chat_id
            or not chat_id.strip()
            or not content
            or not content.strip()
            or not role
            or not role.strip()
        ):
            return "Error: chat_id, role, and content cannot be empty"

        # Validate role
        valid_roles = {"user", "assistant", "tool"}
        if role.lower() not in valid_roles:
            return f"Error: role must be one of {valid_roles}"

        if self.insert_chat_tool:
            try:
                # Use the MCP tool's run_json method for direct execution
                result = await self.insert_chat_tool.run_json(
                    {"chat_id": chat_id, "role": role.lower(), "content": content},
                    CancellationToken(),
                )
                return f"Chat message inserted successfully: {result}"

            except Exception as e:
                return f"Error inserting chat message: {e}"
        else:
            return f"Insert chat tool not available - message not stored: chat_id={chat_id}, role={role}"

    async def search_chat_history(
        self, user_query: str, chat_id: str, role: Union[str, List[str]] = None
    ) -> Union[str, dict]:
        """
        Search the chat history for messages related to the user's query.

        Args:
            user_query: The search query to find relevant chat messages
            chat_id: Unique identifier for the chat session
            role: Optional role filter (user, assistant, tool)

        Returns:
            Search results as formatted string or error message
        """
        # Validate inputs
        if (
            not user_query
            or not user_query.strip()
            or not chat_id
            or not chat_id.strip()
        ):
            return "Error: user_query and chat_id cannot be empty"

        # Validate role if provided
        if role and isinstance(role, str):
            role = role.lower()
            if role not in {"user", "assistant", "tool"}:
                return f"Error: role must be one of user, assistant, tool. Invalid role: {role}"
        elif role and isinstance(role, list):
            # Ensure all roles in the list are valid
            for r in role:
                if r.lower() not in {"user", "assistant", "tool"}:
                    return f"Error: role must be one of user, assistant, tool. Invalid role: {r}"
        elif role is not None:
            return "Error: role must be a string or list of strings"

        if self.search_chat_tool:
            try:
                # Prepare parameters for the MCP tool
                params = {"user_query": user_query, "chat_id": chat_id}

                # Add role filter if provided
                if role:
                    params["role"] = role.lower()

                # Use the MCP tool's run_json method for direct execution
                result = await self.search_chat_tool.run_json(
                    params, CancellationToken()
                )

                return result

            except Exception as e:
                return f"Error searching chat history: {e}"
        else:
            return f"Search chat tool not available - cannot search: query='{user_query}', chat_id='{chat_id}'"

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
