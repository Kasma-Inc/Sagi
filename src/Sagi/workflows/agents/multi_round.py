from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from Sagi.workflows.sagi_memory import SagiMemory


class MultiRoundAgent:

    @classmethod
    async def create(
        cls,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        model_client_stream: bool = True,
        mcp_tools: Optional[Dict[str, List[Any]]] = None,
        web_search: bool = False,
        hirag: bool = False,
    ):
        self = cls()
        self.memory = memory
        self.language = language

        all_tools = []

        if web_search and mcp_tools and "web_search" in mcp_tools:
            all_tools.extend(mcp_tools["web_search"])

        if hirag and mcp_tools and "hirag_retrieval" in mcp_tools:
            all_tools.extend(mcp_tools["hirag_retrieval"])

        system_prompt = self._get_enhanced_system_prompt(web_search, hirag)

        if web_search:
            from Sagi.tools.web_search_agent import WebSearchAgent

            self.agent = WebSearchAgent(
                name="multi_round_agent",
                model_client=model_client,
                model_client_stream=model_client_stream,
                memory=[memory],
                system_message=system_prompt,
                tools=all_tools,
                max_retries=3,
                enable_knowledge_integration=hirag,
            )
        else:
            self.agent = AssistantAgent(
                name="multi_round_agent",
                model_client=model_client,
                model_client_stream=model_client_stream,
                memory=[memory],
                system_message=system_prompt,
                tools=all_tools if all_tools else None,
            )

        self.team = None
        return self

    def _get_enhanced_system_prompt(self, web_search: bool, hirag: bool) -> str:
        """Get enhanced system prompt based on enabled capabilities."""
        base_prompt = self._get_system_prompt()

        enhancements = []
        if web_search:
            enhancements.append(
                "You have web search capabilities with PDF processing and version retrieval."
            )
        if hirag:
            enhancements.append("You have access to knowledge retrieval tools.")

        if enhancements:
            return f"{base_prompt}\n\nEnhanced capabilities: {' '.join(enhancements)}"
        return base_prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt based on language."""
        system_prompt = {
            "en": "You are a helpful assistant that can answer questions and help with tasks. Please use English to answer.",
            "cn-s": "你是一个乐于助人的助手，可以回答问题并帮助完成任务。请用简体中文回答",
            "cn-t": "你是一個樂於助人的助手，可以回答問題並幫助完成任務。請用繁體中文回答",
        }
        return system_prompt.get(self.language, system_prompt["en"])

    def run_workflow(
        self,
        user_input: str,
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):
        # TODO(klma): handle the case of experimental_attachments

        if hasattr(self, "team") and self.team is not None:
            return self.team.run_stream(
                task=user_input,
            )
        else:
            return self.agent.run_stream(
                task=user_input,
            )

    async def cleanup(self):
        if self.agent and hasattr(self.agent, "cleanup"):
            await self.agent.cleanup()
