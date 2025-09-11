from typing import Any, Awaitable, Callable, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import BaseTool

from Sagi.workflows.sagi_memory import SagiMemory


class MultiRoundAgent:
    agent: AssistantAgent
    language: str
    memory: SagiMemory

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        model_client_stream: bool = True,
        tools: Optional[List[BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]]] = None,
    ):
        self.memory = memory
        self.language = language

        # Update system prompt based on whether web search tools are provided
        if self._has_web_search_tools(tools):
            system_prompt = self._get_web_search_system_prompt()
        else:
            system_prompt = self._get_system_prompt()
        
        self.agent = AssistantAgent(
            name="multi_round_agent",
            model_client=model_client,
            model_client_stream=model_client_stream,
            memory=[memory],
            system_message=system_prompt,
            tools=tools,
        )

    def _get_system_prompt(self):
        system_prompt = {
            "en": "You are a helpful assistant that can answer questions and help with tasks. Please use English to answer.",
            "cn-s": "你是一个乐于助人的助手，可以回答问题并帮助完成任务。请用简体中文回答",
            "cn-t": "你是一個樂於助人的助手，可以回答問題並幫助完成任務。請用繁體中文回答",
        }
        return system_prompt.get(self.language, system_prompt["en"])

    def _get_web_search_system_prompt(self):
        """Get system prompt with web search capabilities"""
        base_prompt = self._get_system_prompt()
        web_search_addition = {
            "en": " You have access to web search tools to find current information when needed.",
            "cn-s": " 你可以使用网络搜索工具来查找所需的最新信息。",
            "cn-t": " 你可以使用網路搜尋工具來查找所需的最新資訊。",
        }
        addition = web_search_addition.get(self.language, web_search_addition["en"])
        return base_prompt + addition

    def _has_web_search_tools(self, tools):
        if not tools:
            return False
        
        for tool in tools:
            tool_name = getattr(tool, 'name', '') or getattr(tool, '__name__', '')
            if 'search' in tool_name.lower() or 'brave' in tool_name.lower():
                return True
        return False

    def run_workflow(
        self,
        user_input: str,
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):
        # TODO(klma): handle the case of experimental_attachments
        return self.agent.run_stream(
            task=user_input,
        )

    async def cleanup(self):
        pass