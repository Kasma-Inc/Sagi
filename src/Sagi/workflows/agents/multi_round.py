from typing import Any, Awaitable, Callable, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import BaseTool

from Sagi.utils.prompt import get_multi_round_agent_system_prompt
from Sagi.utils.prompt import (
    get_multi_round_agent_base_prompt,
    get_multi_round_agent_web_search_prompt,
)
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
        markdown_output: bool = False,
        tools: Optional[
            List[
                BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]
            ]
        ] = None,
    ):
        self.memory = memory
        self.language = language

        if markdown_output:
            if self._has_web_search_tools(tools):
                system_prompt = self._get_markdown_web_search_system_prompt()
            else:
                system_prompt = self._get_markdown_system_prompt()
        else:
            if self._has_web_search_tools(tools):
                system_prompt = self._get_web_search_system_prompt(tools)
            else:
                system_prompt = self._get_system_prompt()
        self.agent = AssistantAgent(
            name="multi_round_agent",
            model_client=model_client,
            model_client_stream=model_client_stream,
            memory=[memory],
            system_message=system_prompt,
            tools=tools,
            tool_call_summary_format="{result}",
        )

    def _get_system_prompt(self):
        return get_multi_round_agent_base_prompt(self.language)

    def _get_markdown_system_prompt(self):
        markdown_prompt = get_multi_round_agent_system_prompt()
        return markdown_prompt.get(self.language, markdown_prompt["en"])

    def _get_web_search_system_prompt(self, tools):
        base_prompt = self._get_system_prompt()
        has_pdf_tools = self._has_pdf_tools(tools)
        web_search_addition = get_multi_round_agent_web_search_prompt(
            self.language, has_pdf_tools
        )
        return base_prompt + web_search_addition

    def _get_markdown_web_search_system_prompt(self):
        base_prompt = self._get_markdown_system_prompt()
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
            tool_name = getattr(tool, "name", "") or getattr(tool, "__name__", "")
            if "search" in tool_name.lower() or "brave" in tool_name.lower():
                return True
        return False

    def _has_pdf_tools(self, tools):
        if not tools:
            return False

        for tool in tools:
            tool_name = getattr(tool, "name", "") or getattr(tool, "__name__", "")
            if "pdf" in tool_name.lower() or "extractor" in tool_name.lower():
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
