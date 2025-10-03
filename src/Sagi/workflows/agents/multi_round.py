from typing import Any, Awaitable, Callable, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import BaseTool
from resources.model_client_wrapper import ModelClientWrapper

from Sagi.tools.pdf_extraction.pdf_extraction_tool import PDFExtractionTool
from Sagi.tools.web_search_agent import WebSearchAgent
from Sagi.utils.prompt import (
    get_multi_round_agent_system_prompt,
    get_multi_round_pdf_extraction_agent_prompt,
    get_multi_round_web_search_agent_prompt,
    get_web_search_summary_prompt,
)
from Sagi.workflows.sagi_memory import SagiMemory


class MultiRoundAgentWorkflow:
    @classmethod
    def create(
        cls,
        model_name: str,
        memory: SagiMemory,
        language: str,
        mcp_tools: Dict[
            str,
            List[
                BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]
            ],
        ],
        markdown_output: bool = False,
        is_search: bool = True,
        pdf_extraction: bool = True,
        workspace_id: str | None = None,
        knowledge_base_id: str | None = None,
    ):
        self = cls()
        self.model_client = ModelClientWrapper(model_name, None)
        self.language = language
        self.participant_list = []

        # 1. Web search agent - responsible for searching web content and finding PDF links
        if is_search:
            web_search_agent: WebSearchAgent = WebSearchAgent(
                name="web_search_agent",
                model_client=self.model_client,
                tools=mcp_tools.get("web_search", []),
                system_message=get_multi_round_web_search_agent_prompt(),
                model_client_stream=True,
                max_retries=3,
            )
            self.participant_list.append(web_search_agent)

        # 2. PDF extraction agent - responsible for parsing PDF content
        if pdf_extraction:
            pdf_extraction_agent: AssistantAgent = AssistantAgent(
                name="pdf_extraction_agent",
                model_client=self.model_client,
                tools=[
                    PDFExtractionTool(
                        workspace_id=workspace_id, knowledge_base_id=knowledge_base_id
                    )
                ],
                model_client_stream=False,
                system_message=get_multi_round_pdf_extraction_agent_prompt(),
            )
            self.participant_list.append(pdf_extraction_agent)

        # 3. Multi-round agent - responsible for summarizing and answering
        if is_search:
            system_prompt = get_web_search_summary_prompt(
                language if language in {"en", "zh", "cn-s", "cn-t"} else "en"
            )
        else:
            lang_prompt = {
                "en": "You are a helpful assistant that can answer questions and help with tasks. Please use English to answer.",
                "cn-s": "你是一个乐于助人的助手, 可以回答问题并帮助完成任务。请用简体中文回答",
                "cn-t": "你是一個樂於助人的助手, 可以回答問題並幫助完成任務。請用繁體中文回答",
            }
            if markdown_output:
                markdown_prompt = get_multi_round_agent_system_prompt()
                system_prompt = markdown_prompt.get(language, markdown_prompt["en"])
            else:
                system_prompt = lang_prompt.get(language, lang_prompt["en"])

        multi_round_agent = AssistantAgent(
            name="multi_round_agent",
            model_client=self.model_client,
            model_client_stream=True,
            memory=[memory],
            system_message=system_prompt,
        )
        self.participant_list.append(multi_round_agent)

        self.team = RoundRobinGroupChat(
            participants=self.participant_list,
            termination_condition=TextMessageTermination(source="multi_round_agent"),
        )

        return self

    def set_model(self, model_name: str) -> None:
        self.model_client.set_model_client(model_name=model_name)

    def reset(self):
        pass

    async def cleanup(self):
        if hasattr(self, "session_manager"):
            await self.session_manager.close_all()

    def run_workflow(self, user_input):
        return self.team.run_stream(task=user_input)
