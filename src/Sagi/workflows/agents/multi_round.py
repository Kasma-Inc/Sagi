from typing import Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from Sagi.utils.prompt import get_multi_round_agent_system_prompt
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
    ):

        self.memory = memory
        self.language = language

        system_prompt = self._get_system_prompt(markdown_output)
        self.agent = AssistantAgent(
            name="multi_round_agent",
            model_client=model_client,
            model_client_stream=model_client_stream,
            memory=[memory],
            system_message=system_prompt,
        )

    def _get_system_prompt(self, markdown_output=False):
        lang_prompt = {
            "en": "You are a helpful assistant that can answer questions and help with tasks. Please use English to answer.",
            "cn-s": "你是一个乐于助人的助手, 可以回答问题并帮助完成任务。请用简体中文回答",
            "cn-t": "你是一個樂於助人的助手, 可以回答問題並幫助完成任務。請用繁體中文回答",
        }
        if markdown_output:
            markdown_prompt = get_multi_round_agent_system_prompt()
            return markdown_prompt.get(self.language, markdown_prompt["en"])
        return lang_prompt.get(self.language, lang_prompt["en"])

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