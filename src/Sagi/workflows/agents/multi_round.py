from typing import Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

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
            markdown_prompt = {
                "en": "Please format the output in Markdown, using standard Markdown syntax. Enclose the output in a Markdown code block",
                "cn-s": "請將輸出格式設定為Markdown, 並使用標準Markdown語法。輸出內容需包含在Markdown程式碼區塊中",
                "cn-t": "请将输出格式设置为Markdown, 并使用标准Markdown语法。输出内容需包含在Markdown代码块中",
            }
            return lang_prompt.get(
                self.language, lang_prompt["en"]
            ) + markdown_prompt.get(self.language, markdown_prompt["en"])
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
