from typing import Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from Sagi.workflows.sagi_memory import SagiMemory


class FileEditAgent:
    agent: AssistantAgent
    language: str
    memory: SagiMemory

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        model_client_stream: bool = True,
    ):

        self.memory = memory
        self.language = language

        system_prompt = self._get_system_prompt()
        self.agent = AssistantAgent(
            name="file_edit_agent",
            model_client=model_client,
            model_client_stream=model_client_stream,
            memory=[memory],
            system_message=system_prompt,
        )

    def _get_system_prompt(self):
        system_prompt = {
            "en": "You are a file editor assistant. You help users modify files based on their instructions. You will receive a file input path, highlighted text, and user instructions. Your task is to generate the appropriate edit for the highlighted text according to the user's instructions.",
            "cn-s": "你是一个文件编辑助手。你帮助用户根据他们的指示修改文件。你将收到文件输入路径、高亮文本和用户指令。你的任务是根据用户的指令为高亮文本生成适当的编辑。",
            "cn-t": "你是一個文件編輯助手。你幫助用戶根據他們的指示修改文件。你將收到文件輸入路徑、高亮文本和用戶指令。你的任務是根據用戶的指令為高亮文本生成適當的編輯。",
        }
        return system_prompt.get(self.language, system_prompt["en"])

    def run_workflow(
        self,
        file_input: str,
        highlight_text: str,
        user_instruction: str,
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):

        task_description = (
            f"File Input: {file_input}\n"
            f"Highlighted Text: {highlight_text}\n"
            f"User Instruction: {user_instruction}\n"
            "Please modify the highlighted text section according to the user's instruction and provide ONLY the revised content without any additional explanation."
        )

        return self.agent.run_stream(
            task=task_description,
        )

    async def cleanup(self):
        pass
