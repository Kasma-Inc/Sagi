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
            "en": "You are a file editor assistant. You help users modify files based on their instructions. You will receive file content, highlighted text, and user instruction. Your task is to generate the appropriate edit for the highlighted text according to the user's instructions.",
            "cn-s": "你是一个文件编辑助手。你帮助用户根据他们的指示修改文件。你将收到文件内容、高亮文本和用户指令。你的任务是根据用户的指令为高亮文本生成适当的编辑。",
            "cn-t": "你是一個文件編輯助手。你幫助用戶根據他們的指示修改文件。你將收到文件內容、高亮文本和用戶指令。你的任務是根據用戶的指令為高亮文本生成適當的編輯。",
        }
        return system_prompt.get(self.language, system_prompt["en"])

    def _get_task_description_template(self):
        task_template = {
            "en": "File Content: {file_input}\nHighlighted Text: {highlight_text}\nUser Instruction: {user_instruction}\nPlease modify the highlighted text section according to the user's instruction and provide ONLY the revised content without any additional explanation.",
            "cn-s": "文件内容: {file_input}\n高亮文本: {highlight_text}\n用户指令: {user_instruction}\n请根据用户指令修改高亮文本部分, 并只返回修改后的内容, 不要任何额外解释。",
            "cn-t": "文件內容: {file_input}\n高亮文本: {highlight_text}\n用戶指令: {user_instruction}\n請根據用戶指令修改高亮文本部分, 並只返回修改後的內容, 不要任何額外解釋。",
        }
        return task_template.get(self.language, task_template["en"])

    def run_workflow(
        self,
        file_input: str,
        highlight_text: str,
        user_instruction: str,
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):

        task_template = self._get_task_description_template()
        task_description = task_template.format(
            file_input=file_input,
            highlight_text=highlight_text,
            user_instruction=user_instruction,
        )

        return self.agent.run_stream(
            task=task_description,
        )

    async def cleanup(self):
        pass
