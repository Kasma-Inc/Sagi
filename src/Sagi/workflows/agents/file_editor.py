from typing import Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient, UserMessage
from resources.functions import get_hi_rag_client

from Sagi.utils.prompt import get_file_edit_system_prompt, get_file_edit_task_prompt
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

        return get_file_edit_system_prompt(language=self.language)

    async def run_workflow(
        self,
        file_input: str,
        highlight_text: str,
        user_instruction: str,
        workspace_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):

        # Check if RAG retrieval is needed using LLM
        check_prompt = f"Based on this user instruction, does it require searching a knowledge base for additional context? Answer only YES or NO.\n\nUser instruction: {user_instruction}"
        check_message = UserMessage(content=check_prompt, source="system")

        result = await self.agent._model_client.create([check_message])
        is_need_rag_retrieval = "YES" in result.content.upper()

        rag_context = ""

        if is_need_rag_retrieval and workspace_id and knowledge_base_id:
            rag_instance = get_hi_rag_client()
            await rag_instance.set_language(self.language)

            ret = await rag_instance.query(
                highlight_text,
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                summary=False,
                translation=["en", "zh-TW", "zh"],
            )

            chunks = ret.get("chunks", [])
            if chunks:
                rag_context = "\n".join(
                    f"[{i}] {chunk.get('text', '')}"
                    for i, chunk in enumerate(chunks, start=1)
                )

        final_task_description = get_file_edit_task_prompt(
            file_input=file_input,
            highlight_text=highlight_text,
            user_instruction=user_instruction,
            rag_context=rag_context,
            language=self.language,
        )

        return self.agent.run_stream(
            task=final_task_description,
        )

    async def cleanup(self):
        pass
