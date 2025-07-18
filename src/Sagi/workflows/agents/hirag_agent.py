from typing import Dict, List, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from Sagi.workflows.sagi_memory import SagiMemory
from autogen_ext.tools.mcp._sse import SseMcpToolAdapter
from autogen_ext.tools.mcp._stdio import StdioMcpToolAdapter


class HiragAgent:
    agent: AssistantAgent
    mcp_tools: List[StdioMcpToolAdapter | SseMcpToolAdapter]
    language: str
    memory: SagiMemory
    
    def __init__(
        self, 
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        mcp_tools: List[StdioMcpToolAdapter | SseMcpToolAdapter],
        language: str,
        model_client_stream: bool = True,
    ):

        self.memory = memory
        self.language = language
        self.mcp_tools = mcp_tools

        system_prompt = self._get_system_prompt()

        self.agent = AssistantAgent(
            name="hirag_agent",
            model_client=model_client,
            model_client_stream=True,
            memory=[memory],
            system_message=system_prompt,
            # tools=self.mcp_tools,
        )


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