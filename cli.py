import argparse
import asyncio
import logging
import os
import threading
import uuid

from autogen_agentchat.messages import BaseMessage, ToolCallSummaryMessage
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from sqlalchemy.ext.asyncio import async_sessionmaker

from Sagi.utils.logging_utils import setup_logging
from Sagi.utils.mcp_utils import MCPSessionManager
from Sagi.utils.message_to_memory import get_memory_type_for_message
from Sagi.utils.model_client import ModelClientFactory
from Sagi.utils.model_info import get_model_name_by_api_provider
from Sagi.utils.queries import create_db_engine, saveChats
from Sagi.workflows.agents.hirag_agent import HiragAgent
from Sagi.workflows.agents.multi_round import MultiRoundAgent
from Sagi.workflows.general.general_chat import GeneralChatWorkflow
from Sagi.workflows.planning.planning import PlanningWorkflow
from Sagi.workflows.planning_html.planning_html import PlanningHtmlWorkflow
from Sagi.workflows.sagi_memory import SagiMemory

# Create logging directory if it doesn't exist
os.makedirs("logging", exist_ok=True)
setup_logging()

DEFAULT_TEAM_CONFIG_PATH = "src/Sagi/workflows/planning/team.toml"
DEFAULT_PLANNING_CONFIG_PATH = "src/Sagi/workflows/planning/config.toml"
DEFAULT_GENERAL_CONFIG_PATH = "src/Sagi/workflows/general/config.toml"
DEFAULT_PLANNING_HTML_CONFIG_PATH = "src/Sagi/workflows/planning_html/config.toml"
DEFAULT_TEAM_PLANNING_HTML_CONFIG_PATH = "src/Sagi/workflows/planning_html/team.toml"


def parse_args():
    parser = argparse.ArgumentParser("Sagi CLI")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--planning_config", default=DEFAULT_PLANNING_CONFIG_PATH)
    parser.add_argument("--general_config", default=DEFAULT_GENERAL_CONFIG_PATH)
    parser.add_argument("--team_config", default=DEFAULT_TEAM_CONFIG_PATH)
    parser.add_argument(
        "--team_html_config", default=DEFAULT_TEAM_PLANNING_HTML_CONFIG_PATH
    )
    parser.add_argument(
        "--trace", action="store_true", help="Enable OpenTelemetry tracing"
    )
    parser.add_argument(
        "--trace_endpoint",
        default="http://localhost:4317",
        help="OpenTelemetry collector endpoint",
    )
    parser.add_argument(
        "--trace_service_name",
        default="sagi_tracer",
        help="Service name for OpenTelemetry tracing",
    )
    parser.add_argument(
        "-s",
        "--session-id",
        type=str,
        help="Specify the session ID to load or save; if not provided, one will be generated automatically.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List existing session IDs and exit.",
    )
    parser.add_argument(
        "--template_work_dir",
        type=str,
        help="Specify the template working directory path",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "deep_research_executor",
            "general",
            "web_search",
            "deep_research_html",
            "multi_rounds",
            "hirag",
        ],
        default="deep_research_executor",
        help="Operation mode: deep_research_executor (deep research with code executor), general (general agent only), web_search (web search only), deep_research_html (deep research with html generator)",
    )

    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "cn"],
        default="en",
        help="Language: en (English), cn (Chinese)",
    )
    parser.add_argument(
        "--planning_html_config",
        type=str,
        default=DEFAULT_PLANNING_HTML_CONFIG_PATH,
        help="Specify the planning html config path",
    )
    return parser.parse_args()


# load env variables
load_dotenv("/chatbot/.env", override=True)


def setup_tracing(endpoint: str, service_name: str):
    """Setup OpenTelemetry tracing based on args."""

    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider = TracerProvider(
            resource=Resource.create({ResourceAttributes.SERVICE_NAME: service_name})
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(service_name)
        logging.info(f"OpenTelemetry tracing enabled, exporting to {endpoint}")
        return tracer
    except Exception as e:
        logging.error(f"Failed to setup tracing: {e}")
        return None


def _default_to_text(self) -> str:
    return getattr(self, "content", repr(self))


BaseMessage.to_text = _default_to_text


async def get_input_async():
    # Get user input without blocking the event loop
    loop = asyncio.get_event_loop()
    future = (
        loop.create_future()
    )  # placeholder for the user input that will be available later

    def _get_input():
        try:
            result = input("User: ")
            loop.call_soon_threadsafe(
                future.set_result, result
            )  # safely schedule the future result in the main loop (notify main loop)
        except Exception as e:
            loop.call_soon_threadsafe(future.set_exception, e)

    threading.Thread(target=_get_input, daemon=True).start()
    return await future


async def main_cmd(args: argparse.Namespace):
    engine = await create_db_engine(os.getenv("POSTGRES_URL_NO_SSL_DEV") or "")
    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    chat_id = str(uuid.uuid4())
    # Save the metadata of the chat
    # TODO(klma): get the metadata from config instead of hardcoding it
    async with session_maker() as session:
        await saveChats(
            session=session,
            chat_id=chat_id,
            title="",
            user_id="cli_dev",
            model_name="gpt-4o-mini",
            model_config={},
            model_client_stream=True,
            system_prompt="You are a helpful assistant.",
            visibility="private",
        )

    # Initialize variables for proper cleanup
    workflow = None
    session_manager = None

    if args.mode == "deep_research_executor":
        workflow = await PlanningWorkflow.create(
            args.planning_config,
            args.team_config,
            template_work_dir=args.template_work_dir,
            language=args.language,
            countdown_timer=40,  # time before the docker container is stopped
        )
    elif args.mode == "general":
        workflow = await GeneralChatWorkflow.create(
            args.general_config,
            web_search=False,
        )
    elif args.mode == "web_search":
        workflow = await GeneralChatWorkflow.create(
            args.general_config,
            web_search=True,
        )
    elif args.mode == "deep_research_html":
        workflow = await PlanningHtmlWorkflow.create(
            args.planning_html_config,
            args.team_html_config,
            language=args.language,
        )
    elif args.mode == "multi_rounds":
        model = "gpt-4o-mini"
        model_name = get_model_name_by_api_provider(
            "aiml",
            model,
        )
        model_client = ModelClientFactory.create_model_client(
            {
                "model": model_name,
                "base_url": os.getenv("OPENAI_BASE_URL"),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_tokens": 16000,
            }
        )
    elif args.mode == "hirag":
        model = "gpt-4o-mini"
        model_name = get_model_name_by_api_provider(
            "aiml",
            model,
        )
        model_name = "gpt-4o"
        model_client = ModelClientFactory.create_model_client(
            {
                "model": model_name,
                "base_url": os.getenv("OPENAI_BASE_URL"),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_tokens": 16000,
            }
        )
        memory = SagiMemory(
            chat_id=chat_id,
            model_name=model,
        )
        memory.set_session_maker(session_maker)

        hirag_server_params = StdioServerParams(
            command="mcp-hirag-tool",
            args=[],
            read_timeout_seconds=100,
            env={
                "LLM_API_KEY": os.getenv("OPENAI_API_KEY"),
                "LLM_BASE_URL": os.getenv("OPENAI_BASE_URL"),
                "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY"),
            },
        )

        session_manager = MCPSessionManager()
        hirag_retrieval = await session_manager.create_session(
            "hirag_retrieval", create_mcp_server_session(hirag_server_params)
        )
        await hirag_retrieval.initialize()
        hirag_retrieval_tools = await mcp_server_tools(
            hirag_server_params, session=hirag_retrieval
        )

        # Set language for HiRAG instance
        hirag_set_language_tool = [
            tool for tool in hirag_retrieval_tools if tool.name == "hi_set_language"
        ]

        hirag_retrieval_tools = [
            tool for tool in hirag_retrieval_tools if tool.name == "hi_search"
        ]

    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    try:
        while True:
            user_input = await get_input_async()
            if user_input.lower() in ("quit", "exit", "q"):
                break

            if args.mode == "multi_rounds":
                memory = SagiMemory(
                    chat_id=chat_id,
                    model_name=model,
                )
                memory.set_session_maker(session_maker)

                workflow = MultiRoundAgent(
                    model_client=model_client,
                    memory=memory,
                    language=args.language,
                )

                chat_history = await Console(workflow.run_workflow(user_input))
                messages = chat_history.messages
                messages = [
                    MemoryContent(
                        content=message.content,
                        mime_type=get_memory_type_for_message(message),
                        metadata={"source": message.source},
                    )
                    for message in messages
                ]
                await memory.add(messages)

            elif args.mode == "hirag":
                memory = SagiMemory(
                    chat_id=chat_id,
                    model_name=model,
                )
                memory.set_session_maker(session_maker)

                workflow = HiragAgent(
                    model_client=model_client,
                    memory=memory,
                    mcp_tools=hirag_retrieval_tools,
                    language=args.language,
                )

                # Set language in HiRAG server if the tool is available
                if hirag_set_language_tool:
                    try:
                        language_result = await workflow.set_language(
                            args.language, hirag_set_language_tool[0]
                        )
                        logging.info(f"Language setting result: {language_result}")
                    except Exception as e:
                        logging.error(f"Failed to set language in HiRAG server: {e}")
                        # Continue execution even if language setting fails

                chat_history = await Console(workflow.run_workflow(user_input))
                messages = chat_history.messages
                messages = [
                    MemoryContent(
                        content=workflow.message_to_memory_content(message),
                        mime_type=get_memory_type_for_message(message),
                        metadata={"source": message.source},
                    )
                    for message in messages
                    if not isinstance(message, ToolCallSummaryMessage)
                ]
                await memory.add(messages)
            else:
                await asyncio.create_task(Console(workflow.run_workflow(user_input)))
                await workflow.team.set_id_info("cli_dev", chat_id)
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        # Cleanup session manager first (for hirag mode)
        if session_manager:
            try:
                await session_manager.close_all()
                logging.info("MCP session manager cleaned up successfully.")
            except Exception as session_error:
                logging.error(f"Error cleaning up MCP session manager: {session_error}")
        
        # Cleanup workflow if it exists
        if workflow is not None:
            try:
                await workflow.cleanup()
                logging.info("Workflow cleaned up successfully.")
            except Exception as cleanup_error:
                logging.error(f"Error during workflow cleanup: {cleanup_error}")
        
        # Cleanup database engine
        try:
            await engine.dispose()
            logging.info("Database engine disposed successfully.")
        except Exception as db_error:
            logging.error(f"Error disposing database engine: {db_error}")


if __name__ == "__main__":
    logging.info("------------- run main async---------------------------------------")
    args = parse_args()
    if args.trace:
        tracer = setup_tracing(
            endpoint=args.trace_endpoint, service_name=args.trace_service_name
        )
        with tracer.start_as_current_span("runtime"):
            asyncio.run(main_cmd(args))
    else:
        asyncio.run(main_cmd(args))
