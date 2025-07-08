import argparse
import asyncio
import logging
import os
import threading
import uuid

from autogen_agentchat.messages import BaseMessage
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from Sagi.utils.logging_utils import setup_logging
from Sagi.workflows.general.general_chat import GeneralChatWorkflow
from Sagi.workflows.planning.planning import PlanningWorkflow
from Sagi.workflows.planning_html.planning_html import PlanningHtmlWorkflow

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
load_dotenv(override=True)


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
    chat_id = str(uuid.uuid4())
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
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    try:
        while True:
            user_input = await get_input_async()
            if user_input.lower() in ("quit", "exit", "q"):
                break

            await asyncio.create_task(Console(workflow.run_workflow(user_input)))
            await workflow.team.set_id_info("cli_dev", chat_id)
    finally:
        await workflow.cleanup()
        logging.info("Workflow cleaned up.")


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
