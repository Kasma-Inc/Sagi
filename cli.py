import argparse
import asyncio
import logging
import os

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
from Sagi.workflows.planning.planning import PlanningWorkflow

# Create logging directory if it doesn't exist
os.makedirs("logging", exist_ok=True)
setup_logging()

DEFAULT_TEAM_CONFIG_PATH = "src/Sagi/workflows/team.toml"
DEFAULT_CONFIG_PATH = "src/Sagi/workflows/planning/planning.toml"


def parse_args():
    parser = argparse.ArgumentParser("Sagi CLI")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--team_config", default=DEFAULT_TEAM_CONFIG_PATH)
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
        "--language",
        type=str,
        choices=["en", "cn"],
        default="en",
        help="Language: en (English), cn (Chinese)",
    )
    return parser.parse_args()


# load env variables
load_dotenv(override=True)


def setup_tracing(endpoint: str = None, service_name: str = None):
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


async def main_cmd(args: argparse.Namespace):

    workflow = await PlanningWorkflow.create(
        args.config,
        args.team_config,
        template_work_dir=args.template_work_dir,
        language=args.language,
    )

    try:
        while True:
            user_input = input("User: ")
            if user_input.lower() in ("quit", "exit", "q"):
                break

            await asyncio.create_task(Console(workflow.run_workflow(user_input)))
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
