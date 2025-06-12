import argparse
import asyncio
import logging
import os
import signal
import threading

from autogen_agentchat.messages import BaseMessage
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from Sagi.tools.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)
from Sagi.utils.logging_utils import setup_logging
from Sagi.workflows.planning import PlanningWorkflow

# Create logging directory if it doesn't exist
os.makedirs("logging", exist_ok=True)
setup_logging()

DEFAULT_TEAM_CONFIG_PATH = "src/Sagi/workflows/team.toml"
DEFAULT_CONFIG_PATH = "src/Sagi/workflows/planning.toml"


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
        "--mode",
        type=str,
        choices=["deep_research", "general", "web_search"],
        default="deep_research",
        help="Operation mode: deep_research (full functionality), general (general agent only), or web_search (web search only)",
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

# to check whether the StreamDockerCommandLineCodeExecutor is used
isDockerCommandLine = None


async def setup_graceful_shutdown(stream_code_executor, workflow):
    """Set up signal handlers to stop the container on program termination"""
    loop = asyncio.get_running_loop()

    # Function to handle shutdown
    async def shutdown_handler():
        print("\nStarting graceful shutdown...\n")
        if isDockerCommandLine:
            if await stream_code_executor.is_running():
                await stream_code_executor.stop()

        await workflow.cleanup()
        loop.stop()

    # For graceful shutdown on SIGINT (Ctrl+C) and SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown_handler()))

    print("Graceful shutdown handlers registered")


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

    workflow, stream_code_executor = await PlanningWorkflow.create(
        args.config,
        args.team_config,
        template_work_dir=args.template_work_dir,
        mode=args.mode,
        language=args.language,
    )

    isDockerCommandLine = isinstance(
        stream_code_executor, StreamDockerCommandLineCodeExecutor
    )
    if isDockerCommandLine:
        await stream_code_executor.start()
        print("StreamDockerCommandLine started successfully.")

    # Set up signal handlers for graceful shutdown
    await setup_graceful_shutdown(stream_code_executor, workflow)

    try:
        while True:

            if isDockerCommandLine:
                await stream_code_executor.countdown(1)

            user_input = await get_input_async()
            if user_input.lower() in ("quit", "exit", "q"):
                break

            if isDockerCommandLine:
                await stream_code_executor.resume_docker_container()

            await asyncio.create_task(Console(workflow.run_workflow(user_input)))
    finally:
        await workflow.cleanup()
        if isDockerCommandLine:
            await stream_code_executor.stop_countdown()
            if await stream_code_executor.is_running():
                print("Stopping Docker container...")
                await stream_code_executor.stop()
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
