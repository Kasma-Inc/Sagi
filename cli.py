import argparse
import asyncio
import logging
import os
import uuid
from datetime import datetime

from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_agentchat.messages import BaseMessage
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from Sagi.utils.logging import format_json_string_factory
from Sagi.utils.queries import Database
from Sagi.workflows.planning import PlanningWorkflow

# Create logging directory if it doesn't exist
os.makedirs("logging", exist_ok=True)

logging.setLogRecordFactory(format_json_string_factory)

logging.basicConfig(
    level=logging.INFO,
    filename=f"logging/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

# For trace logging.
trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
trace_logger.addHandler(logging.StreamHandler())
trace_logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("Sagi CLI")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--config", default="src/Sagi/workflows/planning.toml")
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


DB_URL = os.getenv("POSTGRES_URL_NO_SSL_DEV")
if not DB_URL:
    raise RuntimeError("Environment variable POSTGRES_URL_NO_SSL_DEV is not set!")


async def main_cmd(args: argparse.Namespace):

    db = Database(DB_URL)
    await db.init()

    # List sessions and exit
    if args.list_sessions:
        sessions = await db.list_sessions()
        logging.info("Available sessions:", sessions or ["<none>"])
        await db.close()
        return

    session_id = args.session_id or str(uuid.uuid4())
    logging.info(f"use session_id = {session_id!r}")

    workflow = await PlanningWorkflow.create(args.config)

    # Load previous state
    try:
        team_state = await db.load_state(session_id)
        await workflow.team.load_state(team_state)
        logging.info(f"Loaded DB state for session {session_id}")
    except KeyError:
        logging.info(f"No DB state for session {session_id}; starting fresh")
        await workflow.team.reset()
    except Exception as e:
        logging.error(f"DB load error: {e}")

    try:
        while True:
            user_input = input("User: ")
            if user_input.lower() in ("quit", "exit", "q"):
                state = await workflow.team.save_state()
                await db.save_state(session_id, state)
                logging.info(f"Saved DB state before exit for {session_id}")
                break

            await asyncio.create_task(Console(workflow.run_workflow(user_input)))
            state = await workflow.team.save_state()
            await db.save_state(session_id, state)
            logging.info(f"Saved DB state for {session_id}")
    finally:
        await workflow.cleanup()
        await db.close()
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
