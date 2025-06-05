from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
import argparse
import asyncio
import logging
import os
import signal
import uuid

from autogen_agentchat.messages import BaseMessage
from autogen_agentchat.ui import Console
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from Sagi.utils.logging_utils import setup_logging
from Sagi.utils.queries import Database
from Sagi.workflows.planning import PlanningWorkflow

# Create logging directory if it doesn't exist
os.makedirs("logging", exist_ok=True)

# Setup logging using the new utilities
setup_logging()

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


# Extend BaseMessage to text
BaseMessage.to_text = lambda self: getattr(self, "content", repr(self))

DB_URL = os.getenv("POSTGRES_URL_NO_SSL_DEV")
if not DB_URL:
    raise RuntimeError("Environment variable POSTGRES_URL_NO_SSL_DEV is not set!")


_already_shutting_down = False


async def shutdown(db: Database, session_id: str, team):
    global _already_shutting_down
    if _already_shutting_down:
        return
    _already_shutting_down = True

    try:
        logging.info(f"Saving state for session {session_id} before shutdown...")
        await db.save_team_state(session_id, team)
    except Exception as e:
        logging.error(f"Error saving state on shutdown: {e}")
    finally:
        await db.close()
        logging.info("Database connection closed.")


def register_signal_handlers(db: Database, session_id: str, team):
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda sig=sig: asyncio.create_task(shutdown(db, session_id, team))
        )


async def main_cmd(args: argparse.Namespace):

    db = Database(DB_URL)
    await db.init()

    # List sessions and exit
    if args.list_sessions:
        sessions = await db.list_sessions()
        logging.info("Available sessions: %s", sessions or ["<none>"])
        await db.close()
        return

    # Determine session ID
    session_id = args.session_id or str(uuid.uuid4())
    logging.info(f"Using session_id = {session_id!r}")

    # Create workflow and team
    workflow = await PlanningWorkflow.create(args.config)
    team = workflow.team

    # Load previous state
    try:
        await db.load_team_state(session_id, team)
        logging.info(f"Loaded DB state for session {session_id}")
    except KeyError:
        logging.info(f"No DB state for session {session_id}; starting fresh")
        await workflow.team.reset()
    except Exception as e:
        logging.error(f"DB load error: {e}")

    # Register signals for graceful shutdown
    register_signal_handlers(db, session_id, team)

    # Interaction loop
    try:
        while True:
            try:
                user_input = input("User: ")
            except (EOFError, KeyboardInterrupt):
                logging.info("Detected EOF/KeyboardInterrupt, exiting loop.")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                logging.info("Exit command received.")
                break

            # Run workflow and display
            await Console(workflow.run_workflow(user_input))

            # Save state after each turn
            try:
                await db.save_team_state(session_id, team)
                logging.info(f"Saved DB state for session {session_id}")
            except Exception as e:
                logging.error(f"Error saving state: {e}")
    finally:
        # Cleanup
        await shutdown(db, session_id, team)
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
