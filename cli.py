import argparse
import asyncio
import json
import logging
import os
import uuid
from typing import Optional
from datetime import datetime

from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from Sagi.utils.logging import format_json_string_factory
from Sagi.workflows.planning import PlanningWorkflow
from autogen_agentchat.messages import BaseMessage


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
    parser.add_argument("--env", type=str, choices=["dev", "prod"], default="dev")
    parser.add_argument(
        "--config", type=str, default="src/Sagi/workflows/planning.toml"
    )
    parser.add_argument(
        "--trace", action="store_true", help="Enable OpenTelemetry tracing"
    )
    parser.add_argument(
        "--trace_endpoint",
        type=str,
        default="http://localhost:4317",
        help="OpenTelemetry collector endpoint",
    )
    parser.add_argument(
        "--trace_service_name",
        type=str,
        default="sagi_tracer",
        help="Service name for OpenTelemetry tracing",
    )
    parser.add_argument(
        "--session-id", "-s",
        type=str,
        help="Specify the session ID to load or save; if not provided, one will be generated automatically."
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List existing session IDs and exit."
    )
    return parser.parse_args()


# load env variables
load_dotenv()


def setup_tracing(endpoint: str = None, service_name: str = None):
    """Setup OpenTelemetry tracing based on args."""

    try:
        otel_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        tracer_provider = TracerProvider(
            resource=Resource.create({ResourceAttributes.SERVICE_NAME: service_name})
        )
        span_processor = BatchSpanProcessor(otel_exporter)
        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(service_name)
        logging.info(f"OpenTelemetry tracing enabled, exporting to {endpoint}")
        return tracer
    except Exception as e:
        logging.error(f"Failed to setup tracing: {e}")
        return None
    
def _default_to_text(self) -> str:
    return getattr(self, "content", repr(self))

BaseMessage.to_text = _default_to_text

def load_state(file_path:str) -> dict:
    with open(file_path, "r") as f:
        state = json.load(f)
        return state

async def main_cmd(args: argparse.Namespace):

    if args.list_sessions:
        files = [f for f in os.listdir() if f.startswith("state_") and f.endswith(".json")]
        sessions = [f[len("state_"):-5] for f in files]
        print("Available sessions:", sessions or "<none>")
        return

    session_id: str = args.session_id or str(uuid.uuid4())
    state_file = f"state_{session_id}.json"
    print(f" use session_id = {session_id!r},state_file:{state_file}")

    workflow = await PlanningWorkflow.create(args.config)

    try:
        team_state = load_state(state_file)
        await workflow.team.load_state(team_state)
        logging.info(f"Loaded previous state from {state_file}")
    except FileNotFoundError:
        logging.info(f"{state_file} not found; using initial state")
    except Exception as e:
        logging.error(f"Failed to load state.json: {e}")

    try:
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    state = await workflow.team.save_state()
                    with open(state_file, "w") as f:
                        json.dump(state, f)
                    logging.info(f"Saved state to {state_file} before exit")
                    break
                run_task = asyncio.create_task(
                    Console(workflow.run_workflow(user_input))
                )
                await run_task
                state = await workflow.team.save_state()
                with open(state_file, "w") as f:
                    json.dump(state, f)
                logging.info(f"Saved state to {state_file}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Error: {e}")
                break
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
