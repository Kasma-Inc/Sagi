import argparse
import asyncio
import json
import logging
import os
import uuid
import psycopg2  # type: ignore
import psycopg2.extras  # noqa: for DictCursor
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import BaseMessage
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from Sagi.utils.logging import format_json_string_factory
from Sagi.workflows.planning import PlanningWorkflow

# Load environment variables
load_dotenv()
DB_URL = os.getenv("POSTGRES_URL_NO_SSL_DEV")
if not DB_URL:
    raise RuntimeError("Environment variable POSTGRES_URL_NO_SSL_DEV is not set!")

# SQL for table creation
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_states (
    session_id TEXT PRIMARY KEY,
    state      JSONB   NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

def get_conn():
    conn = psycopg2.connect(DB_URL, sslmode="require")
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def load_state_db(session_id: str) -> dict:
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT state FROM session_states WHERE session_id = %s", (session_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return row["state"]
    else:
        raise FileNotFoundError


def save_state_db(session_id: str, state: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO session_states(session_id, state, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (session_id) DO UPDATE
          SET state = EXCLUDED.state,
              updated_at = NOW()
        """,
        (session_id, json.dumps(state)),
    )
    conn.commit()
    cur.close()
    conn.close()

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

def setup_tracing(endpoint: str = None, service_name: str = None):
    """Setup OpenTelemetry tracing."""
    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider = TracerProvider(resource=Resource.create({ResourceAttributes.SERVICE_NAME: service_name}))
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

def parse_args():
    parser = argparse.ArgumentParser("Sagi CLI")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument(
        "--config", default="src/Sagi/workflows/planning.toml"
    )
    parser.add_argument(
        "--trace", action="store_true", help="Enable OpenTelemetry tracing"
    )
    parser.add_argument(
        "--trace_endpoint", default="http://localhost:4317",
        help="OpenTelemetry collector endpoint"
    )
    parser.add_argument(
        "--trace_service_name", default="sagi_tracer",
        help="Service name for OpenTelemetry tracing"
    )
    parser.add_argument(
        "-s", "--session-id", type=str,
        help="Specify the session ID to load or save; if not provided, one will be generated automatically."
    )
    parser.add_argument(
        "--list-sessions", action="store_true",
        help="List existing session IDs and exit."
    )
    return parser.parse_args()

async def main_cmd(args: argparse.Namespace):
    # List sessions and exit
    if args.list_sessions:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT session_id FROM session_states ORDER BY updated_at DESC")
        rows = cur.fetchall()
        cur.close(); conn.close()
        sessions = [r[0] for r in rows]
        print("Available sessions:", sessions or ["<none>"])
        return

    session_id = args.session_id or str(uuid.uuid4())
    print(f"use session_id = {session_id!r}")

    workflow = await PlanningWorkflow.create(args.config)

    # Load previous state
    try:
        team_state = load_state_db(session_id)
        await workflow.team.load_state(team_state)
        logging.info(f"Loaded DB state for session {session_id}")
    except FileNotFoundError:
        logging.info(f"No DB state for session {session_id}; starting fresh")
    except Exception as e:
        logging.error(f"DB load error: {e}")

    try:
        while True:
            user_input = input("User: ")
            if user_input.lower() in ("quit", "exit", "q"):
                state = await workflow.team.save_state()
                save_state_db(session_id, state)
                logging.info(f"Saved DB state before exit for {session_id}")
                break

            await asyncio.create_task(Console(workflow.run_workflow(user_input)))
            state = await workflow.team.save_state()
            save_state_db(session_id, state)
            logging.info(f"Saved DB state for {session_id}")
    finally:
        await workflow.cleanup()
        logging.info("Workflow cleaned up.")

if __name__ == "__main__":
    logging.info("------------- run main async---------------------------------------")
    args = parse_args()
    if args.trace:
        tracer = setup_tracing(
            endpoint=args.trace_endpoint,
            service_name=args.trace_service_name
        )
        with tracer.start_as_current_span("runtime"):
            asyncio.run(main_cmd(args))
    else:
        asyncio.run(main_cmd(args))
