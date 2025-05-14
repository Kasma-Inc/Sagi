import argparse
import asyncio
import json
import logging
import os
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

    workflow = await PlanningWorkflow.create(args.config)
    try:
        team_state = load_state("state.json")
        await workflow.team.load_state(team_state)
        logging.info("Loaded previous state.json")
    except FileNotFoundError:
        logging.info("state.json 不存在，使用初始状态")
    except Exception as e:
        logging.error(f"加载 state.json 失败：{e}")

    try:
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    state = await workflow.team.save_state()
                    with open("state.json", "w") as f:
                        json.dump(state, f)
                    logging.info("退出前已保存 state.json")
                    break
                run_task = asyncio.create_task(
                    Console(workflow.run_workflow(user_input))
                )
                await run_task
                state = await workflow.team.save_state()
                with open("state.json", "w") as f:
                    json.dump(state, f)
                logging.info("已保存 state.json")

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
