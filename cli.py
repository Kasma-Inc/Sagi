import argparse
import asyncio
import logging
import os
from datetime import datetime

from autogen_agentchat.ui import Console
from autogen_core import TRACE_LOGGER_NAME
from dotenv import load_dotenv

from utils.logging import format_json_string_factory
from workflows.planning import PlanningWorkflow

# Create logging directory if it doesn't exist
os.makedirs("logging", exist_ok=True)

logging.setLogRecordFactory(format_json_string_factory)

logging.basicConfig(
    level=logging.INFO,
    filename=f"logging/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("Ofnil Agentic RAG CLI")
    parser.add_argument("--env", type=str, choices=["dev", "prod"], default="dev")
    parser.add_argument("--config", type=str, default="workflows/planning.toml")
    return parser.parse_args()


# load env variables
load_dotenv()


async def main_cmd(args: argparse.Namespace):
    workflow = await PlanningWorkflow.create(args.config)

    try:
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                run_task = asyncio.create_task(
                    Console(workflow.run_workflow(user_input))
                )
                await run_task

            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Error: {e}")
                break
    finally:
        await workflow.cleanup()
        logging.info("Workflow cleaned up.")


if __name__ == "__main__":
    from autogen_agentchat import TRACE_LOGGER_NAME

    # For trace logging.
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    trace_logger.addHandler(logging.StreamHandler())
    trace_logger.setLevel(logging.INFO)
    logging.info("------------- run main async---------------------------------------")
    args = parse_args()
    asyncio.run(main_cmd(args))
