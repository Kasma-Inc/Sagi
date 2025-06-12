import argparse
import asyncio
import logging
import os

from autogen_agentchat.ui import Console
from dotenv import load_dotenv

from Sagi.utils.logging_utils import setup_logging
from Sagi.workflows.general.general_chat import GeneralChatWorkflow

# Create logging directory if it doesn't exist
os.makedirs("logging", exist_ok=True)
setup_logging()

DEFAULT_CONFIG_PATH = "src/Sagi/workflows/general/general.toml"


def parse_args():
    parser = argparse.ArgumentParser("Sagi CLI General Chat")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--web_search", action="store_true", help="Enable web search")

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


async def main_cmd(args: argparse.Namespace):
    if args.web_search:
        workflow = await GeneralChatWorkflow.create(
            args.config,
            web_search=True,
        )
    else:
        workflow = await GeneralChatWorkflow.create(
            args.config,
            web_search=False,
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
    asyncio.run(main_cmd(args))
