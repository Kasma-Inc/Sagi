import json
import logging
import uuid
from dataclasses import dataclass
from typing import List, Literal, TypedDict, Union

import pytest
from autogen_agentchat.messages import (
    CodeExecutionEvent,
    CodeGenerationEvent,
    ModelClientStreamingChunkEvent,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)

from Sagi.workflows.analyzing.analyzing import AnalyzingWorkflow


class NewPlanStepEventArgs(TypedDict):
    stepId: str


class NewPlanStepEventResult(TypedDict):
    content: str
    totalSteps: int


class StateUpdateEventArgs(TypedDict):
    stepId: str


class StateUpdateEventResult(TypedDict):
    content: str


class ToolUsedEventArgs(TypedDict):
    stepId: str
    action: Literal[
        "Executing command",
        "Creating file",
        "Editing file",
        "Reading file",
        "Searching",
    ]
    description: str


class TerminalDetails(TypedDict):
    command: str
    output: str


class TextEditorDetails(TypedDict):
    oldContent: str
    content: str
    action: Literal["read", "write", "update"]
    path: str


class MarkdownDetails(TypedDict):
    content: str
    action: Literal["read"]
    path: str


class SearchResult(TypedDict):
    link: str
    snippet: str
    title: str


class SearchDetails(TypedDict):
    queries: List[str]
    results: List[SearchResult]


class ToolUsedEventResult(TypedDict):
    sandbox: Literal["terminal", "textEditor", "search"]
    details: Union[TerminalDetails, TextEditorDetails, SearchDetails]


class PlanUpdateEventArgs(TypedDict):
    stepId: str


class PlanStatus(TypedDict):
    stepId: str
    status: Literal["todo", "doing", "done"]


class PlanUpdateEventResult(TypedDict):
    planStatus: List[PlanStatus]


class PlanListEventArgs(TypedDict):
    planId: str


class PlanListEventResult(TypedDict):
    content: str
    posFeedback: str
    negFeedback: str


class TextMessageArgs(TypedDict):
    planId: str


class TextMessageResult(TypedDict):
    content: str


@dataclass
class EventArgs:
    toolCallId: str
    toolCallName: str
    args: Union[
        NewPlanStepEventArgs,
        StateUpdateEventArgs,
        ToolUsedEventArgs,
        PlanUpdateEventArgs,
        PlanListEventArgs,
        TextMessageArgs,
    ]

    def to_stream_args(self) -> str:
        payload = {
            "toolCallId": self.toolCallId,
            "toolName": self.toolCallName,
            "args": self.args,
        }
        return f"9:{json.dumps(payload, ensure_ascii=False)}\n"


@dataclass
class EventResult:
    toolCallId: str
    result: Union[
        NewPlanStepEventResult,
        StateUpdateEventResult,
        ToolUsedEventResult,
        PlanUpdateEventResult,
        PlanListEventResult,
        TextMessageResult,
    ]

    def to_stream_result(self) -> str:
        payload = {"toolCallId": self.toolCallId, "result": self.result}
        return f"a:{json.dumps(payload, ensure_ascii=False)}\n"


@pytest.mark.asyncio
async def test_analyze():
    workflow = await AnalyzingWorkflow.create(
        config_path="/chatbot/Sagi/src/Sagi/workflows/analyzing/analyzing.toml"
    )
    res = workflow.run_workflow(
        "Query the first eight pieces of data in the database and analyzing it"
    )
    async for result in res:
        if isinstance(
            result,
            (
                TextMessage,
                ToolCallSummaryMessage,
                CodeGenerationEvent,
                CodeExecutionEvent,
                ToolCallRequestEvent,
                ToolCallExecutionEvent,
                ModelClientStreamingChunkEvent,
            ),
        ):
            match result.source:

                # case "pg_agent":
                #     if (
                #         result.type == "ToolCallRequestEvent"
                #     ):
                #         args = EventArgs(
                #             toolCallId=result.content[0].id,
                #             toolCallName="toolUsed",
                #             args=ToolUsedEventArgs(
                #                 stepId="step_1",
                #                 action="Executing command",
                #                 description=f"Query the database {json.loads(result.content[0].arguments)['query']}",
                #             ),
                #         )
                #         return args.to_stream_args()
                #     if (
                #         result.type == "ToolCallExecutionEvent"
                #     ):
                #         execution_content = result.content[0].content
                #         execution_content_list = json.loads(execution_content)
                #         execution_content = execution_content_list[-1]["text"]
                #         result = EventResult(
                #             toolCallId=result.content[0].call_id,
                #             result=ToolUsedEventResult(
                #                 sandbox="terminal",
                #                 details=TerminalDetails(
                #                     command="Query the database",
                #                     output=execution_content,
                #                 ),
                #             ),
                #         )
                #         return result.to_stream_result()
                case "analyze_agent":
                    logging.info("666666666666666666666666666666666666")
                    logging.info("analyze_agent")
                    logging.info(result)
                    logging.info("666666666666666666666666666666666666")
                    _uuid = str(uuid.uuid4())
                    breakpoint()
                    args = EventArgs(
                        toolCallId=_uuid,
                        toolCallName="text",
                        args=TextMessageArgs(
                            planId="step_1",
                        ),
                    )
                    result = EventResult(
                        toolCallId=_uuid,
                        result=TextMessageResult(
                            content=result.content,
                        ),
                    )
                    logging.info("777777777777777777777777777777777777")
                    logging.info(args.to_stream_args() + result.to_stream_result())
                    logging.info("777777777777777777777777777777777777")
    # breakpoint()
