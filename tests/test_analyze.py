
import pytest

from Sagi.workflows.analyzing.analyzing import AnalyzingWorkflow


@pytest.mark.asyncio
async def test_analyze():
    workflow = await AnalyzingWorkflow.create(
        config_path="/chatbot/Sagi/src/Sagi/workflows/analyzing/analyzing.toml"
    )
    res = workflow.run_workflow(
        "Query the first eight pieces of data in the database and analyzing it"
    )
    async for chunk in res:
        print(chunk)
    breakpoint()
