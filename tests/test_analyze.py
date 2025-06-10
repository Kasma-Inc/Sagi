
import pytest

from Sagi.workflows.analyzing.analyzing import AnalyzingWorkflow


@pytest.mark.asyncio
async def test_analyze():
    analyzing_workflow = await AnalyzingWorkflow.create(
        config_path="/chatbot/Sagi/src/Sagi/workflows/analyzing/config.toml"
    )
    # res = analyzing_workflow.run_workflow("Query the first eight pieces of data in the database and analyzing it")
    # for chunk in res:
    #     print(chunk)
    res = analyzing_workflow.run_workflow(
        "Query the first eight pieces of data in the database and analyzing it"
    )
    async for chunk in res:
        print(chunk)
