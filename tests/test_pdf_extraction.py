import os

import pytest
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from dotenv import load_dotenv

from Sagi.tools.pdf_extraction.pdf_extraction import PDF_Extraction


# If you don't want to test the model (or the model is not available), you can skip this test
@pytest.mark.asyncio
async def test_pdf_extraction_with_model():
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if api_key is None or base_url is None:
        raise ValueError("LLM_API_KEY or LLM_BASE_URL is not set")

    # This is for testing purposes only, so the model here will be as small as possible and the result might not be perfect
    # model_client = OpenAIChatCompletionClient(
    #     model="gpt-4o-mini",
    #     base_url=base_url,
    #     api_key=api_key,
    #     max_tokens=16000,
    # )
    
    model_client = AnthropicChatCompletionClient(
        model="claude-sonnet-4-latest",
        base_url=base_url.replace("/v1", ""),
        auth_token=api_key,
        max_tokens=16000,
    )

    await PDF_Extraction.extract_pdf_with_model(
        input_path="tests/test_pdf_extraction.pdf",
        storage_dir="tests/storage/test_PDF_Extraction_with_model",
        model_client=model_client,
        output_path="tests/storage/test_PDF_Extraction_with_model/output.html",
        save_output_on_s3=False,
    )


@pytest.mark.asyncio
async def test_pdf_extraction_without_model():
    load_dotenv()

    await PDF_Extraction.extract_pdf_without_model(
        input_path="tests/test_pdf_extraction.pdf",
        storage_dir="tests/storage/test_PDF_Extraction_without_model",
        output_path="tests/storage/test_PDF_Extraction_without_model/output.html",
        save_output_on_s3=False,
    )

    # Test the make_editable_document function
    PDF_Extraction.make_editable_document(
        input_path="tests/storage/test_PDF_Extraction_without_model/output.html",
        output_path="tests/storage/test_PDF_Extraction_without_model/editable_output.html",
        title="test_editable_pdf_extraction",
    )

import asyncio

if __name__ == "__main__":
    async def main():
        await test_pdf_extraction_with_model()
    
    asyncio.run(main())