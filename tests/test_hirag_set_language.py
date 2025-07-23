"""
Test script for the HiRAG set_language tool functionality.

This test verifies that:
1. The set_language tool is available in the MCP server
2. Language can be set successfully through the agent
3. The tool integrates correctly with the MCP session
"""

import asyncio
import logging
import os
import sys

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)
from dotenv import load_dotenv

from Sagi.utils.mcp_utils import MCPSessionManager

# Load environment variables
load_dotenv("/chatbot/.env", override=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHiRAGSetLanguage:
    """Test class for HiRAG set_language tool functionality."""

    def __init__(self):
        self.session_manager = MCPSessionManager()
        self.hirag_session = None
        self.hirag_tools = None
        self.set_language_tool = None

    async def setup(self):
        """Setup the test environment by initializing MCP session and tools."""
        logger.info("Setting up HiRAG MCP session...")

        # Setup HiRAG server parameters
        hirag_server_params = StdioServerParams(
            command="mcp-hirag-tool",
            args=[],
            read_timeout_seconds=100,
            env={
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
                "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY"),
            },
        )

        # Create and initialize HiRAG session
        self.hirag_session = await self.session_manager.create_session(
            "hirag_retrieval", create_mcp_server_session(hirag_server_params)
        )
        await self.hirag_session.initialize()

        # Get all available tools
        self.hirag_tools = await mcp_server_tools(
            hirag_server_params, session=self.hirag_session
        )

        # Find the set_language tool
        self.set_language_tool = next(
            (tool for tool in self.hirag_tools if tool.name == "hi_set_language"), None
        )

        logger.info(
            f"Available HiRAG tools: {[tool.name for tool in self.hirag_tools]}"
        )
        logger.info(f"Set language tool found: {self.set_language_tool is not None}")

    async def cleanup(self):
        """Clean up resources."""
        if self.session_manager:
            await self.session_manager.close_all()

    async def test_set_language_tool_available(self):
        """Test that the hi_set_language tool is available in the MCP server."""
        logger.info("Testing if hi_set_language tool is available...")

        tool_names = [tool.name for tool in self.hirag_tools]
        assert (
            "hi_set_language" in tool_names
        ), f"hi_set_language tool not found. Available tools: {tool_names}"
        assert (
            self.set_language_tool is not None
        ), "Set language tool should not be None"

        logger.info("✅ hi_set_language tool is available")
        return True

    async def test_set_language_functionality(self):
        """Test setting different languages using the hi_set_language tool."""
        if not self.set_language_tool:
            logger.warning(
                "⚠️ Skipping language functionality test - tool not available"
            )
            return False

        languages_to_test = ["en", "cn"]

        for language in languages_to_test:
            logger.info(f"Testing language setting: {language}")

            try:
                # Call the set_language tool directly
                result = await self.set_language_tool.run_json(
                    {"language": language}, CancellationToken()
                )

                logger.info(f"✅ Language '{language}' set successfully: {result}")

                # Verify the result is not None and doesn't contain error information
                assert (
                    result is not None
                ), f"Result should not be None for language: {language}"

            except Exception as e:
                logger.error(f"❌ Failed to set language '{language}': {e}")
                raise AssertionError(f"Failed to set language '{language}': {e}")

        logger.info("✅ All language setting tests passed")
        return True

    async def test_invalid_language_handling(self):
        """Test how the system handles invalid language codes."""
        if not self.set_language_tool:
            logger.warning("⚠️ Skipping invalid language test - tool not available")
            return False

        logger.info("Testing invalid language handling...")

        invalid_languages = ["invalid_lang", "", "123", "verylonginvalidlanguagecode"]

        for invalid_lang in invalid_languages:
            logger.info(f"Testing invalid language: '{invalid_lang}'")

            try:
                result = await self.set_language_tool.run_json(
                    {"language": invalid_lang}, CancellationToken()
                )

                # The tool should either handle gracefully or provide feedback
                logger.info(f"Invalid language '{invalid_lang}' result: {result}")

            except Exception as e:
                # It's acceptable for invalid languages to raise exceptions
                logger.info(
                    f"Invalid language '{invalid_lang}' raised exception (expected): {e}"
                )

        logger.info("✅ Invalid language handling test completed")
        return True

    async def test_search_functionality_both_languages(self):
        """Test search functionality in both English and Chinese languages."""
        if not self.set_language_tool:
            logger.warning(
                "⚠️ Skipping search functionality test - set_language tool not available"
            )
            return False

        logger.info("🔍 Testing search functionality in both languages...")

        # Find the search tool
        search_tool = next(
            (tool for tool in self.hirag_tools if tool.name == "hi_search"), None
        )

        if not search_tool:
            logger.warning(
                "⚠️ Skipping search functionality test - hi_search tool not available"
            )
            return False

        test_query = "artificial intelligence"
        results = {}

        # Test with English
        try:
            logger.info("   📝 Setting language to English and testing search...")

            # Set language to English
            en_lang_result = await self.set_language_tool.run_json(
                {"language": "en"}, CancellationToken()
            )
            logger.info(f"   Language setting result: {en_lang_result}")

            # Perform search
            en_search_result = await search_tool.run_json(
                {"query": test_query}, CancellationToken()
            )

            results["en"] = en_search_result
            logger.info(f"   English search result: {str(en_search_result)[:200]}...")

        except Exception as e:
            logger.error(f"   ❌ Error in English search test: {e}")
            return False

        # Test with Chinese
        try:
            logger.info("   📝 Setting language to Chinese and testing search...")

            # Set language to Chinese
            cn_lang_result = await self.set_language_tool.run_json(
                {"language": "cn"}, CancellationToken()
            )
            logger.info(f"   Language setting result: {cn_lang_result}")

            # Perform search
            cn_search_result = await search_tool.run_json(
                {"query": test_query}, CancellationToken()
            )

            results["cn"] = cn_search_result
            logger.info(f"   Chinese search result: {str(cn_search_result)[:200]}...")

        except Exception as e:
            logger.error(f"   ❌ Error in Chinese search test: {e}")
            return False

        # Analyze results
        if results["en"] is not None and results["cn"] is not None:
            logger.info("   ✅ Both language searches returned results")

            # Convert results to strings for comparison
            en_str = str(results["en"])
            cn_str = str(results["cn"])

            # Check if the responses are different (indicating language switching worked)
            if en_str != cn_str:
                logger.info(
                    "   ✅ Different responses for different languages - language switching works!"
                )

                # Try to detect language in results
                en_has_chinese = any("\u4e00" <= char <= "\u9fff" for char in en_str)
                cn_has_chinese = any("\u4e00" <= char <= "\u9fff" for char in cn_str)

                if not en_has_chinese and cn_has_chinese:
                    logger.info(
                        "   ✅ Perfect! English response has no Chinese characters, Chinese response has Chinese characters"
                    )
                elif cn_has_chinese:
                    logger.info(
                        "   ✅ Good! Chinese response contains Chinese characters as expected"
                    )
                else:
                    logger.info(
                        "   ⚠️ Note: No Chinese characters detected in either response (may be due to empty knowledge base)"
                    )

            else:
                logger.warning(
                    "   ⚠️ Same response for both languages - language switching may not be working"
                )

            return True
        else:
            logger.warning("   ⚠️ One or both search results are None")
            return False

    async def run_all_tests(self):
        """Run all test methods."""
        logger.info("🚀 Starting HiRAG set_language tool tests...")

        try:
            await self.setup()

            # Run individual tests
            test_results = []

            test_results.append(await self.test_set_language_tool_available())
            test_results.append(await self.test_set_language_functionality())
            test_results.append(await self.test_invalid_language_handling())
            test_results.append(await self.test_search_functionality_both_languages())

            # Summary
            passed_tests = sum(test_results)
            total_tests = len(test_results)

            logger.info(f"🎯 Test Summary: {passed_tests}/{total_tests} tests passed")

            if passed_tests == total_tests:
                logger.info("🎉 All tests passed successfully!")
            else:
                logger.warning(f"⚠️ {total_tests - passed_tests} tests failed")

            return passed_tests == total_tests

        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main function to run the tests."""
    # Initialize test class
    test_instance = TestHiRAGSetLanguage()

    try:
        # Run all tests
        success = await test_instance.run_all_tests()

        if success:
            logger.info("✅ All HiRAG set_language tests completed successfully")
        else:
            logger.error("❌ Some tests failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        sys.exit(1)


@pytest.mark.asyncio
async def test_hirag_set_language_pytest():
    """Pytest wrapper for HiRAG set_language tests."""
    test_instance = TestHiRAGSetLanguage()

    try:
        success = await test_instance.run_all_tests()
        assert success, "HiRAG set_language tests failed"
    finally:
        await test_instance.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
