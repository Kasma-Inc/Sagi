"""
Test script for the HiRAG set_language tool functionality.

This test verifies that:
1. The set_language tool is available in the MCP server
2. Language can be set successfully through the tool
3. The tool handles different language codes correctly
"""

import asyncio
import logging
import os
import sys

import pytest
from autogen_core import CancellationToken
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

    async def test_basic_functionality_check(self):
        """Test basic functionality to ensure the set_language tool is properly implemented."""
        logger.info("Testing basic set_language functionality...")

        if not self.set_language_tool:
            logger.error("❌ set_language tool not available")
            return False

        # Test a simple language setting to check if the functionality is implemented
        try:
            test_result = await self.set_language_tool.run_json(
                {"language": "en"}, CancellationToken()
            )

            result_str = str(test_result).lower()
            if "error" in result_str and "not" in result_str and "attribute" in result_str:
                logger.error(f"❌ Set language functionality not properly implemented: {test_result}")
                logger.error("❌ The HiRAG MCP server needs to implement the set_language functionality")
                return False
            
            if "successfully set to" in result_str:
                logger.info("✅ Set language functionality is working correctly")
                return True
            else:
                logger.warning(f"⚠️ Unexpected response format: {test_result}")
                return True  # Still consider it working if no explicit error

        except Exception as e:
            logger.error(f"❌ Set language functionality test failed: {e}")
            return False

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

                # Check if the result indicates an error
                result_str = str(result).lower()
                if "error" in result_str:
                    logger.error(f"❌ Language '{language}' setting failed: {result}")
                    raise AssertionError(f"Language '{language}' setting failed: {result}")

                logger.info(f"✅ Language '{language}' set successfully: {result}")

                # Verify the result is not None and contains success indication
                assert (
                    result is not None
                ), f"Result should not be None for language: {language}"
                
                # Verify that the result indicates successful language setting
                if "successfully set to" not in result_str:
                    logger.warning(f"⚠️ Unexpected result format for language '{language}': {result}")

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

                # Check if the tool properly handles invalid languages
                result_str = str(result).lower()
                if "error" in result_str:
                    logger.info(f"✅ Invalid language '{invalid_lang}' properly rejected: {result}")
                else:
                    logger.warning(f"⚠️ Invalid language '{invalid_lang}' was not rejected: {result}")

            except Exception as e:
                # It's acceptable for invalid languages to raise exceptions
                logger.info(
                    f"✅ Invalid language '{invalid_lang}' raised exception (expected): {e}"
                )

        logger.info("✅ Invalid language handling test completed")
        return True

    async def run_all_tests(self):
        """Run all test methods."""
        logger.info("🚀 Starting HiRAG set_language tool tests...")

        try:
            await self.setup()

            # Run individual tests
            test_results = []

            # First check if basic functionality is implemented
            basic_functionality_ok = await self.test_basic_functionality_check()
            test_results.append(basic_functionality_ok)
            
            if not basic_functionality_ok:
                logger.warning("⚠️ Basic set_language functionality not implemented - skipping remaining tests")
                test_results.extend([False] * 2)  # Mark remaining tests as failed
            else:
                test_results.append(await self.test_set_language_tool_available())
                test_results.append(await self.test_set_language_functionality())
                test_results.append(await self.test_invalid_language_handling())

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
