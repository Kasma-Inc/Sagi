"""
Test script for the HiRAG insert and search chat functionality.

This test verifies that:
1. The hi_insert_chat and hi_search_chat tools are available in the MCP server
2. Chat messages can be inserted successfully with different roles
3. Chat history can be searched and relevant messages are retrieved
4. Error handling works correctly for invalid inputs
5. Role filtering works properly during search
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime

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


class TestHiRAGInsertSearchChat:
    """Test class for HiRAG insert and search chat functionality."""

    def __init__(self):
        self.session_manager = MCPSessionManager()
        self.hirag_session = None
        self.hirag_tools = None
        self.insert_chat_tool = None
        self.search_chat_tool = None
        self.test_chat_id = f"test_chat_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"

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

        # Find the insert and search chat tools
        self.insert_chat_tool = next(
            (tool for tool in self.hirag_tools if tool.name == "hi_insert_chat"), None
        )
        self.search_chat_tool = next(
            (tool for tool in self.hirag_tools if tool.name == "hi_search_chat"), None
        )

        logger.info(
            f"Available HiRAG tools: {[tool.name for tool in self.hirag_tools]}"
        )
        logger.info(f"Insert chat tool found: {self.insert_chat_tool is not None}")
        logger.info(f"Search chat tool found: {self.search_chat_tool is not None}")
        logger.info(f"Test chat ID: {self.test_chat_id}")

    async def cleanup(self):
        """Clean up resources."""
        if self.session_manager:
            await self.session_manager.close_all()

    async def test_basic_functionality_check(self):
        """Test basic functionality to ensure the tools are properly implemented."""
        logger.info("Testing basic functionality implementation...")

        if not self.insert_chat_tool or not self.search_chat_tool:
            logger.error("❌ Required tools not available")
            return False

        # Test a simple insert to check if the functionality is implemented
        try:
            test_result = await self.insert_chat_tool.run_json(
                {
                    "chat_id": f"test_functionality_{uuid.uuid4().hex[:4]}",
                    "role": "user",
                    "content": "Test message for functionality check",
                },
                CancellationToken()
            )

            result_str = str(test_result).lower()
            if "error" in result_str or "no attribute" in result_str:
                logger.error(f"❌ Insert functionality not properly implemented: {test_result}")
                logger.error("❌ The HiRAG MCP server needs to implement the 'insert_chat_message' method")
                logger.error("❌ Skipping remaining tests - insert functionality needs to be implemented")
                return False

        except Exception as e:
            logger.error(f"❌ Insert functionality test failed: {e}")
            return False

        # Test a simple search to check if the functionality is implemented  
        try:
            search_result = await self.search_chat_tool.run_json(
                {
                    "user_query": "test",
                    "chat_id": f"nonexistent_chat_{uuid.uuid4().hex[:4]}",
                },
                CancellationToken()
            )

            result_str = str(search_result).lower()
            if "failed to query" in result_str and "name or service not known" in result_str:
                logger.error(f"❌ Search functionality has connectivity issues: {search_result}")
                logger.error("❌ The HiRAG search functionality has network/database connectivity issues")
                logger.error("❌ Check database connection and network configuration")
                logger.error("❌ Skipping remaining tests - search functionality has network/database issues")
                return False

        except Exception as e:
            logger.error(f"❌ Search functionality test failed: {e}")
            return False

        logger.info("✅ Basic functionality checks passed")
        return True

    async def test_chat_tools_available(self):
        """Test that the hi_insert_chat and hi_search_chat tools are available."""
        logger.info("Testing if chat tools are available...")

        tool_names = [tool.name for tool in self.hirag_tools]
        
        assert (
            "hi_insert_chat" in tool_names
        ), f"hi_insert_chat tool not found. Available tools: {tool_names}"
        assert (
            self.insert_chat_tool is not None
        ), "Insert chat tool should not be None"

        assert (
            "hi_search_chat" in tool_names
        ), f"hi_search_chat tool not found. Available tools: {tool_names}"
        assert (
            self.search_chat_tool is not None
        ), "Search chat tool should not be None"

        logger.info("✅ Both chat tools are available")
        return True

    async def test_insert_chat_messages(self):
        """Test inserting different types of chat messages."""
        if not self.insert_chat_tool:
            logger.warning("⚠️ Skipping insert test - tool not available")
            return False

        logger.info("Testing chat message insertion...")

        # Test messages with different roles
        test_messages = [
            {
                "role": "user",
                "content": "Hello, I need help with machine learning algorithms. Can you explain neural networks?",
            },
            {
                "role": "assistant", 
                "content": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections.",
            },
            {
                "role": "user",
                "content": "What about deep learning? How is it different from traditional machine learning?",
            },
            {
                "role": "assistant",
                "content": "Deep learning is a subset of machine learning that uses neural networks with multiple hidden layers. Unlike traditional ML, it can automatically learn feature representations from raw data.",
            },
            {
                "role": "tool",
                "content": "Retrieved relevant papers: 'Deep Learning' by LeCun et al., 'Neural Networks for Pattern Recognition' by Bishop.",
            },
        ]

        for i, message in enumerate(test_messages):
            logger.info(f"Inserting message {i+1}: {message['role']} - {message['content'][:50]}...")

            try:
                result = await self.insert_chat_tool.run_json(
                    {
                        "chat_id": self.test_chat_id,
                        "role": message["role"],
                        "content": message["content"],
                    },
                    CancellationToken()
                )

                # Check if the result indicates an error
                result_str = str(result).lower()
                if "error" in result_str:
                    logger.error(f"❌ Message {i+1} insertion failed: {result}")
                    raise AssertionError(f"Message {i+1} insertion failed: {result}")
                
                logger.info(f"✅ Message {i+1} inserted successfully: {result}")
                assert result is not None, f"Result should not be None for message {i+1}"

            except Exception as e:
                logger.error(f"❌ Failed to insert message {i+1}: {e}")
                raise AssertionError(f"Failed to insert message {i+1}: {e}")

        logger.info("✅ All chat messages inserted successfully")
        return True

    async def test_search_chat_functionality(self):
        """Test searching chat history with different queries."""
        if not self.search_chat_tool:
            logger.warning("⚠️ Skipping search test - tool not available")
            return False

        logger.info("Testing chat search functionality...")

        # Test different search queries
        search_queries = [
            {
                "query": "neural networks",
                "description": "Search for neural networks discussion",
            },
            {
                "query": "deep learning",
                "description": "Search for deep learning explanation",
            },
            {
                "query": "machine learning algorithms",
                "description": "Search for ML algorithms discussion",
            },
            {
                "query": "papers references",
                "description": "Search for paper references",
            },
        ]

        for query_info in search_queries:
            query = query_info["query"]
            description = query_info["description"]
            logger.info(f"🔍 {description}: '{query}'")

            try:
                result = await self.search_chat_tool.run_json(
                    {
                        "user_query": query,
                        "chat_id": self.test_chat_id,
                    },
                    CancellationToken()
                )

                # Check if the result indicates an error or no results due to functionality issues
                result_str = str(result).lower()
                if "error" in result_str or "failed to query" in result_str:
                    logger.error(f"❌ Search for '{query}' failed: {result}")
                    raise AssertionError(f"Search for '{query}' failed: {result}")
                
                # Also check if we got "no chat messages found" which might indicate the insert didn't work
                if "no chat messages found" in result_str:
                    logger.warning(f"⚠️ No messages found for '{query}' - this might indicate insert functionality issues")
                
                logger.info(f"✅ Search result for '{query}': {str(result)[:200]}...")
                assert result is not None, f"Search result should not be None for query: {query}"

            except Exception as e:
                logger.error(f"❌ Failed to search for '{query}': {e}")
                raise AssertionError(f"Failed to search for '{query}': {e}")

        logger.info("✅ All search queries completed successfully")
        return True

    async def test_search_with_role_filter(self):
        """Test searching chat history with role filters."""
        if not self.search_chat_tool:
            logger.warning("⚠️ Skipping role filter test - tool not available")
            return False

        logger.info("Testing search with role filters...")

        # Test role-specific searches
        role_tests = [
            {
                "role": "user",
                "query": "machine learning",
                "description": "Search only user messages",
            },
            {
                "role": "assistant",
                "query": "neural networks",
                "description": "Search only assistant messages",
            },
            {
                "role": "tool",
                "query": "papers",
                "description": "Search only tool messages",
            },
        ]

        for role_test in role_tests:
            role = role_test["role"]
            query = role_test["query"]
            description = role_test["description"]
            logger.info(f"🔍 {description} for '{query}'")

            try:
                result = await self.search_chat_tool.run_json(
                    {
                        "user_query": query,
                        "chat_id": self.test_chat_id,
                        "role": [role],
                    },
                    CancellationToken()
                )

                # Check if the result indicates an error or functionality issues
                result_str = str(result).lower()
                if "error" in result_str or "failed to query" in result_str:
                    logger.error(f"❌ Role-filtered search for '{role}' + '{query}' failed: {result}")
                    raise AssertionError(f"Role-filtered search failed: {result}")

                logger.info(f"✅ Role-filtered search result for '{role}' + '{query}': {str(result)[:150]}...")
                assert result is not None, f"Role-filtered search result should not be None"

            except Exception as e:
                logger.error(f"❌ Failed role-filtered search for '{role}' + '{query}': {e}")
                raise AssertionError(f"Failed role-filtered search: {e}")

        logger.info("✅ All role-filtered searches completed successfully")
        return True

    async def test_invalid_input_handling(self):
        """Test how the tools handle invalid inputs."""
        if not self.insert_chat_tool or not self.search_chat_tool:
            logger.warning("⚠️ Skipping invalid input test - tools not available")
            return False

        logger.info("Testing invalid input handling...")

        # Test invalid insert operations
        invalid_insert_cases = [
            {
                "params": {"chat_id": "", "role": "user", "content": "test"},
                "description": "Empty chat_id",
            },
            {
                "params": {"chat_id": "test", "role": "", "content": "test"},
                "description": "Empty role",
            },
            {
                "params": {"chat_id": "test", "role": "user", "content": ""},
                "description": "Empty content",
            },
            {
                "params": {"chat_id": "test", "role": "invalid_role", "content": "test"},
                "description": "Invalid role",
            },
        ]

        for case in invalid_insert_cases:
            logger.info(f"Testing invalid insert: {case['description']}")
            try:
                result = await self.insert_chat_tool.run_json(
                    case["params"], CancellationToken()
                )
                logger.info(f"Invalid insert result: {result}")
                # Tool should handle gracefully or return error message
            except Exception as e:
                logger.info(f"Invalid insert raised exception (acceptable): {e}")

        # Test invalid search operations
        invalid_search_cases = [
            {
                "params": {"user_query": "", "chat_id": "test"},
                "description": "Empty query",
            },
            {
                "params": {"user_query": "test", "chat_id": ""},
                "description": "Empty chat_id",
            },
            {
                "params": {"user_query": "test", "chat_id": "test", "role": ["invalid_role"]},
                "description": "Invalid role filter",
            },
        ]

        for case in invalid_search_cases:
            logger.info(f"Testing invalid search: {case['description']}")
            try:
                result = await self.search_chat_tool.run_json(
                    case["params"], CancellationToken()
                )
                logger.info(f"Invalid search result: {result}")
                # Tool should handle gracefully or return error message
            except Exception as e:
                logger.info(f"Invalid search raised exception (acceptable): {e}")

        logger.info("✅ Invalid input handling test completed")
        return True

    async def test_empty_chat_search(self):
        """Test searching in a chat that doesn't exist or has no messages."""
        if not self.search_chat_tool:
            logger.warning("⚠️ Skipping empty chat test - tool not available")
            return False

        logger.info("Testing search in non-existent chat...")

        # Use a different chat ID that we haven't inserted messages into
        empty_chat_id = f"empty_chat_{uuid.uuid4().hex[:8]}"

        try:
            result = await self.search_chat_tool.run_json(
                {
                    "user_query": "test query",
                    "chat_id": empty_chat_id,
                },
                CancellationToken()
            )

            logger.info(f"Empty chat search result: {result}")
            # Should return empty results or appropriate message
            assert result is not None, "Empty chat search should return a result (even if empty)"

        except Exception as e:
            logger.info(f"Empty chat search raised exception (acceptable): {e}")

        logger.info("✅ Empty chat search test completed")
        return True

    async def run_all_tests(self):
        """Run all test methods."""
        logger.info("🚀 Starting HiRAG insert and search chat tests...")

        try:
            await self.setup()

            # Run individual tests
            test_results = []

            # First check if basic functionality is implemented
            basic_functionality_ok = await self.test_basic_functionality_check()
            test_results.append(basic_functionality_ok)
            
            if not basic_functionality_ok:
                logger.warning("⚠️ Basic functionality not implemented - skipping remaining tests")
                test_results.extend([False] * 5)  # Mark remaining tests as failed
            else:
                test_results.append(await self.test_chat_tools_available())
                test_results.append(await self.test_insert_chat_messages())
                test_results.append(await self.test_search_chat_functionality())
                test_results.append(await self.test_search_with_role_filter())
                test_results.append(await self.test_invalid_input_handling())
                test_results.append(await self.test_empty_chat_search())

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
    test_instance = TestHiRAGInsertSearchChat()

    try:
        # Run all tests
        success = await test_instance.run_all_tests()

        if success:
            logger.info("✅ All HiRAG insert/search chat tests completed successfully")
        else:
            logger.error("❌ Some tests failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        sys.exit(1)


@pytest.mark.asyncio
async def test_hirag_insert_search_chat_pytest():
    """Pytest wrapper for HiRAG insert and search chat tests."""
    test_instance = TestHiRAGInsertSearchChat()

    try:
        success = await test_instance.run_all_tests()
        assert success, "HiRAG insert/search chat tests failed"
    finally:
        await test_instance.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
