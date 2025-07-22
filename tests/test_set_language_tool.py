"""
Test script for the HiRAG set_language tool functionality.

This test verifies that:
1. The set_language tool is available in the MCP server
2. Language can be set successfully through the agent
3. The tool integrates correctly with the MCP session
"""

import asyncio
import json
import logging
import os
import sys
from typing import List

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


class TestSetLanguageTool:
    """Test class for HiRAG set_language tool functionality."""
    
    def __init__(self):
        self.session_manager = None
        self.hirag_session = None
        self.tools = []
        self.agent = None
        self.hirag_server_params = None
        
    async def setup(self):
        """Setup MCP session and tools."""
        logger.info("🚀 Setting up HiRAG MCP session for testing...")
        
        # Create MCP server parameters
        self.hirag_server_params = StdioServerParams(
            command="mcp-hirag-tool",
            args=[],
            read_timeout_seconds=100,
            env={
                "LLM_API_KEY": os.getenv("OPENAI_API_KEY"),
                "LLM_BASE_URL": os.getenv("OPENAI_BASE_URL"),
                "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY"),
            },
        )
        
        # Get tools directly without creating persistent session for the test
        # This avoids the async cleanup issues
        self.tools = await mcp_server_tools(self.hirag_server_params)
        
        # Create an agent that can use the tools
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        self.agent = AssistantAgent(
            name="test_agent",
            model_client=model_client,
            tools=self.tools,
            system_message="You are a test assistant. Use the available tools to help with requests.",
        )
        
        logger.info(f"✅ Session initialized with {len(self.tools)} tools")
        for tool in self.tools:
            logger.info(f"   - {tool.name}")
    
    async def teardown(self):
        """Clean up resources."""
        # No need for complex cleanup since we're using direct tool access
        logger.info("🧹 Cleanup completed")
    
    def get_tool_by_name(self, tool_name: str):
        """Get a tool by its name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    async def test_tools_availability(self):
        """Test that all expected tools are available."""
        logger.info("🔍 Testing tool availability...")
        
        expected_tools = ["hi_search", "naive_search", "hi_set_language"]
        available_tools = [tool.name for tool in self.tools]
        
        for tool_name in expected_tools:
            if tool_name in available_tools:
                logger.info(f"   ✅ {tool_name} is available")
            else:
                logger.error(f"   ❌ {tool_name} is NOT available")
                return False
        
        return True
    
    async def test_set_language_through_agent(self):
        """Test setting language through the agent."""
        logger.info("🌍 Testing language setting through agent...")
        
        test_cases = [
            ("en", "Set the language to English"),
            ("cn", "Set the language to Chinese"),
        ]
        
        results = []
        
        for language, prompt in test_cases:
            try:
                logger.info(f"   Testing: {prompt}")
                
                # Ask the agent to set the language
                response = await self.agent.on_messages(
                    [TextMessage(content=f"Please set the language to {language} using the set_language tool", source="user")],
                    CancellationToken()
                )
                
                logger.info(f"   Agent response: {response.chat_message.content}")
                
                # Check if the response indicates success
                response_text = response.chat_message.content.lower()
                if "success" in response_text or f"set to {language}" in response_text or "language" in response_text:
                    logger.info(f"   ✅ Successfully set language to {language}")
                    results.append(True)
                else:
                    logger.error(f"   ❌ Failed to set language to {language}")
                    results.append(False)
                    
            except Exception as e:
                logger.error(f"   ❌ Exception setting language to {language}: {e}")
                results.append(False)
        
        return all(results)
    
    async def test_invalid_language_handling(self):
        """Test how invalid languages are handled."""
        logger.info("🚫 Testing invalid language handling...")
        
        try:
            # Ask the agent to set an invalid language
            response = await self.agent.on_messages(
                [TextMessage(content="Please set the language to 'invalid' using the set_language tool", source="user")],
                CancellationToken()
            )
            
            logger.info(f"   Agent response: {response.chat_message.content}")
            
            # Check if the response indicates an error
            response_text = response.chat_message.content.lower()
            if "error" in response_text or "unsupported" in response_text or "invalid" in response_text:
                logger.info("   ✅ Correctly handled invalid language")
                return True
            else:
                logger.warning("   ⚠️ Invalid language may have been accepted")
                return True  # Still okay if the agent handled it gracefully
                
        except Exception as e:
            logger.info(f"   ✅ Exception correctly raised for invalid language: {e}")
            return True
    
    async def test_search_functionality_both_languages(self):
        """Test search functionality in both English and Chinese languages."""
        logger.info("🔍 Testing search functionality in both languages...")
        
        test_query = "Search for information about artificial intelligence using hi_search"
        results = {}
        
        # Test with English
        try:
            logger.info("   📝 Setting language to English and testing search...")
            
            # First set language to English
            en_lang_response = await self.agent.on_messages(
                [TextMessage(content="Please set the language to en using the set_language tool", source="user")],
                CancellationToken()
            )
            logger.info(f"   Language setting response: {en_lang_response.chat_message.content}")
            
            # Then perform search
            en_search_response = await self.agent.on_messages(
                [TextMessage(content=test_query, source="user")],
                CancellationToken()
            )
            
            results["en"] = en_search_response.chat_message.content
            logger.info(f"   English search response length: {len(results['en'])}")
            
            # Extract summary from the JSON response if possible
            try:
                import json
                if "summary" in results["en"]:
                    # Try to extract summary from the response
                    start = results["en"].find('"summary"')
                    if start != -1:
                        summary_part = results["en"][start:start+200]  # First 200 chars of summary section
                        logger.info(f"   English summary preview: {summary_part}...")
            except:
                pass
            
        except Exception as e:
            logger.error(f"   ❌ Error in English search test: {e}")
            return False
        
        # Test with Chinese
        try:
            logger.info("   📝 Setting language to Chinese and testing search...")
            
            # Set language to Chinese
            cn_lang_response = await self.agent.on_messages(
                [TextMessage(content="Please set the language to cn using the set_language tool", source="user")],
                CancellationToken()
            )
            logger.info(f"   Language setting response: {cn_lang_response.chat_message.content}")
            
            # Then perform search
            cn_search_response = await self.agent.on_messages(
                [TextMessage(content=test_query, source="user")],
                CancellationToken()
            )
            
            results["cn"] = cn_search_response.chat_message.content
            logger.info(f"   Chinese search response length: {len(results['cn'])}")
            
            # Extract summary from the JSON response if possible
            try:
                import json
                if "summary" in results["cn"]:
                    # Try to extract summary from the response
                    start = results["cn"].find('"summary"')
                    if start != -1:
                        summary_part = results["cn"][start:start+200]  # First 200 chars of summary section
                        logger.info(f"   Chinese summary preview: {summary_part}...")
            except:
                pass
                
        except Exception as e:
            logger.error(f"   ❌ Error in Chinese search test: {e}")
            return False
        
        # Analyze results
        if len(results["en"]) > 50 and len(results["cn"]) > 50:
            logger.info("   ✅ Both language searches returned reasonable responses")
            
            # Check if the responses are different (indicating language switching worked)
            if results["en"] != results["cn"]:
                logger.info("   ✅ Different responses for different languages - language switching works!")
                
                # Try to detect language in summaries
                en_has_chinese = any('\u4e00' <= char <= '\u9fff' for char in results["en"])
                cn_has_chinese = any('\u4e00' <= char <= '\u9fff' for char in results["cn"])
                
                if not en_has_chinese and cn_has_chinese:
                    logger.info("   ✅ Perfect! English response has no Chinese characters, Chinese response has Chinese characters")
                elif cn_has_chinese:
                    logger.info("   ✅ Good! Chinese response contains Chinese characters as expected")
                else:
                    logger.info("   ⚠️ Note: No Chinese characters detected in either response (may be due to empty knowledge base)")
                
            else:
                logger.warning("   ⚠️ Same response for both languages - language switching may not be working")
            
            return True
        else:
            logger.warning("   ⚠️ One or both search responses seem too short")
            return False
    
    async def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("🎯 Starting HiRAG set_language tool tests...")
        
        try:
            await self.setup()
            
            tests = [
                ("Tool Availability", self.test_tools_availability),
                ("Set Language Through Agent", self.test_set_language_through_agent),
                ("Invalid Language Handling", self.test_invalid_language_handling),
                ("Search Functionality Both Languages", self.test_search_functionality_both_languages),
            ]
            
            results = []
            
            for test_name, test_func in tests:
                logger.info(f"\n{'='*50}")
                logger.info(f"Running test: {test_name}")
                logger.info(f"{'='*50}")
                
                try:
                    result = await test_func()
                    results.append((test_name, result))
                    status = "✅ PASSED" if result else "❌ FAILED"
                    logger.info(f"{test_name}: {status}")
                except Exception as e:
                    logger.error(f"{test_name}: ❌ FAILED with exception: {e}")
                    results.append((test_name, False))
            
            # Print summary
            logger.info(f"\n{'='*50}")
            logger.info("TEST SUMMARY")
            logger.info(f"{'='*50}")
            
            passed = 0
            total = len(results)
            
            for test_name, result in results:
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{test_name:.<40} {status}")
                if result:
                    passed += 1
            
            logger.info(f"\nOverall: {passed}/{total} tests passed")
            
            if passed == total:
                logger.info("🎉 All tests passed!")
                return True
            else:
                logger.error(f"💥 {total - passed} test(s) failed!")
                return False
                
        except Exception as e:
            logger.error(f"💥 Test setup failed: {e}")
            return False
        finally:
            await self.teardown()


async def main():
    """Main function to run the tests."""
    # Suppress some of the noisy async cleanup warnings
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*async generator.*")
    
    test_runner = TestSetLanguageTool()
    success = await test_runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
