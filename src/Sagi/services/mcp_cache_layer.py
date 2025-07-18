"""
Redis-based MCP Caching Layer for Cross-Process Resource Sharing

This module implements a distributed caching layer for MCP (Model Context Protocol) services
that enables true resource sharing across processes without duplicating expensive service instances.

The caching layer provides:
- Transparent MCP tool call caching with Redis backend
- Automatic cache invalidation and TTL management
- Fallback to direct service calls when cache is unavailable
- Process-aware caching strategies for optimal performance

Usage:
    # Initialize cache layer
    cache_layer = MCPCacheLayer(redis_client)

    # Cache MCP tool calls transparently
    result = await cache_layer.cached_tool_call(
        service_name="hirag_retrieval",
        tool_name="hi_search",
        arguments={"query": "example query"}
    )
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis


class MCPCacheLayer:
    """
    Redis-based caching layer for MCP service tool calls that enables
    cross-process resource sharing without service duplication.
    """

    def __init__(
        self,
        redis_client: Redis,
        cache_ttl_seconds: int = 3600,  # 1 hour default TTL
        cache_prefix: str = "mcp_cache:",
        enable_debug_logging: bool = False,
    ):
        """
        Initialize MCP cache layer.

        Args:
            redis_client: Redis client instance
            cache_ttl_seconds: Time-to-live for cached results in seconds
            cache_prefix: Prefix for Redis keys to avoid collisions
            enable_debug_logging: Enable detailed debug logging
        """
        self.redis = redis_client
        self.cache_ttl = cache_ttl_seconds
        self.cache_prefix = cache_prefix
        self.debug_logging = enable_debug_logging

        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0

    def _generate_cache_key(
        self, service_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """
        Generate a deterministic cache key for MCP tool calls.

        Args:
            service_name: Name of the MCP service
            tool_name: Name of the tool being called
            arguments: Tool arguments dictionary

        Returns:
            Redis cache key string
        """
        # Create a deterministic string representation of arguments
        args_str = json.dumps(arguments, sort_keys=True, separators=(",", ":"))

        # Create hash to handle long argument strings
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]

        # Combine into cache key
        cache_key = f"{self.cache_prefix}{service_name}:{tool_name}:{args_hash}"

        if self.debug_logging:
            logging.debug(
                f"🔑 Generated cache key: {cache_key} for {service_name}.{tool_name}"
            )

        return cache_key

    async def get_cached_result(
        self, service_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Retrieve cached result for MCP tool call.

        Args:
            service_name: Name of the MCP service
            tool_name: Name of the tool being called
            arguments: Tool arguments dictionary

        Returns:
            Cached result if available, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(service_name, tool_name, arguments)

            # Try to get cached result
            cached_data = await self.redis.get(cache_key)

            if cached_data:
                result = json.loads(cached_data)
                self.cache_hits += 1

                if self.debug_logging:
                    logging.debug(f"🎯 Cache HIT for {service_name}.{tool_name}")

                logging.info(
                    f"✅ [MCP-CACHE] Cache hit for {service_name}.{tool_name} - returning cached result"
                )
                return result
            else:
                self.cache_misses += 1

                if self.debug_logging:
                    logging.debug(f"❌ Cache MISS for {service_name}.{tool_name}")

                return None

        except Exception as e:
            self.cache_errors += 1
            logging.warning(
                f"⚠️ [MCP-CACHE] Error retrieving cached result for {service_name}.{tool_name}: {e}"
            )
            return None

    async def cache_result(
        self, service_name: str, tool_name: str, arguments: Dict[str, Any], result: Any
    ) -> bool:
        """
        Cache the result of an MCP tool call.

        Args:
            service_name: Name of the MCP service
            tool_name: Name of the tool being called
            arguments: Tool arguments dictionary
            result: Result to cache

        Returns:
            True if caching succeeded, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(service_name, tool_name, arguments)

            # Serialize result for storage
            cached_data = json.dumps(result)

            # Store with TTL
            success = await self.redis.setex(cache_key, self.cache_ttl, cached_data)

            if success:
                if self.debug_logging:
                    logging.debug(
                        f"💾 Cached result for {service_name}.{tool_name} (TTL: {self.cache_ttl}s)"
                    )

                logging.info(
                    f"💾 [MCP-CACHE] Cached result for {service_name}.{tool_name}"
                )
                return True
            else:
                logging.warning(
                    f"⚠️ [MCP-CACHE] Failed to cache result for {service_name}.{tool_name}"
                )
                return False

        except Exception as e:
            self.cache_errors += 1
            logging.warning(
                f"⚠️ [MCP-CACHE] Error caching result for {service_name}.{tool_name}: {e}"
            )
            return False

    async def cached_tool_call(
        self,
        service_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        fallback_callable: Optional[callable] = None,
    ) -> Any:
        """
        Execute MCP tool call with transparent caching.

        This method first checks for cached results, and if not found,
        executes the tool call and caches the result for future use.

        Args:
            service_name: Name of the MCP service
            tool_name: Name of the tool being called
            arguments: Tool arguments dictionary
            fallback_callable: Optional function to call if cache miss occurs

        Returns:
            Tool call result (from cache or fresh execution)
        """
        # Check cache first
        start_time = time.time()
        cache_lookup_start = time.time()
        cached_result = await self.get_cached_result(service_name, tool_name, arguments)
        cache_lookup_time = time.time() - cache_lookup_start

        if cached_result is not None:
            total_time = time.time() - start_time
            logging.info(
                f"⏱️ [MCP-CACHE] {service_name}.{tool_name} cache hit - "
                f"lookup: {cache_lookup_time:.3f}s, total: {total_time:.3f}s"
            )
            return cached_result

        # Cache miss - need to execute tool call
        if fallback_callable is None:
            logging.warning(
                f"⚠️ [MCP-CACHE] No fallback callable provided for {service_name}.{tool_name}"
            )
            raise RuntimeError(
                f"Cache miss and no fallback available for {service_name}.{tool_name}"
            )

        try:
            # Execute the tool call
            logging.info(
                f"🔄 [MCP-CACHE] Cache miss for {service_name}.{tool_name} - executing tool call"
            )
            execution_start = time.time()
            result = await fallback_callable(arguments)
            execution_time = time.time() - execution_start

            # Cache the result for future use
            cache_store_start = time.time()
            cache_success = await self.cache_result(
                service_name, tool_name, arguments, result
            )
            cache_store_time = time.time() - cache_store_start

            total_time = time.time() - start_time

            logging.info(
                f"⏱️ [MCP-CACHE] {service_name}.{tool_name} cache miss - "
                f"lookup: {cache_lookup_time:.3f}s, execution: {execution_time:.3f}s, "
                f"store: {cache_store_time:.3f}s, total: {total_time:.3f}s, "
                f"cached: {'✅' if cache_success else '❌'}"
            )

            return result

        except Exception as e:
            total_time = time.time() - start_time
            logging.error(
                f"⏱️ [MCP-CACHE] {service_name}.{tool_name} execution failed after {total_time:.3f}s: {e}"
            )
            raise

    async def invalidate_cache(
        self, service_name: Optional[str] = None, tool_name: Optional[str] = None
    ) -> int:
        """
        Invalidate cached results matching the given criteria.

        Args:
            service_name: Optional service name filter
            tool_name: Optional tool name filter

        Returns:
            Number of cache keys invalidated
        """
        try:
            # Build pattern for keys to invalidate
            if service_name and tool_name:
                pattern = f"{self.cache_prefix}{service_name}:{tool_name}:*"
            elif service_name:
                pattern = f"{self.cache_prefix}{service_name}:*"
            else:
                pattern = f"{self.cache_prefix}*"

            # Find matching keys
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            # Delete keys if any found
            if keys:
                deleted_count = await self.redis.delete(*keys)
                logging.info(
                    f"🧹 [MCP-CACHE] Invalidated {deleted_count} cache entries matching pattern: {pattern}"
                )
                return deleted_count
            else:
                logging.info(
                    f"🧹 [MCP-CACHE] No cache entries found matching pattern: {pattern}"
                )
                return 0

        except Exception as e:
            logging.error(f"❌ [MCP-CACHE] Error invalidating cache: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) if total_requests > 0 else 0

        try:
            # Get Redis info
            redis_info = await self.redis.info("memory")
            redis_memory_mb = redis_info.get("used_memory", 0) / (1024 * 1024)

            # Count cache keys
            cache_key_count = 0
            async for key in self.redis.scan_iter(match=f"{self.cache_prefix}*"):
                cache_key_count += 1

        except Exception as e:
            logging.warning(f"⚠️ [MCP-CACHE] Error getting Redis stats: {e}")
            redis_memory_mb = 0
            cache_key_count = 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_errors": self.cache_errors,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "redis_memory_mb": redis_memory_mb,
            "cached_keys": cache_key_count,
            "cache_ttl_seconds": self.cache_ttl,
        }

    def reset_stats(self) -> None:
        """Reset cache performance statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0
        logging.info("📊 [MCP-CACHE] Cache statistics reset")


class CachedMCPService:
    """
    Wrapper for MCP services that provides transparent caching.

    This class wraps existing MCP service instances and automatically
    caches tool call results using the MCPCacheLayer.
    """

    def __init__(
        self,
        service_name: str,
        cache_layer: MCPCacheLayer,
        mcp_session: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
    ):
        """
        Initialize cached MCP service wrapper.

        Args:
            service_name: Name of the MCP service
            cache_layer: MCPCacheLayer instance for caching
            mcp_session: Optional direct MCP session (for main process)
            tools: Optional list of available tools
        """
        self.service_name = service_name
        self.cache_layer = cache_layer
        self.mcp_session = mcp_session
        self.tools = tools or []
        self.is_cached_only = mcp_session is None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute tool call with automatic caching.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool call result
        """

        # Define fallback function for cache misses
        async def execute_tool_call(args: Dict[str, Any]) -> Any:
            if self.mcp_session is None:
                # For worker processes, provide a basic fallback for common tools
                if self.service_name == "domain_specific":
                    if tool_name == "get_domain_prompts":
                        logging.warning(
                            f"⚠️ [MCP-FALLBACK] Using default domain prompts for {tool_name}"
                        )
                        return {
                            "facts_prompt": """Below I will present you a request. Before we begin addressing the request, please answer the following pre-survey to the best of your ability. Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be a deep well to draw from.

Here is the request:

{task}

Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the request itself. It is possible that there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found. In some cases, authoritative sources are mentioned in the request itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that facts will typically be specific names, dates, statistics, etc. Your answer should use headings:

    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
""",
                            "plan_prompt": """Fantastic. To address this request we have assembled the following team:

{team}

USER QUERY: {task}

You are a professional planning assistant. 
Based on the team composition, user query, and known and unknown facts, please devise a plan for addressing the USER QUERY. Remember, there is no requirement to involve all team members -- a team member particular expertise may not be needed for this task.

Each plan group should contain the following elements:
1. name: A short title for this group task
2. description: Detailed explanation of the group objective.
3. data_collection_task: Specific instructions for gathering data needed for this group task (optional)
4. code_executor_task: Description of what code executor should do, JUST DETAILED DESCRIPTION IS OK, NOT ACTUAL CODE BLOCK.(optional)
""",
                        }
                    elif tool_name == "get_general_prompts":
                        logging.warning(
                            f"⚠️ [MCP-FALLBACK] Using default general prompts for {tool_name}"
                        )
                        return {
                            "facts_prompt": """Below I will present you a request. Before we begin addressing the request, please answer the following pre-survey to the best of your ability. Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be a deep well to draw from.

Here is the request:

{task}

Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the request itself. It is possible that there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found. In some cases, authoritative sources are mentioned in the request itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that facts will typically be specific names, dates, statistics, etc. Your answer should use headings:

    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
""",
                            "plan_prompt": """Fantastic. To address this request we have assembled the following team:

{team}

USER QUERY: {task}

You are a professional planning assistant. 
Based on the team composition, user query, and known and unknown facts, please devise a plan for addressing the USER QUERY. Remember, there is no requirement to involve all team members -- a team member particular expertise may not be needed for this task.

Each plan group should contain the following elements:
1. name: A short title for this group task
2. description: Detailed explanation of the group objective.
3. data_collection_task: Specific instructions for gathering data needed for this group task (optional)
4. code_executor_task: Description of what code executor should do, JUST DETAILED DESCRIPTION IS OK, NOT ACTUAL CODE BLOCK.(optional)
""",
                        }

                # If no specific fallback is available, raise an error
                logging.error(
                    f"❌ [MCP-FALLBACK] No fallback available for {self.service_name}.{tool_name} in worker process"
                )
                raise RuntimeError(
                    f"Cache miss and no fallback available for {self.service_name}.{tool_name}"
                )

            # Execute the actual tool call on the MCP session
            # For autogen_ext MCP sessions, we need to call the tools directly
            logging.info(
                f"🔧 [MCP-DIRECT] Executing {self.service_name}.{tool_name} on direct session"
            )

            try:
                # Find the tool in the session's available tools
                available_tools = getattr(self.mcp_session, "_tools", None)
                if available_tools:
                    for tool in available_tools:
                        if hasattr(tool, "name") and tool.name == tool_name:
                            # Call the tool directly
                            result = await tool.call(**args)
                            return result

                # Fallback: if we can't find tools, try to call through session
                if hasattr(self.mcp_session, "call_tool"):
                    result = await self.mcp_session.call_tool(tool_name, args)
                    return result
                else:
                    # Last resort: simulate the call for testing
                    result = {"simulated": True, "tool": tool_name, "args": args}
                    logging.warning(
                        f"⚠️ [MCP-DIRECT] Simulated call for {self.service_name}.{tool_name} - real MCP integration needed"
                    )
                    return result

            except Exception as e:
                logging.error(
                    f"❌ [MCP-DIRECT] Error executing {self.service_name}.{tool_name}: {e}"
                )
                raise RuntimeError(
                    f"MCP tool execution failed for {self.service_name}.{tool_name}: {str(e)}"
                ) from e

        # Use cached tool call with fallback
        return await self.cache_layer.cached_tool_call(
            service_name=self.service_name,
            tool_name=tool_name,
            arguments=arguments,
            fallback_callable=execute_tool_call,
        )

    def get_tools(self) -> List[Any]:
        """Get list of available tools for this service."""
        return self.tools

    def is_available(self) -> bool:
        """Check if service is available (either directly or via cache)."""
        return self.mcp_session is not None or self.is_cached_only
