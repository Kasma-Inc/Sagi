import asyncio
import hashlib
import json
import logging
import threading
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from Sagi.factories.model_client_factory import ModelClient
from Sagi.services.model_client_pool import ModelClientPool


class GlobalModelClientManager:
    """
    Global manager for Model Client pools.

    This manager provides a fan interface for managing multiple client pools,
    supporting both async and sync operations with automatic pool creation and management.
    """

    def __init__(self, default_pool_size: int = 10, default_max_idle_time: int = 3600):
        """
        Initialize the GlobalModelClientManager.

        Args:
            default_pool_size: Default pool size for new pools
            default_max_idle_time: Default max idle time for new pools
        """
        # Dictionary to store pools for different client configurations
        self._pools: Dict[str, ModelClientPool] = {}

        # Configuration cache to avoid redundant hashing
        self._config_cache: Dict[str, str] = {}

        # Default pool configuration
        self._default_pool_size = default_pool_size
        self._default_max_idle_time = default_max_idle_time

        # Thread synchronization
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()

        logging.info(
            f"🌐 GlobalModelClientManager initialized with default_pool_size={default_pool_size}, "
            f"default_max_idle_time={default_max_idle_time}s"
        )

    async def get_client_async(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get a client from the appropriate pool asynchronously.

        Args:
            client_type: Type of client to create
            config_path: Path to configuration file
            response_format: Optional response format
            parallel_tool_calls: Whether to enable parallel tool calls

        Returns:
            ModelClient: Available client from pool or newly created
        """
        # Generate pool key based on configuration
        pool_key = await self._get_pool_key_async(client_type, config_path)

        # Get or create pool for this configuration
        pool = await self._get_or_create_pool_async(pool_key)

        # Get client from the pool
        return await pool.get_client_async(
            client_type, config_path, response_format, parallel_tool_calls
        )

    def get_client_sync(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get a client from the appropriate pool synchronously.

        Args:
            client_type: Type of client to create
            config_path: Path to configuration file
            response_format: Optional response format
            parallel_tool_calls: Whether to enable parallel tool calls

        Returns:
            ModelClient: Available client from pool or newly created
        """
        # Generate pool key based on configuration
        pool_key = self._get_pool_key_sync(client_type, config_path)

        # Get or create pool for this configuration
        pool = self._get_or_create_pool_sync(pool_key)

        # Get client from the pool
        return pool.get_client_sync(
            client_type, config_path, response_format, parallel_tool_calls
        )

    async def return_client_async(
        self,
        client: ModelClient,
        client_type: str,
        config_path: str,
    ) -> None:
        """
        Return a client to the appropriate pool asynchronously.

        Args:
            client: Client to return to the pool
            client_type: Type of client
            config_path: Path to configuration file
        """
        # Generate pool key to identify the correct pool
        pool_key = await self._get_pool_key_async(client_type, config_path)

        # Get the pool and return the client
        pool = await self._get_or_create_pool_async(pool_key)
        await pool.return_client_async(client)

    def return_client_sync(
        self,
        client: ModelClient,
        client_type: str,
        config_path: str,
    ) -> None:
        """
        Return a client to the appropriate pool synchronously.

        Args:
            client: Client to return to the pool
            client_type: Type of client
            config_path: Path to configuration file
        """
        # Generate pool key to identify the correct pool
        pool_key = self._get_pool_key_sync(client_type, config_path)

        # Get the pool and return the client
        pool = self._get_or_create_pool_sync(pool_key)
        pool.return_client_sync(client)

    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all managed pools."""
        with self._lock:
            stats = {
                "total_pools": len(self._pools),
                "default_pool_size": self._default_pool_size,
                "default_max_idle_time": self._default_max_idle_time,
                "pools": {},
            }

            for pool_key, pool in self._pools.items():
                pool_stats = pool.get_pool_stats()
                stats["pools"][pool_key] = pool_stats

            return stats

    def cleanup_all_pools(self) -> Dict[str, int]:
        """
        Clean up expired clients in all pools.

        Returns:
            Dict[str, int]: Number of clients cleaned per pool
        """
        cleanup_results = {}

        with self._lock:
            for pool_key, pool in self._pools.items():
                cleaned_count = pool.cleanup_expired_clients()
                if cleaned_count > 0:
                    cleanup_results[pool_key] = cleaned_count

        if cleanup_results:
            total_cleaned = sum(cleanup_results.values())
            logging.info(
                f"🧹 GlobalManager cleaned {total_cleaned} clients across {len(cleanup_results)} pools"
            )

        return cleanup_results

    def clear_all_pools(self) -> None:
        """Clear all clients from all pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.clear_pool()

            self._pools.clear()
            self._config_cache.clear()

            logging.info("🌐🧹 GlobalModelClientManager: all pools cleared")

    async def _get_pool_key_async(self, client_type: str, config_path: str) -> str:
        """Generate a unique key for pool identification asynchronously."""
        async with self._async_lock:
            return self._generate_pool_key(client_type, config_path)

    def _get_pool_key_sync(self, client_type: str, config_path: str) -> str:
        """Generate a unique key for pool identification synchronously."""
        with self._lock:
            return self._generate_pool_key(client_type, config_path)

    def _generate_pool_key(self, client_type: str, config_path: str) -> str:
        """Generate a unique key for pool identification."""
        # Use absolute path for consistency
        import os

        abs_config_path = os.path.abspath(config_path)

        # Create a unique key combining client type and config path
        key_data = f"{client_type}:{abs_config_path}"

        # Use hash for shorter keys while maintaining uniqueness
        key_hash = hashlib.md5(key_data.encode("utf-8")).hexdigest()[:16]

        return f"{client_type}_{key_hash}"

    async def _get_or_create_pool_async(self, pool_key: str) -> ModelClientPool:
        """Get existing pool or create new one asynchronously."""
        async with self._async_lock:
            if pool_key not in self._pools:
                self._pools[pool_key] = ModelClientPool(
                    pool_size=self._default_pool_size,
                    max_idle_time=self._default_max_idle_time,
                )
                logging.info(f"🏊 Created new pool '{pool_key}'")

            return self._pools[pool_key]

    def _get_or_create_pool_sync(self, pool_key: str) -> ModelClientPool:
        """Get existing pool or create new one synchronously."""
        with self._lock:
            if pool_key not in self._pools:
                self._pools[pool_key] = ModelClientPool(
                    pool_size=self._default_pool_size,
                    max_idle_time=self._default_max_idle_time,
                )
                logging.info(f"🏊 Created new pool '{pool_key}' (sync)")

            return self._pools[pool_key]


# Global singleton instance
_global_manager: Optional[GlobalModelClientManager] = None
_manager_lock = threading.Lock()


def get_global_model_client_manager() -> GlobalModelClientManager:
    """
    Get the global singleton instance of ModelClientManager.

    Returns:
        GlobalModelClientManager: The singleton instance
    """
    global _global_manager

    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                _global_manager = GlobalModelClientManager()
                logging.info("🌐 Created global ModelClientManager singleton")

    return _global_manager


def reset_global_model_client_manager() -> None:
    """Reset the global manager (mainly for testing purposes)."""
    global _global_manager

    with _manager_lock:
        if _global_manager is not None:
            _global_manager.clear_all_pools()
            _global_manager = None
            logging.info("🌐 Reset global ModelClientManager singleton")
