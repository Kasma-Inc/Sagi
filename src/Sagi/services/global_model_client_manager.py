import asyncio
import copy
import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from Sagi.factories.model_client_factory import ModelClient
from Sagi.services.model_client_pool import ModelClientPool
from Sagi.utils.load_config import load_toml_with_env_vars


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

        # Configuration file cache
        self._config_cache: Dict[str, Dict[str, Any]] = {}

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

    async def get_client_by_hash(
        self,
        config_hash: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get a client by configuration hash asynchronously.

        Args:
            config_hash: SHA-256 hash of the configuration
            config_path: Path to configuration file
            response_format: Optional response format
            parallel_tool_calls: Whether to enable parallel tool calls

        Returns:
            ModelClient: Available client from pool or newly created
        """
        # Find client_type by hash
        client_type = self._get_client_type_by_hash(config_path, config_hash)

        # Use regular get_client_async with found client_type
        return await self.get_client_async(
            client_type=client_type,
            config_path=config_path,
            response_format=response_format,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_client_by_hash_sync(
        self,
        config_hash: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get a client by configuration hash synchronously.

        Args:
            config_hash: SHA-256 hash of the configuration
            config_path: Path to configuration file
            response_format: Optional response format
            parallel_tool_calls: Whether to enable parallel tool calls

        Returns:
            ModelClient: Available client from pool or newly created
        """
        # Find client_type by hash
        client_type = self._get_client_type_by_hash(config_path, config_hash)

        # Use regular get_client_sync with found client_type
        return self.get_client_sync(
            client_type=client_type,
            config_path=config_path,
            response_format=response_format,
            parallel_tool_calls=parallel_tool_calls,
        )

    def generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate a SHA-256 hash for a configuration dictionary."""
        if not config:
            raise ValueError("Configuration dictionary cannot be empty")

        key_fields = [
            "model",
            "provider",
            "base_url",
            "max_tokens",
            "parallel_tool_calls",
        ]
        normalized_config = {
            field: config[field] for field in key_fields if field in config
        }

        if not normalized_config:
            raise ValueError("No valid configuration fields found for hashing")

        config_json = json.dumps(normalized_config, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def get_config_hash(self, config_path: str, client_type: str) -> str:
        """Get or compute the hash for a specific client configuration."""
        config = self._load_client_config(config_path, client_type)
        return self.generate_config_hash(config)

    def _get_client_type_by_hash(self, config_path: str, config_hash: str) -> str:
        """Find client_type by configuration hash."""
        abs_config_path = os.path.abspath(config_path)

        with self._lock:
            if abs_config_path not in self._config_cache:
                self._config_cache[abs_config_path] = load_toml_with_env_vars(
                    abs_config_path
                )

            config = self._config_cache[abs_config_path]

            if "model_clients" not in config:
                raise ValueError(f"No model_clients section found in {abs_config_path}")

            for client_type, client_config in config["model_clients"].items():
                if self.generate_config_hash(client_config) == config_hash:
                    return client_type

            raise ValueError(
                f"No configuration found with hash '{config_hash[:8]}...'."
            )

    def _load_client_config(self, config_path: str, client_type: str) -> Dict[str, Any]:
        """
        Load client configuration from file with caching.

        Args:
            config_path: Path to configuration file
            client_type: Type of client configuration to load

        Returns:
            Dict containing client configuration

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            KeyError: If client type is not found in configuration
            ValueError: If configuration file is invalid
        """
        # Convert to absolute path for consistency
        abs_config_path = os.path.abspath(config_path)

        # Check if file exists
        if not os.path.exists(abs_config_path):
            raise FileNotFoundError(f"Configuration file not found: {abs_config_path}")

        # Thread-safe configuration loading and caching
        with self._lock:
            if abs_config_path not in self._config_cache:
                logging.debug(f"📄 Loading configuration from {abs_config_path}")
                try:
                    self._config_cache[abs_config_path] = load_toml_with_env_vars(
                        abs_config_path
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to load configuration from {abs_config_path}: {e}"
                    )

            # Get configuration - all access must be inside lock for thread safety
            config = self._config_cache[abs_config_path]

            # Validate configuration structure
            if "model_clients" not in config:
                raise KeyError(
                    f"'model_clients' section not found in {abs_config_path}"
                )

            if client_type not in config["model_clients"]:
                available_clients = list(config["model_clients"].keys())
                raise KeyError(
                    f"Client type '{client_type}' not found in configuration. "
                    f"Available clients: {available_clients}"
                )

            # Return a deep copy to avoid sharing mutable objects across threads
            return copy.deepcopy(config["model_clients"][client_type])

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        stats = self.get_global_stats()

        with self._lock:
            return {
                "global_manager_stats": stats,
                "cached_config_files": len(self._config_cache),
            }

    # Convenience methods
    async def get_client(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """Alias for get_client_async for backward compatibility."""
        return await self.get_client_async(
            client_type, config_path, response_format, parallel_tool_calls
        )

    def clear_cache(self) -> None:
        """Alias for clear_all_pools for backward compatibility."""
        self.clear_all_pools()

    def cleanup_pools(self) -> Dict[str, int]:
        """Alias for cleanup_all_pools for backward compatibility."""
        return self.cleanup_all_pools()

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
