import asyncio
import copy
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel

from Sagi.factories.model_client_factory import ModelClient, ModelClientFactory
from Sagi.services.global_model_client_manager import get_global_model_client_manager
from Sagi.utils.load_config import load_toml_with_env_vars


class ModelClientService:
    """
    Service for managing Model Client lifecycle with improved pool-based architecture.

    This service now uses the GlobalModelClientManager for efficient client pooling,
    supporting both async and sync operations with automatic client reuse and cleanup.
    """

    def __init__(self, max_cache_size: int = 100, cache_ttl: int = 3600):
        # Use global manager for client pooling
        self._global_manager = get_global_model_client_manager()

        # Legacy cache for loaded configuration files (kept for backward compatibility)
        self._config_cache: Dict[str, Dict[str, Any]] = {}

        # Legacy configuration (kept for backward compatibility)
        self._max_cache_size = max_cache_size
        self._cache_ttl = cache_ttl  # TTL in seconds

        # Thread-safe lock for configuration loading
        self._lock = threading.RLock()

        logging.info(
            f"🔧 ModelClientService initialized using GlobalModelClientManager "
            f"(legacy params: max_cache_size={max_cache_size}, cache_ttl={cache_ttl}s)"
        )

    async def get_client(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get or create a Model Client using the global pool manager.

        Args:
            client_type: Type of client to create (e.g., "orchestrator_client")
            config_path: Path to configuration file
            response_format: Optional response format for structured output
            parallel_tool_calls: Whether to enable parallel tool calls

        Returns:
            ModelClient: The requested model client (OpenAI or Anthropic)

        Raises:
            ValueError: If input parameters are invalid
            KeyError: If client type is not found in configuration
            FileNotFoundError: If configuration file doesn't exist
        """
        # Input validation
        if not client_type or not isinstance(client_type, str):
            raise ValueError("client_type must be a non-empty string")
        if not config_path or not isinstance(config_path, str):
            raise ValueError("config_path must be a non-empty string")

        # Use global manager to get client from pool
        return await self._global_manager.get_client_async(
            client_type=client_type,
            config_path=config_path,
            response_format=response_format,
            parallel_tool_calls=parallel_tool_calls,
        )

    async def get_client_by_hash(
        self,
        config_hash: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get or create a Model Client using configuration hash as identifier.

        This method finds the client_type by hash and then uses the pool manager.
        """
        # Input validation
        if not config_hash or not isinstance(config_hash, str):
            raise ValueError("config_hash must be a non-empty string")
        if not config_path or not isinstance(config_path, str):
            raise ValueError("config_path must be a non-empty string")

        try:
            # Find configuration and client_type by hash
            client_type = self._get_client_type_by_hash(config_path, config_hash)

            # Use global manager to get client from pool
            return await self._global_manager.get_client_async(
                client_type=client_type,
                config_path=config_path,
                response_format=response_format,
                parallel_tool_calls=parallel_tool_calls,
            )

        except Exception as e:
            logging.error(
                f"❌ Failed to create model client with hash '{config_hash[:8]}...': {e}"
            )
            raise

    async def return_client_async(
        self,
        client: ModelClient,
        client_type: str,
        config_path: str,
    ) -> None:
        """
        Return a client to the pool asynchronously.

        Args:
            client: Client to return to the pool
            client_type: Type of client
            config_path: Path to configuration file
        """
        await self._global_manager.return_client_async(
            client=client,
            client_type=client_type,
            config_path=config_path,
        )

    def return_client_sync(
        self,
        client: ModelClient,
        client_type: str,
        config_path: str,
    ) -> None:
        """
        Return a client to the pool synchronously.

        Args:
            client: Client to return to the pool
            client_type: Type of client
            config_path: Path to configuration file
        """
        self._global_manager.return_client_sync(
            client=client,
            client_type=client_type,
            config_path=config_path,
        )

    def get_client_sync(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get or create a Model Client synchronously using the global pool manager.

        Args:
            client_type: Type of client to create (e.g., "orchestrator_client")
            config_path: Path to configuration file
            response_format: Optional response format for structured output
            parallel_tool_calls: Whether to enable parallel tool calls

        Returns:
            ModelClient: The requested model client (OpenAI or Anthropic)
        """
        # Input validation
        if not client_type or not isinstance(client_type, str):
            raise ValueError("client_type must be a non-empty string")
        if not config_path or not isinstance(config_path, str):
            raise ValueError("config_path must be a non-empty string")

        # Use global manager to get client from pool
        return self._global_manager.get_client_sync(
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

    def _find_config_by_hash(
        self, config_path: str, target_hash: str
    ) -> Dict[str, Any]:
        """Find a configuration by its hash value."""
        abs_config_path = os.path.abspath(config_path)

        with self._lock:
            if abs_config_path not in self._config_cache:
                self._config_cache[abs_config_path] = load_toml_with_env_vars(
                    abs_config_path
                )

            config = self._config_cache[abs_config_path]

            # All config access must be inside the lock for thread safety
            if "model_clients" not in config:
                raise ValueError(f"No model_clients section found in {abs_config_path}")

            for client_type, client_config in config["model_clients"].items():
                if self.generate_config_hash(client_config) == target_hash:
                    # Return a deep copy to avoid sharing mutable objects across threads
                    return copy.deepcopy(client_config)

            raise ValueError(
                f"No configuration found with hash '{target_hash[:8]}...'."
            )

    def _build_cache_key(
        self,
        client_type: str,
        response_format: Optional[Type[BaseModel]],
        parallel_tool_calls: Optional[bool],
        config_path: str,
    ) -> str:
        """
        Build cache key for client identification.

        Uses absolute path to avoid cache key collisions when multiple config files
        have the same name but different paths.
        """
        # Input validation (defensive programming)
        if not client_type:
            raise ValueError("client_type cannot be empty")
        if not config_path:
            raise ValueError("config_path cannot be empty")

        # Use absolute path to avoid collisions
        abs_config_path = os.path.abspath(config_path)
        key_parts = [client_type, abs_config_path]

        if response_format is not None:
            key_parts.append(f"format_{response_format.__name__}")

        if parallel_tool_calls is not None:
            key_parts.append(f"parallel_{parallel_tool_calls}")

        return "_".join(key_parts)

    def _build_cache_key_by_hash(
        self,
        config_hash: str,
        response_format: Optional[Type[BaseModel]],
        parallel_tool_calls: Optional[bool],
        config_path: str,
    ) -> str:
        """
        Build cache key for client identification using configuration hash.

        Uses config hash as the primary identifier instead of client_type.
        """
        # Input validation (defensive programming)
        if not config_hash:
            raise ValueError("config_hash cannot be empty")
        if not config_path:
            raise ValueError("config_path cannot be empty")

        # Use absolute path to avoid collisions
        abs_config_path = os.path.abspath(config_path)
        key_parts = [f"hash_{config_hash}", abs_config_path]

        if response_format is not None:
            key_parts.append(f"format_{response_format.__name__}")

        if parallel_tool_calls is not None:
            key_parts.append(f"parallel_{parallel_tool_calls}")

        return "_".join(key_parts)

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

    # Legacy methods removed - pool management now handled by GlobalModelClientManager

    def clear_cache(self):
        """Clear all cached clients and configurations."""
        # Clear global pools
        self._global_manager.clear_all_pools()
        
        # Clear local config cache
        with self._lock:
            self._config_cache.clear()
            logging.info("🧹 ModelClientService cache cleared (using global manager)")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current service state (now using global pool manager)."""
        # Get stats from global manager
        global_stats = self._global_manager.get_global_stats()
        
        # Add legacy cache info for backward compatibility
        with self._lock:
            return {
                "global_manager_stats": global_stats,
                "cached_configs": len(self._config_cache),
                "legacy_max_cache_size": self._max_cache_size,
                "legacy_cache_ttl": self._cache_ttl,
            }

    def cleanup_pools(self) -> Dict[str, int]:
        """Clean up expired clients in all pools."""
        return self._global_manager.cleanup_all_pools()
