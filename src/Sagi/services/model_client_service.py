import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel

from Sagi.factories.model_client_factory import ModelClient, ModelClientFactory
from Sagi.utils.load_config import load_toml_with_env_vars


class ModelClientService:
    """
    Service for managing Model Client lifecycle with caching and thread safety.

    This service provides centralized management of OpenAI and Anthropic chat completion clients,
    implementing caching to avoid redundant client creation and ensuring thread safety
    for concurrent access.
    """

    def __init__(self, max_cache_size: int = 100, cache_ttl: int = 3600):
        # LRU cache for created model clients with access tracking
        self._clients: OrderedDict[str, Tuple[ModelClient, float]] = OrderedDict()

        # Cache for loaded configuration files to avoid redundant I/O operations
        self._config_cache: Dict[str, Dict[str, Any]] = {}

        # Cache configuration
        self._max_cache_size = max_cache_size
        self._cache_ttl = cache_ttl  # TTL in seconds

        # Async lock for thread-safe client creation
        self._lock = asyncio.Lock()

        logging.info(
            f"🔧 ModelClientService initialized with max_cache_size={max_cache_size}, cache_ttl={cache_ttl}s"
        )

    async def get_client(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get or create a Model Client with caching support.

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

        # Build cache key
        cache_key = self._build_cache_key(
            client_type, response_format, parallel_tool_calls, config_path
        )

        # Fast cache check without lock
        if cache_key in self._clients:
            client, timestamp = self._clients[cache_key]
            current_time = time.time()

            # Check if cache entry is still valid (TTL)
            if current_time - timestamp < self._cache_ttl:
                # Move to end for LRU (most recently used)
                self._clients.move_to_end(cache_key)
                logging.debug(f"🔍 Model client '{client_type}' found in cache")
                return client
            else:
                # Cache entry expired, remove it
                del self._clients[cache_key]
                logging.debug(f"⏰ Model client '{client_type}' cache expired")

        # Thread-safe client creation
        async with self._lock:
            # Double-check pattern: verify cache again after acquiring lock
            if cache_key in self._clients:
                client, timestamp = self._clients[cache_key]
                current_time = time.time()

                if current_time - timestamp < self._cache_ttl:
                    self._clients.move_to_end(cache_key)
                    logging.debug(
                        f"🔍 Model client '{client_type}' found in cache after lock"
                    )
                    return client
                else:
                    del self._clients[cache_key]

            logging.info(f"🔧 Creating new model client '{client_type}'")

            try:
                # Load configuration
                config = self._load_client_config(config_path, client_type)

                # Handle special logic for single_tool_use_client
                if (
                    parallel_tool_calls is None
                    and client_type == "single_tool_use_client"
                ):
                    parallel_tool_calls_setting = config.get("parallel_tool_calls")
                    if parallel_tool_calls_setting is True:
                        parallel_tool_calls = True

                # Create Model Client using factory
                client = ModelClientFactory.create_model_client(
                    config,
                    response_format=response_format,
                    parallel_tool_calls=parallel_tool_calls,
                )

                # Cache the newly created client with timestamp
                current_time = time.time()
                self._clients[cache_key] = (client, current_time)

                # Evict oldest entries if cache is full
                self._evict_if_needed()

                logging.info(f"✅ Model client '{client_type}' created and cached")
                return client

            except Exception as e:
                logging.error(f"❌ Failed to create model client '{client_type}': {e}")
                raise

    async def get_client_by_hash(
        self,
        config_hash: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """Get or create a Model Client using configuration hash as identifier."""
        # Find configuration by hash
        config = self._find_config_by_hash(config_path, config_hash)
        
        # Create Model Client using factory
        return ModelClientFactory.create_model_client(
            config,
            response_format=response_format,
            parallel_tool_calls=parallel_tool_calls,
        )

    def generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate a SHA-256 hash for a configuration dictionary."""
        if not config:
            raise ValueError("Configuration dictionary cannot be empty")
        
        key_fields = ["model", "provider", "base_url", "max_tokens", "parallel_tool_calls"]
        normalized_config = {field: config[field] for field in key_fields if field in config}
        
        if not normalized_config:
            raise ValueError("No valid configuration fields found for hashing")
        
        config_json = json.dumps(normalized_config, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(config_json.encode('utf-8')).hexdigest()

    def get_config_hash(self, config_path: str, client_type: str) -> str:
        """Get or compute the hash for a specific client configuration."""
        config = self._load_client_config(config_path, client_type)
        return self.generate_config_hash(config)

    def _find_config_by_hash(self, config_path: str, target_hash: str) -> Dict[str, Any]:
        """Find a configuration by its hash value."""
        abs_config_path = os.path.abspath(config_path)
        if abs_config_path not in self._config_cache:
            self._config_cache[abs_config_path] = load_toml_with_env_vars(abs_config_path)
        
        config = self._config_cache[abs_config_path]
        if "model_clients" not in config:
            raise ValueError(f"No model_clients section found in {abs_config_path}")
        
        for client_type, client_config in config["model_clients"].items():
            if self.generate_config_hash(client_config) == target_hash:
                return client_config
        
        raise ValueError(f"No configuration found with hash '{target_hash[:8]}...'.")

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
        # Use absolute path to avoid collisions
        abs_config_path = os.path.abspath(config_path)
        key_parts = [client_type, abs_config_path]

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

        # Load and cache configuration file
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

        # Get configuration
        config = self._config_cache[abs_config_path]

        # Validate configuration structure
        if "model_clients" not in config:
            raise KeyError(f"'model_clients' section not found in {abs_config_path}")

        if client_type not in config["model_clients"]:
            available_clients = list(config["model_clients"].keys())
            raise KeyError(
                f"Client type '{client_type}' not found in configuration. "
                f"Available clients: {available_clients}"
            )

        return config["model_clients"][client_type]

    def _evict_if_needed(self):
        """Evict oldest entries from cache if it exceeds max size."""
        while len(self._clients) > self._max_cache_size:
            # Remove the oldest entry (first in OrderedDict)
            oldest_key = next(iter(self._clients))
            del self._clients[oldest_key]
            logging.debug(f"🗑️ Evicted cache entry: {oldest_key}")

    def _cleanup_expired_entries(self):
        """Remove all expired entries from cache."""
        current_time = time.time()
        expired_keys = []

        for key, (client, timestamp) in self._clients.items():
            if current_time - timestamp >= self._cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._clients[key]
            logging.debug(f"⏰ Removed expired cache entry: {key}")

        if expired_keys:
            logging.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

    def clear_cache(self):
        """Clear all cached clients and configurations."""
        self._clients.clear()
        self._config_cache.clear()
        logging.info("🧹 ModelClientService cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0

        for client, timestamp in self._clients.values():
            if current_time - timestamp < self._cache_ttl:
                valid_entries += 1
            else:
                expired_entries += 1

        return {
            "cached_clients": len(self._clients),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cached_configs": len(self._config_cache),
            "max_cache_size": self._max_cache_size,
            "cache_ttl": self._cache_ttl,
        }
