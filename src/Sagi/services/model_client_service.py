import asyncio
import logging
import os
from typing import Any, Dict, Optional, Type

from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from Sagi.utils.load_config import load_toml_with_env_vars
from Sagi.factories.model_client_factory import ModelClientFactory


class ModelClientService:
    """
    Service for managing Model Client lifecycle with caching and thread safety.
    
    This service provides centralized management of OpenAI chat completion clients,
    implementing caching to avoid redundant client creation and ensuring thread safety
    for concurrent access.
    """
    
    def __init__(self):
        # Cache for created model clients to avoid redundant creation
        self._clients: Dict[str, OpenAIChatCompletionClient] = {}
        
        # Cache for loaded configuration files to avoid redundant I/O operations
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
        # Async lock for thread-safe client creation
        self._lock = asyncio.Lock()
        
        logging.info("🔧 ModelClientService initialized")
    
    async def get_client(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> OpenAIChatCompletionClient:
        """
        Get or create a Model Client with caching support.
        
        Args:
            client_type: Type of client to create (e.g., "orchestrator_client")
            config_path: Path to configuration file
            response_format: Optional response format for structured output
            parallel_tool_calls: Whether to enable parallel tool calls
            
        Returns:
            OpenAIChatCompletionClient: The requested model client
            
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
        cache_key = self._build_cache_key(client_type, response_format, parallel_tool_calls, config_path)
        
        # Fast cache check without lock
        if cache_key in self._clients:
            logging.debug(f"🔍 Model client '{client_type}' found in cache")
            return self._clients[cache_key]
        
        # Thread-safe client creation
        async with self._lock:
            # Double-check pattern: verify cache again after acquiring lock
            if cache_key in self._clients:
                logging.debug(f"🔍 Model client '{client_type}' found in cache after lock")
                return self._clients[cache_key]
            
            logging.info(f"🔧 Creating new model client '{client_type}'")
            
            try:
                # Load configuration
                config = self._load_client_config(config_path, client_type)
                
                # Handle special logic for single_tool_use_client
                if parallel_tool_calls is None and client_type == "single_tool_use_client":
                    parallel_tool_calls_setting = config.get("parallel_tool_calls")
                    if parallel_tool_calls_setting is True:
                        parallel_tool_calls = True
                
                # Create Model Client using factory
                client = ModelClientFactory.create_model_client(
                    config,
                    response_format=response_format,
                    parallel_tool_calls=parallel_tool_calls,
                )
                
                # Cache the newly created client
                self._clients[cache_key] = client
                
                logging.info(f"✅ Model client '{client_type}' created and cached")
                return client
                
            except Exception as e:
                logging.error(f"❌ Failed to create model client '{client_type}': {e}")
                raise
    
    def _build_cache_key(
        self,
        client_type: str,
        response_format: Optional[Type[BaseModel]],
        parallel_tool_calls: Optional[bool],
        config_path: str,
    ) -> str:
        """
        Build cache key for client identification.
        
        Simplified approach using base filename instead of complex hashing
        as per the technical report requirements.
        """
        key_parts = [client_type, os.path.basename(config_path)]
        
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
                self._config_cache[abs_config_path] = load_toml_with_env_vars(abs_config_path)
            except Exception as e:
                raise ValueError(f"Failed to load configuration from {abs_config_path}: {e}")
        
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
    
    def clear_cache(self):
        """Clear all cached clients and configurations."""
        self._clients.clear()
        self._config_cache.clear()
        logging.info("🧹 ModelClientService cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about current cache state."""
        return {
            "cached_clients": len(self._clients),
            "cached_configs": len(self._config_cache)
        }