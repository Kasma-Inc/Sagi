import asyncio
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from Sagi.factories.model_client_factory import ModelClient, ModelClientFactory
from Sagi.utils.load_config import load_toml_with_env_vars


class ModelClientPool:
    """
    Thread-safe object pool for Model Clients.

    This pool manages a collection of model clients for efficient reuse,
    supporting both async and sync operations with proper lifecycle management.
    """

    def __init__(self, pool_size: int = 10, max_idle_time: int = 3600):
        """
        Initialize the ModelClientPool.

        Args:
            pool_size: Maximum number of clients to keep in the pool
            max_idle_time: Maximum time (seconds) a client can stay idle before cleanup
        """
        # Thread-safe queue for available clients
        self._available_clients: queue.Queue[ModelClient] = queue.Queue(
            maxsize=pool_size
        )

        # Track clients currently in use
        self._in_use_clients: Dict[int, ModelClient] = {}

        # Client metadata for tracking creation time and usage
        self._client_metadata: Dict[int, Dict[str, Any]] = {}

        # Pool configuration
        self._pool_size = pool_size
        self._max_idle_time = max_idle_time

        # Pool statistics
        self._created_count = 0
        self._reused_count = 0
        self._cleanup_count = 0

        logging.info(
            f"🏊 ModelClientPool initialized with pool_size={pool_size}, max_idle_time={max_idle_time}s"
        )

    async def get_client_async(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get a client from the pool asynchronously.

        Args:
            client_type: Type of client to create
            config_path: Path to configuration file
            response_format: Optional response format
            parallel_tool_calls: Whether to enable parallel tool calls

        Returns:
            ModelClient: Available client from pool or newly created
        """
        # Try to get an available client from the pool
        try:
            client = self._available_clients.get_nowait()

            # Check if client is still valid (not expired)
            client_id = id(client)
            if self._is_client_valid(client_id):
                # Move client to in-use collection
                self._in_use_clients[client_id] = client
                self._reused_count += 1

                logging.debug(f"🔄 Reused client '{client_type}' from pool")
                return client
            else:
                # Client expired, remove from metadata
                self._client_metadata.pop(client_id, None)
                self._cleanup_count += 1

        except queue.Empty:
            # No available clients in pool
            pass

        # Create new client
        client = await self._create_new_client(
            client_type, config_path, response_format, parallel_tool_calls
        )

        # Track the new client
        client_id = id(client)
        self._in_use_clients[client_id] = client
        self._client_metadata[client_id] = {
            "created_at": time.time(),
            "last_used": time.time(),
            "client_type": client_type,
            "config_path": config_path,
        }
        self._created_count += 1

        logging.info(f"🆕 Created new client '{client_type}' for pool")
        return client

    def get_client_sync(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Get a client from the pool synchronously.

        Args:
            client_type: Type of client to create
            config_path: Path to configuration file
            response_format: Optional response format
            parallel_tool_calls: Whether to enable parallel tool calls

        Returns:
            ModelClient: Available client from pool or newly created
        """
        # Try to get an available client from the pool
        try:
            client = self._available_clients.get_nowait()

            # Check if client is still valid (not expired)
            client_id = id(client)
            if self._is_client_valid(client_id):
                # Move client to in-use collection
                self._in_use_clients[client_id] = client
                self._reused_count += 1

                logging.debug(f"🔄 Reused client '{client_type}' from pool (sync)")
                return client
            else:
                # Client expired, remove from metadata
                self._client_metadata.pop(client_id, None)
                self._cleanup_count += 1

        except queue.Empty:
            # No available clients in pool
            pass

        # Create new client synchronously
        client = self._create_new_client_sync(
            client_type, config_path, response_format, parallel_tool_calls
        )

        # Track the new client
        client_id = id(client)
        self._in_use_clients[client_id] = client
        self._client_metadata[client_id] = {
            "created_at": time.time(),
            "last_used": time.time(),
            "client_type": client_type,
            "config_path": config_path,
        }
        self._created_count += 1

        logging.info(f"🆕 Created new client '{client_type}' for pool (sync)")
        return client

    async def return_client_async(self, client: ModelClient) -> None:
        """
        Return a client to the pool asynchronously.

        Args:
            client: Client to return to the pool
        """
        client_id = id(client)

        # Remove from in-use collection
        if client_id in self._in_use_clients:
            del self._in_use_clients[client_id]

            # Update last used time
            if client_id in self._client_metadata:
                self._client_metadata[client_id]["last_used"] = time.time()

            # Try to return to pool if there's space
            try:
                self._available_clients.put_nowait(client)
                logging.debug(f"↩️ Returned client to pool (async)")
            except queue.Full:
                # Pool is full, discard the client
                self._client_metadata.pop(client_id, None)
                logging.debug(f"🗑️ Pool full, discarded client (async)")
        else:
            logging.warning(f"⚠️ Attempted to return unknown client (async)")

    def return_client_sync(self, client: ModelClient) -> None:
        """
        Return a client to the pool synchronously.

        Args:
            client: Client to return to the pool
        """
        client_id = id(client)

        # Remove from in-use collection
        if client_id in self._in_use_clients:
            del self._in_use_clients[client_id]

            # Update last used time
            if client_id in self._client_metadata:
                self._client_metadata[client_id]["last_used"] = time.time()

            # Try to return to pool if there's space
            try:
                self._available_clients.put_nowait(client)
                logging.debug(f"↩️ Returned client to pool (sync)")
            except queue.Full:
                # Pool is full, discard the client
                self._client_metadata.pop(client_id, None)
                logging.debug(f"🗑️ Pool full, discarded client (sync)")
        else:
            logging.warning(f"⚠️ Attempted to return unknown client (sync)")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        return {
            "pool_size": self._pool_size,
            "available_clients": self._available_clients.qsize(),
            "in_use_clients": len(self._in_use_clients),
            "total_created": self._created_count,
            "total_reused": self._reused_count,
            "total_cleaned": self._cleanup_count,
            "max_idle_time": self._max_idle_time,
        }

    def cleanup_expired_clients(self) -> int:
        """
        Remove expired clients from the pool.

        Returns:
            int: Number of clients cleaned up
        """
        cleaned_count = 0
        current_time = time.time()

        # Clean up available clients that are expired
        available_clients = []

        while not self._available_clients.empty():
            try:
                client = self._available_clients.get_nowait()
                client_id = id(client)

                if self._is_client_valid(client_id, current_time):
                    available_clients.append(client)
                else:
                    # Client expired
                    self._client_metadata.pop(client_id, None)
                    cleaned_count += 1
                    self._cleanup_count += 1

            except queue.Empty:
                break

        # Put back valid clients
        for client in available_clients:
            try:
                self._available_clients.put_nowait(client)
            except queue.Full:
                # Should not happen, but just in case
                break

        if cleaned_count > 0:
            logging.info(f"🧹 Cleaned up {cleaned_count} expired clients from pool")

        return cleaned_count

    def clear_pool(self) -> None:
        """Clear all clients from the pool."""
        # Clear available clients
        while not self._available_clients.empty():
            try:
                self._available_clients.get_nowait()
            except queue.Empty:
                break

        # Clear tracking data
        self._in_use_clients.clear()
        self._client_metadata.clear()

        # Reset statistics
        self._created_count = 0
        self._reused_count = 0
        self._cleanup_count = 0

        logging.info("🧹 ModelClientPool cleared")

    def _is_client_valid(
        self, client_id: int, current_time: Optional[float] = None
    ) -> bool:
        """Check if a client is still valid (not expired)."""
        if current_time is None:
            current_time = time.time()

        metadata = self._client_metadata.get(client_id)
        if not metadata:
            return False

        last_used = metadata.get("last_used", 0)
        return (current_time - last_used) < self._max_idle_time

    async def _create_new_client(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """Create a new client asynchronously."""
        # Load configuration
        config = self._load_client_config(config_path, client_type)

        # Handle special logic for single_tool_use_client
        if parallel_tool_calls is None and client_type == "single_tool_use_client":
            parallel_tool_calls_setting = config.get("parallel_tool_calls")
            if parallel_tool_calls_setting is True:
                parallel_tool_calls = True

        # Add runtime parameters to config if provided
        if response_format is not None:
            config["response_format"] = response_format
        if parallel_tool_calls is not None:
            config["parallel_tool_calls"] = parallel_tool_calls

        # Create client using factory
        return ModelClientFactory.create_model_client(config)

    def _create_new_client_sync(
        self,
        client_type: str,
        config_path: str,
        response_format: Optional[Type[BaseModel]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """Create a new client synchronously."""
        # Load configuration
        config = self._load_client_config(config_path, client_type)

        # Handle special logic for single_tool_use_client
        if parallel_tool_calls is None and client_type == "single_tool_use_client":
            parallel_tool_calls_setting = config.get("parallel_tool_calls")
            if parallel_tool_calls_setting is True:
                parallel_tool_calls = True

        # Add runtime parameters to config if provided
        if response_format is not None:
            config["response_format"] = response_format
        if parallel_tool_calls is not None:
            config["parallel_tool_calls"] = parallel_tool_calls

        # Create client using factory
        return ModelClientFactory.create_model_client(config)

    def _load_client_config(self, config_path: str, client_type: str) -> Dict[str, Any]:
        """Load client configuration from file."""
        # Convert to absolute path for consistency
        abs_config_path = os.path.abspath(config_path)

        # Check if file exists
        if not os.path.exists(abs_config_path):
            raise FileNotFoundError(f"Configuration file not found: {abs_config_path}")

        # Load configuration
        try:
            config = load_toml_with_env_vars(abs_config_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from {abs_config_path}: {e}"
            )

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
