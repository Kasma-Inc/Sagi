"""
Global Resource Manager for Ofnil Agentic RAG System

This module implements a centralized resource management system with singleton pattern
for managing shared resources across the application including database connections,
Redis connection pools, MCP server sessions, workflow instances, and ModelClientService.

Enhanced with support for shared MCP services across worker processes and model client management.
"""

import asyncio
import logging
import os
import threading
import time
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from autogen_ext.tools.mcp import (
    StdioServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)
from redis.asyncio import ConnectionPool, Redis
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from Sagi.services.model_client_service import ModelClientService

from .mcp_cache_layer import CachedMCPService, MCPCacheLayer

if TYPE_CHECKING:
    from Sagi.workflows import PlanningWorkflow


def timing_logger(operation_name: str):
    """Decorator to log execution time of operations."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logging.info(
                    f"⏱️ [TIMING] {operation_name} completed in {duration:.3f}s"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logging.error(
                    f"⏱️ [TIMING] {operation_name} failed after {duration:.3f}s: {e}"
                )
                raise

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logging.info(
                    f"⏱️ [TIMING] {operation_name} completed in {duration:.3f}s"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logging.error(
                    f"⏱️ [TIMING] {operation_name} failed after {duration:.3f}s: {e}"
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class ResourceManager:
    """
    Singleton Global Resource Manager that manages all shared resources
    across the application lifecycle including database connections,
    Redis connection pools, MCP server sessions, workflow instances,
    and ModelClientService.

    Enhanced with ModelClientService management for centralized service access.
    This class follows the singleton pattern to ensure only one instance
    exists throughout the application's lifecycle.
    """

    _instances: Dict[int, "ResourceManager"] = {}  # Per-process instances
    _lock = threading.Lock()
    _model_client_service: Optional[ModelClientService] = None

    def __new__(cls) -> "ResourceManager":
        """Ensure singleton pattern - one instance per process."""
        process_id = os.getpid()

        with cls._lock:  # Move lock outside to prevent race condition
            if process_id not in cls._instances:
                cls._instances[process_id] = super().__new__(cls)
        return cls._instances[process_id]

    def __init__(self):
        """Initialize the resource manager (called only once per process)."""
        # Check if this specific instance has been initialized
        if not hasattr(self, "_instance_initialized") or not self._instance_initialized:
            # Database resources
            self._db_engine: Optional[AsyncEngine] = None
            self._session_maker: Optional[async_sessionmaker] = None

            # Redis resources
            self._redis_pool: Optional[ConnectionPool] = None

            # MCP resources - direct sessions in main process, lightweight in workers
            self._mcp_sessions: Dict[str, Any] = {}
            self._mcp_tools_cache: Dict[str, List[Any]] = {}
            self._mcp_exit_stack: Optional[AsyncExitStack] = None
            self._use_shared_mcp: bool = False

            # MCP Caching Layer for cross-process resource sharing
            self._mcp_cache_layer: Optional[MCPCacheLayer] = None
            self._cached_mcp_services: Dict[str, CachedMCPService] = {}

            # Workflow pools
            self._workflow_pools: Dict[str, "WorkflowPool"] = {}

            # Process tracking
            self._main_process_pid: Optional[int] = None

            # Initialization lock (will be created when first accessed)
            self._init_lock: Optional[asyncio.Lock] = None
            self._lock_creation_lock = threading.Lock()  # Thread-safe lock creation

            # Mark this instance as initialized
            self._instance_initialized = True

            process_type = "[MAIN]" if not self._is_worker_process() else "[WORKER]"
            logging.info(
                f"🚀 {process_type} ResourceManager instance created for process {os.getpid()}"
            )

    def _ensure_init_lock(self) -> asyncio.Lock:
        """Ensure initialization lock exists in a thread-safe manner."""
        if self._init_lock is None:
            with self._lock_creation_lock:
                if self._init_lock is None:
                    self._init_lock = asyncio.Lock()
        return self._init_lock

    @timing_logger("ResourceManager.initialize")
    async def initialize(
        self,
        postgres_url: str,
        redis_url: str,
        mcp_server_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        workflow_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        use_shared_mcp: Optional[bool] = None,
    ) -> None:
        """
        Initialize all shared resources. This method should be called only once
        during application startup in the main process.

        Args:
            postgres_url: PostgreSQL connection URL
            redis_url: Redis connection URL
            mcp_server_configs: Configuration for MCP servers
            workflow_configs: Configuration for workflow pools
            use_shared_mcp: Force use of shared MCP services (auto-detected if None)
        """
        async with self._ensure_init_lock():
            if self._db_engine is not None:
                logging.warning("⚠️ ResourceManager already initialized, skipping...")
                return

            process_type = "[MAIN]" if not self._use_shared_mcp else "[WORKER]"
            logging.info(f"🔄 {process_type} Initializing ResourceManager...")

            # Determine if this is the main process or worker process
            current_pid = os.getpid()

            # Auto-detect if we should use shared MCP services
            if use_shared_mcp is None:
                # Check if we're in a subprocess spawned by ProcessPoolExecutor
                self._use_shared_mcp = self._is_worker_process()
            else:
                self._use_shared_mcp = use_shared_mcp

            if self._use_shared_mcp:
                logging.info(
                    f"🔗 [WORKER-SHARED] Process {current_pid} will use shared MCP services from main process"
                )
            else:
                logging.info(
                    f"🏠 [MAIN-GLOBAL] Process {current_pid} will initialize direct MCP services"
                )
                self._main_process_pid = current_pid

            try:
                # Initialize database engine with connection pool
                await self._initialize_database(postgres_url)

                # Initialize Redis connection pool
                await self._initialize_redis(redis_url)

                # Initialize MCP caching layer after Redis is available
                await self._initialize_mcp_cache_layer()

                # Initialize MCP services (shared or direct)
                if self._use_shared_mcp:
                    await self._initialize_shared_mcp_services()
                else:
                    await self._initialize_mcp_servers(mcp_server_configs)

                # Initialize workflow pools
                if workflow_configs:
                    await self._initialize_workflow_pools(workflow_configs)

                process_type = (
                    "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
                )
                logging.info(
                    f"✅ {process_type} ResourceManager initialization completed successfully"
                )

            except Exception as e:
                logging.error(f"❌ Failed to initialize ResourceManager: {e}")
                # Only cleanup resources that were actually initialized to avoid cleanup errors
                await self._safe_partial_cleanup()
                raise

    @classmethod
    def get_model_client_service(cls) -> ModelClientService:
        """
        Get the singleton ModelClientService instance with thread safety.

        Returns:
            ModelClientService: The singleton service instance
        """
        if cls._model_client_service is None:
            with cls._lock:
                # Double-check pattern for thread safety
                if cls._model_client_service is None:
                    cls._model_client_service = ModelClientService()
                    logging.info("🔧 ModelClientService singleton created")
        return cls._model_client_service

    @timing_logger("Database initialization")
    async def _initialize_database(self, postgres_url: str) -> None:
        """Initialize database engine with optimized connection pool settings."""
        process_type = (
            "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
        )
        logging.info(f"🗄️ {process_type} Initializing database engine...")

        if not postgres_url:
            raise ValueError("PostgreSQL URL is required but not provided")

        # Fix PostgreSQL URL format for async SQLAlchemy
        # Replace postgres:// with postgresql+asyncpg:// for async connections
        if postgres_url.startswith("postgres://"):
            postgres_url = postgres_url.replace(
                "postgres://", "postgresql+asyncpg://", 1
            )
        elif postgres_url.startswith("postgresql://"):
            postgres_url = postgres_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )

        logging.info(f"Using database URL: {postgres_url.split('@')[0]}@***")

        # Create async engine with connection pool
        self._db_engine = create_async_engine(
            postgres_url,
            # Connection pool settings for high performance
            pool_size=20,  # Number of persistent connections
            max_overflow=30,  # Additional connections beyond pool_size
            pool_timeout=30,  # Timeout for getting connection from pool
            pool_recycle=3600,  # Recycle connections every hour
            pool_pre_ping=True,  # Verify connections before use
            echo=False,  # Set to True for SQL logging in development
        )

        # Create session maker
        self._session_maker = async_sessionmaker(
            self._db_engine,
            expire_on_commit=False,
            # Don't pass class_=None, let SQLAlchemy use the default AsyncSession
        )

        process_type = (
            "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
        )
        logging.info(f"✅ {process_type} Database engine initialized successfully")

    @timing_logger("Redis connection pool initialization")
    async def _initialize_redis(self, redis_url: str) -> None:
        """Initialize Redis connection pool."""
        process_type = (
            "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
        )
        logging.info(f"🔗 {process_type} Initializing Redis connection pool...")

        if not redis_url:
            raise ValueError("Redis URL is required but not provided")

        logging.info(
            f"Using Redis URL: {redis_url.split('@')[0] if '@' in redis_url else redis_url.split('://')[0] + '://***'}"
        )

        # Create connection pool with optimized settings
        self._redis_pool = ConnectionPool.from_url(
            redis_url,
            max_connections=50,  # Maximum connections in pool
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30,
        )

        # Test the connection
        redis_client = Redis(connection_pool=self._redis_pool)
        try:
            await redis_client.ping()
            process_type = (
                "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
            )
            logging.info(
                f"✅ {process_type} Redis connection pool initialized successfully"
            )
        except Exception as e:
            logging.error(f"❌ Failed to connect to Redis: {e}")
            raise
        finally:
            await redis_client.aclose()

    @timing_logger("MCP caching layer initialization")
    async def _initialize_mcp_cache_layer(self) -> None:
        """Initialize the MCP caching layer for cross-process resource sharing."""
        process_type = (
            "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
        )

        if self._redis_pool is None:
            logging.warning(
                f"⚠️ {process_type} Redis not available - MCP caching layer disabled"
            )
            return

        try:
            logging.info(f"🚀 {process_type} Initializing MCP caching layer...")

            # Create Redis client for cache layer
            redis_client = Redis(connection_pool=self._redis_pool)

            # Initialize cache layer with configurable settings
            cache_ttl = int(
                os.getenv("MCP_CACHE_TTL_SECONDS", "3600")
            )  # 1 hour default
            enable_debug = os.getenv("MCP_CACHE_DEBUG", "false").lower() == "true"

            self._mcp_cache_layer = MCPCacheLayer(
                redis_client=redis_client,
                cache_ttl_seconds=cache_ttl,
                cache_prefix="ofnil_mcp:",
                enable_debug_logging=enable_debug,
            )

            logging.info(
                f"✅ {process_type} MCP caching layer initialized (TTL: {cache_ttl}s)"
            )

        except Exception as e:
            logging.error(
                f"❌ {process_type} Failed to initialize MCP caching layer: {e}"
            )
            self._mcp_cache_layer = None

    @timing_logger("MCP servers initialization")
    async def _initialize_mcp_servers(
        self, mcp_server_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Initialize MCP server sessions and tools that can be shared globally."""
        logging.info("🔌 [MAIN-GLOBAL] Initializing global MCP server sessions...")

        self._mcp_exit_stack = AsyncExitStack()

        # Initialize predefined MCP services if no custom configs provided
        if not mcp_server_configs:
            logging.info(
                "📋 [MAIN-GLOBAL] No custom MCP configs provided, initializing default MCP services..."
            )
            await self._initialize_default_mcp_services()
            return

        # Initialize custom MCP servers from config
        for server_name, config in mcp_server_configs.items():
            try:
                logging.info(f"🔄 [MAIN-GLOBAL] Initializing MCP server: {server_name}")

                # Create MCP server session
                server_params = StdioServerParams(
                    command=config.get("command", "python"),
                    args=config.get("args", []),
                    env=config.get("env", {}),
                    read_timeout_seconds=config.get("read_timeout_seconds", 30),
                )

                # Check if session initialization is required
                requires_session = config.get("requires_session", True)

                if requires_session:
                    session = await self._mcp_exit_stack.enter_async_context(
                        create_mcp_server_session(server_params)
                    )
                    await session.initialize()
                    self._mcp_sessions[server_name] = session

                    # Get tools with session
                    tools = await mcp_server_tools(server_params, session=session)
                    logging.info(
                        f"🔧 [MAIN-GLOBAL] Loaded {len(tools)} tools for MCP server '{server_name}' (with session)"
                    )
                else:
                    # Get tools without session (for some MCP servers)
                    tools = await mcp_server_tools(server_params)
                    logging.info(
                        f"🔧 [MAIN-GLOBAL] Loaded {len(tools)} tools for MCP server '{server_name}' (without session)"
                    )

                # Apply tool filters if specified
                tool_filters = config.get("tool_filters", [])
                if tool_filters:
                    tools = [tool for tool in tools if tool.name in tool_filters]
                    logging.info(
                        f"🔍 [MAIN-GLOBAL] Filtered tools for '{server_name}': {[tool.name for tool in tools]}"
                    )

                self._mcp_tools_cache[server_name] = tools

                logging.info(
                    f"✅ [MAIN-GLOBAL] MCP server '{server_name}' initialized successfully"
                )

            except Exception as e:
                logging.error(
                    f"❌ [MAIN-GLOBAL] Failed to initialize MCP server '{server_name}': {e}"
                )
                # Continue with other servers instead of failing completely
                continue

        logging.info(
            f"✅ [MAIN-GLOBAL] Initialized {len(self._mcp_sessions)} MCP server sessions and {len(self._mcp_tools_cache)} tool sets"
        )

    @timing_logger("Default MCP services initialization")
    async def _initialize_default_mcp_services(self) -> None:
        """Initialize the default MCP services used by workflows."""
        logging.info("🚀 [MAIN-GLOBAL] Initializing default MCP services...")

        # Initialize each service independently to avoid total failure
        services_initialized = 0

        # 1. Web Search Service
        try:
            await self._initialize_web_search_service()
            services_initialized += 1
        except Exception as e:
            logging.error(
                f"❌ [MAIN-GLOBAL] Failed to initialize web search service: {e}"
            )

        # 2. Domain Specific Service
        try:
            await self._initialize_domain_specific_service()
            services_initialized += 1
        except Exception as e:
            logging.error(
                f"❌ [MAIN-GLOBAL] Failed to initialize domain specific service: {e}"
            )

        # 3. HiRAG Retrieval Service
        try:
            await self._initialize_hirag_service()
            services_initialized += 1
        except Exception as e:
            logging.error(f"❌ [MAIN-GLOBAL] Failed to initialize HiRAG service: {e}")

        if services_initialized > 0:
            logging.info(
                f"✅ [MAIN-GLOBAL] Initialized {services_initialized}/3 default MCP services successfully"
            )
        else:
            logging.warning(
                "⚠️ [MAIN-GLOBAL] No default MCP services were initialized successfully"
            )
            # Don't raise exception - allow application to continue without MCP services

    @timing_logger("Web search service initialization")
    async def _initialize_web_search_service(self) -> None:
        """Initialize web search MCP server session."""
        try:
            logging.info("🔍 [MAIN-GLOBAL] Initializing web search MCP service...")

            brave_api_key = os.getenv("BRAVE_API_KEY")
            if not brave_api_key:
                logging.warning(
                    "⚠️ [MAIN-GLOBAL] BRAVE_API_KEY not found, skipping web search service"
                )
                return

            web_search_server_params = StdioServerParams(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-brave-search"],
                env={"BRAVE_API_KEY": brave_api_key},
                read_timeout_seconds=30,
            )

            web_search_session = await self._mcp_exit_stack.enter_async_context(
                create_mcp_server_session(web_search_server_params)
            )
            await web_search_session.initialize()

            # Store session and tools
            self._mcp_sessions["web_search"] = web_search_session
            self._mcp_tools_cache["web_search"] = await mcp_server_tools(
                web_search_server_params, session=web_search_session
            )

            # Create cached MCP service if cache layer is available
            if self._mcp_cache_layer:
                cached_service = CachedMCPService(
                    service_name="web_search",
                    cache_layer=self._mcp_cache_layer,
                    mcp_session=web_search_session,
                    tools=self._mcp_tools_cache["web_search"],
                )
                self._cached_mcp_services["web_search"] = cached_service
                logging.info(
                    "✅ [MAIN-GLOBAL] Created cached web search MCP service for cross-process sharing"
                )

            logging.info(
                f"✅ [MAIN-GLOBAL] Web search MCP service initialized with {len(self._mcp_tools_cache['web_search'])} tools"
            )

        except Exception as e:
            logging.error(
                f"❌ [MAIN-GLOBAL] Failed to initialize web search MCP service: {e}"
            )
            # Don't raise - allow other services to initialize

    @timing_logger("Domain specific service initialization")
    async def _initialize_domain_specific_service(self) -> None:
        """Initialize domain specific MCP server."""
        try:
            logging.info("🎯 [MAIN-GLOBAL] Initializing domain specific MCP service...")

            # Set env MCP_SERVER_PATH, default is "src/Sagi/mcp_server/"
            mcp_server_path = os.getenv("MCP_SERVER_PATH", "src/Sagi/mcp_server/")

            domain_specific_server_params = StdioServerParams(
                command="uv",
                args=[
                    "--directory",
                    os.path.join(
                        mcp_server_path, "domain_specific_mcp/src/domain_specific_mcp"
                    ),
                    "run",
                    "python",
                    "server.py",
                ],
                read_timeout_seconds=30,
            )

            # For domain specific service, we don't need session, just tools
            domain_specific_tools = await mcp_server_tools(
                domain_specific_server_params
            )

            # Store tools directly (no session needed for this service)
            self._mcp_tools_cache["domain_specific"] = domain_specific_tools

            # Create cached MCP service if cache layer is available (cache-only for domain specific)
            if self._mcp_cache_layer:
                cached_service = CachedMCPService(
                    service_name="domain_specific",
                    cache_layer=self._mcp_cache_layer,
                    mcp_session=None,  # No direct session for domain specific service
                    tools=domain_specific_tools,
                )
                self._cached_mcp_services["domain_specific"] = cached_service
                logging.info(
                    "✅ [MAIN-GLOBAL] Created cached domain specific MCP service for cross-process sharing"
                )

            logging.info(
                f"✅ [MAIN-GLOBAL] Domain specific MCP service initialized with {len(domain_specific_tools)} tools"
            )

        except Exception as e:
            logging.error(
                f"❌ [MAIN-GLOBAL] Failed to initialize domain specific MCP service: {e}"
            )
            # Don't raise - allow other services to initialize

    @timing_logger("HiRAG service initialization")
    async def _initialize_hirag_service(self) -> None:
        """Initialize HiRAG MCP server session."""
        try:
            logging.info("📚 [MAIN-GLOBAL] Initializing HiRAG MCP service...")

            # Check required environment variables
            required_env_vars = ["OPENAI_API_KEY", "VOYAGE_API_KEY"]
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                logging.warning(
                    f"⚠️ [MAIN-GLOBAL] Missing required environment variables for HiRAG: {missing_vars}, skipping..."
                )
                return

            # Try different approaches to initialize HiRAG service
            hirag_server_params = None

            # Approach 1: Try direct Python execution
            try:
                hirag_server_params = StdioServerParams(
                    command="python",
                    args=["-c", "from hirag_prod.server import main; main()"],
                    read_timeout_seconds=100,
                    env={
                        **os.environ,  # Pass all current environment variables
                        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
                        "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY"),
                        "DOC2X_API_KEY": os.getenv("DOC2X_API_KEY", ""),
                    },
                )
                logging.info(
                    "🔧 [MAIN-GLOBAL] Trying direct Python execution for HiRAG service"
                )
            except Exception as e:
                logging.warning(f"⚠️ [MAIN-GLOBAL] Direct Python approach failed: {e}")

                # Approach 2: Fallback to original mcp-hirag-tool command
                hirag_server_params = StdioServerParams(
                    command="mcp-hirag-tool",
                    args=[],
                    read_timeout_seconds=100,
                    env={
                        **os.environ,  # Pass all current environment variables
                        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
                        "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY"),
                        "DOC2X_API_KEY": os.getenv("DOC2X_API_KEY", ""),
                        "PYTHONPATH": os.getenv("PYTHONPATH", "") + ":" + os.getcwd(),
                    },
                )
                logging.info("🔧 [MAIN-GLOBAL] Fallback to mcp-hirag-tool command")

            # Try to initialize with timeout and better error handling
            try:
                hirag_session = await self._mcp_exit_stack.enter_async_context(
                    create_mcp_server_session(hirag_server_params)
                )
                await asyncio.wait_for(hirag_session.initialize(), timeout=60.0)

                # Store session and tools
                self._mcp_sessions["hirag_retrieval"] = hirag_session
                hirag_tools = await mcp_server_tools(
                    hirag_server_params, session=hirag_session
                )
                # Filter to only include hi_search tool
                hirag_tools = [tool for tool in hirag_tools if tool.name == "hi_search"]
                self._mcp_tools_cache["hirag_retrieval"] = hirag_tools

                # Create cached MCP service if cache layer is available
                if self._mcp_cache_layer:
                    cached_service = CachedMCPService(
                        service_name="hirag_retrieval",
                        cache_layer=self._mcp_cache_layer,
                        mcp_session=hirag_session,
                        tools=hirag_tools,
                    )
                    self._cached_mcp_services["hirag_retrieval"] = cached_service
                    logging.info(
                        "✅ [MAIN-GLOBAL] Created cached HiRAG MCP service for cross-process sharing"
                    )

                logging.info(
                    f"✅ [MAIN-GLOBAL] HiRAG MCP service initialized with {len(hirag_tools)} tools"
                )

            except asyncio.TimeoutError:
                logging.warning(
                    "⚠️ [MAIN-GLOBAL] HiRAG MCP service initialization timed out - this may be due to network issues downloading tiktoken encodings"
                )
                # Continue without HiRAG service
                return
            except Exception as session_error:
                logging.warning(
                    f"⚠️ [MAIN-GLOBAL] HiRAG MCP service initialization failed: {session_error}"
                )
                # Continue without HiRAG service
                return

        except Exception as e:
            logging.error(
                f"❌ [MAIN-GLOBAL] Failed to initialize HiRAG MCP service: {e}"
            )
            # Don't raise - allow other services to initialize

    def _is_worker_process(self) -> bool:
        """
        Simple, reliable detection of worker processes.

        Returns:
            True if this appears to be a worker process, False if main process
        """
        # Use a simple environment variable approach for reliable detection
        return os.getenv("OFNIL_WORKER_PROCESS") == "true"

    async def _initialize_shared_mcp_services(self) -> None:
        """
        Lightweight MCP initialization for worker processes using Redis cache layer.

        Worker processes don't create actual MCP sessions but instead use
        cached MCP services that share results through Redis caching.
        """
        logging.info(
            "🔗 [WORKER-SHARED] Worker process detected - using cached MCP services"
        )

        if self._mcp_cache_layer is None:
            logging.warning(
                "⚠️ [WORKER-SHARED] MCP cache layer not available - creating empty service stubs"
            )
            # Fallback to empty tools cache
            expected_services = ["web_search", "domain_specific", "hirag_retrieval"]
            for service in expected_services:
                self._mcp_tools_cache[service] = []
                logging.info(
                    f"🔗 [WORKER-SHARED] Created empty service stub for '{service}'"
                )
            logging.info(
                "✅ [WORKER-SHARED] Worker process MCP initialization completed (stub mode)"
            )
            return

        # Create cache-only MCP services for worker processes
        expected_services = ["web_search", "domain_specific", "hirag_retrieval"]
        for service_name in expected_services:
            try:
                # Create cache-only service (no direct MCP session)
                cached_service = CachedMCPService(
                    service_name=service_name,
                    cache_layer=self._mcp_cache_layer,
                    mcp_session=None,  # Worker processes don't have direct sessions
                    tools=[],  # Tools list not needed for cache-only services
                )
                self._cached_mcp_services[service_name] = cached_service
                self._mcp_tools_cache[service_name] = []  # Empty for compatibility

                logging.info(
                    f"🔗 [WORKER-SHARED] Created cached MCP service for '{service_name}'"
                )

            except Exception as e:
                logging.error(
                    f"❌ [WORKER-SHARED] Failed to create cached service for '{service_name}': {e}"
                )
                # Create empty stub as fallback
                self._mcp_tools_cache[service_name] = []

        logging.info(
            f"✅ [WORKER-SHARED] Worker process MCP initialization completed - {len(self._cached_mcp_services)} cached services available"
        )

    @timing_logger("Workflow pools initialization")
    async def _initialize_workflow_pools(
        self, workflow_configs: Dict[str, Dict[str, Any]]
    ) -> None:
        """Initialize workflow pools for different workflow types."""
        process_type = (
            "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
        )
        logging.info(f"🔄 {process_type} Initializing workflow pools...")

        for workflow_name, config in workflow_configs.items():
            # Adjust pool size for worker processes to minimize resource usage
            if self._use_shared_mcp:
                # Worker process - use smaller pool size (1 instance) and reuse when possible
                pool_size = 1
                logging.info(
                    f"🔧 [WORKER-SHARED] Using reduced pool size (1) for worker process"
                )
            else:
                # Main process - use configured pool size
                pool_size = config.get("pool_size", 3)

            workflow_config_path = config.get("config_path")
            team_config_path = config.get("team_config_path")

            # Create workflow pool
            pool = WorkflowPool(
                workflow_name=workflow_name,
                pool_size=pool_size,
                config_path=workflow_config_path,
                team_config_path=team_config_path,
                resource_manager=self,  # Pass self for shared MCP services
            )

            await pool.initialize()
            self._workflow_pools[workflow_name] = pool

            process_type = (
                "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
            )
            logging.info(
                f"✅ {process_type} Workflow pool '{workflow_name}' initialized with {pool_size} instances"
            )

        process_type = (
            "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
        )
        logging.info(
            f"✅ {process_type} Initialized {len(self._workflow_pools)} workflow pools"
        )

    async def _safe_partial_cleanup(self) -> None:
        """Safely cleanup only the resources that were actually initialized during failed startup."""
        process_type = (
            "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
        )
        logging.info(
            f"🧹 {process_type} Performing safe partial cleanup after initialization failure..."
        )

        try:
            # Cleanup workflow pools if any were created
            if self._workflow_pools:
                for pool in list(self._workflow_pools.values()):
                    try:
                        await pool.cleanup()
                    except Exception as e:
                        logging.error(
                            f"Error cleaning up workflow pool during partial cleanup: {e}"
                        )
                self._workflow_pools.clear()

            # Cleanup MCP resources if any were created
            if self._use_shared_mcp:
                # Worker process - just clear the tools cache
                try:
                    self._mcp_tools_cache.clear()
                    logging.info(
                        "🧹 [WORKER-SHARED] Worker process MCP cache cleared during partial cleanup"
                    )
                except Exception as e:
                    logging.error(
                        f"Error clearing worker MCP cache during partial cleanup: {e}"
                    )
            elif self._mcp_exit_stack:
                # Main process - close MCP sessions with simple error handling
                try:
                    await self._mcp_exit_stack.aclose()
                    logging.info("✅ MCP sessions cleanup completed")
                except RuntimeError as e:
                    if "cancel scope" in str(e):
                        # Known issue with asyncio cancel scopes - log but continue
                        logging.info(
                            "🧹 MCP sessions cleanup completed (with asyncio cancel scope warnings)"
                        )
                    else:
                        logging.error(
                            f"Error closing MCP sessions during partial cleanup: {e}"
                        )
                except Exception as e:
                    logging.error(
                        f"Error closing MCP sessions during partial cleanup: {e}"
                    )
                finally:
                    self._mcp_sessions.clear()
                    self._mcp_tools_cache.clear()
                    self._mcp_exit_stack = None

            # Cleanup Redis pool if it was created
            if self._redis_pool:
                try:
                    # Redis ConnectionPool cleanup (no disconnect method, just set to None)
                    logging.info("🧹 Cleaning up Redis connection pool...")
                except Exception as e:
                    logging.warning(
                        f"Error cleaning up Redis pool during partial cleanup: {e}"
                    )
                finally:
                    self._redis_pool = None

            # Cleanup database engine if it was created
            if self._db_engine:
                try:
                    logging.info("🧹 Cleaning up database engine...")
                    await asyncio.wait_for(self._db_engine.dispose(), timeout=5.0)
                except asyncio.TimeoutError:
                    logging.warning("Database cleanup timed out during partial cleanup")
                except Exception as e:
                    logging.warning(
                        f"Error disposing database engine during partial cleanup: {e}"
                    )
                finally:
                    self._db_engine = None
                    self._session_maker = None

            logging.info("✅ Safe partial cleanup completed")

        except Exception as e:
            logging.error(f"Error during safe partial cleanup: {e}")
            # Don't raise - we're already in an error state

    def get_session_maker(self) -> async_sessionmaker:
        """Get the database session maker."""
        if self._session_maker is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_maker

    def get_redis_client(self) -> Redis:
        """Get a Redis client from the connection pool."""
        if self._redis_pool is None:
            raise RuntimeError("Redis not initialized. Call initialize() first.")
        return Redis(connection_pool=self._redis_pool)

    def get_mcp_session(self, server_name: str) -> Any:
        """Get an MCP server session by name."""
        if self._use_shared_mcp:
            # Worker processes don't have direct MCP sessions
            logging.warning(
                f"⚠️ [WORKER-SHARED] Worker process cannot access MCP session '{server_name}' - sessions only available in main process"
            )
            raise RuntimeError(
                f"MCP sessions not available in worker processes. Use get_mcp_tools() instead."
            )

        if self._mcp_exit_stack is None:
            raise RuntimeError("MCP services not initialized. Call initialize() first.")

        if server_name not in self._mcp_sessions:
            raise KeyError(
                f"MCP server session '{server_name}' not found. Available sessions: {list(self._mcp_sessions.keys())}"
            )

        logging.debug(
            f"🔌 [MAIN-GLOBAL] Accessing MCP session for service: {server_name}"
        )
        return self._mcp_sessions[server_name]

    def get_mcp_tools(self, server_name: str) -> List[Any]:
        """Get MCP tools for a specific service."""
        if self._use_shared_mcp:
            # Worker process - MCP tools are not shared across processes
            # Each worker process creates its own MCP sessions in PlanningWorkflow
            logging.info(
                f"🔧 [WORKER-SHARED] Worker process '{server_name}' - MCP tools not shared across processes"
            )
            return []  # Return empty list to indicate no shared tools
        else:
            # Main process - ensure MCP services are initialized
            if self._mcp_exit_stack is None:
                raise RuntimeError(
                    "MCP services not initialized. Call initialize() first."
                )

        if server_name not in self._mcp_tools_cache:
            available_services = list(self._mcp_tools_cache.keys())
            if self._use_shared_mcp:
                logging.info(
                    f"⚠️ [WORKER-SHARED] MCP service '{server_name}' not available in worker process - each worker creates its own sessions"
                )
                return []  # Return empty list for worker processes
            else:
                raise KeyError(
                    f"MCP service '{server_name}' not found. Available services: {available_services}"
                )

        tools = self._mcp_tools_cache[server_name]

        if self._use_shared_mcp:
            logging.debug(
                f"🔧 [WORKER-SHARED] Worker process MCP access for '{server_name}' (not shared)"
            )
        else:
            logging.debug(
                f"🔧 [MAIN-GLOBAL] Main process MCP access for '{server_name}' ({len(tools)} tools)"
            )

        if not tools and not self._use_shared_mcp:
            logging.warning(
                f"⚠️ [MAIN-GLOBAL] MCP service '{server_name}' has no tools available"
            )

        return tools

    def get_cached_mcp_service(self, service_name: str) -> Optional[CachedMCPService]:
        """
        Get a cached MCP service that provides cross-process resource sharing.

        Args:
            service_name: Name of the MCP service

        Returns:
            CachedMCPService instance if available, None otherwise
        """
        if service_name in self._cached_mcp_services:
            cached_service = self._cached_mcp_services[service_name]

            process_type = (
                "[WORKER-SHARED]" if self._use_shared_mcp else "[MAIN-GLOBAL]"
            )
            logging.debug(
                f"🔧 {process_type} Accessing cached MCP service: {service_name}"
            )

            return cached_service
        else:
            available_services = list(self._cached_mcp_services.keys())
            logging.warning(
                f"⚠️ Cached MCP service '{service_name}' not found. Available: {available_services}"
            )
            return None

    def has_mcp_caching(self) -> bool:
        """Check if MCP caching layer is available."""
        return self._mcp_cache_layer is not None

    async def get_mcp_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get MCP cache performance statistics."""
        if self._mcp_cache_layer is None:
            return None
        return await self._mcp_cache_layer.get_cache_stats()

    def get_workflow_pool(self, workflow_name: str) -> "WorkflowPool":
        """Get a workflow pool by name."""
        if workflow_name not in self._workflow_pools:
            raise KeyError(f"Workflow pool '{workflow_name}' not found")
        return self._workflow_pools[workflow_name]

    def list_mcp_servers(self) -> List[str]:
        """Get list of available MCP server names."""
        return list(self._mcp_sessions.keys())

    def list_mcp_services(self) -> List[str]:
        """Get list of all available MCP service names (sessions + tools-only services)."""
        return list(
            set(list(self._mcp_sessions.keys()) + list(self._mcp_tools_cache.keys()))
        )

    def list_workflow_pools(self) -> List[str]:
        """Get list of available workflow pool names."""
        return list(self._workflow_pools.keys())

    def get_shared_mcp_tools(self) -> Optional[Dict[str, List[Any]]]:
        """
        Get shared MCP tools for workflow creation.

        Returns:
            Dict of MCP tools for main process, None for worker processes
            to trigger workflow's own MCP session creation.
        """
        if self._is_worker_process():
            # Worker process - return None to let workflow create its own MCP sessions
            # This is the intended behavior to avoid expensive cross-process resource sharing
            logging.info(
                "🔧 [WORKER-SHARED] Worker process will let PlanningWorkflow create lightweight MCP sessions"
            )
            return None
        else:
            # Main process - return actual shared MCP tools
            shared_mcp_tools = {
                "web_search": self._mcp_tools_cache.get("web_search", []),
                "domain_specific": self._mcp_tools_cache.get("domain_specific", []),
                "hirag_retrieval": self._mcp_tools_cache.get("hirag_retrieval", []),
            }

            # Log the actual tool counts for debugging
            logging.info(
                f"🔧 [MAIN-GLOBAL] Providing shared MCP tools - web_search: {len(shared_mcp_tools.get('web_search', []))}, "
                f"domain_specific: {len(shared_mcp_tools.get('domain_specific', []))}, "
                f"hirag_retrieval: {len(shared_mcp_tools.get('hirag_retrieval', []))}"
            )

            return shared_mcp_tools

    @classmethod
    def reset(cls):
        """Reset all services - useful for testing. Thread-safe."""
        with cls._lock:
            if cls._model_client_service:
                cls._model_client_service.clear_cache()
            cls._model_client_service = None
            # Reset per-process instances
            for instance in cls._instances.values():
                if hasattr(instance, "_instance_initialized"):
                    instance._instance_initialized = False
            cls._instances.clear()
            logging.info("🔄 ResourceManager reset")

    @classmethod
    def get_service_status(cls) -> dict:
        """Get status of all managed services."""
        status = {"model_client_service": cls._model_client_service is not None}

        if cls._model_client_service:
            status["model_client_cache"] = cls._model_client_service.get_cache_info()

        # Add status for current process instance
        current_pid = os.getpid()
        if current_pid in cls._instances:
            instance = cls._instances[current_pid]
            status.update(
                {
                    "database_initialized": instance._db_engine is not None,
                    "redis_initialized": instance._redis_pool is not None,
                    "mcp_services_count": len(instance._mcp_tools_cache),
                    "workflow_pools_count": len(instance._workflow_pools),
                    "process_id": current_pid,
                    "is_worker_process": instance._use_shared_mcp,
                }
            )

        return status

    async def cleanup(self) -> None:
        """Cleanup all resources. Should be called during application shutdown."""
        process_type = (
            "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
        )
        logging.info(f"🧹 {process_type} Cleaning up ResourceManager...")

        cleanup_errors = []

        # Cleanup workflow pools with individual error handling
        for pool_name, pool in self._workflow_pools.items():
            try:
                process_type = (
                    "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
                )
                logging.info(
                    f"🧹 {process_type} Cleaning up workflow pool: {pool_name}"
                )
                await asyncio.wait_for(pool.cleanup(), timeout=10.0)
            except asyncio.TimeoutError:
                logging.warning(f"Workflow pool '{pool_name}' cleanup timed out")
                cleanup_errors.append(f"Pool {pool_name} cleanup timeout")
            except asyncio.CancelledError:
                logging.warning(f"Workflow pool '{pool_name}' cleanup was cancelled")
                cleanup_errors.append(f"Pool {pool_name} cleanup cancelled")
            except Exception as e:
                logging.error(f"Error cleaning up workflow pool '{pool_name}': {e}")
                cleanup_errors.append(f"Pool {pool_name}: {e}")

        self._workflow_pools.clear()

        # Cleanup MCP resources (direct sessions in main process, lightweight cleanup in workers)
        if self._use_shared_mcp:
            try:
                logging.info(
                    "🧹 [WORKER-SHARED] Cleaning up worker process MCP resources..."
                )
                self._mcp_tools_cache.clear()
                logging.info("✅ [WORKER-SHARED] Worker process MCP cleanup completed")
            except Exception as e:
                logging.error(f"Error cleaning up worker MCP resources: {e}")
                cleanup_errors.append(f"Worker MCP: {e}")
        elif self._mcp_exit_stack:
            try:
                logging.info("🧹 [MAIN-GLOBAL] Cleaning up direct MCP sessions...")

                # Force immediate cleanup without async context managers
                # This prevents cancel scope issues during shutdown
                logging.info(
                    "🧹 [MAIN-GLOBAL] Using immediate cleanup to avoid cancel scope issues"
                )

                # Just clear references and mark as cleaned up
                # The processes will be cleaned up by the OS when the main process exits
                session_count = len(self._mcp_sessions)
                self._mcp_sessions.clear()
                self._mcp_tools_cache.clear()
                self._cached_mcp_services.clear()
                self._mcp_exit_stack = None

                logging.info(
                    f"✅ [MAIN-GLOBAL] Direct MCP cleanup completed - cleared {session_count} sessions"
                )

            except Exception as e:
                logging.error(f"❌ [MAIN-GLOBAL] Error during MCP cleanup: {e}")
                cleanup_errors.append(f"Direct MCP: {e}")
            finally:
                # Ensure cleanup even if errors occur
                self._mcp_sessions.clear()
                self._mcp_tools_cache.clear()
                self._cached_mcp_services.clear()
                self._mcp_exit_stack = None

        # Cleanup Redis pool
        if self._redis_pool:
            try:
                process_type = (
                    "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
                )
                logging.info(f"🧹 {process_type} Cleaning up Redis connection pool...")
                # Redis ConnectionPool doesn't have a disconnect() method, just set to None
                logging.info(f"✅ {process_type} Redis cleanup completed")
            except Exception as e:
                logging.error(f"Error cleaning up Redis pool: {e}")
                cleanup_errors.append(f"Redis: {e}")
            finally:
                self._redis_pool = None

        # Cleanup database engine
        if self._db_engine:
            try:
                process_type = (
                    "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
                )
                logging.info(f"🧹 {process_type} Cleaning up database engine...")
                await asyncio.wait_for(self._db_engine.dispose(), timeout=5.0)
                logging.info(f"✅ {process_type} Database cleanup completed")
            except asyncio.TimeoutError:
                logging.warning("Database cleanup timed out")
                cleanup_errors.append("Database cleanup timeout")
            except Exception as e:
                logging.error(f"Error disposing database engine: {e}")
                cleanup_errors.append(f"Database: {e}")
            finally:
                self._db_engine = None
                self._session_maker = None

        # Log summary
        if cleanup_errors:
            process_type = (
                "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
            )
            logging.warning(
                f"⚠️ {process_type} ResourceManager cleanup completed with {len(cleanup_errors)} errors: {cleanup_errors}"
            )
        else:
            process_type = (
                "[MAIN-GLOBAL]" if not self._use_shared_mcp else "[WORKER-SHARED]"
            )
            logging.info(
                f"✅ {process_type} ResourceManager cleanup completed successfully"
            )

        # Don't raise exceptions during shutdown to prevent blocking application exit


class WorkflowPool:
    """
    Manages a pool of reusable PlanningWorkflow instances to avoid
    expensive initialization overhead for each request.

    This class provides thread-safe access to workflow instances
    using asyncio locks and implements proper resource management.
    """

    def __init__(
        self,
        workflow_name: str,
        pool_size: int = 3,
        config_path: Optional[str] = None,
        team_config_path: Optional[str] = None,
        resource_manager=None,  # Optional ResourceManager for shared services
    ):
        """
        Initialize workflow pool.

        Args:
            workflow_name: Name of the workflow type
            pool_size: Number of workflow instances to maintain in pool
            config_path: Path to workflow configuration file
            team_config_path: Path to team configuration file
            resource_manager: Optional ResourceManager for shared MCP services
        """
        self.workflow_name = workflow_name
        self.pool_size = pool_size
        self.config_path = config_path
        self.team_config_path = team_config_path
        self.resource_manager = resource_manager  # Store resource manager reference

        # Pool storage and synchronization
        self._available_workflows: List["PlanningWorkflow"] = []
        self._in_use_workflows: Dict[str, "PlanningWorkflow"] = {}
        self._pool_lock: Optional[asyncio.Lock] = None
        self._lock_creation_lock = threading.Lock()  # Thread-safe lock creation

        # Initialization flag
        self._initialized = False

        # Detect process type for logging - using same logic as ResourceManager
        self._is_worker_process = os.getenv("OFNIL_WORKER_PROCESS") == "true"
        process_type = "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"

        logging.debug(
            f"🏗️ {process_type} WorkflowPool '{workflow_name}' created with pool size {pool_size}"
        )

    def _ensure_lock(self) -> asyncio.Lock:
        """Ensure async lock exists in a thread-safe manner."""
        if self._pool_lock is None:
            with self._lock_creation_lock:
                if self._pool_lock is None:
                    self._pool_lock = asyncio.Lock()
        return self._pool_lock

    @timing_logger("WorkflowPool.initialize")
    async def initialize(self) -> None:
        """Initialize the workflow pool by creating all workflow instances."""
        async with self._ensure_lock():
            # Check initialization status inside the lock to prevent race conditions
            if self._initialized:
                process_type = (
                    "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
                )
                logging.warning(
                    f"⚠️ {process_type} WorkflowPool '{self.workflow_name}' already initialized"
                )
                return

            process_type = (
                "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
            )
            logging.info(
                f"🔄 {process_type} Initializing WorkflowPool '{self.workflow_name}'..."
            )

            try:
                for i in range(self.pool_size):
                    try:
                        # Get shared MCP tools for the workflow
                        shared_mcp_tools = None
                        if self.resource_manager:
                            if not self.resource_manager._use_shared_mcp:
                                # Main process - use actual shared MCP tools
                                shared_mcp_tools = {
                                    "web_search": self.resource_manager._mcp_tools_cache.get(
                                        "web_search", []
                                    ),
                                    "domain_specific": self.resource_manager._mcp_tools_cache.get(
                                        "domain_specific", []
                                    ),
                                    "hirag_retrieval": self.resource_manager._mcp_tools_cache.get(
                                        "hirag_retrieval", []
                                    ),
                                }

                                # Log the actual tool counts for debugging
                                logging.info(
                                    f"🔧 [MAIN-GLOBAL] WorkflowPool MCP tools - web_search: {len(shared_mcp_tools.get('web_search', []))}, "
                                    f"domain_specific: {len(shared_mcp_tools.get('domain_specific', []))}, "
                                    f"hirag_retrieval: {len(shared_mcp_tools.get('hirag_retrieval', []))}"
                                )
                            else:
                                # Worker process - create empty tools dict to trigger workflow's own MCP session creation
                                # This avoids the expensive initialization that was happening before
                                shared_mcp_tools = None
                                process_type = "[WORKER-SHARED]"
                                logging.info(
                                    f"🔧 {process_type} WorkflowPool letting PlanningWorkflow create lightweight MCP sessions"
                                )

                        # Dynamic import to avoid circular import
                        from Sagi.workflows import PlanningWorkflow

                        workflow = await PlanningWorkflow.create(
                            config_path=self.config_path,
                            team_config_path=self.team_config_path,
                            external_mcp_tools=shared_mcp_tools,
                        )
                        self._available_workflows.append(workflow)
                        process_type = (
                            "[WORKER-SHARED]"
                            if self._is_worker_process
                            else "[MAIN-GLOBAL]"
                        )
                        logging.debug(
                            f"✨ {process_type} Created workflow instance {i+1}/{self.pool_size}"
                        )
                    except Exception as workflow_error:
                        process_type = (
                            "[WORKER-SHARED]"
                            if self._is_worker_process
                            else "[MAIN-GLOBAL]"
                        )
                        logging.error(
                            f"❌ {process_type} Failed to create workflow instance {i+1}/{self.pool_size}: {workflow_error}"
                        )
                        # Cleanup any workflows created so far
                        await self._cleanup_workflows(self._available_workflows)
                        self._available_workflows.clear()
                        raise

                self._initialized = True
                process_type = (
                    "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
                )
                logging.info(
                    f"✅ {process_type} WorkflowPool '{self.workflow_name}' initialized with {len(self._available_workflows)} instances"
                )

            except Exception as e:
                process_type = (
                    "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
                )
                logging.error(
                    f"❌ {process_type} Failed to initialize WorkflowPool '{self.workflow_name}': {e}"
                )
                # Ensure cleanup of any remaining workflows
                if self._available_workflows:
                    await self._cleanup_workflows(self._available_workflows)
                    self._available_workflows.clear()
                raise

    async def get_workflow(self, session_id: str) -> "PlanningWorkflow":
        """
        Get a workflow instance from the pool.

        Args:
            session_id: Unique identifier for the session using this workflow

        Returns:
            "PlanningWorkflow" instance

        Raises:
            RuntimeError: If pool is not initialized or no workflows available
        """
        if not self._initialized:
            raise RuntimeError(f"WorkflowPool '{self.workflow_name}' not initialized")

        async with self._ensure_lock():
            process_type = (
                "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
            )
            logging.info(
                f"📊 {process_type} WorkflowPool '{self.workflow_name}' - Available workflows: {len(self._available_workflows)}"
            )
            if not self._available_workflows:
                # If no workflows available, create a new temporary one
                process_type = (
                    "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
                )
                logging.warning(
                    f"⚠️ {process_type} No workflows available in pool '{self.workflow_name}', creating temporary instance"
                )
                try:
                    # Get shared MCP tools for the temporary workflow
                    shared_mcp_tools = None
                    if self.resource_manager:
                        if not self.resource_manager._use_shared_mcp:
                            # Main process - use actual shared MCP tools
                            shared_mcp_tools = {
                                "web_search": self.resource_manager._mcp_tools_cache.get(
                                    "web_search", []
                                ),
                                "domain_specific": self.resource_manager._mcp_tools_cache.get(
                                    "domain_specific", []
                                ),
                                "hirag_retrieval": self.resource_manager._mcp_tools_cache.get(
                                    "hirag_retrieval", []
                                ),
                            }

                            # Log the actual tool counts for debugging
                            logging.info(
                                f"🔧 [MAIN-GLOBAL] Temporary workflow MCP tools - web_search: {len(shared_mcp_tools.get('web_search', []))}, "
                                f"domain_specific: {len(shared_mcp_tools.get('domain_specific', []))}, "
                                f"hirag_retrieval: {len(shared_mcp_tools.get('hirag_retrieval', []))}"
                            )
                        else:
                            # Worker process - create empty tools dict to trigger workflow's own MCP session creation
                            shared_mcp_tools = None
                            process_type = "[WORKER-SHARED]"
                            logging.info(
                                f"🔧 {process_type} Temporary workflow will create lightweight MCP sessions"
                            )

                    # Dynamic import to avoid circular import
                    from Sagi.workflows import PlanningWorkflow

                    workflow = await PlanningWorkflow.create(
                        config_path=self.config_path,
                        team_config_path=self.team_config_path,
                        external_mcp_tools=shared_mcp_tools,
                    )
                    logging.info(
                        f"✨ {process_type} Created temporary workflow: {workflow}"
                    )
                except Exception as e:
                    logging.error(
                        f"❌ {process_type} Failed to create temporary workflow: {e}"
                    )
                    raise
            else:
                # Get workflow from pool
                workflow = self._available_workflows.pop()
                process_type = (
                    "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
                )
                logging.info(
                    f"🔄 {process_type} Retrieved existing workflow from pool: {workflow}"
                )

            # Verify workflow is not None before tracking
            if workflow is None:
                raise RuntimeError(
                    f"Got None workflow from pool '{self.workflow_name}' for session '{session_id}'"
                )

            # Track workflow as in use
            self._in_use_workflows[session_id] = workflow

            process_type = (
                "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
            )
            logging.info(
                f"✅ {process_type} Retrieved workflow for session '{session_id}' from pool '{self.workflow_name}': {workflow}"
            )
            return workflow

    async def return_workflow(self, session_id: str) -> None:
        """
        Return a workflow instance to the pool.

        Args:
            session_id: Session identifier that was using the workflow
        """
        async with self._ensure_lock():
            if session_id not in self._in_use_workflows:
                process_type = (
                    "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
                )
                logging.warning(
                    f"⚠️ {process_type} Session '{session_id}' not found in in-use workflows"
                )
                return

            workflow = self._in_use_workflows.pop(session_id)

            try:
                # Reset workflow state before returning to pool
                await workflow.team.reset()

                # Return to pool if we haven't exceeded pool size
                if len(self._available_workflows) < self.pool_size:
                    self._available_workflows.append(workflow)
                    process_type = (
                        "[WORKER-SHARED]"
                        if self._is_worker_process
                        else "[MAIN-GLOBAL]"
                    )
                    logging.debug(
                        f"✅ {process_type} Returned workflow for session '{session_id}' to pool '{self.workflow_name}'"
                    )
                else:
                    # Pool is full, cleanup the excess workflow
                    await workflow.cleanup()
                    logging.debug(
                        f"🧹 Pool full, cleaned up excess workflow for session '{session_id}'"
                    )

            except Exception as e:
                logging.error(
                    f"❌ Error resetting workflow for session '{session_id}': {e}"
                )
                # Don't return potentially corrupted workflow to pool
                try:
                    await workflow.cleanup()
                except Exception as cleanup_error:
                    process_type = (
                        "[WORKER-SHARED]"
                        if self._is_worker_process
                        else "[MAIN-GLOBAL]"
                    )
                    logging.error(
                        f"❌ {process_type} Error cleaning up corrupted workflow: {cleanup_error}"
                    )

    def get_pool_status(self) -> Dict[str, int]:
        """Get current pool status for monitoring."""
        return {
            "pool_size": self.pool_size,
            "available": len(self._available_workflows),
            "in_use": len(self._in_use_workflows),
            "total_capacity": self.pool_size,
        }

    async def _cleanup_workflows(self, workflows: List["PlanningWorkflow"]) -> None:
        """Helper method to cleanup a list of workflows."""
        for workflow in workflows:
            try:
                # Use immediate cleanup to avoid cancel scope issues
                process_type = (
                    "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
                )
                logging.info(
                    f"🧹 {process_type} Using immediate workflow cleanup to avoid cancel scope issues"
                )

                # Force immediate cleanup without async context managers
                if hasattr(workflow, "session_manager") and workflow.session_manager:
                    # Clear sessions without async context manager
                    if hasattr(workflow.session_manager, "sessions"):
                        workflow.session_manager.sessions.clear()
                    if hasattr(workflow.session_manager, "exit_stack"):
                        workflow.session_manager.exit_stack = None

                logging.info(f"✅ {process_type} Immediate workflow cleanup completed")

            except Exception as e:
                process_type = (
                    "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
                )
                logging.error(f"❌ {process_type} Error cleaning up workflow: {e}")
                # Continue with emergency cleanup
                try:
                    if (
                        hasattr(workflow, "session_manager")
                        and workflow.session_manager
                    ):
                        if hasattr(workflow.session_manager, "sessions"):
                            workflow.session_manager.sessions.clear()
                        if hasattr(workflow.session_manager, "exit_stack"):
                            workflow.session_manager.exit_stack = None
                except Exception:
                    pass

    async def cleanup(self) -> None:
        """Cleanup all workflows in the pool."""
        process_type = "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
        logging.info(
            f"🧹 {process_type} Cleaning up WorkflowPool '{self.workflow_name}'..."
        )

        async with self._ensure_lock():
            # Cleanup available workflows
            await self._cleanup_workflows(self._available_workflows)
            self._available_workflows.clear()

            # Cleanup in-use workflows
            in_use_list = list(self._in_use_workflows.values())
            await self._cleanup_workflows(in_use_list)
            self._in_use_workflows.clear()

            self._initialized = False

        process_type = "[WORKER-SHARED]" if self._is_worker_process else "[MAIN-GLOBAL]"
        logging.info(
            f"✅ {process_type} WorkflowPool '{self.workflow_name}' cleanup completed"
        )


# Per-process instances for easy access
_resource_managers: Dict[int, ResourceManager] = {}


def get_resource_manager() -> ResourceManager:
    """
    Get the global resource manager instance for this process.

    Returns:
        ResourceManager singleton instance for this process
    """
    process_id = os.getpid()
    if process_id not in _resource_managers:
        _resource_managers[process_id] = ResourceManager()
    return _resource_managers[process_id]


# Convenience functions for common operations
def get_db_session():
    """Get a database session from the global resource manager."""
    manager = get_resource_manager()
    session_maker = manager.get_session_maker()
    return session_maker()


def get_redis():
    """Get a Redis client from the global resource manager."""
    manager = get_resource_manager()
    return manager.get_redis_client()


async def get_workflow(workflow_name: str, session_id: str) -> "PlanningWorkflow":
    """Get a workflow instance from the global resource manager."""
    manager = get_resource_manager()
    pool = manager.get_workflow_pool(workflow_name)
    return await pool.get_workflow(session_id)


async def return_workflow(workflow_name: str, session_id: str) -> None:
    """Return a workflow instance to the global resource manager."""
    manager = get_resource_manager()
    pool = manager.get_workflow_pool(workflow_name)
    await pool.return_workflow(session_id)


def get_mcp_session(service_name: str) -> Any:
    """Get an MCP session from the global resource manager."""
    manager = get_resource_manager()
    return manager.get_mcp_session(service_name)


def get_mcp_tools(service_name: str) -> List[Any]:
    """Get MCP tools from the global resource manager."""
    manager = get_resource_manager()
    return manager.get_mcp_tools(service_name)


def list_mcp_services() -> List[str]:
    """List all available MCP services."""
    manager = get_resource_manager()
    return manager.list_mcp_services()
