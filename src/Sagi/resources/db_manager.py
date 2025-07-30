import logging
from typing import Optional

from redis.asyncio import ConnectionPool, Redis
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from Sagi.utils.settings import settings


class SimpleSingleton:
    _instance = None

    def __new__(cls) -> "SimpleSingleton":
        """Singleton"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True


class PGSQLClient(SimpleSingleton):
    # PostgresSQL
    _pgsql_db_engine: Optional[AsyncEngine] = None
    _pgsql_session_maker: Optional[async_sessionmaker] = None
    _connection_tested: bool = False

    def getDBEngine(self) -> AsyncEngine:
        if self._pgsql_db_engine is None:
            raise ValueError("_pgsql_db_engine is None, call connect first")
        return self._pgsql_db_engine

    def getSessionMaker(self) -> async_sessionmaker:
        if self._pgsql_session_maker is None:
            raise ValueError("_pgsql_session_maker is None, call connect first")
        return self._pgsql_session_maker

    async def connect(self):
        pgsql_url = settings.POSTGRESQL_URL_NO_SSL
        if not pgsql_url:
            raise ValueError("POSTGRESQL_URL is required but not provided")

        # Fix PostgreSQL URL format for async SQLAlchemy
        # Replace postgres:// with postgresql+asyncpg:// for async connections
        if pgsql_url.startswith("postgres://"):
            pgsql_url = pgsql_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif pgsql_url.startswith("postgresql://"):
            pgsql_url = pgsql_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        self._pgsql_db_engine = create_async_engine(
            pgsql_url,
            # Connection pool settings for high performance
            pool_size=20,  # Number of persistent connections
            max_overflow=30,  # Additional connections beyond pool_size
            pool_timeout=30,  # Timeout for getting connection from pool
            pool_recycle=3600,  # Recycle connections every hour
            pool_pre_ping=True,  # Verify connections before use
            echo=False,  # Set to True for SQL logging in development
        )

        self._pgsql_session_maker = async_sessionmaker(
            self._pgsql_db_engine,
            expire_on_commit=False,
            # Don't pass class_=None, let SQLAlchemy use the default AsyncSession
        )

    async def health_check(self):
        if self._connection_tested:
            return

        try:
            async with self._pgsql_db_engine.connect() as conn:
                # Execute a simple query to verify the connection
                result = await conn.execute("SELECT 1")
                logging.info(
                    f"Connection to PostgreSQL successful. Query result: {result.scalar()}"
                )
        except Exception as e:
            logging.error(f"Connection to PostgreSQL failed: {e}")
            return
        finally:
            # Dispose the engine to close all connections in the pool
            await self._pgsql_db_engine.dispose()
            self._connection_tested = True

    async def close(self):
        await self._pgsql_db_engine.dispose()
        logging.info("Connection to PostgreSQL is closed.")


class RedisClient(SimpleSingleton):
    _redis_pool: Optional[ConnectionPool] = None
    _connection_tested: bool = False

    def getRedisPool(self) -> ConnectionPool:
        if self._redis_pool is None:
            raise ValueError("_redis_pool is None, call async init first")
        return self._redis_pool

    async def connect(self):
        redis_url = settings.REDIS_URL
        if not redis_url:
            raise ValueError("REDIS_URL is required but not provided")

        self._redis_pool = ConnectionPool.from_url(
            redis_url,
            max_connections=20,  # Maximum connections in pool
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=60,
        )

    async def health_check(self):
        if self._connection_tested:
            return

        redis = Redis(connection_pool=self._redis_pool)
        try:
            await redis.ping()
            logging.info("Connection to Redis successful.")
        except Exception as e:
            logging.error(f"Connection to Redis failed: {e}")
            raise
        finally:
            await redis.aclose()
            self._connection_tested = True

    async def close(self):
        logging.info("Connection to Redis is closed.")


class DBManager(SimpleSingleton):
    """Placeholder for PGSQLClient andRedisClient"""

    _pgsql_client: Optional[PGSQLClient] = None
    _redis_client: Optional[RedisClient] = None

    def __init__(self):
        self._pgsql_client = PGSQLClient()
        self._redis_client = RedisClient()

    def getPGSQLClient(self) -> PGSQLClient:
        if self._pgsql_client is None:
            raise ValueError("_pgsql_client is None, call async init first")
        return self._pgsql_client

    def getRedisClient(self) -> RedisClient:
        if self._redis_client is None:
            raise ValueError("_redis_client is None, call async init first")
        return self._redis_client

    async def connect(self):
        await self._pgsql_client.connect()
        await self._pgsql_client.health_check()

        await self._redis_client.connect()
        await self._redis_client.health_check()

    async def close(self):
        await self._pgsql_client.close()
        await self._redis_client.close()


dbm = DBManager()
