import pytest

from Sagi.resources import DBManager, PGSQLClient, RedisClient


@pytest.mark.asyncio
async def test_pgsql_client():
    pgsql_client = PGSQLClient()

    await pgsql_client.connect()
    await pgsql_client.health_check()
    await pgsql_client.close()


@pytest.mark.asyncio
async def test_pgsql_client_singleton():
    pgsql_client_1 = PGSQLClient()
    pgsql_client_2 = PGSQLClient()
    assert pgsql_client_1 is pgsql_client_2

    await pgsql_client_1.connect()
    await pgsql_client_1.health_check()
    assert pgsql_client_2._connection_tested is True


@pytest.mark.asyncio
async def test_redis_client():
    redis_client = RedisClient()

    await redis_client.connect()
    await redis_client.health_check()
    await redis_client.close()


@pytest.mark.asyncio
async def test_redis_client_singleton():
    redis_client_1 = RedisClient()
    redis_client_2 = RedisClient()
    assert redis_client_1 is redis_client_2

    await redis_client_1.connect()
    await redis_client_1.health_check()
    assert redis_client_2._connection_tested is True


@pytest.mark.asyncio
async def test_db_manager():
    db_manager = DBManager()

    await db_manager.connect()
    await db_manager.close()
