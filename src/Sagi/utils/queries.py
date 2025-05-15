import asyncpg
import json
import os
from datetime import datetime

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_states (
    session_id TEXT PRIMARY KEY,
    state      JSONB   NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

class Database:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None

    async def init(self):
        self.pool = await asyncpg.create_pool(dsn=self.dsn)
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)

    async def list_sessions(self) -> list[str]:
        assert self.pool, "Database not initialized. Call init() first."
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT session_id FROM session_states ORDER BY updated_at DESC"
            )
        return [r["session_id"] for r in rows]

    async def load_state(self, session_id: str) -> dict:
        assert self.pool, "Database not initialized. Call init() first."
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM session_states WHERE session_id = $1",
                session_id,
            )
        if row:
            return row["state"]
        else:
            raise KeyError(f"Session {session_id!r} not found.")

    async def save_state(self, session_id: str, state: dict):
        assert self.pool, "Database not initialized. Call init() first."
        state_json = json.dumps(state)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO session_states(session_id, state, updated_at)
                VALUES ($1, $2::jsonb, NOW())
                ON CONFLICT (session_id) DO UPDATE
                  SET state = EXCLUDED.state,
                      updated_at = NOW()
                """,
                session_id,
                state_json,
            )

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
