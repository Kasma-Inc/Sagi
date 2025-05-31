import json
from typing import Any, Mapping, Optional

import asyncpg

from Sagi.workflows.planning_group_chat import PlanningGroupChat

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
        self.pool: Optional[asyncpg.Pool] = None

    async def init(self) -> None:
        """Initialize the connection pool and create the table if it does not exist"""
        self.pool = await asyncpg.create_pool(dsn=self.dsn)
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)

    async def list_sessions(self) -> list[str]:
        """List all session_id"""
        assert (
            self.pool
        ), "Please call init() to initialize the database connection first"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT session_id FROM session_states ORDER BY updated_at DESC"
            )
        return [r["session_id"] for r in rows]

    async def load_team_state(self, session_id: str, team: PlanningGroupChat) -> None:
        """
        Load state from the database and pass it to team.load_state(state).
        If the format is incompatible, delete the old record and keep the team in its default empty state.
        """
        assert (
            self.pool
        ), "Please call init() to initialize the database connection first"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM session_states WHERE session_id = $1",
                session_id,
            )
        if not row:
            return

        state = row["state"]
        try:
            await team.load_state(state)
        except Exception:

            async with self.pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM session_states WHERE session_id = $1",
                    session_id,
                )

    async def save_team_state(self, session_id: str, team: PlanningGroupChat) -> None:

        assert (
            self.pool
        ), "Please call init() to initialize the database connection first"

        state: Mapping[str, Any] = await team.save_state()

        state_json = json.dumps(state, ensure_ascii=False)

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

    async def close(self) -> None:

        if self.pool:
            await self.pool.close()
