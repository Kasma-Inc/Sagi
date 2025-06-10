import os

import asyncpg
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# Create an MCP server
mcp = FastMCP("pg_mcp")
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL is not set in environment variables.")


@mcp.tool("pg_query")
async def run_pg_query(query: str) -> str:

    if not query.strip().lower().startswith("select"):
        return ToolResponse(output={"result": "Only SELECT queries are allowed."})

    try:
        conn = await asyncpg.connect(DB_URL)
        rows = await conn.fetch(query)
        await conn.close()

        if not rows:
            return True, "No rows returned."

        headers = rows[0].keys()
        formatted = "\n".join(
            [", ".join(headers)]
            + [", ".join(str(v) for v in row.values()) for row in rows]
        )

        return True, formatted

    except Exception as e:
        return False, f"query fail: {str(e)}"


def main() -> None:
    """Run the MCP server."""
    try:
        mcp.run()
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
