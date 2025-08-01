import logging
from contextlib import AsyncExitStack
from typing import Any, Dict


class MCPSessionManager:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, Any] = {}
        self._closed = False

    async def create_session(self, name: str, context_manager):
        """create and store a session"""
        if self._closed:
            raise RuntimeError("Cannot create session on closed manager")

        session = await self.exit_stack.enter_async_context(context_manager)
        self.sessions[name] = session
        return session

    async def close_all(self):
        """close all sessions with proper error handling"""
        if self._closed:
            return

        self._closed = True

        try:
            # First clear sessions to prevent new operations
            session_count = len(self.sessions)
            self.sessions.clear()

            # Then close the exit stack with error handling
            await self.exit_stack.aclose()
            logging.info(f"Successfully closed {session_count} MCP sessions")

        except RuntimeError as e:
            if "cancel scope" in str(e).lower():
                # This is the known asyncio cancel scope issue - log and continue
                logging.warning(
                    f"MCP cleanup completed with asyncio cancel scope warning: {e}"
                )
            else:
                logging.error(f"Error during MCP session cleanup: {e}")
                raise
        except Exception as e:
            logging.error(f"Unexpected error during MCP session cleanup: {e}")
            # Don't re-raise during cleanup to prevent blocking shutdown
        finally:
            # Ensure sessions are always cleared
            self.sessions.clear()
