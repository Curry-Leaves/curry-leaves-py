"""`_McpSession` — the actual `mcp.ClientSession` + transport + lifecycle management.

One instance per configured server. Lazy: constructing it does no I/O. `connect_*()` is
idempotent and concurrency-safe (a lock guards against two concurrent callers both
spawning the transport); `close()` is idempotent (safe to call more than once, since
every `McpTool` from this server calls it via the shared server object) and, unlike a
one-shot teardown, leaves the session reconnectable — `connect_*()` after a `close()`
opens a fresh transport (used by `McpServerManager.reconnect()`).
"""

from __future__ import annotations

import asyncio
import warnings
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import TYPE_CHECKING, Awaitable, Callable, Literal, Optional

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.message import SessionMessage

if TYPE_CHECKING:
    from mcp import types

_ReadStream = MemoryObjectReceiveStream["SessionMessage | Exception"]
_WriteStream = MemoryObjectSendStream[SessionMessage]


class McpConnectionError(Exception):
    """Raised when a server fails to connect (bad command, handshake failure, etc.)."""


class McpNotConnectedError(Exception):
    """Raised when an operation needs a connected session but `connect()` was never
    (successfully) called."""


class _McpSession:
    def __init__(self, name: str) -> None:
        self.name = name
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect_stdio(
        self, *, command: str, args: list[str], env: dict[str, str], cwd: str | None
    ) -> None:
        async def open_transport(stack: AsyncExitStack) -> tuple[_ReadStream, _WriteStream]:
            params = StdioServerParameters(command=command, args=args, env=env or None, cwd=cwd)
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write

        await self._connect(open_transport)

    async def connect_http(
        self,
        *,
        url: str,
        headers: dict[str, str],
        transport: Literal["http", "sse"],
        connect_timeout_seconds: Optional[float] = None,
    ) -> None:
        async def open_transport(stack: AsyncExitStack) -> tuple[_ReadStream, _WriteStream]:
            if transport == "sse":
                kwargs = {"timeout": connect_timeout_seconds} if connect_timeout_seconds else {}
                read, write = await stack.enter_async_context(
                    sse_client(url, headers=headers or None, **kwargs)  # type: ignore[arg-type]
                )
                return read, write
            http_kwargs = {"timeout": connect_timeout_seconds} if connect_timeout_seconds else {}
            # `streamablehttp_client` is deprecated in newer `mcp` releases in favor of
            # `streamable_http_client` (a different signature: an httpx.AsyncClient
            # instead of headers/timeout kwargs) — but that replacement doesn't exist in
            # `mcp==1.9.x`, our pinned minimum, so we keep the older, still-functional
            # name and just swallow its DeprecationWarning here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                read, write, _get_session_id = await stack.enter_async_context(
                    streamablehttp_client(url, headers=headers or None, **http_kwargs)  # type: ignore[arg-type]
                )
            return read, write

        await self._connect(
            open_transport,
            init_timeout_seconds=connect_timeout_seconds,
        )

    async def _connect(
        self,
        open_transport: Callable[[AsyncExitStack], Awaitable[tuple[_ReadStream, _WriteStream]]],
        *,
        init_timeout_seconds: Optional[float] = None,
    ) -> None:
        async with self._lock:
            if self._session is not None:
                return
            stack = AsyncExitStack()
            try:
                read, write = await open_transport(stack)
                read_timeout = (
                    timedelta(seconds=init_timeout_seconds) if init_timeout_seconds else None
                )
                session = await stack.enter_async_context(
                    ClientSession(read, write, read_timeout_seconds=read_timeout)
                )
                await session.initialize()
            except (Exception, asyncio.CancelledError) as e:
                # CancelledError is caught here too: some transports (notably
                # streamablehttp_client) propagate a connection failure from their own
                # internal anyio task group as a bare CancelledError rather than a
                # concrete exception, once the request fails before a session even
                # exists. Treated as a connection failure, not an outer cancellation of
                # THIS task (nothing outside `_connect` requested cancellation here).
                try:
                    await stack.aclose()
                except (Exception, asyncio.CancelledError):
                    pass  # teardown of an already-broken transport must never raise
                raise McpConnectionError(f"failed to connect MCP server '{self.name}': {e}") from e
            self._session = session
            self._exit_stack = stack

    async def list_tools(self) -> list["types.Tool"]:
        if self._session is None:
            raise McpNotConnectedError(f"MCP server '{self.name}' is not connected")
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict[str, object]) -> "types.CallToolResult":
        if self._session is None:
            raise McpNotConnectedError(f"MCP server '{self.name}' is not connected")
        return await self._session.call_tool(name, arguments)

    async def close(self) -> None:
        async with self._lock:
            session, stack = self._session, self._exit_stack
            self._session = None
            self._exit_stack = None
        if stack is not None:
            try:
                await stack.aclose()
            except (Exception, asyncio.CancelledError):  # noqa: BLE001 - teardown must never raise
                pass
        del session
