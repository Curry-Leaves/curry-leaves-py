"""`McpServerManager` — connects/closes MULTIPLE MCP servers as one unit.

A plain batch-connect convenience for the multi-server case; it does NOT feed into
`Agent` at all (there is no `Agent.mcp_servers` — see `mcp/pick.py`'s `mcp_tools()` for
how servers actually reach an agent). With `drop_failed_servers=True` (default), one
misbehaving server (bad command, dead URL) doesn't take down the whole batch —
`active_servers`/`get()` silently excludes it and `errors[name]` records why.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Optional

from curry_leaves.mcp.server import McpServer


class McpServerManager:
    def __init__(
        self,
        servers: list[McpServer],
        *,
        drop_failed_servers: bool = True,
        connect_in_parallel: bool = True,
        connect_timeout_seconds: Optional[float] = None,
        cleanup_timeout_seconds: Optional[float] = None,
    ) -> None:
        # NOTE: connect_timeout_seconds/cleanup_timeout_seconds are accepted for API
        # stability but are NO-OPS — wrapping server.connect()/close() in
        # asyncio.wait_for() crashes with a cross-task cancel-scope RuntimeError from
        # anyio when the timeout actually fires on a real hang (transports use anyio
        # task groups internally, and cancelling them from an external task/wait_for
        # violates anyio's same-task cancel-scope invariant). Set a timeout on the
        # SERVER itself instead: McpServerHttp(connect_timeout_seconds=...) bounds the
        # HTTP/SSE handshake natively and safely. McpServerStdio has no equivalent (the
        # underlying stdio transport has no native connect timeout) — a hung subprocess
        # handshake needs an external, process-level timeout from the caller.
        if connect_timeout_seconds is not None or cleanup_timeout_seconds is not None:
            warnings.warn(
                "McpServerManager's connect_timeout_seconds/cleanup_timeout_seconds "
                "are no-ops (unsafe to implement via asyncio.wait_for around anyio "
                "transports). Set a timeout on the server itself instead, e.g. "
                "McpServerHttp(connect_timeout_seconds=...).",
                stacklevel=2,
            )
        self._servers = list(servers)
        self._drop_failed_servers = drop_failed_servers
        self._connect_in_parallel = connect_in_parallel
        self._active: list[McpServer] = []
        self._failed: list[McpServer] = []
        self._errors: dict[str, Exception] = {}

    @property
    def active_servers(self) -> list[McpServer]:
        return list(self._active)

    @property
    def failed_servers(self) -> list[McpServer]:
        return list(self._failed)

    @property
    def errors(self) -> dict[str, Exception]:
        return dict(self._errors)

    def get(self, name: str) -> McpServer:
        for server in self._active:
            if server.name == name:
                return server
        if any(s.name == name for s in self._failed):
            raise KeyError(
                f"MCP server '{name}' failed to connect: {self._errors.get(name)}"
            )
        raise KeyError(f"No MCP server named '{name}' was passed to McpServerManager")

    async def __aenter__(self) -> "McpServerManager":
        await self._connect_all(self._servers)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._close_all(self._servers)

    async def reconnect(self, *, failed_only: bool = True) -> None:
        if failed_only:
            targets = self._failed
            self._failed = []
            for server in targets:
                self._errors.pop(server.name, None)
        else:
            targets = self._servers
            await self._close_all(self._servers)
            self._active = []
            self._failed = []
            self._errors = {}
        await self._connect_all(targets)

    async def _connect_all(self, servers: list[McpServer]) -> None:
        async def connect_one(server: McpServer) -> None:
            try:
                await server.connect()
            except Exception as e:
                if not self._drop_failed_servers:
                    raise
                self._failed.append(server)
                self._errors[server.name] = e
                return
            self._active.append(server)

        if self._connect_in_parallel:
            await asyncio.gather(*(connect_one(s) for s in servers))
        else:
            for s in servers:
                await connect_one(s)

    async def _close_all(self, servers: list[McpServer]) -> None:
        async def close_one(server: McpServer) -> None:
            try:
                await server.close()
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass

        await asyncio.gather(*(close_one(s) for s in servers))
