"""The connectable MCP server objects — `McpServerStdio` and `McpServerHttp`.

Construction is pure config — no I/O. Connecting (`connect()` / `async with server:`)
performs the MCP `initialize` handshake and `tools/list`, caching the full, UNFILTERED
remote tool catalog as `McpTool` instances. Tool selection is deliberately NOT a
server-level concern — see `mcp/pick.py`'s `mcp_tools()`, which is how callers narrow
down to specific tools per agent.

## Risk and permissions

Every tool discovered on a server defaults to `risk="exec"` — the most conservative
`Risk` tier (`permission.py`'s fallback rule prompts/blocks anything that isn't
`"read"`), because an arbitrary remote MCP tool's real effect can't be verified
statically the way a local file-write's path can (`contained_approval`'s "exec: never
auto-approve" branch already does the right thing here with no changes needed). Pass
`risk="read"` at server construction if you trust every tool on that server to be
read-only (e.g. a docs-search server); leave it as `"exec"` (or set `"write"`) for
anything that mutates state:

    docs = McpServerStdio(name="docs", command="npx", args=[...], risk="read")
    github = McpServerStdio(name="github", command="npx", args=[...])  # stays "exec"

Fine-grained per-tool verdicts still go through the framework's existing
`agent.permissions` map, keyed by the tool's namespaced name
(`mcp__<server>__<tool>`) — no MCP-specific permission mechanism exists or is needed:

    agent = Agent(model="...", tools=[*await_mcp_picks],
                  permissions={"mcp__github__search_issues": "allow",
                               "mcp__github__create_pull_request": "ask"})

Known limitation: `agent.permissions` matches EXACT tool names, so "allow every tool on
server X" currently means listing each of that server's picked tools individually
(there's no `"mcp__github__*"` wildcard). This is a pre-existing property of the
permissions map's design, not something MCP support changes — a future prefix/glob
convention would need to land in `PermissionEngine._decide`, out of scope here.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Protocol, runtime_checkable

from curry_leaves.core.tools import Risk, Tool
from curry_leaves.mcp.client import McpNotConnectedError, _McpSession
from curry_leaves.mcp.tool import McpTool


@runtime_checkable
class McpServer(Protocol):
    """The common shape both `McpServerStdio` and `McpServerHttp` satisfy."""

    name: str

    async def connect(self) -> None: ...
    async def __aenter__(self) -> "McpServer": ...
    async def __aexit__(self, *exc: object) -> None: ...
    async def list_tools(self) -> "list[Tool[Any]]": ...
    async def close(self) -> None: ...


class _BaseServer:
    """Shared discovery/caching/teardown logic for the concrete server classes below.
    Not part of the public API — construct `McpServerStdio`/`McpServerHttp` directly.
    """

    name: str
    _risk: Risk
    _timeout: Optional[float]
    _cache_tools_list: bool

    def __init__(self) -> None:
        self._session = _McpSession(self.name)
        self._tools: Optional[list[McpTool]] = None

    async def _do_connect(self) -> None:
        """Subclasses call this after opening the transport, to discover (or reuse the
        cached) tool list."""
        if self._tools is not None and self._cache_tools_list:
            return
        remote_tools = await self._session.list_tools()
        self._tools = [
            McpTool(self._session, self.name, t, risk=self._risk, timeout=self._timeout)
            for t in remote_tools
        ]

    def invalidate_tools_cache(self) -> None:
        """Force the next `connect()`/`list_tools()` to re-fetch `tools/list`, even
        when `cache_tools_list=True`."""
        self._tools = None

    async def list_tools(self) -> "list[Tool[Any]]":
        if self._tools is None:
            raise McpNotConnectedError(
                f"MCP server '{self.name}' is not connected — call `await server.connect()` "
                f"or use `async with server:` before listing/picking its tools."
            )
        return list(self._tools)

    async def close(self) -> None:
        await self._session.close()
        if not self._cache_tools_list:
            self._tools = None


class McpServerStdio(_BaseServer):
    """An MCP server launched as a local subprocess over stdio."""

    def __init__(
        self,
        *,
        name: str,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        risk: Risk = "exec",
        timeout: Optional[float] = None,
        cache_tools_list: bool = False,
    ) -> None:
        self.name = name
        self._command = command
        self._args = list(args) if args is not None else []
        self._env = dict(env) if env is not None else {}
        self._cwd = cwd
        self._risk: Risk = risk
        self._timeout = timeout
        self._cache_tools_list = cache_tools_list
        super().__init__()

    async def connect(self) -> None:
        await self._session.connect_stdio(
            command=self._command, args=self._args, env=self._env, cwd=self._cwd
        )
        await self._do_connect()

    async def __aenter__(self) -> "McpServerStdio":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class McpServerHttp(_BaseServer):
    """An MCP server reached over HTTP (streamable-http, the modern default) or
    Server-Sent Events (the legacy transport, for older servers)."""

    def __init__(
        self,
        *,
        name: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        transport: Literal["http", "sse"] = "http",
        risk: Risk = "exec",
        timeout: Optional[float] = None,
        connect_timeout_seconds: Optional[float] = None,
        cache_tools_list: bool = False,
    ) -> None:
        self.name = name
        self._url = url
        self._headers = dict(headers) if headers is not None else {}
        self._transport: Literal["http", "sse"] = transport
        self._risk: Risk = risk
        self._timeout = timeout
        self._connect_timeout_seconds = connect_timeout_seconds
        self._cache_tools_list = cache_tools_list
        super().__init__()

    async def connect(self) -> None:
        await self._session.connect_http(
            url=self._url,
            headers=self._headers,
            transport=self._transport,
            connect_timeout_seconds=self._connect_timeout_seconds,
        )
        await self._do_connect()

    async def __aenter__(self) -> "McpServerHttp":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
