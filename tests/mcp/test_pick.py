from __future__ import annotations

from typing import Any

import pytest
from mcp import types

from curry_leaves.mcp.pick import mcp_tools
from curry_leaves.mcp.server import McpServer
from curry_leaves.mcp.tool import McpTool


class _FakeSession:
    def __init__(self) -> None:
        self.call_count = 0

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")], isError=False)

    async def close(self) -> None:
        pass


class _FakeServer:
    """A minimal stand-in satisfying `McpServer`'s structural shape, with a fixed,
    already-"discovered" tool list (no real connection).
    """

    def __init__(self, name: str, tool_names: list[str]) -> None:
        self.name = name
        self._session = _FakeSession()
        self.list_tools_calls = 0
        self._tools = [
            McpTool(self._session, name, types.Tool(name=n, description=f"{n} tool", inputSchema={"type": "object"}))
            for n in tool_names
        ]

    async def connect(self) -> None:
        pass

    async def __aenter__(self) -> "_FakeServer":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    async def list_tools(self) -> "list[Any]":
        self.list_tools_calls += 1
        return list(self._tools)

    async def close(self) -> None:
        pass


def test_fake_server_satisfies_protocol() -> None:
    server: McpServer = _FakeServer("github", ["search_issues"])
    assert isinstance(server, McpServer)


async def test_picks_named_tools_only() -> None:
    server = _FakeServer("github", ["search_issues", "create_pull_request", "merge_pull_request"])
    picked = await mcp_tools(server, "search_issues")
    assert [t.name for t in picked] == ["mcp__github__search_issues"]


async def test_pick_does_zero_additional_io() -> None:
    server = _FakeServer("github", ["a", "b"])
    await mcp_tools(server, "a")
    assert server.list_tools_calls == 1  # one call to list_tools, no re-fetch per name


async def test_unknown_name_raises_value_error() -> None:
    server = _FakeServer("github", ["a", "b"])
    with pytest.raises(ValueError, match="nonexistent"):
        await mcp_tools(server, "nonexistent")


async def test_two_disjoint_picks_share_one_server() -> None:
    server = _FakeServer("github", ["search_issues", "create_pull_request", "merge_pull_request"])
    read_only = await mcp_tools(server, "search_issues")
    write_only = await mcp_tools(server, "create_pull_request", "merge_pull_request")
    assert [t.name for t in read_only] == ["mcp__github__search_issues"]
    assert [t.name for t in write_only] == [
        "mcp__github__create_pull_request",
        "mcp__github__merge_pull_request",
    ]
    # same underlying server object, not reconnected
    assert server.list_tools_calls == 2
