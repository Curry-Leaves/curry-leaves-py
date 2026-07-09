from __future__ import annotations

import sys
from pathlib import Path

import pytest

from curry_leaves.mcp.client import McpNotConnectedError
from curry_leaves.mcp.server import McpServerHttp, McpServerStdio

FIXTURE = str(Path(__file__).parent / "fixtures" / "echo_server.py")


def test_construction_is_io_free() -> None:
    # Bogus command — if construction did any I/O, this would raise immediately.
    server = McpServerStdio(name="bogus", command="/no/such/binary", args=["--nope"])
    assert server.name == "bogus"


async def test_list_tools_before_connect_raises() -> None:
    server = McpServerStdio(name="bogus", command="/no/such/binary")
    with pytest.raises(McpNotConnectedError):
        await server.list_tools()


async def test_connect_discovers_full_unfiltered_catalog() -> None:
    server = McpServerStdio(name="echo", command=sys.executable, args=[FIXTURE])
    async with server:
        tools = await server.list_tools()
        names = sorted(t.name for t in tools)
        assert names == ["mcp__echo__add", "mcp__echo__echo", "mcp__echo__fail"]


async def test_close_is_idempotent() -> None:
    server = McpServerStdio(name="echo", command=sys.executable, args=[FIXTURE])
    await server.connect()
    await server.close()
    await server.close()  # must not raise


def test_http_server_construction_is_io_free() -> None:
    server = McpServerHttp(name="docs", url="https://example.invalid/mcp")
    assert server.name == "docs"


async def test_http_server_connect_discovers_catalog(http_fixture_url: str) -> None:
    server = McpServerHttp(name="http_echo", url=http_fixture_url)
    async with server:
        tools = await server.list_tools()
        assert sorted(t.name for t in tools) == ["mcp__http_echo__add", "mcp__http_echo__echo"]


async def test_http_server_connect_failure_raises_mcp_connection_error() -> None:
    from curry_leaves.mcp.client import McpConnectionError

    server = McpServerHttp(
        name="unreachable", url="http://127.0.0.1:1/mcp", connect_timeout_seconds=2
    )
    with pytest.raises(McpConnectionError):
        await server.connect()
    await server.close()
