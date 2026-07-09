"""Real end-to-end test: a genuine MCP stdio subprocess (the fixture echo server),
connected via the official `mcp` package's own stdio_client through our
`McpServerStdio`/`_McpSession`. No network, no mocks — this is the one test that
exercises the actual transport.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from curry_leaves.mcp.server import McpServerStdio

FIXTURE = str(Path(__file__).parent / "fixtures" / "echo_server.py")


async def test_connect_list_call_close_real_subprocess() -> None:
    server = McpServerStdio(name="echo", command=sys.executable, args=[FIXTURE])
    await server.connect()
    try:
        tools = await server.list_tools()
        assert sorted(t.name for t in tools) == [
            "mcp__echo__add",
            "mcp__echo__echo",
            "mcp__echo__fail",
        ]

        echo_tool = next(t for t in tools if t.name == "mcp__echo__echo")
        args = echo_tool.schema.model_validate({"text": "hello mcp"})
        result = await echo_tool.run(args, None, asyncio.Event())  # type: ignore[arg-type]
        assert result.content == "hello mcp"
        assert result.is_error is False

        add_tool = next(t for t in tools if t.name == "mcp__echo__add")
        args2 = add_tool.schema.model_validate({"a": 19, "b": 23})
        result2 = await add_tool.run(args2, None, asyncio.Event())  # type: ignore[arg-type]
        assert result2.content == "42"

        fail_tool = next(t for t in tools if t.name == "mcp__echo__fail")
        args3 = fail_tool.schema.model_validate({"message": "kaboom"})
        result3 = await fail_tool.run(args3, None, asyncio.Event())  # type: ignore[arg-type]
        assert result3.is_error is True
        assert "kaboom" in result3.content
    finally:
        await server.close()


async def test_context_manager_connects_and_closes() -> None:
    async with McpServerStdio(name="echo", command=sys.executable, args=[FIXTURE]) as server:
        tools = await server.list_tools()
        assert len(tools) == 3
    # After exiting, the underlying subprocess/session is torn down; a second close()
    # (e.g. via Runner.close()) must not raise.
    await server.close()
