"""Full path: McpServerManager over real fixture stdio servers, mcp_tools() picks from
each, Agent(tools=[...picked]) — confirm the ToolRegistry contains exactly the picked
tools (correctly namespaced) from both servers, a Runner builds an executor from them,
and Runner.close()/manager teardown doesn't error on double-close.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterable, Optional

from curry_leaves.core.agent import Agent
from curry_leaves.mcp import McpServerManager, McpServerStdio, mcp_tools
from curry_leaves.providers.base import Context, Model, StreamEvent, StreamOpts
from curry_leaves.runner import Runner

FIXTURE = str(Path(__file__).parent / "fixtures" / "echo_server.py")


class _StubProvider:
    """Never actually streams — this test only exercises tool registration/teardown,
    not a real model turn.
    """

    async def stream(
        self, ctx: Context, model: Model, opts: Optional[StreamOpts] = None
    ) -> AsyncIterable[StreamEvent]:
        raise AssertionError("stream() should not be called in this test")
        yield  # pragma: no cover - makes this an async generator


_STUB_MODEL = Model(
    id="stub-model", provider="stub", max_output_tokens=1024, context_window=8192, supports_thinking=False
)


async def test_agent_registry_contains_only_picked_tools() -> None:
    server_a = McpServerStdio(name="echoA", command=sys.executable, args=[FIXTURE])
    server_b = McpServerStdio(name="echoB", command=sys.executable, args=[FIXTURE])

    async with McpServerManager([server_a, server_b]) as manager:
        a = manager.get("echoA")
        b = manager.get("echoB")

        picked_a = await mcp_tools(a, "echo")
        picked_b = await mcp_tools(b, "add", "fail")

        agent = Agent(
            model=_STUB_MODEL,
            provider=_StubProvider(),
            tools=[*picked_a, *picked_b],
        )

        registered_names = sorted(t.name for t in agent.tools.tools())
        assert registered_names == ["mcp__echoA__echo", "mcp__echoB__add", "mcp__echoB__fail"]

        runner = Runner(agent)
        # Runner._bind_agent already built an executor registry from agent.tools —
        # confirm the same picked tools are present there too (not just on Agent).
        exec_names = sorted(t.name for t in runner._exec_registry.tools())
        assert "mcp__echoA__echo" in exec_names
        assert "mcp__echoB__add" in exec_names
        assert "mcp__echoB__fail" in exec_names

        await runner.close()

    # Manager teardown + Runner.close() both ran — closing again must not raise.
    await server_a.close()
    await server_b.close()


async def test_two_agents_share_one_server_disjoint_tools() -> None:
    async with McpServerManager([McpServerStdio(name="echo", command=sys.executable, args=[FIXTURE])]) as manager:
        server = manager.get("echo")

        research_agent = Agent(
            model=_STUB_MODEL, provider=_StubProvider(), tools=[*await mcp_tools(server, "echo")]
        )
        release_agent = Agent(
            model=_STUB_MODEL, provider=_StubProvider(), tools=[*await mcp_tools(server, "add", "fail")]
        )

        assert [t.name for t in research_agent.tools.tools()] == ["mcp__echo__echo"]
        assert sorted(t.name for t in release_agent.tools.tools()) == ["mcp__echo__add", "mcp__echo__fail"]
