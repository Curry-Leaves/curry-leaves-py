#!/usr/bin/env python3
"""07_mcp_tools.py — give an agent tools from an MCP (Model Context Protocol) server.

Connect a server, then pick specific tools from it by name with `mcp_tools()` — the
result is a plain `list[Tool]`, spliced straight into `Agent(tools=[...])` exactly like
any built-in preset. `Agent` itself needs no special MCP support: it never sees a
server or does any MCP I/O, only the tools `mcp_tools()` already resolved.

This example bundles its own tiny MCP server (`tests/mcp/fixtures/echo_server.py`, a
couple of trivial tools) as a local subprocess, so it runs with no external MCP server,
no network, and no extra secrets beyond your usual ANTHROPIC_API_KEY/OPENAI_API_KEY.

Risk/permissions: every MCP tool defaults to `risk="exec"` (always prompts/needs an
explicit verdict) since a remote tool's real effect can't be verified statically —
see `agent.permissions` below for per-tool overrides, and `mcp/server.py`'s docstring
for the full risk/permissions story including the "no wildcard yet" limitation.

Run:  python3 examples/07_mcp_tools.py "Use the echo tool to say hi, then add 19 and 23"

To instead load server definitions from `.curry-leaves/settings.json`'s "mcpServers"
key (see `mcp/config.py`'s docstring for the file format) rather than constructing
`McpServerStdio(...)` in code:

    python3 examples/07_mcp_tools.py --from-settings "..."
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from curry_leaves import Agent, Runner, coding_tools
from curry_leaves.mcp import McpServerManager, McpServerStdio, load_mcp_servers, mcp_tools

MODEL = os.environ.get("CURRY_LEAVES_MODEL", "claude-sonnet-4-5")
FIXTURE_SERVER = str(Path(__file__).resolve().parent.parent / "tests" / "mcp" / "fixtures" / "echo_server.py")


async def run_from_settings(prompt: str) -> None:
    """The settings.json-driven path: servers come from load_mcp_servers() instead of
    being constructed in code. Requires a .curry-leaves/settings.json with an
    "mcpServers" entry (see mcp/config.py's docstring for the exact shape) — this only
    demonstrates the wiring, not a specific server, so it prints what it finds rather
    than assuming any particular tool exists.
    """
    servers = load_mcp_servers()
    if not servers:
        print(
            "No mcpServers configured in .curry-leaves/settings.json "
            "(or ~/.curry-leaves/settings.json). Nothing to connect."
        )
        return

    async with McpServerManager(list(servers.values())) as manager:
        print(f"Connected: {[s.name for s in manager.active_servers]}")
        if manager.failed_servers:
            print(f"Failed: {[s.name for s in manager.failed_servers]} — {manager.errors}")

        all_tools = []
        for server in manager.active_servers:
            discovered = await server.list_tools()
            print(f"  {server.name}: {[t.name for t in discovered]}")
            all_tools.extend(discovered)  # no filtering here — real usage would mcp_tools() specific names

        agent = Agent(model=MODEL, tools=[*coding_tools(), *all_tools])
        result = await Runner(agent).run(prompt)
        print(result.output_text)


async def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--from-settings":
        await run_from_settings(" ".join(args[1:]) or "What tools do you have available?")
        return

    prompt = " ".join(args) or "Use the echo tool to say hi, then add 19 and 23."

    # A server object is pure config at construction — no I/O happens until connected.
    echo_server = McpServerStdio(name="echo", command=sys.executable, args=[FIXTURE_SERVER])

    # McpServerManager connects one or many servers as a batch and drops any that fail
    # to connect rather than failing the whole run (see manager.failed_servers/.errors).
    # For a single server, `async with echo_server:` works standalone too.
    async with McpServerManager([echo_server]) as manager:
        server = manager.get("echo")

        # mcp_tools() picks specific tools by name from the already-connected server —
        # unlisted tools (e.g. the fixture's `fail` tool) never become agent-visible.
        picked = await mcp_tools(server, "echo", "add")

        agent = Agent(
            model=MODEL,
            instructions="You are a helpful assistant. Use the provided tools rather than guessing.",
            tools=[*coding_tools(), *picked],
            permissions={"mcp__echo__echo": "allow", "mcp__echo__add": "allow"},
        )

        result = await Runner(agent).run(prompt)
        print(result.output_text)

        if manager.failed_servers:
            print("Skipped (couldn't connect):", [s.name for s in manager.failed_servers], manager.errors)

    # Two agents sharing one connection with disjoint tool access — the same server,
    # picked twice with different names:
    async with McpServerStdio(name="echo2", command=sys.executable, args=[FIXTURE_SERVER]) as shared:
        agent_a = Agent(model=MODEL, tools=[*await mcp_tools(shared, "echo")])
        agent_b = Agent(model=MODEL, tools=[*await mcp_tools(shared, "add")])
        print(f"\nagent_a tools: {[t.name for t in agent_a.tools.tools()]}")
        print(f"agent_b tools: {[t.name for t in agent_b.tools.tools()]}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001
        print(e, file=sys.stderr)
        sys.exit(1)
