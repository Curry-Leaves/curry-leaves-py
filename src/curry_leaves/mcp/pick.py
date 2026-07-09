"""`mcp_tools` — the one function callers use to get agent-ready tools from an
already-connected MCP server, for splicing into `Agent(tools=[...])`.

No I/O: it filters `server`'s already-discovered, cached tool list (populated at
connect time). Calling it more than once against the same connected server, with
different names each time, is how two different agents share one live connection with
disjoint tool access.
"""

from __future__ import annotations

from typing import Any

from curry_leaves.core.tools import Tool
from curry_leaves.mcp.server import McpServer


async def mcp_tools(server: McpServer, *names: str) -> "list[Tool[Any]]":
    """Pick specific tools by name from an already-connected `server`. Raises
    `ValueError` immediately if any name isn't among the server's discovered tools,
    rather than silently returning fewer tools than asked for.
    """
    available = await server.list_tools()
    by_remote_name = {_remote_name(t, server.name): t for t in available}

    picked: list[Tool[Any]] = []
    missing: list[str] = []
    for name in names:
        tool = by_remote_name.get(name)
        if tool is None:
            missing.append(name)
        else:
            picked.append(tool)

    if missing:
        known = ", ".join(sorted(by_remote_name.keys())) or "(none)"
        raise ValueError(
            f"MCP server '{server.name}' has no tool(s) named {missing!r}. "
            f"Available tools: {known}"
        )
    return picked


def _remote_name(tool: "Tool[Any]", server_name: str) -> str:
    """The bare remote tool name, stripped of the `mcp__<server>__` namespacing prefix
    `McpTool` applies — so callers pick by the name MCP's own `tools/list` reports, not
    the internal qualified form.
    """
    prefix = f"mcp__{server_name}__"
    return tool.name[len(prefix):] if tool.name.startswith(prefix) else tool.name
