"""MCP (Model Context Protocol) client support — connect to MCP servers and pick
specific tools from them for an `Agent`.

    from curry_leaves.mcp import McpServerStdio, McpServerManager, mcp_tools

    async with McpServerStdio(name="github", command="npx", args=[...]) as gh:
        agent = Agent(model="claude-sonnet-4-5",
                       tools=[*coding_tools(), *await mcp_tools(gh, "search_issues")])

`Agent` itself is untouched — `mcp_tools()` returns a plain `list[Tool]` for the
existing `tools=[...]`/`deferred_tools=[...]` constructor args.

Servers can also be loaded from the layered `settings.json`'s `"mcpServers"` key
instead of constructing them in code:

    from curry_leaves.mcp import load_mcp_servers, McpServerManager, mcp_tools

    async with McpServerManager(list(load_mcp_servers().values())) as manager:
        gh = manager.get("github")
        tools = await mcp_tools(gh, "search_issues")
"""

from __future__ import annotations

from curry_leaves.mcp.client import McpConnectionError, McpNotConnectedError
from curry_leaves.mcp.config import McpServerConfigError, load_mcp_servers
from curry_leaves.mcp.manager import McpServerManager
from curry_leaves.mcp.pick import mcp_tools
from curry_leaves.mcp.server import McpServer, McpServerHttp, McpServerStdio

__all__ = [
    "McpServer",
    "McpServerStdio",
    "McpServerHttp",
    "McpServerManager",
    "mcp_tools",
    "load_mcp_servers",
    "McpConnectionError",
    "McpNotConnectedError",
    "McpServerConfigError",
]
