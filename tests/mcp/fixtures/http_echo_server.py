"""A minimal MCP streamable-http fixture server for tests. Run directly:

    python http_echo_server.py <port>

Serves the same tools as echo_server.py but over streamable-http, so HTTP transport
tests don't need network access or a real remote MCP server.
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
mcp = FastMCP("http-echo-fixture", host="127.0.0.1", port=port, stateless_http=True)


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back."""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
