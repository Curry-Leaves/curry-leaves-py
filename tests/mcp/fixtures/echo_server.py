"""A minimal MCP stdio fixture server for tests — a couple of trivial tools, no
network/secrets required. Run directly: `python echo_server.py`.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-fixture")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back."""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def fail(message: str = "boom") -> str:
    """Always raises, to exercise error-path handling."""
    raise RuntimeError(message)


if __name__ == "__main__":
    mcp.run(transport="stdio")
