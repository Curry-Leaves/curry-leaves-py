"""`search_tools` — the client-side tool-search pattern. Deferred tools aren't advertised
upfront; the model calls this to discover them by keyword, and matches are ACTIVATED so
they appear (and become callable) from the next turn on.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

import pydantic

from curry_leaves.core.tools import Risk, ToolRegistry, ToolResult
from curry_leaves.providers.base import Context


class SearchToolsArgs(pydantic.BaseModel):
    query: str = pydantic.Field(
        description="Keywords for the capability you need, e.g. 'image', 'current time', 'database'."
    )


class SearchToolsTool:
    """Structurally satisfies the `Tool` protocol (see core/tools.py)."""

    name = "search_tools"
    risk: Optional[Risk] = "read"
    description = (
        "Search for more tools by keyword when the currently available tools aren't enough. Returns "
        "matching tools, which then become available for you to call."
    )
    schema: type[pydantic.BaseModel] = SearchToolsArgs
    timeout: Optional[float] = None

    def __init__(self, registry: ToolRegistry, activate: Callable[[str], None]) -> None:
        self._registry = registry
        self._activate = activate

    async def run(self, args: SearchToolsArgs, ctx: Context, signal: asyncio.Event) -> ToolResult:
        matches = self._registry.search(args.query)
        if len(matches) == 0:
            return ToolResult(content=f"No tools found for '{args.query}'.")
        lines: list[str] = []
        for tool in matches:
            self._activate(tool.name)  # advertised from the next turn on
            lines.append(f"- {tool.name}: {tool.description}")
        return ToolResult(content="Found these tools (now available to call):\n" + "\n".join(lines))

    async def close(self) -> None:
        pass
