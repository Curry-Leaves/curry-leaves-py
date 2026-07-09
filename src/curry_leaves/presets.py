"""Presets — batteries-included tool bundles so consumers don't re-derive the wiring.

Plain functions returning plain lists: inspectable, filterable, no magic.

    from curry_leaves import Agent, Runner, coding_tools, web_tools
    agent = Agent(model=model, instructions="…",
                   tools=coding_tools(), deferred_tools=web_tools())

Subtract what you don't want: `[t for t in coding_tools() if t.name != "bash"]`.
"""

from __future__ import annotations

from typing import Any

from curry_leaves.core.tools import Tool
from curry_leaves.tools.ask import ask_tool
from curry_leaves.tools.bash import bash_tool
from curry_leaves.tools.current_time import current_time_tool
from curry_leaves.tools.edit import edit_tool
from curry_leaves.tools.find import find_tool
from curry_leaves.tools.read import read_tool
from curry_leaves.tools.search import search_tool
from curry_leaves.tools.tasks import task_tools
from curry_leaves.tools.web import web_fetch_tool, web_search_tool
from curry_leaves.tools.write import write_tool


def coding_tools() -> list[Tool[Any]]:
    """The standard always-on coding toolset (filesystem, exec, search, tasks, ask)."""
    return [
        read_tool(),
        find_tool(),
        search_tool(),
        write_tool(),
        edit_tool(),
        bash_tool(),
        *task_tools(),
        ask_tool(),
    ]


def web_tools() -> list[Tool[Any]]:
    """Network / time tools — usually registered as deferred_tools (hidden until searched)."""
    return [current_time_tool(), web_fetch_tool(), web_search_tool()]
