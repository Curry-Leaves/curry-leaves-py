"""`load_mcp_servers` — build `McpServerStdio`/`McpServerHttp` objects from the
existing layered `settings.json` (`~/.curry-leaves/settings.json` < project
`.curry-leaves/settings.json`), under the `"mcpServers"` key — the same key Claude
Desktop/Claude Code/OpenAI Agents SDK already use, so a config is portable.

Fully optional: the primary, documented way to get a server is constructing
`McpServerStdio(...)`/`McpServerHttp(...)` directly in code (see `mcp/server.py`). This
is a convenience for teams that would rather keep server definitions in a config file.
Reuses `settings.py`'s `load_settings()` deep-merge as-is — no new config-loading
mechanism, no schema knowledge added to `settings.py` itself.

    {
      "mcpServers": {
        "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_TOKEN": "..."}},
        "docs":    {"url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer ${DOCS_API_KEY}"}}
      }
    }

`transport` is auto-detected when omitted: `command` present -> stdio; `url` present ->
http. Header values support `${VAR}` interpolation against `os.environ`, so secrets
don't need to be committed to settings.json literally (mirrors how stdio's own `env`
already lets a subprocess inherit/receive real env values).
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Optional

from curry_leaves.mcp.server import McpServer, McpServerHttp, McpServerStdio
from curry_leaves.settings import load_settings

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env(value: str) -> str:
    """Replace `${VAR}` references with `os.environ["VAR"]`. An unset var is left as
    the literal `${VAR}` text (never silently dropped, so a misconfigured/missing
    secret is visible rather than turning into an empty string)."""

    def repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        return os.environ.get(name, m.group(0))

    return _VAR_PATTERN.sub(repl, value)


def _interpolate_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: _interpolate_env(v) for k, v in headers.items()}


class McpServerConfigError(Exception):
    """Raised when a `mcpServers` entry in settings.json is malformed."""


def _build_server(name: str, entry: dict[str, Any]) -> McpServer:
    command = entry.get("command")
    url = entry.get("url")
    transport = entry.get("transport")

    if transport is None:
        transport = "stdio" if command is not None else ("http" if url is not None else None)

    if transport == "stdio":
        if not isinstance(command, str):
            raise McpServerConfigError(
                f"mcpServers.{name}: stdio server requires a string 'command'"
            )
        args = entry.get("args") or []
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise McpServerConfigError(f"mcpServers.{name}: 'args' must be a list of strings")
        env = entry.get("env") or {}
        if not isinstance(env, dict):
            raise McpServerConfigError(f"mcpServers.{name}: 'env' must be an object")
        return McpServerStdio(
            name=name,
            command=command,
            args=list(args),
            env={k: str(v) for k, v in env.items()},
            cwd=entry.get("cwd"),
            risk=entry.get("risk", "exec"),
            timeout=entry.get("timeout"),
            cache_tools_list=bool(entry.get("cache_tools_list", False)),
        )

    if transport in ("http", "sse"):
        if not isinstance(url, str):
            raise McpServerConfigError(
                f"mcpServers.{name}: {transport} server requires a string 'url'"
            )
        headers = entry.get("headers") or {}
        if not isinstance(headers, dict):
            raise McpServerConfigError(f"mcpServers.{name}: 'headers' must be an object")
        resolved_transport: Literal["http", "sse"] = "sse" if transport == "sse" else "http"
        return McpServerHttp(
            name=name,
            url=url,
            headers=_interpolate_headers({k: str(v) for k, v in headers.items()}),
            transport=resolved_transport,
            risk=entry.get("risk", "exec"),
            timeout=entry.get("timeout"),
            connect_timeout_seconds=entry.get("connect_timeout_seconds"),
            cache_tools_list=bool(entry.get("cache_tools_list", False)),
        )

    raise McpServerConfigError(
        f"mcpServers.{name}: could not determine transport — provide 'command' (stdio), "
        f"'url' (http/sse), or an explicit 'transport'"
    )


def load_mcp_servers(cwd: Optional[str] = None) -> dict[str, McpServer]:
    """Resolve `mcpServers` from the layered settings (user < project < env) into
    ready-to-connect server objects, keyed by name. Never raises for a MISSING
    `mcpServers` key (returns `{}`) — matches `load_settings`'s "never throw on absent
    config" philosophy. A malformed individual entry raises `McpServerConfigError`
    immediately (unlike settings.py's "swallow and treat as empty," a bad MCP server
    definition is a configuration bug worth surfacing loudly, not silently dropping).
    """
    settings = load_settings(cwd)
    raw = settings.get("mcpServers")
    if not isinstance(raw, dict):
        return {}
    servers: dict[str, McpServer] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise McpServerConfigError(f"mcpServers.{name}: must be an object")
        servers[name] = _build_server(name, entry)
    return servers
