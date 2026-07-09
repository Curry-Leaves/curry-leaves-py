from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from curry_leaves.mcp.config import McpServerConfigError, load_mcp_servers
from curry_leaves.mcp.server import McpServerHttp, McpServerStdio
from curry_leaves.util import paths


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path) -> Iterator[None]:
    """Every test gets its own fake `~/.curry-leaves` so it never reads/writes the
    real user's settings.json."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    paths.set_home(str(fake_home))
    yield
    paths.set_home(None)


def _write_project_settings(tmp_path: Path, data: dict) -> Path:
    project_dir = tmp_path / "project"
    (project_dir / ".curry-leaves").mkdir(parents=True)
    settings_file = project_dir / ".curry-leaves" / "settings.json"
    settings_file.write_text(json.dumps(data))
    return project_dir


def test_missing_mcp_servers_key_returns_empty(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {"model": "claude-sonnet-4-5"})
    servers = load_mcp_servers(cwd=str(project_dir))
    assert servers == {}


def test_no_settings_file_at_all_returns_empty(tmp_path: Path) -> None:
    empty_dir = tmp_path / "nowhere"
    empty_dir.mkdir()
    servers = load_mcp_servers(cwd=str(empty_dir))
    assert servers == {}


def test_stdio_server_parsed(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {
        "mcpServers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "abc123"},
            }
        }
    })
    servers = load_mcp_servers(cwd=str(project_dir))
    assert set(servers.keys()) == {"github"}
    assert isinstance(servers["github"], McpServerStdio)
    assert servers["github"].name == "github"


def test_http_server_parsed_transport_auto_detected(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {
        "mcpServers": {"docs": {"url": "https://example.com/mcp"}}
    })
    servers = load_mcp_servers(cwd=str(project_dir))
    assert isinstance(servers["docs"], McpServerHttp)


def test_sse_transport_explicit(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {
        "mcpServers": {"legacy": {"url": "https://example.com/sse", "transport": "sse"}}
    })
    servers = load_mcp_servers(cwd=str(project_dir))
    assert isinstance(servers["legacy"], McpServerHttp)


def test_header_env_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCS_API_KEY", "sekret-value")
    project_dir = _write_project_settings(tmp_path, {
        "mcpServers": {
            "docs": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer ${DOCS_API_KEY}"},
            }
        }
    })
    servers = load_mcp_servers(cwd=str(project_dir))
    docs = servers["docs"]
    assert isinstance(docs, McpServerHttp)
    assert docs._headers == {"Authorization": "Bearer sekret-value"}  # type: ignore[attr-defined]


def test_header_env_interpolation_unset_var_left_literal(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {
        "mcpServers": {
            "docs": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer ${NEVER_SET_XYZ}"},
            }
        }
    })
    servers = load_mcp_servers(cwd=str(project_dir))
    docs = servers["docs"]
    assert isinstance(docs, McpServerHttp)
    assert docs._headers == {"Authorization": "Bearer ${NEVER_SET_XYZ}"}  # type: ignore[attr-defined]


def test_malformed_entry_not_an_object_raises(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {"mcpServers": {"bad": "not-an-object"}})
    with pytest.raises(McpServerConfigError):
        load_mcp_servers(cwd=str(project_dir))


def test_stdio_missing_command_raises(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {
        "mcpServers": {"bad": {"transport": "stdio"}}
    })
    with pytest.raises(McpServerConfigError):
        load_mcp_servers(cwd=str(project_dir))


def test_no_command_or_url_raises(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {"mcpServers": {"bad": {}}})
    with pytest.raises(McpServerConfigError):
        load_mcp_servers(cwd=str(project_dir))


def test_risk_and_timeout_passthrough(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {
        "mcpServers": {
            "readonly": {"command": "npx", "args": [], "risk": "read", "timeout": 30}
        }
    })
    servers = load_mcp_servers(cwd=str(project_dir))
    server = servers["readonly"]
    assert isinstance(server, McpServerStdio)
    assert server._risk == "read"  # type: ignore[attr-defined]
    assert server._timeout == 30  # type: ignore[attr-defined]


def test_multiple_servers(tmp_path: Path) -> None:
    project_dir = _write_project_settings(tmp_path, {
        "mcpServers": {
            "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
            "docs": {"url": "https://example.com/mcp"},
        }
    })
    servers = load_mcp_servers(cwd=str(project_dir))
    assert set(servers.keys()) == {"github", "docs"}
