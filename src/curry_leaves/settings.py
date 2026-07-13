"""Layered settings — one resolved config from several sources, later layers winning:

  defaults  <  user (~/.curry-leaves/settings.json)  <  project (.curry-leaves/settings.json)  <  env

Dicts deep-merge; scalars replace. JSON keeps it dependency-free. This is where durable
permission state lives: ``approvals.allow`` is the standing "always allow (everywhere)" list
that seeds the PermissionEngine so those tools never prompt again. ``add_global_approval``
appends to the USER file (the global layer) and is what a frontend calls on an "always" choice.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .util.paths import find_up, home, join

_USER_REL = "settings.json"
_PROJECT_REL = join(".curry-leaves", "settings.json")

# env var → dotted settings path
_ENV_MAP: dict[str, str] = {
    "CURRY_LEAVES_PROVIDER": "provider",
    "CURRY_LEAVES_MODEL": "model",
}


class Settings(dict[str, Any]):
    """Resolved settings dict.

    Mirrors the TS ``Settings`` interface (``provider``, ``model``, ``approvals.allow``,
    ``auto.hosts``, plus an open index signature "room to grow without schema churn") — a
    plain dict subclass is the natural Python analogue of a TS interface with a `[key: string]:
    unknown` catch-all, since attribute access was never used on it in the original either.
    """


def user_settings_path() -> str:
    return join(home(), _USER_REL)


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}  # missing or corrupt → treat as empty; never crash startup


def _is_object(v: Any) -> bool:
    return isinstance(v, dict)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if _is_object(v) and _is_object(out.get(k)) else v
    return out


def _env_layer() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for var_name, key in _ENV_MAP.items():
        val = os.environ.get(var_name)
        if val:
            out[key] = val
    return out


def load_settings(cwd: str | None = None) -> Settings:
    """Resolve settings for `cwd` (defaults → user → project → env). Never throws."""
    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, _read_json(user_settings_path()))
    project = find_up(_PROJECT_REL, cwd)
    if project:
        merged = _deep_merge(merged, _read_json(project))
    merged = _deep_merge(merged, _env_layer())
    return Settings(merged)


# The default local (Ollama) model used when no cloud API key is present. Chosen for solid
# tool-use support. Overridable per-run with $CURRY_LEAVES_MODEL.
DEFAULT_LOCAL_MODEL = "qwen3"


def resolve_default_model(cwd: str | None = None) -> str:
    """Resolve the model to start with — the zero-config "just works" entry point.

      1. an explicit choice: settings `model` (which env `$CURRY_LEAVES_MODEL` feeds) or a
         remembered `/model` pick — always wins.
      2. detect an available cloud provider by key presence: ANTHROPIC_API_KEY → claude,
         OPENAI_API_KEY → gpt-*.
      3. otherwise fall back to a local Ollama model — no API key needed, works immediately.

    The user can change the model any time (env or `/model`); a `/model` pick is remembered via
    `save_model_choice`, so this only decides the very first run.
    """
    explicit = load_settings(cwd).get("model")
    if explicit:
        return str(explicit)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-5"
    return DEFAULT_LOCAL_MODEL


def _write_user_settings(next_settings: dict[str, Any]) -> None:
    """Persist the whole USER settings file. Best-effort — a write failure is swallowed, since
    recording preferences is never the run's job.
    """
    path = user_settings_path()
    try:
        dir_ = os.path.dirname(path)
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
        with open(path, "w", encoding="utf8") as f:
            f.write(json.dumps(next_settings, indent=2) + "\n")
    except Exception:
        pass  # best-effort


def save_model_choice(model: str) -> None:
    """Remember the user's model choice by writing it to the USER settings file (the same layer
    as `add_global_approval`). Best-effort; a write failure is swallowed. Reads the user file
    directly so we only ever touch our own layer, never the merged view.
    """
    current = _read_json(user_settings_path())
    if current.get("model") == model:
        return
    _write_user_settings({**current, "model": model})


def global_approvals(cwd: str | None = None) -> list[str]:
    """The standing global allowlist (tools always permitted across sessions)."""
    s = load_settings(cwd)
    approvals = s.get("approvals")
    if isinstance(approvals, dict):
        allow = approvals.get("allow")
        if isinstance(allow, list):
            return list(allow)
    return []


def auto_hosts(cwd: str | None = None) -> list[str]:
    """Network hosts auto-approved when auto mode is on (from settings.json `auto.hosts`)."""
    s = load_settings(cwd)
    auto = s.get("auto")
    if isinstance(auto, dict):
        hosts = auto.get("hosts")
        if isinstance(hosts, list):
            return list(hosts)
    return []


def add_global_approval(tool: str) -> None:
    """Append a tool to the USER settings' global allowlist and persist it, so future sessions
    never prompt for it. Idempotent; best-effort (a write failure is swallowed — recording is not
    the run's job). Reads the user file directly (not the merged view) so we only ever write our
    layer.
    """
    current = _read_json(user_settings_path())
    approvals = current.get("approvals")
    existing = approvals.get("allow") if isinstance(approvals, dict) else None
    allow = set(existing) if isinstance(existing, list) else set()
    if tool in allow:
        return
    allow.add(tool)
    next_approvals: dict[str, Any] = {**(approvals if isinstance(approvals, dict) else {}), "allow": list(allow)}
    _write_user_settings({**current, "approvals": next_approvals})
