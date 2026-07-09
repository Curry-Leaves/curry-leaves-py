"""Resource URLs — the small scheme layer behind `read`.

  artifact://<id>   full output a tool offloaded to the blob store
  local://<slug>    a session resource (a plan, etc.)
  skill://<name>    a skill's SKILL.md body (or a bundled file)
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from curry_leaves.core.blobs import BlobStore

_SCHEMES = {"artifact", "local", "skill"}


def is_resource_url(s: str) -> bool:
    i = s.find("://")
    if i < 0:
        return False
    return s[:i] in _SCHEMES


class Resolvers(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    blobs: BlobStore | None = None
    resolve_local: Callable[[str], str | None] | None = None
    resolve_skill: Callable[[str], str | None] | None = None


def resolve_url(url: str, r: Resolvers) -> str:
    """Resolve a resource URL to text. Raises ValueError (model-readable) when it can't."""
    i = url.find("://")
    scheme = url[:i]
    rest = url[i + 3 :]

    if scheme == "artifact":
        if r.blobs is None:
            raise ValueError("no artifact store available")
        text = r.blobs.get_text(rest)
        if text is None:
            raise ValueError(f"unknown artifact id: {rest}")
        return text
    if scheme == "local":
        if r.resolve_local is None:
            raise ValueError("no local resolver available")
        text = r.resolve_local(rest)
        if text is None:
            raise ValueError(f"unknown local resource: {rest}")
        return text
    if scheme == "skill":
        if r.resolve_skill is None:
            raise ValueError("no skill resolver available")
        text = r.resolve_skill(rest)
        if text is None:
            raise ValueError(f"unknown skill: {rest}")
        return text
    raise ValueError(f"unknown resource scheme: {scheme}")
