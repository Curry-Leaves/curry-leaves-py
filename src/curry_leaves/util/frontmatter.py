"""Minimal YAML-ish frontmatter parsing for markdown assets (skills).

Handles the common ``--- key: value --- body`` header. Flat string keys only.
"""

from __future__ import annotations


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[3:end].split("\n"):
        i = line.find(":")
        if i >= 0:
            meta[line[:i].strip()] = line[i + 1 :].strip()
    body = text[end + 4 :]
    return meta, body.lstrip("\n")
