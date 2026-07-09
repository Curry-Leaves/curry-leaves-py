"""Skills — lazily-loaded capability packages (progressive disclosure).

A skill is a folder with a `SKILL.md` (frontmatter: name, description; markdown body of
instructions). Only the name + description (a teaser) goes into the system prompt; the
model `read`s `skill://<name>` to pull the full body into context ONLY when relevant.

Discovered from `~/.curry-leaves/skills/<name>/SKILL.md` (user) and
`.curry-leaves/skills/<name>/SKILL.md` (project, walking up). `hide: true` keeps a skill
reachable via `skill://` but out of the prompt listing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .util.frontmatter import parse_frontmatter
from .util.paths import find_up_dir, home, is_absolute, join, resolve


class Skill(BaseModel):
    name: str
    description: str
    path: str  # the SKILL.md
    source: Literal["user", "project"]
    hide: bool


class SkillRegistry:
    def __init__(self, *, discover: bool = False, cwd: str | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        if discover:
            self.discover(cwd)

    def discover(self, cwd: str | None = None) -> int:
        """Scan user + project skill dirs. Project overrides user on a name clash. Returns count."""
        loaded = 0
        for dir_, source in self._skill_dirs(cwd):
            try:
                if not os.path.isdir(dir_):
                    continue
                entries = sorted(os.listdir(dir_))
            except OSError:
                continue
            for sub in entries:
                md = join(dir_, sub, "SKILL.md")
                try:
                    if not os.path.isfile(md):
                        continue
                    text = Path(md).read_text(encoding="utf-8")
                except OSError:
                    continue
                meta, _ = parse_frontmatter(text)
                name = meta.get("name") or sub
                self._skills[name] = Skill(
                    name=name,
                    description=meta.get("description") or "",
                    path=md,
                    source=source,
                    hide=(meta.get("hide") or "").lower() in ("true", "1", "yes"),
                )
                loaded += 1
        return loaded

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def teasers(self) -> list[tuple[str, str]]:
        """(name, description) for skills that should appear in the system-prompt listing."""
        return [(s.name, s.description) for s in self.all() if not s.hide]

    def body(self, skill: Skill) -> str:
        """The SKILL.md markdown body (without frontmatter)."""
        _, body = parse_frontmatter(Path(skill.path).read_text(encoding="utf-8"))
        return body

    def read(self, ref: str) -> str | None:
        """
        Resolve a `skill://` reference. `<name>` -> the SKILL.md body; `<name>/<path>` -> a
        bundled file inside the skill's directory (path-traversal guarded). None if unresolvable.
        """
        slash = ref.find("/")
        name = ref if slash == -1 else ref[:slash]
        sub = "" if slash == -1 else ref[slash + 1 :]
        skill = self._skills.get(name)
        if not skill:
            return None
        if not sub:
            return self.body(skill)
        return self._read_asset(skill, sub)

    def _read_asset(self, skill: Skill, sub: str) -> str | None:
        if is_absolute(sub) or ".." in sub.replace("\\", "/").split("/"):
            return None
        base = resolve(os.path.dirname(skill.path))
        target = resolve(join(base, sub))
        rel = os.path.relpath(target, base)
        if rel.startswith("..") or is_absolute(rel):
            return None  # escaped the skill dir
        try:
            if not os.path.isfile(target):
                return None
            return Path(target).read_text(encoding="utf-8")
        except OSError:
            return None  # missing or binary

    def _skill_dirs(self, cwd: str | None = None) -> list[tuple[str, Literal["user", "project"]]]:
        dirs: list[tuple[str, Literal["user", "project"]]] = [(join(home(), "skills"), "user")]
        project = find_up_dir(join(".curry-leaves", "skills"), cwd)
        if project:
            dirs.append((project, "project"))
        return dirs
