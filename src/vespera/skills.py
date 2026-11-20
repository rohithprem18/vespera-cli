"""Progressive-disclosure skills stored as ``<skills_dir>/<name>/SKILL.md``.

Only a one-line index (name + description) goes into the system prompt. The
full instructions are loaded on demand through ``read_skill`` so the context
stays small no matter how many skills accumulate.

Each ``SKILL.md`` looks like::

    ---
    name: deploy-docs
    description: Build and publish the docs site
    ---
    1. Run ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SKILL_FILENAME = "SKILL.md"
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 200

_FRONTMATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path


def sanitize_name(name: str) -> str:
    """Make a safe, lowercase, hyphenated folder name (no path traversal)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    slug = slug[:MAX_NAME_LENGTH].strip("-")
    if not slug:
        raise ValueError(f"invalid skill name: {name!r}")
    return slug


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split ``SKILL.md`` text into (frontmatter dict, markdown body)."""
    match = _FRONTMATTER.match(text.lstrip("﻿"))
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, match.group(2)


def render_skill(name: str, description: str, instructions: str) -> str:
    front = yaml.safe_dump(
        {"name": name, "description": description},
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    return f"---\n{front}---\n\n{instructions.strip()}\n"


class SkillRegistry:
    """Index, read, and write skills under a directory."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        self.refresh()

    def refresh(self) -> None:
        found: dict[str, Skill] = {}
        for skill_file in sorted(self.skills_dir.glob(f"*/{SKILL_FILENAME}")):
            try:
                text = skill_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            meta, _ = parse_frontmatter(text)
            folder = skill_file.parent.name
            name = str(meta.get("name") or folder).strip()
            description = " ".join(str(meta.get("description") or "").split())
            found[folder] = Skill(name=name, description=description, path=skill_file)
        self._skills = found

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def index_text(self) -> str:
        """The lightweight index injected into the system prompt."""
        if not self._skills:
            return "(no skills saved yet)"
        return "\n".join(
            f"- {s.name}: {s.description or '(no description)'}" for s in self._skills.values()
        )

    def find(self, name: str) -> Skill | None:
        try:
            key = sanitize_name(name)
        except ValueError:
            return None
        if key in self._skills:
            return self._skills[key]
        for skill in self._skills.values():
            if skill.name == name or sanitize_name(skill.name) == key:
                return skill
        return None

    def read(self, name: str) -> str:
        """Full SKILL.md text. Raises ``KeyError`` when the skill is unknown."""
        skill = self.find(name)
        if skill is None:
            self.refresh()  # picked up a skill written by another process?
            skill = self.find(name)
        if skill is None:
            raise KeyError(name)
        return skill.path.read_text(encoding="utf-8", errors="replace")

    def save(self, name: str, description: str, instructions: str) -> Skill:
        """Create or overwrite a skill and make it visible to later lookups."""
        slug = sanitize_name(name)
        description = " ".join(description.split())[:MAX_DESCRIPTION_LENGTH]
        if not description:
            raise ValueError("description must not be empty")
        if not instructions.strip():
            raise ValueError("instructions must not be empty")
        folder = self.skills_dir / slug
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / SKILL_FILENAME
        path.write_text(render_skill(slug, description, instructions), encoding="utf-8")
        skill = Skill(name=slug, description=description, path=path)
        self._skills[slug] = skill
        return skill
