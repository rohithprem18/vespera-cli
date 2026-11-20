"""Paths, limits, and environment configuration for Vespera.

Everything that touches the filesystem goes through ``pathlib.Path`` so the
same code runs on Linux, macOS, and Windows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Hard caps that keep the system prompt small and cache-friendly.
MEMORY_CHAR_LIMIT = 2200
USER_CHAR_LIMIT = 1375

# Caps that keep tool output from flooding the context window.
SHELL_OUTPUT_LIMIT = 8000
FILE_READ_LIMIT = 20000
SHELL_TIMEOUT_SECONDS = 60

# Safety valve for runaway tool loops.
MAX_TOOL_ITERATIONS = 25

# A turn with at least this many tool calls triggers the reflection pass.
REFLECTION_TOOL_THRESHOLD = 3

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

MEMORY_TEMPLATE = """# Memory
Durable facts about the environment, projects, and lessons learned.
"""

USER_TEMPLATE = """# User
Preferences and working style of the person using Vespera.
"""


def resolve_home() -> Path:
    """Return the Vespera state directory, honouring ``$VESPERA_HOME``."""
    override = os.environ.get("VESPERA_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".vespera"


@dataclass
class Config:
    """Resolved runtime configuration."""

    home: Path = field(default_factory=resolve_home)
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    auto_approve: bool = False

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def memory_file(self) -> Path:
        return self.home / "MEMORY.md"

    @property
    def user_file(self) -> Path:
        return self.home / "USER.md"

    @property
    def db_path(self) -> Path:
        return self.home / "sessions.db"

    @classmethod
    def from_env(cls, home: Path | None = None) -> "Config":
        return cls(
            home=home or resolve_home(),
            api_key=os.environ.get("VESPERA_API_KEY"),
            base_url=os.environ.get("VESPERA_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("VESPERA_MODEL", DEFAULT_MODEL),
        )

    def ensure_dirs(self) -> None:
        """Create the state directory and starter files if they are missing."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text(MEMORY_TEMPLATE, encoding="utf-8")
        if not self.user_file.exists():
            self.user_file.write_text(USER_TEMPLATE, encoding="utf-8")


# Module-level defaults resolved at import time, mirroring the PRD names.
VESPERA_HOME = resolve_home()
SKILLS_DIR = VESPERA_HOME / "skills"
MEMORY_FILE = VESPERA_HOME / "MEMORY.md"
USER_FILE = VESPERA_HOME / "USER.md"
DB_PATH = VESPERA_HOME / "sessions.db"
