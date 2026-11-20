"""Tool schemas and their local execution handlers.

Every handler returns a plain string that is fed back to the model. Handlers
never raise: failures, denials, and bad arguments come back as ``Error: ...``
text so the agent loop can keep going and let the model recover.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from .config import FILE_READ_LIMIT, SHELL_OUTPUT_LIMIT, SHELL_TIMEOUT_SECONDS
from .memory import MemoryStore
from .skills import SkillRegistry

# confirm(command, high_risk) -> approved?
ConfirmFn = Callable[[str, bool], bool]


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _fn(
        "run_shell",
        "Run a shell command on the user's machine and return exit code, stdout and stderr. "
        "Large output is truncated. The user may be asked to approve the command.",
        {"command": {"type": "string", "description": "The command line to execute."}},
        ["command"],
    ),
    _fn(
        "read_file",
        "Read a UTF-8 text file. Long files are truncated.",
        {"path": {"type": "string", "description": "Absolute or working-directory-relative path."}},
        ["path"],
    ),
    _fn(
        "write_file",
        "Create or overwrite a UTF-8 text file, creating parent directories as needed.",
        {
            "path": {"type": "string", "description": "Destination path."},
            "content": {"type": "string", "description": "Full file content to write."},
        },
        ["path", "content"],
    ),
    _fn(
        "search_sessions",
        "Full-text search over previous conversation sessions. Returns short snippets.",
        {
            "query": {"type": "string", "description": "Keywords to search for."},
            "limit": {"type": "integer", "description": "Max results (default 5).", "minimum": 1},
        },
        ["query"],
    ),
    _fn(
        "read_skill",
        "Load the full instructions of a saved skill listed in the skills index.",
        {"name": {"type": "string", "description": "Skill name from the index."}},
        ["name"],
    ),
    _fn(
        "save_skill",
        "Create or update a reusable skill: a step-by-step playbook for a repeatable workflow.",
        {
            "name": {"type": "string", "description": "Short kebab-case name."},
            "description": {"type": "string", "description": "One line: when to use this skill."},
            "instructions": {"type": "string", "description": "Markdown steps to follow."},
        },
        ["name", "description", "instructions"],
    ),
    _fn(
        "update_memory",
        "Persist a durable fact. target='memory' for environment/project facts, "
        "target='user' for the user's preferences. Takes effect from the next session.",
        {
            "target": {"type": "string", "enum": ["memory", "user"]},
            "content": {"type": "string", "description": "Text to write."},
            "mode": {
                "type": "string",
                "enum": ["append", "replace"],
                "description": "append (default) adds a line; replace rewrites the whole file.",
            },
        },
        ["target", "content"],
    ),
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]

# Commands that always need an explicit "yes", even with --yes.
HIGH_RISK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\brm\s+(-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*)\b",  # rm -rf / rm -fr
        r"\brm\s+.*--recursive\b.*--force\b|\brm\s+.*--force\b.*--recursive\b",
        r"\bmkfs(\.\w+)?\b",
        r"\bdd\s+.*\bif=",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bhalt\b|\bpoweroff\b",
        r":\s*\(\s*\)\s*\{.*:\s*\|\s*:.*&.*\}",  # :(){ :|:& };:
        r"\bformat\s+[a-z]:",  # Windows: format C:
        r"\b(del|erase)\b.*\s/[sq]\b",  # Windows: del /s /q
        r"\b(rd|rmdir)\b.*\s/s\b",  # Windows: rd /s
        r"\bremove-item\b.*-recurse\b.*-force\b|\bremove-item\b.*-force\b.*-recurse\b",
        r"\bstop-computer\b|\brestart-computer\b",
    )
]


def is_high_risk(command: str) -> bool:
    return any(p.search(command) for p in HIGH_RISK_PATTERNS)


def truncate(text: str, limit: int) -> str:
    """Keep the head and tail of ``text`` so it fits in ``limit`` chars."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    dropped = len(text) - head - tail
    return f"{text[:head]}\n... [truncated {dropped} chars] ...\n{text[-tail:]}"


class ToolExecutor:
    """Dispatches tool calls to local handlers with guardrails."""

    def __init__(
        self,
        memory: MemoryStore,
        skills: SkillRegistry,
        confirm: ConfirmFn | None = None,
        auto_approve: bool = False,
        cwd: Path | None = None,
        shell_timeout: int = SHELL_TIMEOUT_SECONDS,
    ):
        self.memory = memory
        self.skills = skills
        self.confirm = confirm
        self.auto_approve = auto_approve
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.shell_timeout = shell_timeout
        self.current_session_id: str | None = None
        self._handlers: dict[str, Callable[..., str]] = {
            "run_shell": self.run_shell,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "search_sessions": self.search_sessions,
            "read_skill": self.read_skill,
            "save_skill": self.save_skill,
            "update_memory": self.update_memory,
        }

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def schemas_for(self, names: list[str]) -> list[dict]:
        return [t for t in TOOL_SCHEMAS if t["function"]["name"] in names]

    def execute(self, name: str, arguments: str | dict[str, Any] | None) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self._handlers)}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return f"Error: tool arguments are not valid JSON ({exc.msg})."
        if not isinstance(arguments, dict):
            return "Error: tool arguments must be a JSON object."
        try:
            return handler(**arguments)
        except TypeError as exc:
            return f"Error: bad arguments for {name}: {exc}"
        except Exception as exc:  # a tool must never take the loop down
            return f"Error: {name} failed: {type(exc).__name__}: {exc}"

    # ----------------------------------------------------------------- shell
    def _approved(self, command: str) -> bool:
        high_risk = is_high_risk(command)
        if self.auto_approve and not high_risk:
            return True
        if self.confirm is None:
            return False
        return bool(self.confirm(command, high_risk))

    def run_shell(self, command: str) -> str:
        if not command.strip():
            return "Error: empty command."
        if not self._approved(command):
            return "Error: command denied by the user. Do not retry it; ask or try another way."
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.shell_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {self.shell_timeout}s."
        half = SHELL_OUTPUT_LIMIT // 2
        stdout = truncate(proc.stdout or "", half)
        stderr = truncate(proc.stderr or "", half)
        return f"exit_code: {proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".rstrip()

    # ----------------------------------------------------------------- files
    def _resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else self.cwd / p

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            return f"Error: file not found: {target}"
        if target.is_dir():
            return f"Error: {target} is a directory, not a file."
        with target.open("r", encoding="utf-8", errors="replace") as fh:
            data = fh.read(FILE_READ_LIMIT + 1)
        if len(data) > FILE_READ_LIMIT:
            size = target.stat().st_size
            return (
                data[:FILE_READ_LIMIT]
                + f"\n... [truncated: showing first {FILE_READ_LIMIT} chars of ~{size} bytes]"
            )
        return data

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {target}"

    # ------------------------------------------------------- memory & skills
    def search_sessions(self, query: str, limit: int = 5) -> str:
        hits = self.memory.search_sessions(query, limit, exclude_session=self.current_session_id)
        if not hits:
            return f"No past sessions matched '{query}'."
        lines = [
            f"[{h.session_id}] {h.created_at} {h.title!r} ({h.role}): {h.snippet}" for h in hits
        ]
        return "\n".join(lines)

    def read_skill(self, name: str) -> str:
        try:
            return self.skills.read(name)
        except KeyError:
            names = ", ".join(s.name for s in self.skills.list()) or "none"
            return f"Error: no skill named '{name}'. Known skills: {names}"

    def save_skill(self, name: str, description: str, instructions: str) -> str:
        try:
            skill = self.skills.save(name, description, instructions)
        except ValueError as exc:
            return f"Error: {exc}"
        return f"Saved skill '{skill.name}' to {skill.path}"

    def update_memory(self, target: str, content: str, mode: str = "append") -> str:
        try:
            text = self.memory.update_file(target, content, mode=mode)
        except ValueError as exc:
            return f"Error: {exc}"
        limit = self.memory.limits[target]
        return f"Updated {target} ({len(text)}/{limit} chars). Visible from the next session."
