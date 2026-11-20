"""The core agent: a tool-calling loop plus a post-turn reflection pass.

Phase 1 (execution): send system prompt + history + tool schemas to the model;
run every tool call it emits, append the results, and repeat until the model
answers with plain text.

Phase 2 (reflection): after a non-trivial turn (several tool calls), ask the
model -- with only ``update_memory`` / ``save_skill`` available -- whether
anything durable was learned. Reflection traffic is not added to the session
history, so it never costs context in later turns.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import MAX_TOOL_ITERATIONS, REFLECTION_TOOL_THRESHOLD
from .memory import MemoryStore
from .skills import SkillRegistry
from .tools import ToolExecutor

# on_event(kind, **data). Kinds:
#   text_delta(text)            streamed assistant text
#   assistant_end(content)      an assistant message finished
#   tool_call(name, arguments)  a tool is about to run
#   tool_result(name, result)   a tool finished
#   reflection_start()          reflection pass begins
#   reflection_end(actions)     reflection pass finished
EventFn = Callable[..., None]

REFLECTION_TOOLS = ["update_memory", "save_skill"]
NOTHING_TO_SAVE = "NOTHING_TO_SAVE"

SYSTEM_TEMPLATE = """You are Vespera, an autonomous command-line agent running on the user's machine.
You act by calling tools: run shell commands, read and write files, search past sessions,
and load or author skills. Work step by step, verify results, and keep answers concise.

Guidelines:
- Prefer doing over describing: if a tool can answer the question, call it.
- Before a task that matches a skill in the index, call read_skill to load its playbook.
- If the user refers to earlier work you do not see, call search_sessions.
- Shell commands may be denied by the user; if so, do not retry the same command.
- Keep tool output usage lean: inspect only what you need.

Environment:
- OS: {os_name}
- Working directory: {cwd}

<memory>
{memory}
</memory>

<user_profile>
{user}
</user_profile>

<skills_index>
{skills}
</skills_index>"""

REFLECTION_PROMPT = f"""[reflection] The turn above is complete. Decide whether anything should be kept for future sessions:
- A durable user preference or environment fact -> call update_memory (target 'user' or 'memory', mode 'append').
- A repeatable multi-step workflow that worked -> call save_skill with concise, general step-by-step instructions.
Do not save one-off details, secrets, or facts already present in memory. If nothing is worth keeping,
reply with exactly {NOTHING_TO_SAVE}."""


@dataclass
class TurnResult:
    text: str
    tool_calls: int = 0
    reflection_actions: list[str] = field(default_factory=list)
    stopped_early: bool = False


def build_system_prompt(memory: MemoryStore, skills: SkillRegistry, cwd: str | None = None) -> str:
    return SYSTEM_TEMPLATE.format(
        os_name=f"{platform.system()} {platform.release()}".strip(),
        cwd=cwd or os.getcwd(),
        memory=memory.memory_snapshot.strip() or "(empty)",
        user=memory.user_snapshot.strip() or "(empty)",
        skills=skills.index_text(),
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an SDK object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class Agent:
    def __init__(
        self,
        client: Any,
        model: str,
        memory: MemoryStore,
        skills: SkillRegistry,
        tools: ToolExecutor,
        *,
        stream: bool = True,
        on_event: EventFn | None = None,
        reflection_threshold: int = REFLECTION_TOOL_THRESHOLD,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ):
        self.client = client
        self.model = model
        self.memory = memory
        self.skills = skills
        self.tools = tools
        self.stream = stream
        self.on_event = on_event or (lambda kind, **data: None)
        self.reflection_threshold = reflection_threshold
        self.max_iterations = max_iterations
        # Built once per process: keeps the prompt prefix stable for caching.
        self.system_prompt = build_system_prompt(memory, skills, str(tools.cwd))
        self.session_id: str | None = None
        self.messages: list[dict[str, Any]] = []

    # --------------------------------------------------------------- sessions
    def new_session(self) -> None:
        """Forget the active history; a new session row is created lazily."""
        self.session_id = None
        self.messages = []
        self.tools.current_session_id = None

    def resume(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages = self.memory.load_messages(session_id)
        self.tools.current_session_id = session_id

    def _record(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        if self.session_id:
            self.memory.append_message(self.session_id, message)

    # ------------------------------------------------------------ model call
    def _complete(
        self, messages: list[dict[str, Any]], tools: list[dict], stream: bool
    ) -> tuple[str, list[dict[str, Any]]]:
        """One chat completion. Returns (text, tool_calls in OpenAI dict form)."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if not stream:
            response = self.client.chat.completions.create(**kwargs)
            msg = _get(response.choices[0], "message")
            text = _get(msg, "content") or ""
            calls = []
            for tc in _get(msg, "tool_calls") or []:
                fn = _get(tc, "function")
                calls.append(
                    {
                        "id": _get(tc, "id") or f"call_{len(calls)}",
                        "type": "function",
                        "function": {
                            "name": _get(fn, "name") or "",
                            "arguments": _get(fn, "arguments") or "{}",
                        },
                    }
                )
            if text:
                self.on_event("text_delta", text=text)
            return text, calls

        parts: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        for chunk in self.client.chat.completions.create(stream=True, **kwargs):
            choices = _get(chunk, "choices") or []
            if not choices:
                continue
            delta = _get(choices[0], "delta")
            if delta is None:
                continue
            piece = _get(delta, "content")
            if piece:
                parts.append(piece)
                self.on_event("text_delta", text=piece)
            for pos, tc in enumerate(_get(delta, "tool_calls") or []):
                index = _get(tc, "index")
                index = pos if index is None else index
                slot = slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if _get(tc, "id"):
                    slot["id"] = _get(tc, "id")
                fn = _get(tc, "function")
                if fn is not None:
                    slot["name"] += _get(fn, "name") or ""
                    slot["arguments"] += _get(fn, "arguments") or ""
        calls = [
            {
                "id": s["id"] or f"call_{i}",
                "type": "function",
                "function": {"name": s["name"], "arguments": s["arguments"] or "{}"},
            }
            for i, s in sorted(slots.items())
        ]
        return "".join(parts), calls

    # ------------------------------------------------------------ main loop
    def run_turn(self, user_input: str) -> TurnResult:
        if self.session_id is None:
            title = " ".join(user_input.split())[:60]
            self.session_id = self.memory.new_session(title)
            self.tools.current_session_id = self.session_id
        turn_start = len(self.messages)
        self._record({"role": "user", "content": user_input})

        tool_calls_made = 0
        text = ""
        stopped_early = False
        for _ in range(self.max_iterations):
            convo = [{"role": "system", "content": self.system_prompt}, *self.messages]
            text, calls = self._complete(convo, self.tools.schemas, self.stream)
            assistant: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                assistant["tool_calls"] = calls
            self._record(assistant)
            self.on_event("assistant_end", content=text)
            if not calls:
                break
            for call in calls:
                tool_calls_made += 1
                name = call["function"]["name"]
                raw_args = call["function"]["arguments"]
                self.on_event("tool_call", name=name, arguments=_preview_args(raw_args))
                result = self.tools.execute(name, raw_args)
                self.on_event("tool_result", name=name, result=result)
                self._record({"role": "tool", "tool_call_id": call["id"], "content": result})
        else:
            stopped_early = True
            text = (
                f"Stopped after {self.max_iterations} tool iterations without a final answer. "
                "Ask me to continue if needed."
            )
            self._record({"role": "assistant", "content": text})
            self.on_event("text_delta", text=text)
            self.on_event("assistant_end", content=text)

        result = TurnResult(text=text, tool_calls=tool_calls_made, stopped_early=stopped_early)
        if tool_calls_made >= self.reflection_threshold:
            result.reflection_actions = self.reflect(self.messages[turn_start:])
        return result

    # ------------------------------------------------------------ reflection
    def reflect(self, turn_messages: list[dict[str, Any]], max_rounds: int = 3) -> list[str]:
        """Let the model persist durable lessons. Errors never escape."""
        self.on_event("reflection_start")
        actions: list[str] = []
        convo: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *turn_messages,
            {"role": "user", "content": REFLECTION_PROMPT},
        ]
        schemas = self.tools.schemas_for(REFLECTION_TOOLS)
        try:
            for _ in range(max_rounds):
                text, calls = self._complete_quietly(convo, schemas)
                if not calls:
                    break
                convo.append({"role": "assistant", "content": text or None, "tool_calls": calls})
                for call in calls:
                    name = call["function"]["name"]
                    if name not in REFLECTION_TOOLS:
                        out = f"Error: {name} is not available during reflection."
                    else:
                        out = self.tools.execute(name, call["function"]["arguments"])
                        if not out.startswith("Error"):
                            actions.append(f"{name}: {out}")
                    convo.append({"role": "tool", "tool_call_id": call["id"], "content": out})
        except Exception as exc:  # reflection is best-effort
            actions.append(f"reflection skipped: {type(exc).__name__}: {exc}")
        self.on_event("reflection_end", actions=actions)
        return actions

    def _complete_quietly(self, messages: list[dict[str, Any]], tools: list[dict]):
        """Like _complete but without streaming text to the UI."""
        saved = self.on_event
        self.on_event = lambda kind, **data: None
        try:
            return self._complete(messages, tools, stream=False)
        finally:
            self.on_event = saved


def _preview_args(raw: str) -> dict[str, Any] | str:
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return raw
    return parsed if isinstance(parsed, dict) else raw
