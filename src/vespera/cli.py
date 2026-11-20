"""Interactive terminal front end for Vespera."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from . import __version__
from .agent import Agent
from .config import Config
from .memory import MemoryStore
from .skills import SkillRegistry
from .tools import ToolExecutor

ACCENT = "#b48cff"
TOOL_COLOR = "#4fd1c5"
PROMPT = "vespera › "

COMMANDS = {
    "/help": "Show this help.",
    "/new": "Start a fresh session (history is kept on disk).",
    "/model <name>": "Switch the active model for this session.",
    "/skills": "List skills indexed from the skills directory.",
    "/memory": "Show MEMORY.md / USER.md character usage and limits.",
    "/exit": "Quit Vespera (Ctrl-D also works).",
}

LOGO = r"""
 _   _____  ___ _ __   ___ _ __ __ _
| | / / _ \/ __| '_ \ / _ \ '__/ _` |
| |/ /  __/\__ \ |_) |  __/ | | (_| |
|___/ \___||___/ .__/ \___|_|  \__,_|
               |_|"""


# ------------------------------------------------------------------ rendering
class Renderer:
    """Turns agent events into Rich output.

    With ``live=True`` assistant text is streamed as progressively rendered
    Markdown; otherwise it is printed once the message is complete.
    """

    def __init__(self, console: Console, live: bool = True):
        self.console = console
        self.live_enabled = live
        self._buffer: list[str] = []
        self._live: Live | None = None

    def __call__(self, kind: str, **data: Any) -> None:
        handler = getattr(self, f"on_{kind}", None)
        if handler:
            handler(**data)

    def on_text_delta(self, text: str) -> None:
        self._buffer.append(text)
        if not self.live_enabled:
            return
        if self._live is None:
            self._live = Live(
                self._render_text(), console=self.console, refresh_per_second=12, transient=False
            )
            self._live.start()
        else:
            self._live.update(self._render_text())

    def on_assistant_end(self, content: str) -> None:
        if self._live is not None:
            self._live.update(self._render_text())
            self._live.stop()
            self._live = None
        elif self._buffer:
            self.console.print(self._render_text())
        self._buffer = []

    def _render_text(self) -> Group:
        return Group(Text("◆ vespera", style=f"bold {ACCENT}"), Markdown("".join(self._buffer)))

    def on_tool_call(self, name: str, arguments: Any) -> None:
        if isinstance(arguments, dict):
            if name == "run_shell":
                detail = f"$ {arguments.get('command', '')}"
            else:
                detail = ", ".join(f"{k}={_short(v)}" for k, v in arguments.items())
        else:
            detail = _short(arguments)
        self.console.print(
            Text.assemble(("  ⚙ ", TOOL_COLOR), (name, f"bold {TOOL_COLOR}"), ("  " + detail, "dim"))
        )

    def on_tool_result(self, name: str, result: str) -> None:
        lines = result.strip().splitlines() or [""]
        shown = lines[:8]
        if len(lines) > 8:
            shown.append(f"… {len(lines) - 8} more lines")
        style = "red" if result.startswith("Error") else "grey62"
        for line in shown:
            self.console.print(
                Text("    │ " + line[:200], style=style), no_wrap=True, overflow="ellipsis"
            )

    def on_reflection_start(self) -> None:
        self.console.print(Text("  ✦ reflecting on this turn…", style=f"italic {ACCENT}"))

    def on_reflection_end(self, actions: list[str]) -> None:
        if not actions:
            self.console.print(Text("  ✦ nothing new to remember", style="dim italic"))
        for action in actions:
            self.console.print(Text(f"  ✦ learned → {action}", style=f"{ACCENT}"))


def _short(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def make_confirm(console: Console):
    def confirm(command: str, high_risk: bool) -> bool:
        if high_risk:
            console.print(
                Panel(
                    Text(command, style="bold"),
                    title="⚠ high-risk command",
                    border_style="red",
                    expand=False,
                )
            )
            prompt = "[red]This looks destructive. Run it anyway?[/red]"
        else:
            console.print(Text.assemble(("  ? ", "yellow"), ("run: ", "yellow"), (command, "bold")))
            prompt = "  Allow this command?"
        try:
            return Confirm.ask(prompt, console=console, default=False)
        except (EOFError, KeyboardInterrupt):
            return False

    return confirm


def render_banner(console: Console, agent: Agent, config: Config, resumed: bool) -> None:
    usage = agent.memory.usage()
    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column()
    info.add_row("model", Text(agent.model, style=f"bold {ACCENT}"))
    info.add_row("endpoint", config.base_url)
    info.add_row("home", str(config.home))
    info.add_row(
        "memory",
        f"{usage['memory']['snapshot']}/{usage['memory']['limit']} · "
        f"user {usage['user']['snapshot']}/{usage['user']['limit']} chars",
    )
    info.add_row("skills", f"{len(agent.skills)} indexed")
    session = agent.session_id if resumed else "new (created on first message)"
    info.add_row("session", str(session))
    info.add_row("shell", "auto-approve (risky still asks)" if config.auto_approve else "ask first")
    console.print(
        Panel(
            Group(Text(LOGO.strip("\n"), style=f"bold {ACCENT}"), Text(""), info),
            title=f"[bold]vespera[/bold] v{__version__}",
            subtitle="/help for commands",
            border_style=ACCENT,
            expand=False,
        )
    )


def render_help(console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style=f"bold {ACCENT}")
    table.add_column()
    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(Panel(table, title="commands", border_style=ACCENT, expand=False))


def render_skills(console: Console, skills: SkillRegistry) -> None:
    skills.refresh()
    if not len(skills):
        console.print(f"[dim]No skills yet in {skills.skills_dir}[/dim]")
        return
    table = Table(title=f"skills ({len(skills)})", border_style=ACCENT, title_style=ACCENT)
    table.add_column("name", style="bold")
    table.add_column("description")
    for s in skills.list():
        table.add_row(s.name, s.description)
    console.print(table)


def render_memory(console: Console, memory: MemoryStore) -> None:
    table = Table(title="memory", border_style=ACCENT, title_style=ACCENT)
    table.add_column("file")
    table.add_column("on disk", justify="right")
    table.add_column("frozen snapshot", justify="right")
    table.add_column("limit", justify="right")
    table.add_column("used", justify="right")
    for target, fname in (("memory", "MEMORY.md"), ("user", "USER.md")):
        u = memory.usage()[target]
        pct = 100 * u["live"] / u["limit"]
        color = "green" if pct < 75 else "yellow" if pct < 100 else "red"
        table.add_row(
            fname, str(u["live"]), str(u["snapshot"]), str(u["limit"]), f"[{color}]{pct:.0f}%[/]"
        )
    console.print(table)


# ------------------------------------------------------------------- commands
def handle_command(line: str, agent: Agent, console: Console) -> bool:
    """Run a slash command. Returns False when the REPL should exit."""
    cmd, _, arg = line.strip().partition(" ")
    cmd = cmd.lower()
    arg = arg.strip()
    if cmd in ("/exit", "/quit"):
        return False
    if cmd == "/help":
        render_help(console)
    elif cmd == "/new":
        agent.new_session()
        console.print(f"[{ACCENT}]✦ started a new session[/]")
    elif cmd == "/model":
        if not arg:
            console.print(f"current model: [bold]{agent.model}[/bold]  (usage: /model <name>)")
        else:
            agent.model = arg
            console.print(f"[{ACCENT}]✦ model switched to[/] [bold]{arg}[/bold]")
    elif cmd == "/skills":
        render_skills(console, agent.skills)
    elif cmd == "/memory":
        render_memory(console, agent.memory)
    else:
        console.print(f"[red]unknown command {cmd}[/red] — try /help")
    return True


# ------------------------------------------------------------------------ app
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vespera",
        description="Vespera: a terminal agent with memory, session recall, and self-written skills.",
    )
    parser.add_argument("prompt", nargs="*", help="Run a single prompt non-interactively and exit.")
    parser.add_argument("-c", "--continue", dest="resume", action="store_true",
                        help="Resume the most recently updated session.")
    parser.add_argument("-m", "--model", help="Override the model for this run.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Auto-approve shell commands (high-risk patterns still ask).")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    parser.add_argument("--version", action="version", version=f"vespera {__version__}")
    return parser.parse_args(argv)


def build_client(config: Config):
    from openai import OpenAI

    # Local servers (llama.cpp, Ollama) accept any key; the SDK just needs one.
    return OpenAI(api_key=config.api_key or "not-needed", base_url=config.base_url)


def _is_local(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))


def read_line(session: Any, console: Console) -> str:
    if session is None:
        return console.input(f"[bold {ACCENT}]{PROMPT}[/]")
    return session.prompt([("class:prompt", PROMPT)])


def make_prompt_session(config: Config):
    if not sys.stdin.isatty():
        return None
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    completer = WordCompleter([c.split()[0] for c in COMMANDS], sentence=True)
    return PromptSession(
        history=FileHistory(str(config.home / "prompt_history")),
        completer=completer,
        style=Style.from_dict({"prompt": f"bold {ACCENT}"}),
    )


def run_once(agent: Agent, console: Console, text: str) -> None:
    try:
        agent.run_turn(text)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
    except Exception as exc:  # network/auth/model errors should not kill the REPL
        console.print(f"[red]model error:[/red] {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.from_env()
    if args.model:
        config.model = args.model
    config.auto_approve = args.yes
    config.ensure_dirs()

    console = Console(highlight=False)
    if not config.api_key and not _is_local(config.base_url):
        console.print(
            "[yellow]VESPERA_API_KEY is not set.[/yellow] Set it (or point VESPERA_BASE_URL at a "
            "local server) before sending prompts."
        )

    memory = MemoryStore(config)
    skills = SkillRegistry(config.skills_dir)
    tools = ToolExecutor(memory, skills, confirm=make_confirm(console), auto_approve=config.auto_approve)
    renderer = Renderer(console, live=not args.no_stream and console.is_terminal)
    agent = Agent(
        build_client(config), config.model, memory, skills, tools,
        stream=not args.no_stream, on_event=renderer,
    )

    resumed = False
    if args.resume:
        latest = memory.latest_session_id()
        if latest:
            agent.resume(latest)
            resumed = True
        else:
            console.print("[dim]No previous session to continue; starting fresh.[/dim]")

    try:
        if args.prompt:
            run_once(agent, console, " ".join(args.prompt))
            return 0

        render_banner(console, agent, config, resumed)
        if resumed:
            console.print(f"[dim]resumed session with {len(agent.messages)} messages[/dim]")
        session = make_prompt_session(config)
        while True:
            try:
                line = read_line(session, console).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            if not line:
                continue
            if line.startswith("/"):
                if not handle_command(line, agent, console):
                    break
                continue
            run_once(agent, console, line)
            console.print()
        console.print(f"[{ACCENT}]goodbye ✦[/]")
        return 0
    finally:
        memory.close()


if __name__ == "__main__":
    sys.exit(main())
