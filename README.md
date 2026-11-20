<div align="center">

# vespera

**A command-line-first autonomous agent that remembers, recalls, and writes its own playbooks.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-7c5cff.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-b48cff.svg)](LICENSE)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-4fd1c5.svg)](tests/)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macOS-555.svg)

</div>

Vespera is a persistent terminal agent, not a stateless chatbot. It runs real
local tools, searches its own past sessions with SQLite FTS5, keeps a small
cache-friendly memory, and gets better over time by saving reusable `SKILL.md`
playbooks after non-trivial work.

It talks to any **OpenAI-compatible Chat Completions endpoint**: OpenRouter,
OpenAI, or a local `llama.cpp` / Ollama server.

## Features

- **Tool-calling loop.** Shell, file read/write, session search, and skill and
  memory tools run in a loop until the model gives a final answer.
- **Bounded, frozen memory.** `MEMORY.md` (2,200 chars) and `USER.md`
  (1,375 chars) are read once per session. The system prompt prefix stays
  byte-identical across turns, so provider prompt caching keeps working.
- **Session recall.** Every message is stored in SQLite and indexed with FTS5.
  The agent finds earlier context with `search_sessions` instead of replaying old
  transcripts.
- **Progressive-disclosure skills.** Only a one-line index of skills goes into
  the prompt. Full instructions load on demand with `read_skill`.
- **Self-improvement.** After busy turns (3 or more tool calls), a reflection
  pass may save a new skill or a durable preference.
- **Guardrails.** Shell commands need your approval. With `--yes`, destructive
  patterns (`rm -rf`, `mkfs`, `dd if=`, `shutdown`, `reboot`, fork bombs, and
  Windows equivalents) still ask. Shell and file output is truncated to protect
  the context window.
- **Cross-platform.** All paths use `pathlib`, so the same code runs on Linux,
  macOS, and Windows.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/rohithprem18/vespera-cli.git
cd vespera-cli
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

This puts a `vespera` command on your `PATH`. On first run it creates
`~/.vespera/` with starter memory files.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VESPERA_API_KEY` | — | API key for the endpoint. Local servers don't need one. |
| `VESPERA_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible base URL |
| `VESPERA_MODEL` | `openai/gpt-4o-mini` | Model name sent to the endpoint |
| `VESPERA_HOME` | `~/.vespera` | Where memory, skills, and the session database live |

**Linux / macOS**

```bash
export VESPERA_API_KEY="sk-or-..."
export VESPERA_MODEL="openai/gpt-4o"
vespera
```

**Windows (PowerShell)**

```powershell
$env:VESPERA_API_KEY = "sk-or-..."
$env:VESPERA_MODEL = "openai/gpt-4o-mini"
vespera
```

**Local model (Ollama / llama.cpp)**

```bash
export VESPERA_BASE_URL="http://localhost:11434/v1"
export VESPERA_MODEL="qwen2.5-coder:7b"
vespera
```

## Usage

```text
vespera                     start an interactive session
vespera -c                  continue the most recently updated session
vespera -m <model>          override the model for this run
vespera -y                  auto-approve shell commands (risky ones still ask)
vespera "summarise README"  run one prompt and exit
vespera --no-stream         print whole responses instead of streaming
```

| Flag | Description |
| --- | --- |
| `-c`, `--continue` | Resume the most recently updated session |
| `-m`, `--model` | Override the model for the run |
| `-y`, `--yes` | Auto-approve standard shell commands (high-risk patterns still prompt) |
| `--no-stream` | Disable token streaming |
| `--version` | Print the version |

### Slash commands

| Command | Description |
| --- | --- |
| `/help` | List commands and usage |
| `/new` | Start a new session and reset the active history |
| `/model <name>` | Switch the active model mid-session |
| `/skills` | Show skills indexed from `~/.vespera/skills` |
| `/memory` | Show character usage and limits for `MEMORY.md` and `USER.md` |
| `/exit` | Exit cleanly (Ctrl-D works too) |

## How it works

```text
            ┌────────────────────── session startup ──────────────────────┐
            │ load config → create ~/.vespera → open sessions.db (FTS5)   │
            │ freeze MEMORY.md + USER.md → build skills index → new / -c  │
            └──────────────────────────────┬──────────────────────────────┘
                                           ▼
 user ──► ┌──────── phase 1: execution loop ────────┐
          │ system prompt + history + tool schemas  │
          │            │                            │
          │            ▼                            │
          │      model response ── tool calls? ─yes─┼─► run tools, append results ─┐
          │            │ no                         │                              │
          │            ▼                            │◄─────────────────────────────┘
          │      final answer ──► user              │
          └────────────┬────────────────────────────┘
                       ▼ (≥ 3 tool calls this turn)
          ┌──────── phase 2: reflection ────────────┐
          │ tools: update_memory, save_skill only   │
          │ not added to history; failures ignored  │
          └─────────────────────────────────────────┘
```

| Module | Responsibility |
| --- | --- |
| [`config.py`](src/vespera/config.py) | Paths, character limits, env config, directory bootstrap |
| [`memory.py`](src/vespera/memory.py) | Frozen `MEMORY.md`/`USER.md` snapshot, SQLite sessions + FTS5 search |
| [`skills.py`](src/vespera/skills.py) | `SKILL.md` index, on-demand reader, sanitized writer |
| [`tools.py`](src/vespera/tools.py) | Tool JSON schemas, handlers, confirmation and risk guardrails |
| [`agent.py`](src/vespera/agent.py) | Tool-calling loop (streaming and non-streaming) and reflection pass |
| [`cli.py`](src/vespera/cli.py) | `prompt_toolkit` + `rich` REPL, flags, slash commands |

### Tools

| Tool | What it does |
| --- | --- |
| `run_shell(command)` | Runs a command with a 60 s timeout; stdout/stderr head+tail truncated |
| `read_file(path)` | Reads UTF-8 text, capped at 20,000 chars, clear missing-file errors |
| `write_file(path, content)` | Writes UTF-8, creating parent directories |
| `search_sessions(query, limit?)` | FTS5 `MATCH` over past sessions, returns `snippet()` excerpts |
| `read_skill(name)` | Loads a skill's full `SKILL.md` |
| `save_skill(name, description, instructions)` | Creates or updates a skill |
| `update_memory(target, content, mode?)` | Appends to or replaces `MEMORY.md` / `USER.md` within the caps |

Tool handlers never raise. Errors, denials, and malformed arguments come back to
the model as `Error: ...` text, so a bad step never crashes the session.

### State directory

```text
~/.vespera/
├── MEMORY.md          # environment & project facts   (≤ 2,200 chars)
├── USER.md            # your preferences              (≤ 1,375 chars)
├── sessions.db        # sessions, messages, messages_fts
├── prompt_history     # REPL input history
└── skills/
    └── <skill-name>/
        └── SKILL.md
```

## Skill format

Each skill is a folder with a `SKILL.md`: YAML frontmatter followed by Markdown
instructions.

```markdown
---
name: release-notes
description: Draft release notes from git history
---

1. Find the last tag: `git describe --tags --abbrev=0`
2. List changes: `git log --oneline <tag>..HEAD`
3. Group entries under Features / Fixes / Docs and drop merge commits.
```

Only `- release-notes: Draft release notes from git history` goes into the system
prompt. The body is read when the agent calls `read_skill("release-notes")`.
You can write skills by hand, or let Vespera save them during reflection. Names
are slugified (`Release Notes!` becomes `release-notes`), which also blocks path
traversal.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite covers memory caps and snapshot freezing, FTS5 search, skill parsing
and writing, guardrail patterns, truncation, the multi-step loop, streaming
tool-call deltas, reflection, and session resume. A fake client stands in for
the model API, so no network access is needed.

## License

[MIT](LICENSE) © Rohith Prem Sivaraman
