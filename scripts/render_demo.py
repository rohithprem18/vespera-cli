"""Render the README screenshot and the 3D demo GIF from a real Vespera run.

The session is driven by a scripted model (no API key needed), but everything
else is the real thing: the CLI renderer, the tool executor (which really runs
``pytest`` on this repository), the SQLite FTS5 store, and the reflection pass.

    python scripts/render_demo.py

Outputs ``docs/assets/screenshot.png`` and ``docs/assets/demo.gif``.
Requires Pillow and NumPy (``pip install pillow numpy``).
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rich.console import Console
from rich.segment import Segment
from rich.terminal_theme import MONOKAI as THEME
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
sys.path.insert(0, str(ROOT / "src"))

from vespera.agent import Agent  # noqa: E402
from vespera.cli import ACCENT, PROMPT, Renderer, handle_command, render_banner  # noqa: E402
from vespera.config import Config  # noqa: E402
from vespera.memory import MemoryStore  # noqa: E402
from vespera.skills import SkillRegistry  # noqa: E402
from vespera.tools import ToolExecutor  # noqa: E402

WIDTH = 92  # terminal columns

# ------------------------------------------------------------------ palette
BG = (17, 14, 28)
FG = (222, 220, 235)
TERM_BG = (22, 19, 36)
TITLE_BG = (34, 29, 52)


# ------------------------------------------------------------ scripted model
def _reply(text=None, calls=()):
    tool_calls = [
        NS(id=f"call_{i}", type="function",
           function=NS(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls)
    ] or None
    return NS(choices=[NS(message=NS(content=text, tool_calls=tool_calls))])


class ScriptedModel:
    """Plays a believable agent. Final answers are derived from real tool output."""

    def __init__(self):
        self.chat = NS(completions=NS(create=self.create))

    def create(self, messages, tools=None, **_):
        names = {t["function"]["name"] for t in tools or []}
        last = messages[-1]
        if names == {"update_memory", "save_skill"}:  # reflection pass
            if last["role"] == "user":
                return _reply(calls=[("update_memory", {
                    "target": "user", "mode": "append",
                    "content": "- Prefers `pytest -q` with a one-line pass/fail summary.",
                })])
            return _reply("NOTHING_TO_SAVE")

        user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        prompt = messages[user_idx]["content"]
        tool_results = [m["content"] for m in messages[user_idx:] if m["role"] == "tool"]

        if "run the tests" in prompt:
            plan = [
                ("search_sessions", {"query": "pytest failures", "limit": 2}),
                ("read_skill", {"name": "run-tests"}),
                ("run_shell", {"command": "python -m pytest -q"}),
            ]
            if len(tool_results) < len(plan):
                return _reply(calls=[plan[len(tool_results)]])
            summary = re.search(r"(\d+) passed[^\n]*", tool_results[-1])
            passed = summary.group(0).strip() if summary else "tests finished"
            return _reply(
                f"All green — **{passed}**.\n\n"
                "- Last time `test_memory.py` failed on a missing FTS5 build; that is fixed now.\n"
                "- Followed the `run-tests` skill: quiet mode, summary only."
            )
        if "commit" in prompt:
            if not tool_results:
                return _reply(calls=[("run_shell", {"command": "git log --oneline -4"})])
            stdout = tool_results[-1].split("STDOUT:")[1].split("STDERR:")[0]
            count = len([ln for ln in stdout.splitlines() if ln.strip()])
            return _reply(f"The last **{count} commits** are listed above, newest first.")
        return _reply("Done.")


# ---------------------------------------------------------- recording session
def record_session() -> list[tuple[str, list[list[Segment]]]]:
    """Run a demo session and return ordered steps of (kind, lines)."""
    tmp = Path(tempfile.mkdtemp(prefix="vespera-demo-"))
    config = Config(home=tmp / ".vespera", auto_approve=True)
    config.ensure_dirs()
    # What the banner shows: the real paths are a throwaway temp directory.
    shown_config = Config(home=Path("~/.vespera"), auto_approve=True)
    config.memory_file.write_text(
        "# Memory\n- vespera-cli: tests live in tests/, run with pytest.\n"
        "- Windows + Linux; use pathlib for paths.\n",
        encoding="utf-8",
    )
    skills = SkillRegistry(config.skills_dir)
    skills.save("run-tests", "Run the project test suite and summarise the result",
                "1. Run `python -m pytest -q` from the repo root.\n2. Report pass/fail in one line.")
    skills.save("release-notes", "Draft release notes from git history",
                "1. `git log --oneline <last-tag>..HEAD`\n2. Group by feat/fix/docs.")

    memory = MemoryStore(config)
    old = memory.new_session("fix failing memory tests")
    memory.append_message(old, {"role": "user", "content": "pytest failures in test_memory.py: no such module fts5"})

    console = Console(record=True, width=WIDTH, force_terminal=True, color_system="truecolor",
                      file=io.StringIO(), highlight=False, legacy_windows=False)
    tools = ToolExecutor(memory, skills, auto_approve=True, cwd=ROOT)
    agent = Agent(ScriptedModel(), config.model, memory, skills, tools, stream=False,
                  on_event=Renderer(console, live=False))

    steps: list[tuple[str, list[list[Segment]]]] = []

    def flush(kind: str) -> None:
        segs = [s for s in console._record_buffer if not s.control]
        console._record_buffer.clear()
        steps.append((kind, list(Segment.split_lines(segs))))

    def say(text: str) -> None:
        console.print(Text.assemble((PROMPT, f"bold {ACCENT}"), (text, "bold")))
        flush("type")

    render_banner(console, agent, shown_config, resumed=False)
    flush("output")
    say("run the tests and tell me if anything is failing")
    agent.run_turn("run the tests and tell me if anything is failing")
    console.print()
    flush("output")
    say("/memory")
    handle_command("/memory", agent, console)
    flush("output")
    say("/skills")
    handle_command("/skills", agent, console)
    flush("output")
    say("show me the latest commits")
    agent.run_turn("show me the latest commits")
    console.print()
    flush("output")
    say("/exit")
    console.print(f"[{ACCENT}]goodbye ✦[/]")
    flush("output")
    memory.close()
    return steps


# -------------------------------------------------------------- text drawing
FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"


def _font(names: list[str], size: int) -> ImageFont.FreeTypeFont:
    for name in names:
        for path in (FONT_DIR / name, Path("/usr/share/fonts/truetype/dejavu") / name):
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


class TermPainter:
    def __init__(self, size: int = 15):
        self.font = _font(["CascadiaMono.ttf", "consola.ttf", "DejaVuSansMono.ttf"], size)
        self.bold = _font(["CascadiaMono.ttf", "consolab.ttf", "DejaVuSansMono-Bold.ttf"], size)
        self.symbol = _font(["seguisym.ttf", "DejaVuSans.ttf"], size - 1)
        self.cw = round(self.font.getlength("M"))
        self.ch = round(size * 1.38)

    def _uses_symbol_font(self, c: str) -> bool:
        o = ord(c)
        return o >= 0x2190 and not (0x2500 <= o <= 0x257F)

    def draw_line(self, draw: ImageDraw.ImageDraw, x0: int, y0: int, line: list[Segment],
                  max_chars: int | None = None) -> None:
        col = 0
        for seg in line:
            style = seg.style
            fg, bg, bold, dim, italic = FG, None, False, False, False
            if style is not None:
                if style.color is not None and not style.color.is_default:
                    fg = tuple(style.color.get_truecolor(THEME))
                if style.bgcolor is not None and not style.bgcolor.is_default:
                    bg = tuple(style.bgcolor.get_truecolor(THEME, foreground=False))
                bold, dim, italic = bool(style.bold), bool(style.dim), bool(style.italic)
            if dim:
                fg = tuple(int(c * 0.55 + TERM_BG[i] * 0.45) for i, c in enumerate(fg))
            for c in seg.text:
                if max_chars is not None and col >= max_chars:
                    return
                x = x0 + col * self.cw
                if bg:
                    draw.rectangle([x, y0, x + self.cw, y0 + self.ch], fill=bg)
                if c != " ":
                    if self._uses_symbol_font(c):
                        draw.text((x + self.cw / 2, y0 + self.ch / 2), c, font=self.symbol,
                                  fill=fg, anchor="mm")
                    else:
                        draw.text((x, y0 + 2), c, font=self.bold if bold else self.font, fill=fg)
                col += 1

    def terminal(self, lines: list[list[Segment]], rows: int, partial: tuple[int, int] | None = None,
                 cursor: bool = False) -> Image.Image:
        """A macOS-style terminal window showing the last ``rows`` lines."""
        pad, title_h = 18, 34
        w = WIDTH * self.cw + pad * 2
        h = rows * self.ch + pad * 2 + title_h
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=14, fill=255)
        body = Image.new("RGBA", (w, h), TERM_BG + (255,))
        d = ImageDraw.Draw(body)
        d.rectangle([0, 0, w, title_h], fill=TITLE_BG)
        for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            cx = 20 + i * 22
            d.ellipse([cx - 6, title_h / 2 - 6, cx + 6, title_h / 2 + 6], fill=color)
        d.text((w / 2, title_h / 2), "vespera — ~/vespera-cli", font=self.font,
               fill=(170, 160, 200), anchor="mm")
        visible = lines[-rows:] if len(lines) > rows else lines
        top = title_h + pad
        for r, line in enumerate(visible):
            limit = None
            if partial is not None and r == len(visible) - 1:
                limit = partial[1]
            self.draw_line(d, pad, top + r * self.ch, line, limit)
        if cursor and visible:
            last = visible[-1]
            n = sum(len(s.text) for s in last)
            if partial is not None:
                n = partial[1]
            x = pad + n * self.cw
            y = top + (len(visible) - 1) * self.ch
            d.rectangle([x + 1, y + 3, x + self.cw - 1, y + self.ch - 1], fill=(180, 140, 255))
        img.paste(body, (0, 0), mask)
        return img


# ---------------------------------------------------------- flat screenshot
def render_screenshot(steps, painter: TermPainter, path: Path) -> None:
    lines = [ln for _, chunk in steps for ln in chunk]
    term = painter.terminal(lines, rows=len(lines))
    margin = 48
    canvas = Image.new("RGBA", (term.width + margin * 2, term.height + margin * 2), BG + (255,))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [margin, margin + 12, margin + term.width, margin + term.height + 12], 16, fill=(0, 0, 0, 200))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle(
        [margin - 4, margin - 4, margin + term.width + 4, margin + term.height + 4], 18,
        fill=(140, 90, 255, 90))
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(24)))
    canvas.alpha_composite(term, (margin, margin))
    canvas.convert("RGB").save(path, optimize=True)


# ---------------------------------------------------------------- 3D GIF
def _perspective_coeffs(dst: list[tuple[float, float]], src: list[tuple[float, float]]):
    """Coefficients mapping output pixels (dst quad) back to input pixels (src)."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y]); rhs.append(v)
    return np.linalg.solve(np.array(rows, float), np.array(rhs, float)).tolist()


def _project(w, h, yaw, pitch, roll, scale, cx, cy, dist=1900.0):
    pts = []
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        x, y = x * cr - y * sr, x * sr + y * cr
        x, z = x * cy_, -x * sy_
        y, z = y * cp - z * sp, y * sp + z * cp
        f = dist / (dist + z)
        pts.append((cx + x * f * scale, cy + y * f * scale))
    return pts


def _background(size) -> Image.Image:
    w, h = size
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(-1, 1, w)[None, :]
    base = np.zeros((h, w, 3))
    top, bottom = np.array([24, 12, 48]), np.array([6, 5, 14])
    base[:] = top * (1 - yy[..., None]) + bottom * yy[..., None]
    glow = np.exp(-((xx * 1.1) ** 2 + ((yy - 0.42) * 2.4) ** 2))[..., None]
    base += glow * np.array([70, 40, 120])
    img = Image.fromarray(np.clip(base, 0, 255).astype("uint8")).convert("RGBA")
    d = ImageDraw.Draw(img)
    horizon, vx = int(h * 0.62), w / 2
    for i in range(-14, 15):  # floor grid converging to a vanishing point
        d.line([(vx + i * 22, horizon), (vx + i * 180, h)], fill=(120, 80, 220, 255), width=1)
    for k in range(1, 12):
        y = horizon + (h - horizon) * (k / 11) ** 2.2
        d.line([(0, y), (w, y)], fill=(120, 80, 220, 255), width=1)
    rng = np.random.default_rng(7)
    for _ in range(90):
        x, y = rng.uniform(0, w), rng.uniform(0, horizon)
        r = rng.uniform(0.6, 1.8)
        a = int(rng.uniform(80, 220))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(220, 210, 255, a))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def _timeline(steps):
    """Yield (lines, partial, cursor) snapshots describing the typing/printing."""
    shown: list[list[Segment]] = []
    for kind, chunk in steps:
        if kind == "type":
            line = chunk[0]
            total = sum(len(s.text) for s in line)
            start = len(PROMPT)
            for _ in range(4):  # idle prompt with blinking cursor
                yield shown + [line], (0, start), True
            for n in range(start, total + 1, 4):
                yield shown + [line], (0, n), True
            yield shown + [line], (0, total), True
            shown.append(line)
            for extra in chunk[1:]:
                shown.append(extra)
        else:
            step = 3 if len(chunk) > 6 else 2
            for i in range(0, len(chunk), step):
                yield shown + chunk[: i + step], None, False
            shown.extend(chunk)
            for _ in range(6):  # let the output breathe
                yield list(shown), None, False


def render_gif(steps, painter: TermPainter, path: Path, rows: int = 24) -> None:
    out_w, out_h = 1120, 700
    bg = _background((out_w, out_h))
    snaps = list(_timeline(steps))
    intro, outro = 14, 16
    frames: list[Image.Image] = []
    total = intro + len(snaps) + outro
    for f in range(total):
        t = f / total
        if f < intro:  # swing the card in from the side
            k = 1 - (1 - f / intro) ** 3
            lines, partial, cursor = snaps[0][0], snaps[0][1], False
            yaw, pitch, scale = math.radians(-62 + 44 * k), math.radians(14 - 6 * k), 0.55 + 0.23 * k
        else:
            idx = min(f - intro, len(snaps) - 1)
            lines, partial, cursor = snaps[idx]
            sway = math.sin(t * math.tau * 1.5)
            yaw = math.radians(-18 + 7 * sway)
            pitch = math.radians(8 + 2 * math.cos(t * math.tau))
            scale = 0.78
            if f >= intro + len(snaps):  # outro: settle face-on
                k = (f - intro - len(snaps)) / outro
                k = k * k * (3 - 2 * k)
                yaw = yaw * (1 - k)
                pitch = pitch * (1 - k)
                scale = 0.78 + 0.05 * k
        if cursor and (f // 3) % 2:
            cursor = False
        term = painter.terminal(lines, rows, partial, cursor)
        w, h = term.size
        roll = math.radians(-1.2) * math.cos(t * math.tau)
        quad = _project(w, h, yaw, pitch, roll, scale, out_w / 2, out_h * 0.47)
        coeffs = _perspective_coeffs(quad, [(0, 0), (w, 0), (w, h), (0, h)])
        card = term.transform((out_w, out_h), Image.Transform.PERSPECTIVE, coeffs,
                              Image.Resampling.BICUBIC)
        frame = bg.copy()
        alpha = card.getchannel("A")
        glow = Image.new("RGBA", (out_w, out_h), (150, 100, 255, 0))
        glow.putalpha(alpha.filter(ImageFilter.GaussianBlur(26)).point(lambda a: int(a * 0.55)))
        shadow = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(20)).point(lambda a: int(a * 0.7)))
        frame.alpha_composite(shadow, (14, 30))
        frame.alpha_composite(glow)
        frame.alpha_composite(card)
        frames.append(frame.convert("RGB").resize((out_w * 3 // 4, out_h * 3 // 4),
                                                  Image.Resampling.LANCZOS))

    # One shared palette, sampled from frames across the whole animation.
    samples = frames[:: max(1, len(frames) // 8)]
    montage = Image.new("RGB", (samples[0].width, samples[0].height * len(samples)))
    for i, fr in enumerate(samples):
        montage.paste(fr, (0, i * fr.height))
    palette_src = montage.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    paletted = [fr.quantize(palette=palette_src, dither=Image.Dither.NONE) for fr in frames]
    durations = [70] * len(paletted)
    durations[-1] = 2600
    paletted[0].save(path, save_all=True, append_images=paletted[1:], duration=durations,
                     loop=0, optimize=True, disposal=1)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    steps = record_session()
    painter = TermPainter()
    render_screenshot(steps, painter, ASSETS / "screenshot.png")
    render_gif(steps, painter, ASSETS / "demo.gif")
    for name in ("screenshot.png", "demo.gif"):
        p = ASSETS / name
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
