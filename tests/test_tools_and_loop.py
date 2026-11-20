import json
import sys
from types import SimpleNamespace as NS

import pytest

from vespera.agent import NOTHING_TO_SAVE, Agent
from vespera.config import FILE_READ_LIMIT, SHELL_OUTPUT_LIMIT
from vespera.tools import TOOL_NAMES, ToolExecutor, is_high_risk, truncate

PY = f'"{sys.executable}"'


# ----------------------------------------------------------------- fakes
def call(name, args, cid=None):
    return {"name": name, "arguments": json.dumps(args), "id": cid or f"call_{name}"}


def reply(text=None, calls=()):
    tool_calls = [
        NS(id=c["id"], type="function", function=NS(name=c["name"], arguments=c["arguments"]))
        for c in calls
    ]
    return NS(choices=[NS(message=NS(content=text, tool_calls=tool_calls or None))])


class FakeClient:
    """Mimics ``client.chat.completions.create`` with scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(memory, skills, replies, tmp_path, **kw):
    tools = ToolExecutor(memory, skills, confirm=kw.pop("confirm", None),
                         auto_approve=kw.pop("auto_approve", True), cwd=tmp_path)
    events = []
    client = FakeClient(replies)
    agent = Agent(client, "test-model", memory, skills, tools, stream=False,
                  on_event=lambda kind, **d: events.append((kind, d)), **kw)
    return agent, client, events


# ----------------------------------------------------------------- tools
def test_schemas_cover_required_tools():
    assert set(TOOL_NAMES) == {
        "run_shell", "read_file", "write_file", "search_sessions",
        "read_skill", "save_skill", "update_memory",
    }


@pytest.mark.parametrize("cmd", [
    "rm -rf /", "sudo rm -fr ~/x", "mkfs.ext4 /dev/sda1", "dd if=/dev/zero of=/dev/sda",
    "shutdown -h now", "reboot", ":(){ :|:& };:", "format C:", "rd /s /q C:\\x",
    "Remove-Item -Recurse -Force C:\\x",
])
def test_high_risk_patterns(cmd):
    assert is_high_risk(cmd)


@pytest.mark.parametrize("cmd", ["ls -la", "git status", "rm notes.txt", "dir /b", "echo done"])
def test_safe_commands_not_flagged(cmd):
    assert not is_high_risk(cmd)


def test_truncate_keeps_head_and_tail():
    text = "A" * 100 + "B" * 100
    out = truncate(text, 60)
    assert out.startswith("A" * 40) and out.endswith("B" * 20) and "truncated" in out
    assert truncate("short", 60) == "short"


def test_run_shell_executes_and_truncates(memory, skills, tmp_path):
    ex = ToolExecutor(memory, skills, auto_approve=True, cwd=tmp_path)
    out = ex.run_shell(f"{PY} -c \"print('hi')\"")
    assert "exit_code: 0" in out and "hi" in out
    big = ex.run_shell(f"{PY} -c \"print('x' * {SHELL_OUTPUT_LIMIT * 3})\"")
    assert "truncated" in big and len(big) < SHELL_OUTPUT_LIMIT + 300


def test_run_shell_timeout(memory, skills, tmp_path):
    ex = ToolExecutor(memory, skills, auto_approve=True, cwd=tmp_path, shell_timeout=1)
    assert "timed out" in ex.run_shell(f"{PY} -c \"import time; time.sleep(5)\"")


def test_run_shell_requires_confirmation(memory, skills, tmp_path):
    asked = []
    ex = ToolExecutor(memory, skills, confirm=lambda c, r: asked.append((c, r)) or False, cwd=tmp_path)
    assert "denied" in ex.run_shell("echo hi")
    assert asked == [("echo hi", False)]
    # Without any confirm callback, nothing runs.
    assert "denied" in ToolExecutor(memory, skills, cwd=tmp_path).run_shell("echo hi")


def test_high_risk_always_confirms_even_with_yes(memory, skills, tmp_path):
    asked = []
    ex = ToolExecutor(memory, skills, confirm=lambda c, r: asked.append(r) or False,
                      auto_approve=True, cwd=tmp_path)
    assert "denied" in ex.execute("run_shell", {"command": "rm -rf ./build"})
    assert asked == [True]
    ex.execute("run_shell", {"command": "echo safe"})
    assert asked == [True]  # safe command was auto-approved without asking


def test_file_tools(memory, skills, tmp_path):
    ex = ToolExecutor(memory, skills, cwd=tmp_path)
    assert "Wrote 5 chars" in ex.write_file("nested/dir/a.txt", "hello")
    assert ex.read_file("nested/dir/a.txt") == "hello"
    assert "not found" in ex.read_file("missing.txt")
    assert "directory" in ex.read_file("nested")
    ex.write_file("big.txt", "y" * (FILE_READ_LIMIT + 10))
    assert "truncated" in ex.read_file("big.txt")


def test_execute_never_raises(memory, skills, tmp_path):
    ex = ToolExecutor(memory, skills, cwd=tmp_path)
    assert ex.execute("nope", {}).startswith("Error: unknown tool")
    assert "not valid JSON" in ex.execute("read_file", "{bad json")
    assert "bad arguments" in ex.execute("read_file", {"wrong": 1})
    assert ex.execute("update_memory", {"target": "user", "content": "z" * 5000}).startswith("Error")


def test_memory_and_skill_tools(memory, skills, tmp_path):
    ex = ToolExecutor(memory, skills, cwd=tmp_path)
    assert "Updated user" in ex.update_memory("user", "- likes dark mode")
    assert "dark mode" in memory.read_live("user")
    assert "Saved skill 'lint-fix'" in ex.save_skill("Lint Fix", "fix lint", "1. run ruff --fix")
    assert "ruff" in ex.read_skill("lint-fix")
    assert "no skill named" in ex.read_skill("nope")

    sid = memory.new_session("old")
    memory.append_message(sid, {"role": "user", "content": "configure nginx reverse proxy"})
    assert sid in ex.search_sessions("nginx")
    ex.current_session_id = sid
    assert "No past sessions" in ex.search_sessions("nginx")


# ------------------------------------------------------------------ loop
def test_plain_answer_without_tools(memory, skills, tmp_path):
    agent, client, events = make_agent(memory, skills, [reply("Hello!")], tmp_path)
    result = agent.run_turn("hi")
    assert result.text == "Hello!" and result.tool_calls == 0
    assert client.requests[0]["messages"][0]["role"] == "system"
    assert {t["function"]["name"] for t in client.requests[0]["tools"]} == set(TOOL_NAMES)
    assert ("text_delta", {"text": "Hello!"}) in events
    # persisted
    assert [m["role"] for m in memory.load_messages(agent.session_id)] == ["user", "assistant"]


def test_multi_step_tool_loop(memory, skills, tmp_path):
    replies = [
        reply(calls=[call("write_file", {"path": "x.txt", "content": "42"}, "c1")]),
        reply(calls=[call("read_file", {"path": "x.txt"}, "c2")]),
        reply("The file contains 42."),
    ]
    agent, client, events = make_agent(memory, skills, replies, tmp_path, reflection_threshold=99)
    result = agent.run_turn("write and read")
    assert result.text == "The file contains 42." and result.tool_calls == 2
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert tool_msgs[1] == {"role": "tool", "tool_call_id": "c2", "content": "42"}
    # Each request carries the growing history.
    assert len(client.requests[2]["messages"]) == 1 + 1 + 4
    kinds = [k for k, _ in events]
    assert kinds.count("tool_call") == 2 and kinds.count("tool_result") == 2


def test_tool_errors_and_denials_do_not_crash(memory, skills, tmp_path):
    replies = [
        reply(calls=[call("run_shell", {"command": "echo hi"}, "c1"),
                     call("missing_tool", {}, "c2")]),
        reply("Could not run it."),
    ]
    agent, _, _ = make_agent(memory, skills, replies, tmp_path, auto_approve=False,
                             confirm=lambda c, r: False, reflection_threshold=99)
    result = agent.run_turn("go")
    tool_msgs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert "denied" in tool_msgs[0] and "unknown tool" in tool_msgs[1]
    assert result.text == "Could not run it."


def test_iteration_cap(memory, skills, tmp_path):
    looping = [reply(calls=[call("read_file", {"path": "a"}, f"c{i}")]) for i in range(3)]
    agent, _, _ = make_agent(memory, skills, looping, tmp_path, max_iterations=3,
                             reflection_threshold=99)
    result = agent.run_turn("loop forever")
    assert result.stopped_early and "Stopped after 3" in result.text


def test_reflection_runs_after_busy_turn(memory, skills, tmp_path):
    replies = [
        reply(calls=[call("write_file", {"path": f"f{i}.txt", "content": "x"}, f"c{i}") for i in range(3)]),
        reply("Done."),
        # reflection round: save a skill and a preference
        reply(calls=[
            call("save_skill", {"name": "scaffold files", "description": "create starter files",
                                "instructions": "1. write f0..f2"}, "r1"),
            call("update_memory", {"target": "user", "content": "- likes scaffolds"}, "r2"),
            call("run_shell", {"command": "echo sneaky"}, "r3"),
        ]),
        reply(NOTHING_TO_SAVE),
    ]
    agent, client, events = make_agent(memory, skills, replies, tmp_path)
    result = agent.run_turn("scaffold three files")
    assert result.tool_calls == 3
    assert len(result.reflection_actions) == 2
    assert skills.find("scaffold-files") is not None
    assert "likes scaffolds" in memory.read_live("user")
    # Reflection only exposes the learning tools and is not added to history.
    reflection_tools = {t["function"]["name"] for t in client.requests[2]["tools"]}
    assert reflection_tools == {"update_memory", "save_skill"}
    assert not any("[reflection]" in str(m.get("content")) for m in agent.messages)
    assert ("reflection_start", {}) in events


def test_reflection_skipped_for_trivial_turn(memory, skills, tmp_path):
    replies = [reply(calls=[call("read_file", {"path": "a"})]), reply("ok")]
    agent, client, _ = make_agent(memory, skills, replies, tmp_path)
    assert agent.run_turn("one tool").reflection_actions == []
    assert len(client.requests) == 2


def test_reflection_failure_is_contained(memory, skills, tmp_path):
    replies = [
        reply(calls=[call("read_file", {"path": "a"}, f"c{i}") for i in range(3)]),
        reply("done"),
        RuntimeError("provider down"),
    ]
    agent, _, _ = make_agent(memory, skills, replies, tmp_path)
    result = agent.run_turn("x")
    assert result.text == "done"
    assert "reflection skipped" in result.reflection_actions[0]


def test_system_prompt_is_stable_and_progressive(memory, skills, tmp_path):
    skills.save("deploy", "ship it", "VERY LONG BODY")
    replies = [reply("a"), reply("b")]
    agent, client, _ = make_agent(memory, skills, replies, tmp_path)
    agent.run_turn("one")
    memory.update_file("memory", "- changed mid-session")
    skills.save("later", "added later", "zzz")
    agent.run_turn("two")
    first, second = (r["messages"][0]["content"] for r in client.requests)
    assert first == second
    assert "- deploy: ship it" in first and "VERY LONG BODY" not in first


def test_resume_and_new_session(memory, skills, tmp_path):
    agent, _, _ = make_agent(memory, skills, [reply("first")], tmp_path)
    agent.run_turn("remember me")
    sid = agent.session_id

    agent2, client2, _ = make_agent(memory, skills, [reply("again")], tmp_path)
    agent2.resume(memory.latest_session_id())
    assert agent2.session_id == sid
    agent2.run_turn("continue")
    sent = [m["content"] for m in client2.requests[0]["messages"][1:]]
    assert sent == ["remember me", "first", "continue"]

    agent2.new_session()
    assert agent2.session_id is None and agent2.messages == []


def test_streaming_accumulates_text_and_tool_calls(memory, skills, tmp_path):
    def chunk(content=None, tool_calls=None):
        return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls))])

    stream_1 = [
        chunk(tool_calls=[NS(index=0, id="s1", function=NS(name="write_", arguments='{"path": '))]),
        chunk(tool_calls=[NS(index=0, id=None, function=NS(name="file", arguments='"s.txt", "content": "ok"}'))]),
        NS(choices=[]),  # usage-only chunk
    ]
    stream_2 = [chunk("All "), chunk("done.")]
    agent, _, events = make_agent(memory, skills, [iter(stream_1), iter(stream_2)], tmp_path,
                                  reflection_threshold=99)
    agent.stream = True
    result = agent.run_turn("stream it")
    assert result.text == "All done."
    assert (tmp_path / "s.txt").read_text() == "ok"
    assert [d["text"] for k, d in events if k == "text_delta"] == ["All ", "done."]
