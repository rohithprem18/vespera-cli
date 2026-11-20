import pytest

from vespera.config import MEMORY_CHAR_LIMIT, USER_CHAR_LIMIT, Config
from vespera.memory import MemoryStore


def test_ensure_dirs_creates_starter_files(tmp_path):
    cfg = Config(home=tmp_path / "home")
    cfg.ensure_dirs()
    assert cfg.skills_dir.is_dir()
    assert cfg.memory_file.read_text(encoding="utf-8").startswith("# Memory")
    assert cfg.user_file.read_text(encoding="utf-8").startswith("# User")


def test_vespera_home_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VESPERA_HOME", str(tmp_path / "custom"))
    cfg = Config.from_env()
    assert cfg.home == (tmp_path / "custom").resolve()
    assert cfg.db_path.name == "sessions.db"


def test_snapshot_is_frozen_for_the_session(memory):
    before = memory.memory_snapshot
    memory.update_file("memory", "# Memory\n- project uses uv\n")
    assert memory.memory_snapshot == before
    assert "uv" in memory.read_live("memory")


def test_next_session_sees_update(config, memory):
    memory.update_file("user", "# User\n- prefers terse answers\n")
    with MemoryStore(config) as fresh:
        assert "terse" in fresh.user_snapshot


def test_snapshot_truncated_to_limit(config):
    config.memory_file.write_text("x" * (MEMORY_CHAR_LIMIT + 500), encoding="utf-8")
    config.user_file.write_text("y" * (USER_CHAR_LIMIT + 500), encoding="utf-8")
    with MemoryStore(config) as store:
        assert len(store.memory_snapshot) == MEMORY_CHAR_LIMIT
        assert len(store.user_snapshot) == USER_CHAR_LIMIT


def test_update_rejects_over_limit(memory):
    with pytest.raises(ValueError, match="limit"):
        memory.update_file("user", "z" * (USER_CHAR_LIMIT + 1))
    memory.update_file("user", "z" * USER_CHAR_LIMIT)  # exactly at the cap is fine


def test_append_mode_and_invalid_target(memory):
    memory.update_file("memory", "- one", mode="replace")
    memory.update_file("memory", "- two", mode="append")
    assert memory.read_live("memory") == "- one\n- two\n"
    with pytest.raises(ValueError):
        memory.update_file("secrets", "nope")


def test_usage_reports_limits(memory):
    usage = memory.usage()
    assert usage["memory"]["limit"] == MEMORY_CHAR_LIMIT
    assert usage["user"]["limit"] == USER_CHAR_LIMIT


def test_sessions_persist_and_reload(memory):
    sid = memory.new_session("first")
    msgs = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "run_shell", "arguments": '{"command": "ls"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "README.md"},
    ]
    for m in msgs:
        memory.append_message(sid, m)
    assert memory.load_messages(sid) == msgs
    assert memory.latest_session_id() == sid


def test_latest_session_follows_updates(memory):
    a = memory.new_session("a")
    b = memory.new_session("b")
    assert memory.latest_session_id() == b
    memory.conn.execute("UPDATE sessions SET updated_at = '2999-01-01' WHERE id = ?", (a,))
    assert memory.latest_session_id() == a


def test_fts_search_returns_snippets(memory):
    s1 = memory.new_session("docker")
    memory.append_message(s1, {"role": "user", "content": "How do I prune dangling docker images?"})
    s2 = memory.new_session("python")
    memory.append_message(s2, {"role": "user", "content": "Set up a python virtualenv please"})

    hits = memory.search_sessions("docker images")
    assert hits and hits[0].session_id == s1
    assert "[docker]" in hits[0].snippet
    assert memory.search_sessions("docker", exclude_session=s1) == []


def test_fts_search_handles_punctuation_and_empty(memory):
    sid = memory.new_session("t")
    memory.append_message(sid, {"role": "user", "content": "error: AND OR NOT ( quote\""})
    assert isinstance(memory.search_sessions('"(AND*'), list)  # must not raise on FTS syntax
    assert memory.search_sessions("   ") == []
    assert memory.search_sessions("quote")[0].session_id == sid


def test_tool_call_arguments_are_searchable(memory):
    sid = memory.new_session("t")
    memory.append_message(sid, {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "1", "type": "function",
                        "function": {"name": "run_shell", "arguments": '{"command": "kubectl get pods"}'}}],
    })
    assert memory.search_sessions("kubectl")[0].session_id == sid
