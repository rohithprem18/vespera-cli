import pytest

from vespera.skills import SkillRegistry, parse_frontmatter, sanitize_name


def test_sanitize_name():
    assert sanitize_name("Deploy Docs!") == "deploy-docs"
    assert sanitize_name("../../etc/passwd") == "etc-passwd"
    assert len(sanitize_name("a" * 200)) == 64
    with pytest.raises(ValueError):
        sanitize_name("!!!")


def test_parse_frontmatter():
    meta, body = parse_frontmatter("---\nname: x\ndescription: does x\n---\n\n1. step\n")
    assert meta == {"name": "x", "description": "does x"}
    assert body.strip() == "1. step"
    assert parse_frontmatter("no frontmatter")[0] == {}


def test_parse_frontmatter_crlf_and_bad_yaml():
    meta, body = parse_frontmatter("---\r\nname: win\r\n---\r\nbody")
    assert meta["name"] == "win" and body == "body"
    assert parse_frontmatter("---\nname: [unclosed\n---\nbody")[0] == {}


def test_empty_index(skills):
    assert len(skills) == 0
    assert "no skills" in skills.index_text()


def test_save_and_read_roundtrip(skills):
    skill = skills.save("Release Checklist", "Cut a release\nof the package", "1. Bump version\n2. Tag")
    assert skill.name == "release-checklist"
    assert skill.path.name == "SKILL.md"
    assert skills.find("release checklist") is skill
    text = skills.read("release-checklist")
    meta, body = parse_frontmatter(text)
    assert meta == {"name": "release-checklist", "description": "Cut a release of the package"}
    assert "Bump version" in body


def test_index_contains_only_name_and_description(skills):
    skills.save("alpha", "first skill", "SECRET BODY CONTENT")
    index = skills.index_text()
    assert "- alpha: first skill" in index
    assert "SECRET BODY CONTENT" not in index


def test_save_overwrites_existing(skills):
    skills.save("tidy", "v1", "old")
    skills.save("tidy", "v2", "new")
    assert len(skills) == 1
    assert "new" in skills.read("tidy") and "old" not in skills.read("tidy")


def test_registry_scans_existing_folders(config):
    folder = config.skills_dir / "handwritten"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: handwritten\ndescription: made by hand\n---\nDo it.\n", encoding="utf-8"
    )
    (config.skills_dir / "no-skill-file").mkdir()
    reg = SkillRegistry(config.skills_dir)
    assert [s.name for s in reg.list()] == ["handwritten"]


def test_read_unknown_and_late_written_skill(config, skills):
    with pytest.raises(KeyError):
        skills.read("ghost")
    # A skill written by another process is found after an automatic refresh.
    other = SkillRegistry(config.skills_dir)
    other.save("ghost", "appears later", "boo")
    assert "boo" in skills.read("ghost")


def test_save_validates_inputs(skills):
    with pytest.raises(ValueError):
        skills.save("x", "   ", "body")
    with pytest.raises(ValueError):
        skills.save("x", "desc", "  ")
