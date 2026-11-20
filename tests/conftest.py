from pathlib import Path

import pytest

from vespera.config import Config
from vespera.memory import MemoryStore
from vespera.skills import SkillRegistry


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(home=tmp_path / "vespera-home")
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def memory(config: Config):
    store = MemoryStore(config)
    yield store
    store.close()


@pytest.fixture
def skills(config: Config) -> SkillRegistry:
    return SkillRegistry(config.skills_dir)
