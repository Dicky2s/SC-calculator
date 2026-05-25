from pathlib import Path

import yaml

from sc_mining.domain.models import BuildProfile, HeadConfig, ModuleConfig


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_heads(path: str | Path) -> dict[str, HeadConfig]:
    raw = load_yaml(path)
    return {
        head_id: HeadConfig(**payload)
        for head_id, payload in raw["heads"].items()
    }


def load_modules(path: str | Path) -> dict[str, ModuleConfig]:
    raw = load_yaml(path)
    return {
        module_id: ModuleConfig(**payload)
        for module_id, payload in raw["modules"].items()
    }


def load_build(path: str | Path) -> BuildProfile:
    raw = load_yaml(path)
    return BuildProfile(**raw)