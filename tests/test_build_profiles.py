from pathlib import Path

import pytest

from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import BeamState, CalculationInput, RockInput


BUILD_DIR = Path("configs/builds")
HEADS_PATH = Path("configs/heads.yaml")
MODULES_PATH = Path("configs/modules.yaml")


EXPECTED_PRESET_BUILDS = {
    "prospector_helix_2x_rieger_v1",
    "prospector_helix_rieger_fltr_v1",
    "prospector_helix_rieger_torrent_v1",
    "prospector_helix_rieger_torrent_iii_v1",
    "mole_helix_2x_rieger_v1",
    "mole_helix_rieger_fltr_v1",
    "mole_helix_rieger_torrent_v1",
    "mole_helix_rieger_torrent_iii_v1",
    "golem_pitman_2x_rieger_v1",
    "golem_pitman_rieger_fltr_v1",
    "golem_pitman_rieger_torrent_v1",
    "golem_pitman_rieger_torrent_iii_v1",
}


@pytest.fixture()
def heads():
    return load_heads(HEADS_PATH)


@pytest.fixture()
def modules():
    return load_modules(MODULES_PATH)


def test_expected_build_presets_exist():
    build_ids = {
        load_build(path).build_id
        for path in BUILD_DIR.glob("*.yaml")
    }

    assert EXPECTED_PRESET_BUILDS.issubset(build_ids)


def test_all_build_profiles_reference_existing_heads_and_modules(heads, modules):
    for path in BUILD_DIR.glob("*.yaml"):
        build = load_build(path)
        assert build.heads, f"{path} has no heads"

        for head in build.heads:
            assert head.head_id in heads, f"{path} references missing head {head.head_id}"
            for module_id in head.modules:
                assert module_id in modules, f"{path} references missing module {module_id}"


def test_preset_builds_cover_existing_ship_types():
    ships_by_build_id = {
        load_build(path).build_id: load_build(path).ship_type
        for path in BUILD_DIR.glob("*.yaml")
    }

    for ship_type in {"prospector", "mole", "golem"}:
        matching = [
            build_id
            for build_id, build_ship_type in ships_by_build_id.items()
            if build_ship_type == ship_type and build_id in EXPECTED_PRESET_BUILDS
        ]
        assert len(matching) == 4


def test_rieger_and_torrent_presets_are_calculable(heads, modules):
    rock = RockInput(
        mass=25000,
        resistance=0.28,
        instability=0.18,
        distance=30,
    )

    for build_name in [
        "prospector_helix_2x_rieger.yaml",
        "prospector_helix_rieger_fltr.yaml",
        "prospector_helix_rieger_torrent.yaml",
        "golem_pitman_2x_rieger.yaml",
        "mole_helix_rieger_torrent.yaml",
    ]:
        build = load_build(BUILD_DIR / build_name)
        beams = [
            BeamState(slot=head.slot, power_percent=65)
            for head in build.heads
        ]
        result = calculate(
            CalculationInput(rock=rock, build=build, beams=beams),
            heads=heads,
            modules=modules,
        )

        assert result.required_power > 0
        assert result.effective_power > 0
        assert result.verdict in {"take", "risky", "skip", "need_more_power"}
