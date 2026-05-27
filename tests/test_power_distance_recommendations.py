from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import BeamState, CalculationInput, RockInput
from sc_mining.domain.recommendations import (
    build_power_distance_recommendation,
    build_recommendation_scan_input,
    calculation_input_for_candidate,
    select_recommendation_candidate,
)


def test_power_distance_recommendation_finds_warmup_and_stable_pairs():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/prospector_helix_2x_rieger.yaml")

    calc_input = CalculationInput(
        rock=RockInput(mass=13040, resistance=0, instability=0.12, distance=45),
        build=build,
        beams=[BeamState(slot="main", power_percent=20)],
    )

    recommendation = build_power_distance_recommendation(calc_input, heads=heads, modules=modules)

    assert recommendation.scanned_count > 0
    assert recommendation.minimum_warmup is not None
    assert recommendation.stable_hold is not None
    assert 20 <= recommendation.minimum_warmup.power_percent <= 100
    assert 10 <= recommendation.minimum_warmup.distance <= 120


def test_recommendation_ignores_current_ui_power_slider_value():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/prospector_helix_2x_rieger.yaml")

    low_slider = CalculationInput(
        rock=RockInput(mass=20728, resistance=0, instability=0.2, distance=15),
        build=build,
        beams=[BeamState(slot="main", power_percent=20)],
    )
    high_slider = CalculationInput(
        rock=low_slider.rock,
        build=build,
        beams=[BeamState(slot="main", power_percent=81)],
    )

    low = build_power_distance_recommendation(low_slider, heads=heads, modules=modules)
    high = build_power_distance_recommendation(high_slider, heads=heads, modules=modules)

    assert low.minimum_warmup == high.minimum_warmup
    assert low.stable_hold == high.stable_hold
    assert low.scanned_count == high.scanned_count


def test_recommendation_scan_input_keeps_only_enabled_beams():
    build = load_build("configs/builds/mole_helix_rieger_torrent.yaml")
    calc_input = CalculationInput(
        rock=RockInput(mass=40000, resistance=0.2, instability=0.1, distance=35),
        build=build,
        beams=[BeamState(slot="left", power_percent=87, active_modules=["torrent"])],
    )

    scan_input = build_recommendation_scan_input(calc_input)

    assert len(scan_input.beams) == 1
    assert scan_input.beams[0].slot == "left"
    assert scan_input.beams[0].power_percent == 20
    assert scan_input.beams[0].active_modules == ["torrent"]


def test_recommendation_scan_input_returns_no_beams_when_all_disabled():
    build = load_build("configs/builds/mole_helix_2x_rieger.yaml")
    calc_input = CalculationInput(
        rock=RockInput(mass=40000, resistance=0.2, instability=0.1, distance=35),
        build=build,
        beams=[],
    )

    scan_input = build_recommendation_scan_input(calc_input)

    assert scan_input.beams == []


def test_recommendation_explains_when_current_build_is_underpowered():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/prospector_helix_2x_rieger.yaml")

    calc_input = CalculationInput(
        rock=RockInput(mass=50000, resistance=0.0, instability=0.1, distance=15),
        build=build,
        beams=[BeamState(slot="main", power_percent=100)],
    )

    recommendation = build_power_distance_recommendation(calc_input, heads=heads, modules=modules)

    assert recommendation.minimum_warmup is None
    assert recommendation.stable_hold is None
    assert recommendation.best_available is not None
    assert recommendation.best_available.margin < 0
    assert recommendation.limiting_reason == "not_enough_max_power"
    assert recommendation.required_multiplier is not None
    assert recommendation.required_multiplier > 1.0


def test_selected_recommendation_candidate_can_build_saved_formula_input():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/prospector_helix_2x_rieger.yaml")
    calc_input = CalculationInput(
        rock=RockInput(mass=5245, resistance=0.39, instability=1.1216, distance=15),
        build=build,
        beams=[BeamState(slot="main", power_percent=20)],
    )

    recommendation = build_power_distance_recommendation(calc_input, heads=heads, modules=modules)
    selected = select_recommendation_candidate(recommendation)
    saved_input = calculation_input_for_candidate(calc_input, selected)

    assert selected is not None
    assert saved_input.rock.distance == selected.distance
    assert saved_input.beams[0].power_percent == selected.power_percent
