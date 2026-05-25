from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import BeamState, CalculationInput, RockInput
from sc_mining.domain.recommendations import (
    build_power_distance_recommendation,
    build_recommendation_scan_input,
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


def test_recommendation_scan_input_uses_build_slots_when_no_beam_enabled():
    build = load_build("configs/builds/mole_helix_2x_rieger.yaml")
    calc_input = CalculationInput(
        rock=RockInput(mass=40000, resistance=0.2, instability=0.1, distance=35),
        build=build,
        beams=[],
    )

    scan_input = build_recommendation_scan_input(calc_input)

    assert len(scan_input.beams) == len(build.heads)
    assert {beam.slot for beam in scan_input.beams} == {head.slot for head in build.heads}
    assert all(beam.power_percent == 20 for beam in scan_input.beams)
