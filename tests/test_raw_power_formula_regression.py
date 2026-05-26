from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_heads, load_modules
from sc_mining.domain.models import BeamState, BuildProfile, CalculationInput, HeadBuild, RockInput


def golem_pitman_rieger_stampede_build() -> BuildProfile:
    return BuildProfile(
        build_id="golem_pitman_rieger_stampede_v1",
        ship_type="golem",
        heads=[
            HeadBuild(
                slot="main",
                head_id="pitman_s1",
                modules=["rieger_c3", "stampede"],
            )
        ],
    )


def test_golem_manual_event_is_not_take_at_20_percent_without_active_stampede():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    calc_input = CalculationInput(
        rock=RockInput(mass=13560, resistance=0.15, instability=0.2489, distance=15),
        build=golem_pitman_rieger_stampede_build(),
        beams=[BeamState(slot="main", power_percent=20, active_modules=[])],
    )

    result = calculate(calc_input, heads=heads, modules=modules)

    assert result.required_power > 3000
    assert result.effective_power < result.required_power
    assert result.verdict == "need_more_power"


def test_golem_manual_power_distance_observations_are_plausible():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = golem_pitman_rieger_stampede_build()
    rock_base = dict(mass=13560, resistance=0.15, instability=0.2489)

    no_warmup_near = calculate(
        CalculationInput(
            rock=RockInput(**rock_base, distance=17),
            build=build,
            beams=[BeamState(slot="main", power_percent=20, active_modules=[])],
        ),
        heads=heads,
        modules=modules,
    )
    warmup_with_stampede = calculate(
        CalculationInput(
            rock=RockInput(**rock_base, distance=15),
            build=build,
            beams=[BeamState(slot="main", power_percent=67, active_modules=["stampede"])],
        ),
        heads=heads,
        modules=modules,
    )
    no_warmup_far = calculate(
        CalculationInput(
            rock=RockInput(**rock_base, distance=34),
            build=build,
            beams=[BeamState(slot="main", power_percent=30, active_modules=[])],
        ),
        heads=heads,
        modules=modules,
    )

    assert no_warmup_near.verdict == "need_more_power"
    assert warmup_with_stampede.effective_power > warmup_with_stampede.required_power
    assert warmup_with_stampede.verdict in {"take", "edge_take", "risky"}
    assert no_warmup_far.verdict == "need_more_power"
