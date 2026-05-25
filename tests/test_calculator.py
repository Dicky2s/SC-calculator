from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import BeamState, CalculationInput, RockInput


def test_easy_rock_should_be_take_or_risky():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/prospector_manual.yaml")

    calc_input = CalculationInput(
        rock=RockInput(
            mass=5000,
            resistance=0.1,
            instability=0.05,
            distance=80,
        ),
        build=build,
        beams=[
            BeamState(slot="main", power_percent=70),
        ],
    )

    result = calculate(calc_input, heads=heads, modules=modules)

    assert result.effective_power > result.required_power
    assert result.verdict in {"take", "risky"}


def test_hard_rock_should_not_be_easy_take():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/prospector_manual.yaml")

    calc_input = CalculationInput(
        rock=RockInput(
            mass=50000,
            resistance=0.8,
            instability=0.4,
            distance=140,
        ),
        build=build,
        beams=[
            BeamState(slot="main", power_percent=50),
        ],
    )

    result = calculate(calc_input, heads=heads, modules=modules)

    assert result.verdict in {"need_more_power", "skip", "risky"}


def test_mole_two_beams_should_have_more_power_than_prospector():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")

    prospector = load_build("configs/builds/prospector_manual.yaml")
    mole = load_build("configs/builds/mole_manual.yaml")

    rock = RockInput(
        mass=25000,
        resistance=0.35,
        instability=0.15,
        distance=100,
    )

    prospector_input = CalculationInput(
        rock=rock,
        build=prospector,
        beams=[
            BeamState(slot="main", power_percent=70),
        ],
    )

    mole_input = CalculationInput(
        rock=rock,
        build=mole,
        beams=[
            BeamState(slot="left", power_percent=60),
            BeamState(slot="center", power_percent=50),
        ],
    )

    prospector_result = calculate(prospector_input, heads=heads, modules=modules)
    mole_result = calculate(mole_input, heads=heads, modules=modules)

    assert mole_result.effective_power > prospector_result.effective_power


def test_unknown_beam_slot_should_raise_error():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/prospector_manual.yaml")

    calc_input = CalculationInput(
        rock=RockInput(
            mass=5000,
            resistance=0.1,
            instability=0.05,
            distance=80,
        ),
        build=build,
        beams=[
            BeamState(slot="wrong_slot", power_percent=70),
        ],
    )

    try:
        calculate(calc_input, heads=heads, modules=modules)
    except ValueError as error:
        assert "Unknown beam slot" in str(error)
    else:
        raise AssertionError("Expected ValueError for unknown beam slot")