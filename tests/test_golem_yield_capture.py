from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import (
    BeamState,
    CalculationInput,
    OutcomeFeedback,
    ResourceYieldFeedback,
    RockInput,
    RunContext,
)
from sc_mining.storage.event_logger import save_calculation_event
from sc_mining.storage.event_reader import load_events_dataframe


EXPECTED_MINING_MODULE_IDS = {
    "brandt",
    "fltr",
    "fltr_l",
    "fltr_xl",
    "focus",
    "focus_ii",
    "focus_iii",
    "forel",
    "lifeline",
    "optimum",
    "rieger",
    "rieger_c2",
    "rieger_c3",
    "rime",
    "roc",
    "stampede",
    "surge",
    "torpid",
    "torrent",
    "torrent_ii",
    "torrent_iii",
    "vaux",
    "vaux_c2",
    "vaux_c3",
    "xtr",
    "xtr_l",
    "xtr_xl",
}


def test_all_known_ship_mining_modules_are_in_config():
    modules = load_modules("configs/modules.yaml")

    assert EXPECTED_MINING_MODULE_IDS.issubset(set(modules))
    assert len(EXPECTED_MINING_MODULE_IDS) == 27


def test_golem_build_can_be_calculated_with_pitman_head():
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/golem_manual.yaml")

    calc_input = CalculationInput(
        rock=RockInput(
            mass=18000,
            resistance=0.22,
            instability=0.10,
            distance=35,
        ),
        build=build,
        beams=[
            BeamState(
                slot="main",
                power_percent=70,
                active_modules=["stampede"],
            ),
        ],
    )

    result = calculate(calc_input, heads=heads, modules=modules)

    assert build.ship_type == "golem"
    assert result.effective_power > 0
    assert result.verdict in {"take", "edge_take", "almost", "risky", "skip", "need_more_power"}


def test_event_reader_flattens_run_context_and_resource_yield(tmp_path):
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build("configs/builds/golem_manual.yaml")
    path = tmp_path / "events.jsonl"

    calc_input = CalculationInput(
        rock=RockInput(mass=18000, resistance=0.22, instability=0.10, distance=35),
        build=build,
        beams=[BeamState(slot="main", power_percent=70, active_modules=["stampede"])],
    )
    result = calculate(calc_input, heads=heads, modules=modules)

    save_calculation_event(
        path=path,
        session_id="real_duo_test",
        calc_input=calc_input,
        result=result,
        source="golem_quick_page",
        outcome=OutcomeFeedback(actual_outcome="good", comment="good golem test"),
        resource_yield=ResourceYieldFeedback(
            primary_resource="taranite",
            resource_percent=38,
            raw_scu_estimate=10.5,
            refined_scu_estimate=7.4,
            estimated_value_auec=118000,
            mining_time_seconds=420,
            comment="synthetic unit-test yield",
        ),
        run_context=RunContext(operator_name="pilot_a", crew_size=2, run_tag="duo"),
    )

    df = load_events_dataframe(path)
    row = df.iloc[0]

    assert row["source"] == "golem_quick_page"
    assert row["operator_name"] == "pilot_a"
    assert row["crew_size"] == 2
    assert row["run_tag"] == "duo"
    assert row["primary_resource"] == "taranite"
    assert row["resource_percent"] == 38
    assert row["refined_scu_estimate"] == 7.4
    assert row["estimated_value_auec"] == 118000
