import json

from sc_mining.domain.models import (
    CalculationInput,
    CalculationResult,
    BeamState,
    BuildProfile,
    HeadBuild,
    RefinedResourceOutput,
    RefineryFeedback,
    ResourceComponent,
    ResourceYieldFeedback,
    RockInput,
)
from sc_mining.storage.event_logger import save_calculation_event
from sc_mining.storage.event_reader import flatten_event, load_events_dataframe


def _calc_input() -> CalculationInput:
    return CalculationInput(
        rock=RockInput(mass=12000, resistance=0.1, instability=0.05, distance=30),
        build=BuildProfile(
            build_id="prospector_test",
            ship_type="prospector",
            heads=[HeadBuild(slot="main", head_id="helix_s1", modules=[])],
        ),
        beams=[BeamState(slot="main", power_percent=30)],
    )


def _result() -> CalculationResult:
    return CalculationResult(
        required_power=10,
        effective_power=15,
        margin=5,
        risk_score=0.1,
        verdict="take",
    )


def test_event_stores_multiple_resources_and_refinery(tmp_path):
    path = tmp_path / "events.jsonl"
    resource_yield = ResourceYieldFeedback(
        primary_resource="taranite",
        resource_percent=42,
        raw_scu_estimate=8.5,
        resources=[
            ResourceComponent(resource_name="taranite", resource_percent=42, raw_scu_estimate=8.5),
            ResourceComponent(resource_name="gold", resource_percent=18, raw_scu_estimate=2.1),
        ],
    )
    refinery = RefineryFeedback(
        refinery_method="dinix",
        refinery_location="ARC-L1",
        refined_scu_actual=7.2,
        sell_value_auec=95000,
        refined_resources=[
            RefinedResourceOutput(resource_name="taranite", refined_scu_actual=5.1, sell_value_auec=70000),
            RefinedResourceOutput(resource_name="gold", refined_scu_actual=2.1, sell_value_auec=25000),
        ],
    )

    event = save_calculation_event(
        path=path,
        session_id="real_run_01",
        calc_input=_calc_input(),
        result=_result(),
        resource_yield=resource_yield,
        refinery=refinery,
    )

    assert event["resource_yield"]["resources"][0]["resource_name"] == "taranite"
    assert event["refinery"]["refinery_method"] == "dinix"

    df = load_events_dataframe(path)
    row = df.iloc[0]
    assert row["resource_count"] == 2
    assert row["resource_names"] == "taranite, gold"
    assert row["total_resource_percent"] == 60
    assert row["refinery_method"] == "dinix"
    assert row["sell_value_auec"] == 95000
    assert row["refined_resource_count"] == 2
    assert row["refined_resource_names"] == "taranite, gold"
    assert row["total_refined_scu_actual"] == 7.2
    assert row["total_resource_sell_value_auec"] == 95000


def test_flatten_legacy_single_resource_event_still_works():
    event = {
        "event_id": "legacy-1",
        "session_id": "s1",
        "timestamp": "2026-05-25T00:00:00+00:00",
        "source": "manual_ui",
        "build": {"build_id": "b1", "ship_type": "prospector"},
        "rock": {"mass": 1000, "resistance": 0, "instability": 0, "distance": 25},
        "beams": [{"slot": "main", "power_percent": 20}],
        "result": {"required_power": 1, "effective_power": 2, "margin": 1, "risk_score": 0.1, "verdict": "take"},
        "outcome": {"actual_outcome": "unknown", "comment": ""},
        "resource_yield": {"primary_resource": "gold", "resource_percent": 25, "raw_scu_estimate": 3.3},
    }

    row = flatten_event(event)

    assert row["primary_resource"] == "gold"
    assert row["resource_count"] == 1
    assert row["resource_names"] == "gold"
    assert json.loads(row["resources_json"])[0]["resource_name"] == "gold"
