from sc_mining.domain.models import (
    BeamState,
    BuildProfile,
    CalculationInput,
    CalculationResult,
    HeadBuild,
    RefinedResourceOutput,
    RefineryFeedback,
    ResourceComponent,
    ResourceYieldFeedback,
    RockInput,
)
from sc_mining.storage.event_logger import save_calculation_event
from sc_mining.storage.event_reader import load_events_dataframe, read_jsonl
from sc_mining.storage.refinery_labeler import has_refinery_result, update_event_refinery


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


def test_update_event_refinery_adds_later_result(tmp_path):
    path = tmp_path / "events.jsonl"
    event = save_calculation_event(
        path=path,
        session_id="real_run_01",
        calc_input=_calc_input(),
        result=_result(),
        resource_yield=ResourceYieldFeedback(
            resources=[
                ResourceComponent(resource_name="taranite", resource_percent=35),
                ResourceComponent(resource_name="gold", resource_percent=15),
            ],
        ),
    )

    update_result = update_event_refinery(
        path=path,
        event_id=event["event_id"],
        refinery=RefineryFeedback(
            refinery_method="dinix",
            refinery_location="ARC-L1",
            refined_scu_actual=8.4,
            sell_value_auec=125000,
            refined_resources=[
                RefinedResourceOutput(resource_name="taranite", refined_scu_actual=6.2, sell_value_auec=98000),
                RefinedResourceOutput(resource_name="gold", refined_scu_actual=2.2, sell_value_auec=27000),
            ],
        ),
    )

    assert update_result["has_refinery_result"] is True

    records = read_jsonl(path)
    assert records[0]["refinery"]["refinery_method"] == "dinix"
    assert records[0]["refinery"]["refined_resources"][0]["resource_name"] == "taranite"
    assert records[0]["refinery_labeling"]["has_refinery_result"] is True

    df = load_events_dataframe(path)
    row = df.iloc[0]
    assert row["refined_resource_count"] == 2
    assert row["refined_resource_names"] == "taranite, gold"
    assert row["total_refined_scu_actual"] == 8.4
    assert row["total_resource_sell_value_auec"] == 125000
    assert "taranite" in row["refined_resources_json"]


def test_update_event_refinery_requires_existing_event(tmp_path):
    path = tmp_path / "events.jsonl"
    save_calculation_event(
        path=path,
        session_id="real_run_01",
        calc_input=_calc_input(),
        result=_result(),
    )

    try:
        update_event_refinery(
            path=path,
            event_id="missing",
            refinery=RefineryFeedback(refinery_method="dinix"),
        )
    except ValueError as exc:
        assert "Event not found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing event")


def test_has_refinery_result_detects_empty_and_filled_values():
    assert has_refinery_result({}) is False
    assert has_refinery_result({"refinery_method": "unknown"}) is False
    assert has_refinery_result({"refinery_method": "dinix"}) is True
    assert has_refinery_result({"refined_resources": [{"resource_name": "gold"}]}) is True
