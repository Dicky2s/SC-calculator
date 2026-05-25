import json

from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import (
    BeamState,
    CalculationInput,
    OutcomeFeedback,
    RockInput,
)
from sc_mining.storage.event_logger import save_calculation_event


def make_calculation():
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
    return calc_input, result


def test_save_calculation_event_creates_jsonl_record(tmp_path):
    calc_input, result = make_calculation()

    output_path = tmp_path / "events.jsonl"

    event = save_calculation_event(
        path=output_path,
        session_id="test_session",
        calc_input=calc_input,
        result=result,
        source="test",
    )

    assert output_path.exists()
    assert event["session_id"] == "test_session"
    assert event["source"] == "test"
    assert event["build"]["build_id"] == "prospector_helix_rieger_focus_v1"
    assert event["outcome"]["actual_outcome"] == "unknown"
    assert event["outcome"]["comment"] == ""

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    loaded = json.loads(lines[0])
    assert loaded["event_id"] == event["event_id"]
    assert loaded["result"]["verdict"] in {
        "take",
        "risky",
        "skip",
        "need_more_power",
    }
    assert loaded["outcome"]["actual_outcome"] == "unknown"


def test_save_calculation_event_writes_manual_outcome(tmp_path):
    calc_input, result = make_calculation()

    output_path = tmp_path / "events.jsonl"
    outcome = OutcomeFeedback(
        actual_outcome="good",
        comment="fractured fine and was worth taking",
    )

    event = save_calculation_event(
        path=output_path,
        session_id="test_session",
        calc_input=calc_input,
        result=result,
        source="test",
        outcome=outcome,
    )

    loaded = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert event["outcome"]["actual_outcome"] == "good"
    assert event["outcome"]["comment"] == "fractured fine and was worth taking"
    assert loaded["outcome"] == event["outcome"]
