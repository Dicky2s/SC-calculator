import json

from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import BeamState, CalculationInput, RockInput
from sc_mining.storage.event_logger import save_calculation_event


def test_save_calculation_event_creates_jsonl_record(tmp_path):
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