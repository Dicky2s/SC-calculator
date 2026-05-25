import json

from sc_mining.dataset.exporter import build_dataset
from sc_mining.domain.models import ResourceComponent, ResourceYieldFeedback
from sc_mining.storage.event_logger import build_calculation_event
from sc_mining.domain.models import BuildProfile, HeadBuild, BeamState, CalculationInput, CalculationResult, RockInput


def test_resource_yield_supports_total_scu_estimate():
    payload = ResourceYieldFeedback(
        primary_resource="copper",
        resource_percent=50.0,
        raw_scu_estimate=12.0,
        total_scu_estimate=23.87,
        resources=[ResourceComponent(resource_name="copper", resource_percent=50.0, raw_scu_estimate=12.0)],
    )
    assert payload.total_scu_estimate == 23.87


def test_dataset_exports_total_scu_estimate(tmp_path):
    event = {
        "event_id": "e1",
        "session_id": "s1",
        "timestamp": "2026-05-26T00:00:00+00:00",
        "source": "manual_ui",
        "run_context": {"operator_name": "", "crew_size": 1, "run_tag": ""},
        "build": {"build_id": "b1", "ship_type": "prospector", "heads": []},
        "rock": {"mass": 10000, "resistance": 0.0, "instability": 0.1, "distance": 20},
        "beams": [{"slot": "main", "power_percent": 20.0, "active_modules": []}],
        "result": {"required_power": 10.0, "effective_power": 20.0, "margin": 10.0, "risk_score": 0.1, "verdict": "take", "notes": []},
        "ml_prediction": {"model_available": False},
        "outcome": {"actual_outcome": "good", "comment": ""},
        "resource_yield": {
            "primary_resource": "copper",
            "resource_percent": 50.0,
            "raw_scu_estimate": 11.935,
            "total_scu_estimate": 23.87,
            "resources": [{"resource_name": "copper", "resource_percent": 50.0, "raw_scu_estimate": 11.935, "comment": ""}],
        },
        "refinery": {"refinery_method": "unknown", "refined_resources": []},
        "calibration": {"formula_issue_flag": False, "observations": []},
        "labeling": {"label_source": "initial_save_ui", "labeled_at": "2026-05-26T00:00:00+00:00", "is_labeled": True},
    }
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event), encoding="utf-8")
    dataset = build_dataset(path)
    assert dataset.iloc[0]["total_scu_estimate"] == 23.87
