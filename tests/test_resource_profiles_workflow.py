import json
from pathlib import Path

from sc_mining.dataset.exporter import build_dataset
from sc_mining.domain.config_loader import load_resources
from sc_mining.domain.models import ResourceComponent, ResourceYieldFeedback


def test_resource_profiles_load_all_core_ores():
    resources = load_resources("configs/resources.yaml")

    expected = {
        "agricium",
        "aluminum",
        "beryl",
        "bexalite",
        "borase",
        "copper",
        "corundum",
        "diamond",
        "gold",
        "hephaestanite",
        "inert_materials",
        "iron",
        "laranite",
        "quantainium",
        "quartz",
        "taranite",
        "tin",
        "titanium",
        "tungsten",
    }

    assert expected.issubset(set(resources))
    assert resources["beryl"].window_hint in {"tiny", "small", "normal", "wide", "unknown"}


def test_resource_component_stores_observed_behavior():
    item = ResourceComponent(
        resource_name="beryl",
        resource_percent=20.0,
        raw_scu_estimate=4.2,
        observed_window_size="tiny",
        observed_charge_behavior="jumping",
    )

    assert item.observed_window_size == "tiny"
    assert item.observed_charge_behavior == "jumping"


def test_dataset_exports_resource_behavior_fields(tmp_path):
    event = {
        "event_id": "e1",
        "session_id": "s1",
        "timestamp": "2026-05-26T00:00:00+00:00",
        "source": "manual_ui",
        "run_context": {"operator_name": "", "crew_size": 1, "run_tag": ""},
        "build": {"build_id": "b1", "ship_type": "prospector", "heads": []},
        "rock": {"mass": 31000, "resistance": 0.43, "instability": 2.8474, "distance": 15},
        "beams": [{"slot": "main", "power_percent": 20.0, "active_modules": []}],
        "result": {"required_power": 10.0, "effective_power": 20.0, "margin": 10.0, "risk_score": 0.1, "verdict": "take", "notes": []},
        "ml_prediction": {"model_available": False},
        "outcome": {"actual_outcome": "good", "comment": ""},
        "resource_yield": {
            "primary_resource": "beryl",
            "resource_percent": 5.57,
            "raw_scu_estimate": 0.547,
            "total_scu_estimate": 25.49,
            "resources": [
                {
                    "resource_name": "beryl",
                    "resource_percent": 5.57,
                    "raw_scu_estimate": 0.547,
                    "observed_window_size": "tiny",
                    "observed_charge_behavior": "jumping",
                    "comment": "",
                }
            ],
        },
        "refinery": {"refinery_method": "unknown", "refined_resources": []},
        "calibration": {"formula_issue_flag": False, "observations": []},
        "labeling": {"label_source": "initial_save_ui", "labeled_at": "2026-05-26T00:00:00+00:00", "is_labeled": True},
    }
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event), encoding="utf-8")

    dataset = build_dataset(path)
    row = dataset.iloc[0]

    assert row["dominant_resource_window_hint"] == "tiny"
    assert row["dominant_resource_charge_behavior"] == "jumping"
    assert "beryl" in row["resources_json"]
