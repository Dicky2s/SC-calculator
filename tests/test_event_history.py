import json

from sc_mining.storage.event_history import (
    build_event_detail_payload,
    build_event_timeline,
    get_event_by_id,
    load_event_records,
)


def sample_event() -> dict:
    return {
        "event_id": "event-1",
        "session_id": "session-1",
        "timestamp": "2026-05-25T12:00:00+00:00",
        "source": "manual_ui",
        "build": {"build_id": "prospector_helix_2x_rieger_v1", "ship_type": "prospector"},
        "rock": {"mass": 20728, "resistance": 0, "instability": 0.2, "distance": 15},
        "beams": [{"slot": "main", "power_percent": 20, "active_modules": []}],
        "result": {
            "required_power": 24.044,
            "effective_power": 33.333,
            "margin": 9.289,
            "risk_score": 0.321,
            "verdict": "take",
            "notes": ["Distance efficiency: distance=15.0, factor=1.667"],
        },
        "ml_prediction": {
            "model_available": True,
            "model_source": "synthetic_smoke_test",
            "prediction": "good",
            "good_probability": 0.7058,
            "captured_at": "2026-05-25T12:00:01+00:00",
        },
        "outcome": {"actual_outcome": "good", "comment": "stable at 81%"},
        "labeling": {
            "label_source": "initial_save_ui",
            "labeled_at": "2026-05-25T12:00:02+00:00",
            "is_labeled": True,
        },
        "resource_yield": {
            "primary_resource": "copper",
            "resources": [
                {"resource_name": "copper", "resource_percent": 57, "raw_scu_estimate": 1.2, "comment": ""}
            ],
        },
        "refinery": {"refinery_method": "unknown", "refined_resources": []},
        "calibration": {
            "formula_issue_flag": True,
            "observed_min_warmup_power_percent": 78,
            "observed_stable_power_percent": 81,
            "observed_distance": 15,
            "comment": "20% did not warm up",
            "observations": [
                {"distance": 14, "power_percent": 20, "observation": "no_warmup", "beam_warmed": False, "held_stable": False, "comment": ""},
                {"distance": 15, "power_percent": 81, "observation": "stable_hold", "beam_warmed": True, "held_stable": True, "comment": ""},
            ],
        },
        "calibration_labeling": {
            "label_source": "manual_calibration_review_ui",
            "updated_at": "2026-05-25T12:00:03+00:00",
            "has_calibration_observations": True,
        },
    }


def test_load_event_records_and_get_event_by_id(tmp_path):
    path = tmp_path / "events.jsonl"
    records = [sample_event(), {"event_id": "event-2"}]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    loaded = load_event_records(path)
    found = get_event_by_id(path, "event-1")
    missing = get_event_by_id(path, "missing")

    assert len(loaded) == 2
    assert found is not None
    assert found["event_id"] == "event-1"
    assert missing is None


def test_build_event_detail_payload_keeps_nested_sections():
    payload = build_event_detail_payload(sample_event())

    assert payload["summary"]["event_id"] == "event-1"
    assert payload["summary"]["ship_type"] == "prospector"
    assert payload["rock"]["mass"] == 20728
    assert payload["result"]["verdict"] == "take"
    assert payload["resources"]["resources"][0]["resource_name"] == "copper"
    assert payload["calibration"]["observations"][1]["observation"] == "stable_hold"
    assert payload["flat"]["calibration_attempt_count"] == 2


def test_build_event_timeline_uses_available_lifecycle_timestamps():
    payload = build_event_detail_payload(sample_event())
    timeline = build_event_timeline(payload)

    stages = [row["stage"] for row in timeline]

    assert stages == [
        "event_captured",
        "ml_prediction_logged",
        "outcome_labeled",
        "calibration_updated",
    ]
