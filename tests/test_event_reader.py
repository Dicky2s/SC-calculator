import json

from sc_mining.storage.event_reader import (
    flatten_event,
    get_events_summary,
    load_events_dataframe,
    read_jsonl,
)


def test_read_jsonl_returns_records(tmp_path):
    path = tmp_path / "events.jsonl"

    records = [
        {"event_id": "1", "session_id": "s1"},
        {"event_id": "2", "session_id": "s1"},
    ]

    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    loaded = read_jsonl(path)

    assert len(loaded) == 2
    assert loaded[0]["event_id"] == "1"
    assert loaded[1]["event_id"] == "2"


def test_load_events_dataframe_flattens_event(tmp_path):
    path = tmp_path / "events.jsonl"

    event = {
        "event_id": "event-1",
        "session_id": "session-1",
        "timestamp": "2026-05-25T12:00:00+00:00",
        "source": "manual_ui",
        "build": {
            "build_id": "prospector_helix_rieger_focus_v1",
            "ship_type": "prospector",
        },
        "rock": {
            "mass": 12600,
            "resistance": 0.34,
            "instability": 0.12,
            "distance": 92,
        },
        "beams": [
            {
                "slot": "main",
                "power_percent": 65,
                "active_modules": [],
            }
        ],
        "result": {
            "required_power": 22.4,
            "effective_power": 77.2,
            "margin": 54.8,
            "risk_score": 0.1,
            "verdict": "take",
        },
        "outcome": {
            "actual_outcome": "good",
            "comment": "fractured fine",
        },
    }

    path.write_text(json.dumps(event), encoding="utf-8")

    df = load_events_dataframe(path)

    assert len(df) == 1
    assert df.iloc[0]["event_id"] == "event-1"
    assert df.iloc[0]["build_id"] == "prospector_helix_rieger_focus_v1"
    assert df.iloc[0]["ship_type"] == "prospector"
    assert df.iloc[0]["mass"] == 12600
    assert df.iloc[0]["beam_count"] == 1
    assert df.iloc[0]["beam_slots"] == "main"
    assert df.iloc[0]["verdict"] == "take"
    assert df.iloc[0]["actual_outcome"] == "good"
    assert df.iloc[0]["outcome_comment"] == "fractured fine"


def test_get_events_summary_for_empty_dataframe(tmp_path):
    path = tmp_path / "missing.jsonl"

    df = load_events_dataframe(path)
    summary = get_events_summary(df)

    assert summary["event_count"] == 0
    assert summary["labeled_event_count"] == 0
    assert summary["unlabeled_event_count"] == 0
    assert summary["session_count"] == 0
    assert summary["build_count"] == 0
    assert summary["ship_count"] == 0


def test_get_events_summary_counts_labeled_and_unlabeled_events(tmp_path):
    path = tmp_path / "events.jsonl"

    records = [
        {
            "event_id": "event-1",
            "session_id": "s1",
            "build": {"build_id": "b1", "ship_type": "prospector"},
            "outcome": {"actual_outcome": "unknown", "comment": ""},
        },
        {
            "event_id": "event-2",
            "session_id": "s1",
            "build": {"build_id": "b1", "ship_type": "prospector"},
            "outcome": {"actual_outcome": "good", "comment": ""},
        },
        {
            "event_id": "event-3",
            "session_id": "s2",
            "build": {"build_id": "b2", "ship_type": "mole"},
            "outcome": {"actual_outcome": "too_unstable", "comment": ""},
        },
    ]

    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    summary = get_events_summary(load_events_dataframe(path))

    assert summary["event_count"] == 3
    assert summary["labeled_event_count"] == 2
    assert summary["unlabeled_event_count"] == 1
    assert summary["session_count"] == 2
    assert summary["build_count"] == 2
    assert summary["ship_count"] == 2


def test_flatten_event_handles_missing_fields():
    row = flatten_event({"event_id": "event-1"})

    assert row["event_id"] == "event-1"
    assert row["beam_count"] == 0
    assert row["beam_slots"] == ""
    assert row["beam_power_sum"] == 0
    assert row["actual_outcome"] == "unknown"
    assert row["outcome_comment"] == ""


def test_load_events_dataframe_flattens_ml_prediction_snapshot(tmp_path):
    path = tmp_path / "events.jsonl"

    event = {
        "event_id": "event-ml-1",
        "session_id": "session-1",
        "timestamp": "2026-05-25T12:00:00+00:00",
        "source": "manual_ui",
        "build": {"build_id": "b1", "ship_type": "prospector"},
        "rock": {"mass": 10000, "resistance": 0.2, "instability": 0.1, "distance": 90},
        "beams": [{"slot": "main", "power_percent": 70}],
        "result": {
            "required_power": 20,
            "effective_power": 30,
            "margin": 10,
            "risk_score": 0.2,
            "verdict": "take",
        },
        "ml_prediction": {
            "model_available": True,
            "model_version": "baseline_rf_v1",
            "model_path": "models/model.joblib",
            "model_source": "manual_real_data",
            "formula_expected_outcome": "good",
            "prediction": "not_good",
            "good_probability": 0.33,
            "confidence_band": "weak_not_good",
            "agreement_label": "ml_warns_against_formula_take",
            "captured_at": "2026-05-25T12:00:01+00:00",
        },
        "outcome": {"actual_outcome": "unknown", "comment": ""},
    }
    path.write_text(json.dumps(event), encoding="utf-8")

    row = load_events_dataframe(path).iloc[0]

    assert bool(row["ml_model_available"]) is True
    assert row["ml_model_version"] == "baseline_rf_v1"
    assert row["ml_model_source"] == "manual_real_data"
    assert row["ml_formula_expected_outcome"] == "good"
    assert row["ml_prediction"] == "not_good"
    assert row["ml_good_probability"] == 0.33
    assert row["ml_confidence_band"] == "weak_not_good"
    assert row["ml_agreement_label"] == "ml_warns_against_formula_take"
