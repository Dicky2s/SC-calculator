import json

import pandas as pd

from sc_mining.dataset.exporter import (
    DATASET_COLUMNS,
    build_dataset,
    export_dataset,
    get_dataset_export_summary,
)


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )


def sample_event(event_id: str, verdict: str, actual_outcome: str) -> dict:
    return {
        "event_id": event_id,
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
            "verdict": verdict,
        },
        "outcome": {
            "actual_outcome": actual_outcome,
            "comment": "manual label",
        },
    }


def test_build_dataset_returns_expected_columns(tmp_path):
    events_path = tmp_path / "events.jsonl"
    write_jsonl(events_path, [sample_event("event-1", "take", "good")])

    dataset = build_dataset(events_path)

    assert list(dataset.columns) == DATASET_COLUMNS
    assert len(dataset) == 1
    assert dataset.iloc[0]["event_id"] == "event-1"
    assert dataset.iloc[0]["actual_outcome"] == "good"
    assert bool(dataset.iloc[0]["is_labeled"]) is True


def test_build_dataset_can_filter_labeled_only(tmp_path):
    events_path = tmp_path / "events.jsonl"
    write_jsonl(
        events_path,
        [
            sample_event("event-1", "take", "unknown"),
            sample_event("event-2", "take", "good"),
            sample_event("event-3", "risky", "too_unstable"),
        ],
    )

    dataset = build_dataset(events_path, labeled_only=True)

    assert len(dataset) == 2
    assert set(dataset["event_id"]) == {"event-2", "event-3"}
    assert set(dataset["actual_outcome"]) == {"good", "too_unstable"}


def test_export_dataset_writes_csv(tmp_path):
    events_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "datasets" / "mining_events.csv"
    write_jsonl(events_path, [sample_event("event-1", "take", "good")])

    exported = export_dataset(events_path, output_path)
    loaded = pd.read_csv(output_path)

    assert output_path.exists()
    assert len(exported) == 1
    assert len(loaded) == 1
    assert loaded.iloc[0]["event_id"] == "event-1"
    assert loaded.iloc[0]["mass"] == 12600


def test_get_dataset_export_summary_counts_rows(tmp_path):
    events_path = tmp_path / "events.jsonl"
    write_jsonl(
        events_path,
        [
            sample_event("event-1", "take", "unknown"),
            sample_event("event-2", "take", "good"),
            sample_event("event-3", "need_more_power", "not_enough_power"),
        ],
    )

    dataset = build_dataset(events_path)
    summary = get_dataset_export_summary(dataset)

    assert summary["row_count"] == 3
    assert summary["labeled_count"] == 2
    assert summary["unlabeled_count"] == 1
    assert summary["verdict_distribution"] == {"take": 2, "need_more_power": 1}
    assert summary["actual_outcome_distribution"] == {
        "unknown": 1,
        "good": 1,
        "not_enough_power": 1,
    }


def test_build_dataset_returns_empty_schema_for_missing_file(tmp_path):
    dataset = build_dataset(tmp_path / "missing.jsonl")

    assert dataset.empty
    assert list(dataset.columns) == DATASET_COLUMNS
