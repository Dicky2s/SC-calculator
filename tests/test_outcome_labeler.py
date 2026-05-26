import json

import pytest

from sc_mining.domain.models import OutcomeFeedback
from sc_mining.storage.event_reader import load_events_dataframe
from sc_mining.storage.outcome_labeler import (
    LABEL_SOURCE_MANUAL_REVIEW,
    is_labeled_outcome,
    update_event_outcome,
)


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )


def sample_event(event_id: str, actual_outcome: str = "unknown") -> dict:
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
            "verdict": "take",
        },
        "outcome": {
            "actual_outcome": actual_outcome,
            "comment": "",
        },
    }


def test_is_labeled_outcome_detects_unknown_and_known_values():
    assert is_labeled_outcome("unknown") is False
    assert is_labeled_outcome(None) is False
    assert is_labeled_outcome("good") is True
    assert is_labeled_outcome("too_unstable") is True


def test_update_event_outcome_updates_matching_jsonl_record(tmp_path):
    events_path = tmp_path / "manual_events.jsonl"
    write_jsonl(
        events_path,
        [
            sample_event("event-1", "unknown"),
            sample_event("event-2", "unknown"),
        ],
    )

    result = update_event_outcome(
        path=events_path,
        event_id="event-2",
        outcome=OutcomeFeedback(
            actual_outcome="too_unstable",
            comment="jumped too much above 70%",
        ),
    )

    records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]

    assert result["event_id"] == "event-2"
    assert result["previous_outcome"] == "unknown"
    assert result["actual_outcome"] == "too_unstable"
    assert result["is_labeled"] is True
    assert result["label_source"] == LABEL_SOURCE_MANUAL_REVIEW
    assert records[0]["outcome"]["actual_outcome"] == "unknown"
    assert records[1]["outcome"]["actual_outcome"] == "too_unstable"
    assert records[1]["outcome"]["comment"] == "jumped too much above 70%"
    assert records[1]["labeling"]["label_source"] == LABEL_SOURCE_MANUAL_REVIEW
    assert records[1]["labeling"]["is_labeled"] is True
    assert records[1]["labeling"]["labeled_at"]


def test_update_event_outcome_metadata_is_visible_in_event_dataframe(tmp_path):
    events_path = tmp_path / "manual_events.jsonl"
    write_jsonl(events_path, [sample_event("event-1", "unknown")])

    update_event_outcome(
        path=events_path,
        event_id="event-1",
        outcome=OutcomeFeedback(actual_outcome="good", comment="worth taking"),
    )

    df = load_events_dataframe(events_path)

    assert len(df) == 1
    assert df.iloc[0]["actual_outcome"] == "good"
    assert df.iloc[0]["outcome_comment"] == "worth taking"
    assert df.iloc[0]["label_source"] == LABEL_SOURCE_MANUAL_REVIEW
    assert df.iloc[0]["labeled_at"]


def test_update_event_outcome_raises_for_missing_event_id(tmp_path):
    events_path = tmp_path / "manual_events.jsonl"
    write_jsonl(events_path, [sample_event("event-1", "unknown")])

    with pytest.raises(ValueError, match="Event not found"):
        update_event_outcome(
            path=events_path,
            event_id="missing-event",
            outcome=OutcomeFeedback(actual_outcome="bad", comment="not worth it"),
        )


def test_update_event_outcome_raises_for_empty_log(tmp_path):
    with pytest.raises(FileNotFoundError, match="No event records"):
        update_event_outcome(
            path=tmp_path / "missing.jsonl",
            event_id="event-1",
            outcome=OutcomeFeedback(actual_outcome="bad", comment="not worth it"),
        )
