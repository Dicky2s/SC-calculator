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


def test_get_events_summary_for_empty_dataframe(tmp_path):
    path = tmp_path / "missing.jsonl"

    df = load_events_dataframe(path)
    summary = get_events_summary(df)

    assert summary["event_count"] == 0
    assert summary["session_count"] == 0
    assert summary["build_count"] == 0
    assert summary["ship_count"] == 0


def test_flatten_event_handles_missing_fields():
    row = flatten_event({"event_id": "event-1"})

    assert row["event_id"] == "event-1"
    assert row["beam_count"] == 0
    assert row["beam_slots"] == ""
    assert row["beam_power_sum"] == 0