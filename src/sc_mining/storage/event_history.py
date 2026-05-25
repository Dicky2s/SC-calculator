from __future__ import annotations

from pathlib import Path
from typing import Any

from sc_mining.storage.event_reader import flatten_event, read_jsonl


DETAIL_SECTIONS = [
    "summary",
    "rock",
    "build",
    "beams",
    "result",
    "ml_prediction",
    "outcome",
    "resources",
    "refinery",
    "calibration",
    "labeling",
]


def load_event_records(path: str | Path) -> list[dict[str, Any]]:
    """Load raw event records from JSONL in chronological append order."""

    return read_jsonl(path)


def get_event_by_id(path: str | Path, event_id: str) -> dict[str, Any] | None:
    """Return one raw event record by event_id, or None when it is absent."""

    if not event_id:
        return None

    for record in load_event_records(path):
        if str(record.get("event_id")) == str(event_id):
            return record

    return None


def build_event_detail_payload(event: dict[str, Any]) -> dict[str, Any]:
    """Build a UI-friendly detailed payload for one raw event.

    The table view intentionally flattens events for analytics. This payload keeps
    nested sections intact so Streamlit can show one event as a readable history
    card: rock, build, beams, formula result, ML snapshot, labels, resources,
    refinery and calibration data.
    """

    flat = flatten_event(event)

    return {
        "summary": {
            "event_id": flat.get("event_id"),
            "timestamp": flat.get("timestamp"),
            "session_id": flat.get("session_id"),
            "source": flat.get("source"),
            "ship_type": flat.get("ship_type"),
            "build_id": flat.get("build_id"),
            "actual_outcome": flat.get("actual_outcome"),
            "verdict": flat.get("verdict"),
        },
        "rock": event.get("rock", {}),
        "build": event.get("build", {}),
        "beams": event.get("beams", []),
        "result": event.get("result", {}),
        "ml_prediction": event.get("ml_prediction", {}),
        "outcome": event.get("outcome", {}),
        "resources": event.get("resource_yield", {}),
        "refinery": event.get("refinery", {}),
        "calibration": event.get("calibration", {}),
        "labeling": {
            "outcome": event.get("labeling", {}),
            "refinery": event.get("refinery_labeling", {}),
            "calibration": event.get("calibration_labeling", {}),
        },
        "flat": flat,
    }


def build_event_timeline(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a small chronological lifecycle view for one event.

    JSONL stores the latest state of an event, not every previous edit. This
    timeline therefore shows lifecycle timestamps that are actually present in
    the record: initial capture, outcome label update, refinery update,
    calibration update, and ML snapshot capture.
    """

    summary = payload.get("summary", {})
    labeling = payload.get("labeling", {})
    ml_prediction = payload.get("ml_prediction", {})

    rows: list[dict[str, Any]] = []

    if summary.get("timestamp"):
        rows.append(
            {
                "time": summary.get("timestamp"),
                "stage": "event_captured",
                "source": summary.get("source", ""),
                "details": f"{summary.get('ship_type', '')} / {summary.get('build_id', '')}",
            }
        )

    if ml_prediction.get("captured_at"):
        rows.append(
            {
                "time": ml_prediction.get("captured_at"),
                "stage": "ml_prediction_logged",
                "source": ml_prediction.get("model_source", ""),
                "details": f"prediction={ml_prediction.get('prediction', '')}, p_good={ml_prediction.get('good_probability', '')}",
            }
        )

    outcome_labeling = labeling.get("outcome", {}) or {}
    if outcome_labeling.get("labeled_at"):
        rows.append(
            {
                "time": outcome_labeling.get("labeled_at"),
                "stage": "outcome_labeled",
                "source": outcome_labeling.get("label_source", ""),
                "details": f"actual_outcome={payload.get('outcome', {}).get('actual_outcome', 'unknown')}",
            }
        )

    refinery_labeling = labeling.get("refinery", {}) or {}
    if refinery_labeling.get("updated_at"):
        rows.append(
            {
                "time": refinery_labeling.get("updated_at"),
                "stage": "refinery_updated",
                "source": refinery_labeling.get("label_source", ""),
                "details": f"has_refinery_result={refinery_labeling.get('has_refinery_result', False)}",
            }
        )

    calibration_labeling = labeling.get("calibration", {}) or {}
    if calibration_labeling.get("updated_at"):
        rows.append(
            {
                "time": calibration_labeling.get("updated_at"),
                "stage": "calibration_updated",
                "source": calibration_labeling.get("label_source", ""),
                "details": f"has_calibration={calibration_labeling.get('has_calibration_observations', False)}",
            }
        )

    return sorted(rows, key=lambda row: str(row.get("time", "")))
