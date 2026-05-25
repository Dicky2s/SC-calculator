from pathlib import Path
from typing import Any

from sc_mining.domain.models import CalibrationFeedback
from sc_mining.storage.event_logger import model_to_dict, utc_now_iso
from sc_mining.storage.event_reader import read_jsonl
from sc_mining.storage.outcome_labeler import write_jsonl_records


CALIBRATION_LABEL_SOURCE_MANUAL_REVIEW = "manual_calibration_review_ui"


def has_calibration_observations(calibration: dict | None) -> bool:
    if not calibration:
        return False

    if bool(calibration.get("formula_issue_flag", False)):
        return True

    scalar_fields = [
        "observed_min_warmup_power_percent",
        "observed_stable_power_percent",
        "observed_distance",
    ]
    if any(calibration.get(field) not in (None, "", 0, 0.0) for field in scalar_fields):
        return True

    if str(calibration.get("comment", "") or "").strip():
        return True

    for row in calibration.get("observations") or []:
        if row.get("distance") not in (None, "") and row.get("power_percent") not in (None, ""):
            return True
        if str(row.get("observation", "unknown") or "unknown") != "unknown":
            return True
        if str(row.get("comment", "") or "").strip():
            return True

    return False


def update_event_calibration(
    path: str | Path,
    event_id: str,
    calibration: CalibrationFeedback,
    label_source: str = CALIBRATION_LABEL_SOURCE_MANUAL_REVIEW,
) -> dict[str, Any]:
    """Update real-game power/distance calibration data for one existing event."""

    if not event_id:
        raise ValueError("event_id is required")

    input_path = Path(path)
    records = read_jsonl(input_path)
    if not records:
        raise FileNotFoundError(f"No event records found at {input_path}")

    target_index: int | None = None
    previous_calibration = None

    for index, record in enumerate(records):
        if str(record.get("event_id")) == str(event_id):
            target_index = index
            previous_calibration = record.get("calibration", {})
            break

    if target_index is None:
        raise ValueError(f"Event not found: {event_id}")

    now = utc_now_iso()
    target = records[target_index]
    target["calibration"] = model_to_dict(calibration)
    target["calibration_labeling"] = {
        "label_source": label_source,
        "updated_at": now,
        "has_calibration_observations": has_calibration_observations(target["calibration"]),
    }

    records[target_index] = target
    write_jsonl_records(input_path, records)

    return {
        "event_id": event_id,
        "previous_has_calibration_observations": has_calibration_observations(previous_calibration),
        "has_calibration_observations": has_calibration_observations(target["calibration"]),
        "label_source": label_source,
        "updated_at": now,
    }
