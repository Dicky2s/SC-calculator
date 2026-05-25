from pathlib import Path
from typing import Any
import json

from sc_mining.domain.models import RefineryFeedback
from sc_mining.storage.event_logger import model_to_dict, utc_now_iso
from sc_mining.storage.event_reader import read_jsonl
from sc_mining.storage.outcome_labeler import write_jsonl_records


REFINERY_LABEL_SOURCE_MANUAL_REVIEW = "manual_refinery_review_ui"


def has_refinery_result(refinery: dict | None) -> bool:
    if not refinery:
        return False

    scalar_fields = [
        "refined_scu_actual",
        "refined_value_auec",
        "refinery_fee_auec",
        "sell_value_auec",
    ]
    if any(refinery.get(field) not in (None, "", 0, 0.0) for field in scalar_fields):
        return True

    if str(refinery.get("refinery_method", "unknown") or "unknown") != "unknown":
        return True

    if str(refinery.get("refinery_location", "") or "").strip():
        return True

    if str(refinery.get("comment", "") or "").strip():
        return True

    for row in refinery.get("refined_resources") or []:
        if str(row.get("resource_name", "unknown") or "unknown") != "unknown":
            return True
        if row.get("refined_scu_actual") not in (None, "", 0, 0.0):
            return True
        if row.get("sell_value_auec") not in (None, "", 0, 0.0):
            return True

    return False


def update_event_refinery(
    path: str | Path,
    event_id: str,
    refinery: RefineryFeedback,
    label_source: str = REFINERY_LABEL_SOURCE_MANUAL_REVIEW,
) -> dict[str, Any]:
    """Update refinery/final-yield data for one existing JSONL event.

    This supports the real workflow where mining is captured immediately, but
    refinery completion, final SCU, and sell value are known later.
    """

    if not event_id:
        raise ValueError("event_id is required")

    input_path = Path(path)
    records = read_jsonl(input_path)
    if not records:
        raise FileNotFoundError(f"No event records found at {input_path}")

    target_index: int | None = None
    previous_refinery = None

    for index, record in enumerate(records):
        if str(record.get("event_id")) == str(event_id):
            target_index = index
            previous_refinery = record.get("refinery", {})
            break

    if target_index is None:
        raise ValueError(f"Event not found: {event_id}")

    now = utc_now_iso()
    target = records[target_index]
    target["refinery"] = model_to_dict(refinery)

    target["refinery_labeling"] = {
        "label_source": label_source,
        "updated_at": now,
        "has_refinery_result": has_refinery_result(target["refinery"]),
    }

    records[target_index] = target
    write_jsonl_records(input_path, records)

    return {
        "event_id": event_id,
        "previous_has_refinery_result": has_refinery_result(previous_refinery),
        "has_refinery_result": has_refinery_result(target["refinery"]),
        "label_source": label_source,
        "updated_at": now,
    }
