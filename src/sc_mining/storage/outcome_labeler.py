from pathlib import Path
from typing import Any
import json

from sc_mining.domain.models import OutcomeFeedback
from sc_mining.storage.event_logger import model_to_dict, utc_now_iso
from sc_mining.storage.event_reader import LABELED_OUTCOME_VALUES, read_jsonl


LABEL_SOURCE_MANUAL_REVIEW = "manual_review_ui"


def write_jsonl_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Rewrite a JSONL file with the provided records."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_labeled_outcome(actual_outcome: str | None) -> bool:
    return str(actual_outcome or "unknown") in LABELED_OUTCOME_VALUES


def update_event_outcome(
    path: str | Path,
    event_id: str,
    outcome: OutcomeFeedback,
    label_source: str = LABEL_SOURCE_MANUAL_REVIEW,
) -> dict[str, Any]:
    """Update actual_outcome/comment for one existing event in a JSONL event log.

    This is intentionally an in-place rewrite because manual labels are part of the
    same event record used later for dataset export and supervised ML training.
    """

    if not event_id:
        raise ValueError("event_id is required")

    input_path = Path(path)
    records = read_jsonl(input_path)
    if not records:
        raise FileNotFoundError(f"No event records found at {input_path}")

    target_index: int | None = None
    previous_outcome = "unknown"

    for index, record in enumerate(records):
        if str(record.get("event_id")) == str(event_id):
            target_index = index
            previous_outcome = str(
                record.get("outcome", {}).get("actual_outcome", "unknown")
            )
            break

    if target_index is None:
        raise ValueError(f"Event not found: {event_id}")

    now = utc_now_iso()
    target = records[target_index]
    target["outcome"] = model_to_dict(outcome)
    target["labeling"] = {
        "label_source": label_source,
        "labeled_at": now,
        "is_labeled": is_labeled_outcome(outcome.actual_outcome),
    }

    records[target_index] = target
    write_jsonl_records(input_path, records)

    return {
        "event_id": event_id,
        "previous_outcome": previous_outcome,
        "actual_outcome": outcome.actual_outcome,
        "is_labeled": is_labeled_outcome(outcome.actual_outcome),
        "label_source": label_source,
        "labeled_at": now,
    }
