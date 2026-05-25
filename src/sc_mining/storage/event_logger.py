from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import json

from sc_mining.domain.models import (
    CalculationInput,
    CalculationResult,
    OutcomeFeedback,
    ResourceYieldFeedback,
    RefineryFeedback,
    CalibrationFeedback,
    RunContext,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    if hasattr(model, "dict"):
        return model.dict()

    raise TypeError(f"Object is not a Pydantic model: {type(model)}")


def normalize_ml_prediction_snapshot(snapshot: dict | None, captured_at: str) -> dict:
    """Normalize formula-vs-ML comparison payload before storing it in an event."""

    if not snapshot:
        return {
            "model_available": False,
            "model_version": "",
            "model_path": "",
            "model_source": "",
            "formula_expected_outcome": "",
            "prediction": "",
            "good_probability": None,
            "confidence_band": "",
            "agreement_label": "",
            "recommendation": "",
            "captured_at": "",
        }

    return {
        "model_available": bool(snapshot.get("model_available", False)),
        "model_version": snapshot.get("model_version") or "",
        "model_path": snapshot.get("model_path") or "",
        "model_source": snapshot.get("model_source") or "",
        "formula_expected_outcome": snapshot.get("formula_expected_outcome") or "",
        "prediction": snapshot.get("ml_prediction") or snapshot.get("prediction") or "",
        "good_probability": snapshot.get("ml_good_probability", snapshot.get("good_probability")),
        "confidence_band": snapshot.get("confidence_band") or snapshot.get("ml_confidence_band") or "",
        "agreement_label": snapshot.get("agreement_label") or snapshot.get("formula_ml_agreement") or "",
        "recommendation": snapshot.get("recommendation") or "",
        "captured_at": captured_at,
    }


def build_calculation_event(
    session_id: str,
    calc_input: CalculationInput,
    result: CalculationResult,
    source: str = "manual_ui",
    outcome: OutcomeFeedback | None = None,
    ml_prediction_snapshot: dict | None = None,
    resource_yield: ResourceYieldFeedback | None = None,
    refinery: RefineryFeedback | None = None,
    calibration: CalibrationFeedback | None = None,
    run_context: RunContext | None = None,
) -> dict:
    outcome_feedback = outcome or OutcomeFeedback()
    resource_feedback = resource_yield or ResourceYieldFeedback()
    refinery_feedback = refinery or RefineryFeedback()
    calibration_feedback = calibration or CalibrationFeedback()
    context = run_context or RunContext()
    event_timestamp = utc_now_iso()
    is_labeled = outcome_feedback.actual_outcome != "unknown"

    ml_prediction = normalize_ml_prediction_snapshot(ml_prediction_snapshot, event_timestamp)

    return {
        "event_id": str(uuid4()),
        "session_id": session_id,
        "timestamp": event_timestamp,
        "source": source,
        "run_context": model_to_dict(context),
        "build": {
            "build_id": calc_input.build.build_id,
            "ship_type": calc_input.build.ship_type,
            "heads": [model_to_dict(head) for head in calc_input.build.heads],
        },
        "rock": model_to_dict(calc_input.rock),
        "beams": [model_to_dict(beam) for beam in calc_input.beams],
        "result": model_to_dict(result),
        "ml_prediction": ml_prediction,
        "outcome": model_to_dict(outcome_feedback),
        "resource_yield": model_to_dict(resource_feedback),
        "refinery": model_to_dict(refinery_feedback),
        "calibration": model_to_dict(calibration_feedback),
        "labeling": {
            "label_source": "initial_save_ui" if is_labeled else "",
            "labeled_at": event_timestamp if is_labeled else "",
            "is_labeled": is_labeled,
        },
    }


def append_jsonl(path: str | Path, record: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_calculation_event(
    path: str | Path,
    session_id: str,
    calc_input: CalculationInput,
    result: CalculationResult,
    source: str = "manual_ui",
    outcome: OutcomeFeedback | None = None,
    ml_prediction_snapshot: dict | None = None,
    resource_yield: ResourceYieldFeedback | None = None,
    refinery: RefineryFeedback | None = None,
    calibration: CalibrationFeedback | None = None,
    run_context: RunContext | None = None,
) -> dict:
    event = build_calculation_event(
        session_id=session_id,
        calc_input=calc_input,
        result=result,
        source=source,
        outcome=outcome,
        ml_prediction_snapshot=ml_prediction_snapshot,
        resource_yield=resource_yield,
        refinery=refinery,
        calibration=calibration,
        run_context=run_context,
    )

    append_jsonl(path, event)

    return event
