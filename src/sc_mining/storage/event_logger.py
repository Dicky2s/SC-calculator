from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import json

from sc_mining.domain.models import CalculationInput, CalculationResult


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_to_dict(model: Any) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    if hasattr(model, "dict"):
        return model.dict()

    raise TypeError(f"Object is not a Pydantic model: {type(model)}")


def build_calculation_event(
    session_id: str,
    calc_input: CalculationInput,
    result: CalculationResult,
    source: str = "manual_ui",
) -> dict:
    return {
        "event_id": str(uuid4()),
        "session_id": session_id,
        "timestamp": utc_now_iso(),
        "source": source,
        "build": {
            "build_id": calc_input.build.build_id,
            "ship_type": calc_input.build.ship_type,
            "heads": [model_to_dict(head) for head in calc_input.build.heads],
        },
        "rock": model_to_dict(calc_input.rock),
        "beams": [model_to_dict(beam) for beam in calc_input.beams],
        "result": model_to_dict(result),
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
) -> dict:
    event = build_calculation_event(
        session_id=session_id,
        calc_input=calc_input,
        result=result,
        source=source,
    )

    append_jsonl(path, event)

    return event
