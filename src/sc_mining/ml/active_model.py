from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sc_mining.ml.registry import (
    ModelArtifactSpec,
    default_model_artifact_specs,
    existing_model_artifacts,
    model_source_warning,
)


ACTIVE_MODEL_CONFIG_PATH = Path("models") / "active_model.json"


@dataclass(frozen=True)
class ActiveModelSelection:
    label: str
    model_path: str
    report_path: str
    model_source: str
    safe_for_gameplay_review: bool
    selected_at: str
    model_exists: bool
    report_exists: bool
    warning: str | None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def active_selection_from_spec(
    spec: ModelArtifactSpec,
    selected_at: str | None = None,
) -> ActiveModelSelection:
    return ActiveModelSelection(
        label=spec.label,
        model_path=str(spec.model_path),
        report_path=str(spec.report_path),
        model_source=spec.model_source,
        safe_for_gameplay_review=spec.safe_for_gameplay_review,
        selected_at=selected_at or _now_iso(),
        model_exists=spec.model_path.exists(),
        report_exists=spec.report_path.exists(),
        warning=model_source_warning(spec.model_source),
    )


def active_selection_to_dict(selection: ActiveModelSelection) -> dict:
    return asdict(selection)


def write_active_model_config(
    spec: ModelArtifactSpec,
    path: str | Path = ACTIVE_MODEL_CONFIG_PATH,
) -> ActiveModelSelection:
    """Persist which trained model should be used by default for inference."""

    selection = active_selection_from_spec(spec)
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(active_selection_to_dict(selection), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return selection


def read_active_model_config(
    path: str | Path = ACTIVE_MODEL_CONFIG_PATH,
) -> ActiveModelSelection | None:
    target_path = Path(path)
    if not target_path.exists():
        return None

    payload = json.loads(target_path.read_text(encoding="utf-8"))
    return ActiveModelSelection(
        label=str(payload.get("label", "Unknown active model")),
        model_path=str(payload.get("model_path", "")),
        report_path=str(payload.get("report_path", "")),
        model_source=str(payload.get("model_source", "unknown")),
        safe_for_gameplay_review=bool(payload.get("safe_for_gameplay_review", False)),
        selected_at=str(payload.get("selected_at", "")),
        model_exists=Path(str(payload.get("model_path", ""))).exists(),
        report_exists=Path(str(payload.get("report_path", ""))).exists(),
        warning=model_source_warning(str(payload.get("model_source", "unknown"))),
    )


def clear_active_model_config(path: str | Path = ACTIVE_MODEL_CONFIG_PATH) -> bool:
    target_path = Path(path)
    if not target_path.exists():
        return False
    target_path.unlink()
    return True


def find_matching_spec(
    model_path: str | Path,
    specs: Iterable[ModelArtifactSpec] | None = None,
) -> ModelArtifactSpec | None:
    target = Path(model_path)
    for spec in list(specs or default_model_artifact_specs()):
        if spec.model_path == target:
            return spec
    return None


def build_active_model_status(
    path: str | Path = ACTIVE_MODEL_CONFIG_PATH,
    specs: Iterable[ModelArtifactSpec] | None = None,
) -> dict:
    available_specs = existing_model_artifacts(specs)
    active = read_active_model_config(path)

    if active is None:
        return {
            "configured": False,
            "valid": False,
            "reason": "No active model selected.",
            "active_model": None,
            "available_count": len(available_specs),
            "available_models": [spec.label for spec in available_specs],
        }

    model_exists = Path(active.model_path).exists()
    report_exists = Path(active.report_path).exists()

    if not model_exists:
        valid = False
        reason = f"Active model file is missing: {active.model_path}"
    else:
        valid = True
        reason = "Active model is available."

    active_payload = active_selection_to_dict(active)
    active_payload["model_exists"] = model_exists
    active_payload["report_exists"] = report_exists

    return {
        "configured": True,
        "valid": valid,
        "reason": reason,
        "active_model": active_payload,
        "available_count": len(available_specs),
        "available_models": [spec.label for spec in available_specs],
    }
