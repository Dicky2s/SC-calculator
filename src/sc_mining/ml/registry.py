from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MODEL_SOURCE_MANUAL_REAL = "manual_real_data"
MODEL_SOURCE_SYNTHETIC = "synthetic_smoke_test"
MODEL_SOURCE_LEGACY_MANUAL = "legacy_manual_baseline"
MODEL_SOURCE_UNKNOWN = "unknown"

MANUAL_MODEL_PATH = Path("models") / "mining_outcome_baseline_manual.joblib"
MANUAL_MODEL_REPORT_PATH = Path("reports") / "baseline_model_report_manual.json"

LEGACY_MANUAL_MODEL_PATH = Path("models") / "mining_outcome_baseline.joblib"
LEGACY_MANUAL_MODEL_REPORT_PATH = Path("reports") / "baseline_model_report.json"

SYNTHETIC_MODEL_PATH = Path("models") / "mining_outcome_baseline_synthetic.joblib"
SYNTHETIC_MODEL_REPORT_PATH = Path("reports") / "baseline_model_report_synthetic.json"


@dataclass(frozen=True)
class ModelArtifactSpec:
    label: str
    model_path: Path
    report_path: Path
    model_source: str
    usage: str
    safe_for_gameplay_review: bool


def default_model_artifact_specs() -> list[ModelArtifactSpec]:
    return [
        ModelArtifactSpec(
            label="Manual real-data baseline model",
            model_path=MANUAL_MODEL_PATH,
            report_path=MANUAL_MODEL_REPORT_PATH,
            model_source=MODEL_SOURCE_MANUAL_REAL,
            usage="Train from manually labeled gameplay events. Use for gameplay review only after enough real labels are collected.",
            safe_for_gameplay_review=True,
        ),
        ModelArtifactSpec(
            label="Legacy manual baseline model",
            model_path=LEGACY_MANUAL_MODEL_PATH,
            report_path=LEGACY_MANUAL_MODEL_REPORT_PATH,
            model_source=MODEL_SOURCE_LEGACY_MANUAL,
            usage="Backward-compatible artifact from older blocks. Prefer retraining into the manual real-data path.",
            safe_for_gameplay_review=False,
        ),
        ModelArtifactSpec(
            label="Synthetic smoke-test model",
            model_path=SYNTHETIC_MODEL_PATH,
            report_path=SYNTHETIC_MODEL_REPORT_PATH,
            model_source=MODEL_SOURCE_SYNTHETIC,
            usage="Validate the ML pipeline only. Do not use as a gameplay recommendation model.",
            safe_for_gameplay_review=False,
        ),
    ]


def infer_model_source(model_path: str | Path | None) -> str:
    if model_path is None:
        return MODEL_SOURCE_UNKNOWN

    normalized = str(model_path).replace("\\", "/").lower()
    filename = Path(normalized).name

    if "synthetic" in normalized:
        return MODEL_SOURCE_SYNTHETIC
    if "manual" in normalized:
        return MODEL_SOURCE_MANUAL_REAL
    if filename == LEGACY_MANUAL_MODEL_PATH.name.lower():
        return MODEL_SOURCE_LEGACY_MANUAL
    if filename.endswith(".joblib"):
        return MODEL_SOURCE_UNKNOWN
    return MODEL_SOURCE_UNKNOWN


def model_source_warning(model_source: str) -> str | None:
    if model_source == MODEL_SOURCE_SYNTHETIC:
        return (
            "Synthetic smoke-test model. It validates the pipeline, "
            "not real gameplay decision quality."
        )
    if model_source == MODEL_SOURCE_LEGACY_MANUAL:
        return (
            "Legacy manual model artifact. Retrain into the manual real-data model path "
            "to keep model sources explicit."
        )
    if model_source == MODEL_SOURCE_UNKNOWN:
        return "Unknown model source. Treat predictions as inspection output only."
    return None


def is_gameplay_review_model(model_source: str) -> bool:
    return model_source == MODEL_SOURCE_MANUAL_REAL


def existing_model_artifacts(
    specs: Iterable[ModelArtifactSpec] | None = None,
) -> list[ModelArtifactSpec]:
    candidates = list(specs or default_model_artifact_specs())
    return [spec for spec in candidates if spec.model_path.exists()]


def spec_to_dict(spec: ModelArtifactSpec) -> dict:
    payload = asdict(spec)
    payload["model_path"] = str(spec.model_path)
    payload["report_path"] = str(spec.report_path)
    return payload
