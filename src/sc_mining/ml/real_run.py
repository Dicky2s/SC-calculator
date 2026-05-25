from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sc_mining.dataset.exporter import export_dataset
from sc_mining.dataset.quality import build_quality_report
from sc_mining.ml.active_model import ACTIVE_MODEL_CONFIG_PATH, write_active_model_config
from sc_mining.ml.baseline import (
    DEFAULT_TEST_SIZE,
    MIN_LABELED_ROWS_FOR_TRAINING,
    check_training_readiness,
    result_to_dict,
    train_baseline_model,
)
from sc_mining.ml.prediction_evaluation import build_prediction_evaluation_summary
from sc_mining.ml.promotion import PromotionCriteria, evaluate_model_promotion, promotion_decision_to_dict
from sc_mining.ml.registry import (
    MANUAL_MODEL_PATH,
    MANUAL_MODEL_REPORT_PATH,
    MODEL_SOURCE_MANUAL_REAL,
    ModelArtifactSpec,
)
from sc_mining.ml.tracking import TRAINING_RUNS_PATH, append_training_run
from sc_mining.storage.event_reader import load_events_dataframe


DEFAULT_EVENTS_PATH = Path("data") / "sessions" / "manual_events.jsonl"
DEFAULT_DATASET_PATH = Path("data") / "datasets" / "mining_events.csv"


@dataclass(frozen=True)
class RealMLRunConfig:
    """Configuration for one real/manual ML pipeline run.

    The run intentionally targets the manual real-data model path. Synthetic models are
    handled separately and should not be promoted as gameplay-review candidates.
    """

    events_path: Path = DEFAULT_EVENTS_PATH
    dataset_path: Path = DEFAULT_DATASET_PATH
    model_path: Path = MANUAL_MODEL_PATH
    report_path: Path = MANUAL_MODEL_REPORT_PATH
    training_runs_path: Path = TRAINING_RUNS_PATH
    active_model_path: Path = ACTIVE_MODEL_CONFIG_PATH
    min_labeled_rows: int = MIN_LABELED_ROWS_FOR_TRAINING
    min_test_rows: int = 5
    min_accuracy: float = 0.60
    max_false_good_rate: float = 0.25
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = 42
    train_if_ready: bool = True
    promote_if_passed: bool = False
    notes: str = "real ML run"


@dataclass(frozen=True)
class RealMLRunResult:
    status: str
    dataset_path: str
    model_path: str
    report_path: str
    active_model_path: str
    exported_rows: int
    labeled_rows: int
    unknown_rows: int
    quality_status: str
    quality_issue_count: int
    training_ready: bool
    training_reason: str
    trained: bool
    training_result: dict[str, Any] | None = None
    training_run_id: str | None = None
    promotion_status: str | None = None
    can_promote: bool = False
    promoted: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_issues: list[dict[str, Any]] = field(default_factory=list)
    prediction_evaluation_summary: dict[str, Any] = field(default_factory=dict)
    promotion_decision: dict[str, Any] | None = None


def _status_from_result(
    *,
    trained: bool,
    training_ready: bool,
    promoted: bool,
    promotion_status: str | None,
    quality_status: str,
) -> str:
    if promoted:
        return "promoted"
    if trained:
        return "trained"
    if not training_ready:
        return "not_ready"
    if quality_status == "fail":
        return "quality_failed"
    if promotion_status == "fail":
        return "trained_or_existing_model_failed_gate"
    return "ready"


def _manual_spec(model_path: str | Path, report_path: str | Path) -> ModelArtifactSpec:
    return ModelArtifactSpec(
        label="Manual real-data baseline model",
        model_path=Path(model_path),
        report_path=Path(report_path),
        model_source=MODEL_SOURCE_MANUAL_REAL,
        usage="Train from manually labeled gameplay events. Use for gameplay review only after promotion checks pass.",
        safe_for_gameplay_review=True,
    )


def run_real_ml_pipeline(config: RealMLRunConfig | None = None) -> RealMLRunResult:
    """Run the manual real-data ML workflow end-to-end.

    Steps:
    1. Export raw event JSONL into a flat CSV dataset.
    2. Build data-quality and training-readiness reports.
    3. Train the manual baseline model when the dataset is ready and training is enabled.
    4. Append a training-run record.
    5. Evaluate the promotion gate for the manual model.
    6. Optionally promote the manual model to active_model.json if the gate passes.
    """

    config = config or RealMLRunConfig()

    dataset = export_dataset(
        events_path=config.events_path,
        output_path=config.dataset_path,
        labeled_only=False,
    )
    events = load_events_dataframe(config.events_path)

    quality_report = build_quality_report(
        dataset,
        min_labeled_rows=config.min_labeled_rows,
    )
    readiness = check_training_readiness(
        dataset,
        min_labeled_rows=config.min_labeled_rows,
    )
    prediction_eval_summary = build_prediction_evaluation_summary(events)

    training_result_payload: dict[str, Any] | None = None
    training_run_id: str | None = None
    trained = False
    reasons: list[str] = []
    warnings: list[str] = []

    if config.train_if_ready:
        if readiness.ready:
            training_result = train_baseline_model(
                dataset=dataset,
                model_path=config.model_path,
                report_path=config.report_path,
                min_labeled_rows=config.min_labeled_rows,
                test_size=config.test_size,
                random_state=config.random_state,
                model_source=MODEL_SOURCE_MANUAL_REAL,
            )
            trained = True
            training_result_payload = result_to_dict(training_result)
            run_record = append_training_run(
                training_result=training_result,
                path=config.training_runs_path,
                notes=config.notes,
            )
            training_run_id = run_record.run_id
        else:
            reasons.append(readiness.reason)
    else:
        warnings.append("Training was disabled for this run; only export/readiness/promotion checks were executed.")

    criteria = PromotionCriteria(
        min_rows_used=config.min_labeled_rows,
        min_test_rows=config.min_test_rows,
        min_accuracy=config.min_accuracy,
        max_false_good_rate=config.max_false_good_rate,
    )
    promotion_decision = evaluate_model_promotion(
        model_path=config.model_path,
        report_path=config.report_path,
        criteria=criteria,
        prediction_evaluation_summary=prediction_eval_summary,
    )

    reasons.extend(promotion_decision.reasons)
    warnings.extend(promotion_decision.warnings)

    promoted = False
    if config.promote_if_passed and promotion_decision.can_promote:
        write_active_model_config(_manual_spec(config.model_path, config.report_path), config.active_model_path)
        promoted = True
    elif config.promote_if_passed and not promotion_decision.can_promote:
        warnings.append("Promotion was requested, but the promotion gate did not pass.")

    status = _status_from_result(
        trained=trained,
        training_ready=readiness.ready,
        promoted=promoted,
        promotion_status=promotion_decision.status,
        quality_status=str(quality_report.get("status", "unknown")),
    )

    return RealMLRunResult(
        status=status,
        dataset_path=str(config.dataset_path),
        model_path=str(config.model_path),
        report_path=str(config.report_path),
        active_model_path=str(config.active_model_path),
        exported_rows=int(len(dataset)),
        labeled_rows=int(quality_report.get("labeled_count", 0)),
        unknown_rows=int(quality_report.get("unknown_outcome_count", 0)),
        quality_status=str(quality_report.get("status", "unknown")),
        quality_issue_count=int(len(quality_report.get("issues", []))),
        training_ready=readiness.ready,
        training_reason=readiness.reason,
        trained=trained,
        training_result=training_result_payload,
        training_run_id=training_run_id,
        promotion_status=promotion_decision.status,
        can_promote=promotion_decision.can_promote,
        promoted=promoted,
        reasons=reasons,
        warnings=warnings,
        quality_issues=list(quality_report.get("issues", [])),
        prediction_evaluation_summary=prediction_eval_summary,
        promotion_decision=promotion_decision_to_dict(promotion_decision),
    )


def real_ml_run_result_to_dict(result: RealMLRunResult) -> dict[str, Any]:
    return asdict(result)


def result_to_dataframe(result: RealMLRunResult) -> pd.DataFrame:
    """Return a compact two-column view for UI/debug output."""

    payload = real_ml_run_result_to_dict(result)
    compact_keys = [
        "status",
        "exported_rows",
        "labeled_rows",
        "unknown_rows",
        "quality_status",
        "training_ready",
        "training_reason",
        "trained",
        "promotion_status",
        "can_promote",
        "promoted",
        "dataset_path",
        "model_path",
        "report_path",
        "active_model_path",
    ]
    return pd.DataFrame(
        [{"metric": key, "value": payload.get(key)} for key in compact_keys]
    )
