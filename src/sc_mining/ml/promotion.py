from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sc_mining.ml.registry import MODEL_SOURCE_MANUAL_REAL


@dataclass(frozen=True)
class PromotionCriteria:
    """Minimum checks before a model can become the active gameplay-review model."""

    min_rows_used: int = 30
    min_test_rows: int = 5
    min_accuracy: float = 0.60
    required_model_source: str = MODEL_SOURCE_MANUAL_REAL
    require_two_binary_classes: bool = True
    max_false_good_rate: float | None = 0.25
    min_evaluable_predictions: int = 0


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    can_promote: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_path: str = ""
    report_path: str = ""


def load_training_report(report_path: str | Path) -> dict[str, Any]:
    target_path = Path(report_path)
    if not target_path.exists():
        return {}
    return json.loads(target_path.read_text(encoding="utf-8"))


def _binary_target_class_count(target_distribution: dict[str, Any]) -> int:
    return sum(1 for value in target_distribution.values() if int(value) > 0)


def _false_good_rate(prediction_evaluation_summary: dict[str, Any] | None) -> float | None:
    if not prediction_evaluation_summary:
        return None

    evaluable = int(prediction_evaluation_summary.get("evaluable_prediction_count", 0) or 0)
    if evaluable <= 0:
        return None

    false_good = int(prediction_evaluation_summary.get("false_good_count", 0) or 0)
    return false_good / evaluable


def evaluate_model_promotion(
    model_path: str | Path,
    report_path: str | Path,
    criteria: PromotionCriteria | None = None,
    prediction_evaluation_summary: dict[str, Any] | None = None,
) -> PromotionDecision:
    """Evaluate whether a trained model should be promoted to the active model pointer.

    The gate is intentionally conservative. It does not prove that the model is good;
    it prevents obvious mistakes such as promoting a synthetic model, a one-class model,
    or a model trained on too few labeled rows.
    """

    criteria = criteria or PromotionCriteria()
    target_model_path = Path(model_path)
    target_report_path = Path(report_path)

    reasons: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {
        "model_exists": target_model_path.exists(),
        "report_exists": target_report_path.exists(),
        "criteria": asdict(criteria),
    }

    if not target_model_path.exists():
        reasons.append(f"Model artifact is missing: {target_model_path}")

    if not target_report_path.exists():
        reasons.append(f"Training report is missing: {target_report_path}")
        return PromotionDecision(
            status="fail",
            can_promote=False,
            reasons=reasons,
            warnings=warnings,
            metrics=metrics,
            model_path=str(target_model_path),
            report_path=str(target_report_path),
        )

    try:
        report = load_training_report(target_report_path)
    except json.JSONDecodeError as exc:
        reasons.append(f"Training report is not valid JSON: {exc}")
        return PromotionDecision(
            status="fail",
            can_promote=False,
            reasons=reasons,
            warnings=warnings,
            metrics=metrics,
            model_path=str(target_model_path),
            report_path=str(target_report_path),
        )

    model_source = str(report.get("model_source", "unknown"))
    rows_used = int(report.get("rows_used", 0) or 0)
    test_rows = int(report.get("test_rows", 0) or 0)
    accuracy = float(report.get("accuracy", 0.0) or 0.0)
    target_distribution = dict(report.get("target_distribution", {}) or {})
    binary_class_count = _binary_target_class_count(target_distribution)
    false_good_rate = _false_good_rate(prediction_evaluation_summary)

    metrics.update(
        {
            "model_source": model_source,
            "rows_used": rows_used,
            "test_rows": test_rows,
            "accuracy": round(accuracy, 4),
            "target_distribution": target_distribution,
            "binary_target_class_count": binary_class_count,
            "false_good_rate": None if false_good_rate is None else round(false_good_rate, 4),
        }
    )

    if model_source != criteria.required_model_source:
        reasons.append(
            f"Model source is {model_source}, expected {criteria.required_model_source}."
        )

    if rows_used < criteria.min_rows_used:
        reasons.append(
            f"Only {rows_used} rows used for training; required at least {criteria.min_rows_used}."
        )

    if test_rows < criteria.min_test_rows:
        reasons.append(
            f"Only {test_rows} test rows; required at least {criteria.min_test_rows}."
        )

    if accuracy < criteria.min_accuracy:
        reasons.append(
            f"Accuracy {accuracy:.4f} is below required {criteria.min_accuracy:.4f}."
        )

    if criteria.require_two_binary_classes and binary_class_count < 2:
        reasons.append("Training target has fewer than two binary classes.")

    if prediction_evaluation_summary:
        evaluable = int(prediction_evaluation_summary.get("evaluable_prediction_count", 0) or 0)
        metrics["evaluable_predictions"] = evaluable
        if criteria.min_evaluable_predictions and evaluable < criteria.min_evaluable_predictions:
            warnings.append(
                f"Only {evaluable} historical predictions are evaluable; target is {criteria.min_evaluable_predictions}."
            )

        if false_good_rate is not None and criteria.max_false_good_rate is not None:
            if false_good_rate > criteria.max_false_good_rate:
                warnings.append(
                    f"False-good rate {false_good_rate:.4f} is above review threshold {criteria.max_false_good_rate:.4f}."
                )

    if reasons:
        status = "fail"
        can_promote = False
    elif warnings:
        status = "warn"
        can_promote = True
    else:
        status = "pass"
        can_promote = True

    return PromotionDecision(
        status=status,
        can_promote=can_promote,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        model_path=str(target_model_path),
        report_path=str(target_report_path),
    )


def promotion_decision_to_dict(decision: PromotionDecision) -> dict[str, Any]:
    return asdict(decision)
