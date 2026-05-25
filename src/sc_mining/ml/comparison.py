from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from sc_mining.domain.models import CalculationInput, CalculationResult
from sc_mining.ml.baseline import (
    MODEL_VERSION,
    load_baseline_model,
    predict_good_probability,
)


MODEL_SOURCE_MANUAL = "manual_baseline"
MODEL_SOURCE_SYNTHETIC = "synthetic_smoke_test"
MODEL_SOURCE_UNKNOWN = "unknown"

COMPARISON_EXPORT_COLUMNS = [
    "event_id",
    "session_id",
    "ship_type",
    "build_id",
    "mass",
    "resistance",
    "instability",
    "distance",
    "margin",
    "risk_score",
    "verdict",
    "formula_expected_outcome",
    "ml_prediction",
    "ml_good_probability",
    "ml_confidence_band",
    "formula_ml_agreement",
    "actual_outcome",
    "model_source",
]


@dataclass(frozen=True)
class FormulaMlComparison:
    model_available: bool
    reason: str
    model_version: str | None
    model_path: str | None
    model_source: str
    model_warning: str | None
    formula_verdict: str
    formula_expected_outcome: str
    ml_prediction: str | None
    ml_good_probability: float | None
    confidence_band: str | None
    agreement_label: str | None
    recommendation: str | None


def infer_model_source(model_path: str | Path | None) -> str:
    """Infer whether a model artifact is a real/manual baseline or a synthetic smoke-test artifact."""

    if model_path is None:
        return MODEL_SOURCE_UNKNOWN

    normalized = str(model_path).replace("\\", "/").lower()
    if "synthetic" in normalized:
        return MODEL_SOURCE_SYNTHETIC
    if normalized.endswith(".joblib"):
        return MODEL_SOURCE_MANUAL
    return MODEL_SOURCE_UNKNOWN


def model_source_warning(model_source: str) -> str | None:
    if model_source == MODEL_SOURCE_SYNTHETIC:
        return (
            "Synthetic smoke-test model. It validates the pipeline, "
            "not real gameplay decision quality."
        )
    if model_source == MODEL_SOURCE_UNKNOWN:
        return "Unknown model source. Treat predictions as inspection output only."
    return None


def _sum_beam_power_percent(calc_input: CalculationInput) -> float:
    return round(sum(float(beam.power_percent) for beam in calc_input.beams), 3)


def build_inference_dataset_row(
    calc_input: CalculationInput,
    result: CalculationResult,
) -> pd.DataFrame:
    """Build a one-row dataframe that matches the baseline ML feature schema."""

    beam_slots = ",".join(beam.slot for beam in calc_input.beams)

    row = {
        "event_id": "inference-current",
        "session_id": "inference",
        "timestamp": "",
        "source": "manual_ui_inference",
        "build_id": calc_input.build.build_id,
        "ship_type": calc_input.build.ship_type,
        "mass": float(calc_input.rock.mass),
        "resistance": float(calc_input.rock.resistance),
        "instability": float(calc_input.rock.instability),
        "distance": float(calc_input.rock.distance),
        "beam_count": len(calc_input.beams),
        "beam_slots": beam_slots,
        "beam_power_sum": _sum_beam_power_percent(calc_input),
        "required_power": float(result.required_power),
        "effective_power": float(result.effective_power),
        "margin": float(result.margin),
        "risk_score": float(result.risk_score),
        "verdict": result.verdict,
        "actual_outcome": "unknown",
        "is_labeled": False,
        "outcome_comment": "",
    }

    return pd.DataFrame([row])


def formula_expected_outcome(verdict: str) -> str:
    """Map rule-based formula verdict to the binary baseline ML target language."""

    return "good" if verdict == "take" else "not_good"


def confidence_band(good_probability: float) -> str:
    if good_probability >= 0.75:
        return "high_good"
    if good_probability >= 0.55:
        return "weak_good"
    if good_probability <= 0.25:
        return "high_not_good"
    if good_probability <= 0.45:
        return "weak_not_good"
    return "uncertain"


def agreement_label(formula_verdict: str, ml_prediction: str) -> str:
    expected = formula_expected_outcome(formula_verdict)

    if expected == "good" and ml_prediction == "good":
        return "formula_and_ml_take"
    if expected == "good" and ml_prediction == "not_good":
        return "ml_warns_against_formula_take"
    if expected == "not_good" and ml_prediction == "good":
        return "ml_sees_possible_opportunity"
    return "formula_and_ml_avoid"


def recommendation_from_comparison(
    formula_verdict: str,
    ml_prediction: str,
    probability: float,
) -> str:
    label = agreement_label(formula_verdict, ml_prediction)
    band = confidence_band(probability)

    if label == "formula_and_ml_take":
        return "Both formula and ML lean positive. Still validate manually until the model is trained on real data."
    if label == "ml_warns_against_formula_take":
        return "Formula says take, but ML predicts not-good. Treat as a review case and check risk/margin before saving outcome."
    if label == "ml_sees_possible_opportunity":
        return "Formula is cautious, but ML predicts good. Treat as a possible missed opportunity, not an instruction."
    if band == "high_not_good":
        return "Both systems lean negative. Likely avoid or test only if the row is useful for data collection."
    return "Both systems lean negative/cautious. Use the result mainly as another labeled example."


def compare_formula_with_model(
    calc_input: CalculationInput,
    result: CalculationResult,
    model_path: str | Path,
    model_source: str | None = None,
) -> FormulaMlComparison:
    target_model_path = Path(model_path)
    resolved_model_source = model_source or infer_model_source(target_model_path)
    warning = model_source_warning(resolved_model_source)

    if not target_model_path.exists():
        return FormulaMlComparison(
            model_available=False,
            reason=f"Model file not found: {target_model_path}",
            model_version=None,
            model_path=str(target_model_path),
            model_source=resolved_model_source,
            model_warning=warning,
            formula_verdict=result.verdict,
            formula_expected_outcome=formula_expected_outcome(result.verdict),
            ml_prediction=None,
            ml_good_probability=None,
            confidence_band=None,
            agreement_label=None,
            recommendation="Train a baseline model first, or generate/train the synthetic smoke-test model.",
        )

    model = load_baseline_model(target_model_path)
    inference_row = build_inference_dataset_row(calc_input, result)
    predictions = predict_good_probability(model, inference_row)

    prediction = str(predictions.iloc[0]["ml_prediction"])
    probability = round(float(predictions.iloc[0]["ml_good_probability"]), 4)

    return FormulaMlComparison(
        model_available=True,
        reason="Model prediction completed.",
        model_version=MODEL_VERSION,
        model_path=str(target_model_path),
        model_source=resolved_model_source,
        model_warning=warning,
        formula_verdict=result.verdict,
        formula_expected_outcome=formula_expected_outcome(result.verdict),
        ml_prediction=prediction,
        ml_good_probability=probability,
        confidence_band=confidence_band(probability),
        agreement_label=agreement_label(result.verdict, prediction),
        recommendation=recommendation_from_comparison(result.verdict, prediction, probability),
    )


def cleanup_export_dataframe(dataset: pd.DataFrame) -> pd.DataFrame:
    """Remove spreadsheet/UI export artifacts such as pandas index columns."""

    if dataset.empty:
        return dataset.copy()

    output = dataset.copy()
    index_like_columns = [
        column
        for column in output.columns
        if str(column).startswith("Unnamed:") or str(column) in {"index", "level_0"}
    ]
    if index_like_columns:
        output = output.drop(columns=index_like_columns)
    return output


def comparison_actual_outcome_coverage(dataset: pd.DataFrame) -> dict[str, int | float]:
    if dataset.empty or "actual_outcome" not in dataset.columns:
        return {
            "row_count": int(len(dataset)),
            "known_outcome_count": 0,
            "unknown_outcome_count": int(len(dataset)),
            "known_outcome_ratio": 0.0,
        }

    actual = dataset["actual_outcome"].fillna("unknown").astype(str)
    unknown_count = int((actual == "unknown").sum())
    row_count = int(len(dataset))
    known_count = row_count - unknown_count
    ratio = round(known_count / row_count, 4) if row_count else 0.0

    return {
        "row_count": row_count,
        "known_outcome_count": known_count,
        "unknown_outcome_count": unknown_count,
        "known_outcome_ratio": ratio,
    }


def apply_formula_ml_comparison_to_dataset(
    dataset: pd.DataFrame,
    model: Pipeline,
    model_source: str = MODEL_SOURCE_UNKNOWN,
) -> pd.DataFrame:
    """Attach ML predictions and formula-vs-ML labels to an existing dataset."""

    cleaned = cleanup_export_dataframe(dataset)
    if cleaned.empty:
        return cleaned.copy()

    predictions = predict_good_probability(model, cleaned)
    output = predictions.copy()
    output["formula_expected_outcome"] = output["verdict"].apply(formula_expected_outcome)
    output["formula_ml_agreement"] = [
        agreement_label(str(verdict), str(prediction))
        for verdict, prediction in zip(output["verdict"], output["ml_prediction"])
    ]
    output["ml_confidence_band"] = output["ml_good_probability"].apply(confidence_band)
    output["model_source"] = model_source
    return output


def build_comparison_export_dataframe(compared: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, clean dataframe for CSV download/export."""

    cleaned = cleanup_export_dataframe(compared)
    existing_columns = [column for column in COMPARISON_EXPORT_COLUMNS if column in cleaned.columns]
    return cleaned[existing_columns].copy()


def comparison_export_csv(compared: pd.DataFrame) -> str:
    """Serialize comparison rows without pandas index columns."""

    export_frame = build_comparison_export_dataframe(compared)
    return export_frame.to_csv(index=False)


def comparison_to_dict(comparison: FormulaMlComparison) -> dict[str, Any]:
    return asdict(comparison)
