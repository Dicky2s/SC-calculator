from __future__ import annotations

from typing import Any

import pandas as pd

from sc_mining.dataset.exporter import NUMERIC_COLUMNS
from sc_mining.storage.event_reader import LABELED_OUTCOME_VALUES


GOOD_OUTCOME_VALUES = {"good"}
NOT_GOOD_OUTCOME_VALUES = LABELED_OUTCOME_VALUES - GOOD_OUTCOME_VALUES
AVOID_VERDICTS = {"skip", "need_more_power"}
TAKE_VERDICTS = {"take"}
RISKY_VERDICTS = {"risky"}


ANALYTICS_COLUMNS = [
    "event_id",
    "session_id",
    "timestamp",
    "build_id",
    "ship_type",
    "mass",
    "resistance",
    "instability",
    "distance",
    "beam_power_sum",
    "required_power",
    "effective_power",
    "margin",
    "risk_score",
    "verdict",
    "actual_outcome",
    "formula_decision",
    "outcome_quality",
    "analytics_label",
    "outcome_comment",
]


FEATURE_SIGNAL_COLUMNS = [
    "feature",
    "mean_good",
    "mean_not_good",
    "difference_good_minus_not_good",
    "correlation_with_good",
]


OUTCOME_NUMERIC_SUMMARY_COLUMNS = [
    "actual_outcome",
    "row_count",
    "feature",
    "mean",
    "median",
    "min",
    "max",
]


FORMULA_OUTCOME_MATRIX_COLUMNS = ["verdict", "actual_outcome", "count"]


def _normalize_outcome(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str)


def _normalize_verdict(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str)


def _formula_decision(verdict: str) -> str:
    if verdict in TAKE_VERDICTS:
        return "take"
    if verdict in AVOID_VERDICTS:
        return "avoid"
    if verdict in RISKY_VERDICTS:
        return "caution"
    return "unknown"


def _outcome_quality(actual_outcome: str) -> str:
    if actual_outcome in GOOD_OUTCOME_VALUES:
        return "good"
    if actual_outcome in NOT_GOOD_OUTCOME_VALUES:
        return "not_good"
    return "unknown"


def _analytics_label(row: pd.Series) -> str:
    decision = row["formula_decision"]
    quality = row["outcome_quality"]

    if decision == "take" and quality == "good":
        return "correct_take"
    if decision == "take" and quality == "not_good":
        return "dangerous_take"
    if decision == "avoid" and quality == "not_good":
        return "correct_avoid"
    if decision == "avoid" and quality == "good":
        return "missed_opportunity"
    if decision == "caution" and quality == "good":
        return "risky_good"
    if decision == "caution" and quality == "not_good":
        return "risky_bad"
    return "unknown"


def build_labeled_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with a real manual outcome label."""

    if dataset.empty or "actual_outcome" not in dataset.columns:
        return dataset.iloc[0:0].copy()

    labeled = dataset.copy()
    labeled["actual_outcome"] = _normalize_outcome(labeled["actual_outcome"])
    labeled = labeled[labeled["actual_outcome"].isin(LABELED_OUTCOME_VALUES)].copy()

    return labeled


def enrich_with_analytics_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add formula-vs-outcome diagnostic labels to a labeled dataset."""

    labeled = build_labeled_dataset(dataset)

    if labeled.empty:
        return pd.DataFrame(columns=ANALYTICS_COLUMNS)

    labeled["verdict"] = _normalize_verdict(labeled["verdict"])
    labeled["actual_outcome"] = _normalize_outcome(labeled["actual_outcome"])
    labeled["formula_decision"] = labeled["verdict"].map(_formula_decision)
    labeled["outcome_quality"] = labeled["actual_outcome"].map(_outcome_quality)
    labeled["analytics_label"] = labeled.apply(_analytics_label, axis=1)

    for column in ANALYTICS_COLUMNS:
        if column not in labeled.columns:
            labeled[column] = ""

    return labeled[ANALYTICS_COLUMNS]


def build_formula_outcome_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build a compact verdict-vs-actual_outcome count table."""

    labeled = build_labeled_dataset(dataset)

    if labeled.empty:
        return pd.DataFrame(columns=FORMULA_OUTCOME_MATRIX_COLUMNS)

    matrix = (
        labeled.groupby(["verdict", "actual_outcome"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["verdict", "actual_outcome"])
        .reset_index(drop=True)
    )

    return matrix[FORMULA_OUTCOME_MATRIX_COLUMNS]


def build_formula_diagnostics(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return labeled rows with diagnostic labels for formula analysis."""

    enriched = enrich_with_analytics_labels(dataset)

    if enriched.empty:
        return enriched

    return enriched.sort_values(
        ["analytics_label", "timestamp"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)


def build_feature_signal_table(
    dataset: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compare numeric feature means for good vs not-good labeled outcomes."""

    features = features or [
        "mass",
        "resistance",
        "instability",
        "distance",
        "beam_power_sum",
        "required_power",
        "effective_power",
        "margin",
        "risk_score",
    ]

    labeled = build_labeled_dataset(dataset)

    if labeled.empty:
        return pd.DataFrame(columns=FEATURE_SIGNAL_COLUMNS)

    labeled = labeled.copy()
    labeled["is_good_outcome"] = labeled["actual_outcome"].isin(GOOD_OUTCOME_VALUES).astype(int)

    rows: list[dict[str, Any]] = []

    for feature in features:
        if feature not in labeled.columns:
            continue

        values = pd.to_numeric(labeled[feature], errors="coerce")
        good_values = values[labeled["is_good_outcome"] == 1]
        not_good_values = values[labeled["is_good_outcome"] == 0]

        mean_good = float(good_values.mean()) if not good_values.empty else None
        mean_not_good = float(not_good_values.mean()) if not not_good_values.empty else None

        if mean_good is None or mean_not_good is None:
            difference = None
        else:
            difference = mean_good - mean_not_good

        valid = pd.DataFrame(
            {
                "value": values,
                "is_good_outcome": labeled["is_good_outcome"],
            }
        ).dropna()
        if (
            len(valid) >= 2
            and valid["value"].nunique() >= 2
            and valid["is_good_outcome"].nunique() >= 2
        ):
            correlation = valid["value"].corr(valid["is_good_outcome"])
            correlation_value = None if pd.isna(correlation) else float(correlation)
        else:
            correlation_value = None

        rows.append(
            {
                "feature": feature,
                "mean_good": None if mean_good is None else round(mean_good, 4),
                "mean_not_good": None if mean_not_good is None else round(mean_not_good, 4),
                "difference_good_minus_not_good": None if difference is None else round(difference, 4),
                "correlation_with_good": None
                if correlation_value is None
                else round(correlation_value, 4),
            }
        )

    return pd.DataFrame(rows, columns=FEATURE_SIGNAL_COLUMNS)


def build_outcome_numeric_summary(
    dataset: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize numeric feature ranges by actual_outcome."""

    features = features or NUMERIC_COLUMNS
    labeled = build_labeled_dataset(dataset)

    if labeled.empty:
        return pd.DataFrame(columns=OUTCOME_NUMERIC_SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []

    for actual_outcome, group in labeled.groupby("actual_outcome", dropna=False):
        for feature in features:
            if feature not in group.columns:
                continue

            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if values.empty:
                continue

            rows.append(
                {
                    "actual_outcome": str(actual_outcome),
                    "row_count": int(len(values)),
                    "feature": feature,
                    "mean": round(float(values.mean()), 4),
                    "median": round(float(values.median()), 4),
                    "min": round(float(values.min()), 4),
                    "max": round(float(values.max()), 4),
                }
            )

    return pd.DataFrame(rows, columns=OUTCOME_NUMERIC_SUMMARY_COLUMNS)


def build_basic_analytics_report(dataset: pd.DataFrame) -> dict[str, Any]:
    """Build high-level analytics numbers for the Streamlit dashboard."""

    diagnostics = build_formula_diagnostics(dataset)

    if diagnostics.empty:
        return {
            "labeled_row_count": 0,
            "good_count": 0,
            "not_good_count": 0,
            "dangerous_take_count": 0,
            "missed_opportunity_count": 0,
            "risky_bad_count": 0,
            "risky_good_count": 0,
            "correct_take_count": 0,
            "correct_avoid_count": 0,
            "diagnostic_distribution": {},
        }

    label_counts = diagnostics["analytics_label"].value_counts(dropna=False).to_dict()
    quality_counts = diagnostics["outcome_quality"].value_counts(dropna=False).to_dict()

    return {
        "labeled_row_count": int(len(diagnostics)),
        "good_count": int(quality_counts.get("good", 0)),
        "not_good_count": int(quality_counts.get("not_good", 0)),
        "dangerous_take_count": int(label_counts.get("dangerous_take", 0)),
        "missed_opportunity_count": int(label_counts.get("missed_opportunity", 0)),
        "risky_bad_count": int(label_counts.get("risky_bad", 0)),
        "risky_good_count": int(label_counts.get("risky_good", 0)),
        "correct_take_count": int(label_counts.get("correct_take", 0)),
        "correct_avoid_count": int(label_counts.get("correct_avoid", 0)),
        "diagnostic_distribution": {str(key): int(value) for key, value in label_counts.items()},
    }
