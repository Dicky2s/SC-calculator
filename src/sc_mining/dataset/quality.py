from __future__ import annotations

from typing import Any

import pandas as pd

from sc_mining.dataset.exporter import DATASET_COLUMNS, NUMERIC_COLUMNS
from sc_mining.storage.event_reader import LABELED_OUTCOME_VALUES


MIN_LABELED_ROWS_FOR_BASELINE = 30
IMBALANCE_WARNING_SHARE = 0.80
OPTIONAL_MISSING_COLUMNS = {
    "operator_name",
    "crew_size",
    "run_tag",
    "outcome_comment",
    "primary_resource",
    "resource_percent",
    "raw_scu_estimate",
    "total_scu_estimate",
    "refined_scu_estimate",
    "estimated_value_auec",
    "mining_time_seconds",
    "resource_comment",
    "resource_count",
    "resource_names",
    "total_resource_percent",
    "resources_json",
    "refinery_method",
    "refinery_location",
    "refinery_start_at",
    "refinery_complete_at",
    "refined_scu_actual",
    "refined_value_auec",
    "refinery_fee_auec",
    "sell_value_auec",
    "refinery_comment",
    "refined_resource_count",
    "refined_resource_names",
    "total_refined_scu_actual",
    "total_resource_sell_value_auec",
    "refined_resources_json",
    "formula_issue_flag",
    "observed_min_warmup_power_percent",
    "observed_stable_power_percent",
    "observed_distance",
    "calibration_comment",
    "calibration_attempt_count",
    "calibration_no_warmup_count",
    "calibration_warmup_count",
    "calibration_stable_hold_count",
    "calibration_attempts_json",
}

NUMERIC_SANITY_RULES: dict[str, dict[str, float | None]] = {
    "mass": {"min": 1.0, "max": 500_000.0},
    "resistance": {"min": 0.0, "max": 1.0},
    "instability": {"min": 0.0, "max": 1.0},
    "distance": {"min": 1.0, "max": 500.0},
    "beam_count": {"min": 0.0, "max": 3.0},
    "beam_power_sum": {"min": 0.0, "max": 300.0},
    "required_power": {"min": 0.0, "max": None},
    "effective_power": {"min": 0.0, "max": None},
    "risk_score": {"min": 0.0, "max": 1.0},
}


def _safe_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _value_distribution(series: pd.Series) -> dict[str, int]:
    counts = series.fillna("missing").astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _dominant_share(series: pd.Series) -> float:
    if series.empty:
        return 0.0

    counts = series.fillna("missing").astype(str).value_counts(dropna=False)
    if counts.empty:
        return 0.0

    return float(counts.iloc[0] / counts.sum())


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    check: str,
    message: str,
    count: int | None = None,
    column: str | None = None,
) -> None:
    issues.append(
        {
            "severity": severity,
            "check": check,
            "column": column or "",
            "count": count if count is not None else "",
            "message": message,
        }
    )


def build_quality_report(
    dataset: pd.DataFrame,
    min_labeled_rows: int = MIN_LABELED_ROWS_FOR_BASELINE,
) -> dict[str, Any]:
    """Build a compact data quality report for analytics and ML readiness."""

    issues: list[dict[str, Any]] = []
    row_count = int(len(dataset))

    dataset = dataset.copy()
    optional_columns = OPTIONAL_MISSING_COLUMNS
    optional_missing_columns = [
        column for column in DATASET_COLUMNS
        if column not in dataset.columns and column in optional_columns
    ]
    for column in optional_missing_columns:
        dataset[column] = None

    missing_columns = [
        column for column in DATASET_COLUMNS
        if column not in dataset.columns and column not in optional_columns
    ]
    if missing_columns:
        for column in missing_columns:
            _add_issue(
                issues,
                severity="fail",
                check="required_column",
                column=column,
                message=f"Required dataset column is missing: {column}",
            )

        return {
            "status": "fail",
            "row_count": row_count,
            "labeled_count": 0,
            "unlabeled_count": row_count,
            "labeled_ratio": 0.0,
            "unknown_outcome_count": 0,
            "duplicate_event_id_count": 0,
            "missing_values": {},
            "verdict_distribution": {},
            "actual_outcome_distribution": {},
            "numeric_summary": {},
            "issues": issues,
        }

    if row_count == 0:
        _add_issue(
            issues,
            severity="fail",
            check="row_count",
            count=0,
            message="Dataset is empty. There is nothing to analyze or train on.",
        )

    normalized_outcome = dataset["actual_outcome"].fillna("unknown").astype(str)
    labeled_mask = normalized_outcome.isin(LABELED_OUTCOME_VALUES)
    labeled_count = int(labeled_mask.sum())
    unlabeled_count = int(row_count - labeled_count)
    labeled_ratio = round(labeled_count / row_count, 4) if row_count else 0.0
    unknown_outcome_count = int((normalized_outcome == "unknown").sum())

    if row_count > 0 and labeled_count == 0:
        _add_issue(
            issues,
            severity="fail",
            check="labeled_rows",
            count=0,
            message="No labeled rows found. Supervised ML cannot be trained yet.",
        )
    elif 0 < labeled_count < min_labeled_rows:
        _add_issue(
            issues,
            severity="warn",
            check="labeled_rows",
            count=labeled_count,
            message=(
                f"Only {labeled_count} labeled rows found. "
                f"Collect at least {min_labeled_rows} for a weak baseline."
            ),
        )

    if row_count > 0 and unknown_outcome_count / row_count > 0.5:
        _add_issue(
            issues,
            severity="warn",
            check="unknown_outcomes",
            count=unknown_outcome_count,
            message="More than half of the dataset has unknown actual_outcome.",
        )

    duplicate_event_id_count = int(dataset["event_id"].duplicated().sum())
    if duplicate_event_id_count:
        _add_issue(
            issues,
            severity="warn",
            check="duplicate_event_id",
            count=duplicate_event_id_count,
            column="event_id",
            message="Duplicate event_id values found. Check whether events were appended twice.",
        )

    missing_values = {
        column: int(dataset[column].isna().sum())
        for column in DATASET_COLUMNS
        if column in dataset.columns
    }

    for column, missing_count in missing_values.items():
        if missing_count and column not in OPTIONAL_MISSING_COLUMNS:
            _add_issue(
                issues,
                severity="warn",
                check="missing_values",
                column=column,
                count=missing_count,
                message=f"Column {column} contains missing values.",
            )

    for column, rule in NUMERIC_SANITY_RULES.items():
        values = pd.to_numeric(dataset[column], errors="coerce")
        invalid_numeric_count = int(values.isna().sum())
        if invalid_numeric_count:
            _add_issue(
                issues,
                severity="warn",
                check="numeric_parse",
                column=column,
                count=invalid_numeric_count,
                message=f"Column {column} contains values that cannot be parsed as numbers.",
            )

        min_value = rule.get("min")
        max_value = rule.get("max")

        if min_value is not None:
            below_min_count = int((values < float(min_value)).sum())
            if below_min_count:
                _add_issue(
                    issues,
                    severity="fail",
                    check="numeric_min",
                    column=column,
                    count=below_min_count,
                    message=f"Column {column} has values below {min_value}.",
                )

        if max_value is not None:
            above_max_count = int((values > float(max_value)).sum())
            if above_max_count:
                severity = "warn" if column == "mass" else "fail"
                _add_issue(
                    issues,
                    severity=severity,
                    check="numeric_max",
                    column=column,
                    count=above_max_count,
                    message=f"Column {column} has values above {max_value}.",
                )

    labeled_outcomes = normalized_outcome[labeled_mask]
    labeled_outcome_class_count = int(labeled_outcomes.nunique())

    if labeled_count > 0 and labeled_outcome_class_count < 2:
        _add_issue(
            issues,
            severity="warn",
            check="target_classes",
            count=labeled_outcome_class_count,
            column="actual_outcome",
            message="Labeled data has fewer than 2 actual_outcome classes.",
        )

    if labeled_count >= 2:
        dominant_outcome_share = _dominant_share(labeled_outcomes)
        if dominant_outcome_share >= IMBALANCE_WARNING_SHARE:
            _add_issue(
                issues,
                severity="warn",
                check="target_imbalance",
                column="actual_outcome",
                message=(
                    "One actual_outcome class dominates the labeled data. "
                    f"Dominant share: {dominant_outcome_share:.0%}."
                ),
            )

    verdict_share = _dominant_share(dataset["verdict"])
    if row_count >= 2 and verdict_share >= IMBALANCE_WARNING_SHARE:
        _add_issue(
            issues,
            severity="warn",
            check="verdict_imbalance",
            column="verdict",
            message=(
                "One verdict dominates the dataset. "
                f"Dominant share: {verdict_share:.0%}."
            ),
        )

    ship_count = int(dataset["ship_type"].fillna("missing").astype(str).nunique())
    build_count = int(dataset["build_id"].fillna("missing").astype(str).nunique())

    if row_count > 0 and ship_count < 2:
        _add_issue(
            issues,
            severity="warn",
            check="ship_coverage",
            column="ship_type",
            count=ship_count,
            message="Dataset contains only one ship_type. Model will not learn cross-ship differences.",
        )

    if row_count > 0 and build_count < 2:
        _add_issue(
            issues,
            severity="warn",
            check="build_coverage",
            column="build_id",
            count=build_count,
            message="Dataset contains only one build_id. Model will not learn build differences.",
        )

    has_failures = any(issue["severity"] == "fail" for issue in issues)
    has_warnings = any(issue["severity"] == "warn" for issue in issues)
    status = "fail" if has_failures else "warn" if has_warnings else "ok"

    numeric_summary = (
        dataset[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce").describe().round(3).to_dict()
        if row_count
        else {}
    )

    return {
        "status": status,
        "row_count": row_count,
        "labeled_count": labeled_count,
        "unlabeled_count": unlabeled_count,
        "labeled_ratio": labeled_ratio,
        "unknown_outcome_count": unknown_outcome_count,
        "duplicate_event_id_count": duplicate_event_id_count,
        "missing_values": missing_values,
        "verdict_distribution": _value_distribution(dataset["verdict"]),
        "actual_outcome_distribution": _value_distribution(normalized_outcome),
        "numeric_summary": numeric_summary,
        "issues": issues,
    }


def quality_issues_to_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    """Convert report issues into a Streamlit/pandas-friendly table."""

    columns = ["severity", "check", "column", "count", "message"]
    issues = report.get("issues", [])

    if not issues:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(issues)[columns]


def distribution_to_dataframe(distribution: dict[str, int], label_column: str) -> pd.DataFrame:
    """Convert a value-count dictionary into a stable table."""

    if not distribution:
        return pd.DataFrame(columns=[label_column, "count"])

    return pd.DataFrame(
        [{label_column: key, "count": value} for key, value in distribution.items()]
    )
