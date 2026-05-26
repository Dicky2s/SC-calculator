from pathlib import Path

import pandas as pd

from sc_mining.storage.event_reader import LABELED_OUTCOME_VALUES, load_events_dataframe


DATASET_COLUMNS = [
    "event_id",
    "session_id",
    "timestamp",
    "source",
    "operator_name",
    "crew_size",
    "run_tag",
    "build_id",
    "ship_type",
    "mass",
    "resistance",
    "instability",
    "distance",
    "beam_count",
    "beam_slots",
    "beam_power_sum",
    "required_power",
    "effective_power",
    "margin",
    "risk_score",
    "verdict",
    "actual_outcome",
    "is_labeled",
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
    "dominant_resource_window_hint",
    "dominant_resource_charge_behavior",
    "resource_window_behaviors",
    "resource_charge_behaviors",
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
]

NUMERIC_COLUMNS = [
    "mass",
    "resistance",
    "instability",
    "distance",
    "beam_count",
    "beam_power_sum",
    "required_power",
    "effective_power",
    "margin",
    "risk_score",
    "crew_size",
    "resource_percent",
    "raw_scu_estimate",
    "total_scu_estimate",
    "refined_scu_estimate",
    "estimated_value_auec",
    "mining_time_seconds",
    "resource_count",
    "total_resource_percent",
    "refined_scu_actual",
    "refined_value_auec",
    "refinery_fee_auec",
    "sell_value_auec",
    "refined_resource_count",
    "total_refined_scu_actual",
    "total_resource_sell_value_auec",
    "observed_min_warmup_power_percent",
    "observed_stable_power_percent",
    "observed_distance",
    "calibration_attempt_count",
    "calibration_no_warmup_count",
    "calibration_warmup_count",
    "calibration_stable_hold_count",
]

TEXT_COLUMNS = [
    "event_id",
    "session_id",
    "timestamp",
    "source",
    "operator_name",
    "run_tag",
    "build_id",
    "ship_type",
    "beam_slots",
    "verdict",
    "actual_outcome",
    "outcome_comment",
    "primary_resource",
    "resource_comment",
    "resource_names",
    "dominant_resource_window_hint",
    "dominant_resource_charge_behavior",
    "resource_window_behaviors",
    "resource_charge_behaviors",
    "resources_json",
    "refinery_method",
    "refinery_location",
    "refinery_start_at",
    "refinery_complete_at",
    "refinery_comment",
    "refined_resource_names",
    "refined_resources_json",
    "calibration_comment",
    "calibration_attempts_json",
]


def build_dataset(events_path: str | Path, labeled_only: bool = False) -> pd.DataFrame:
    """Build an ML/analytics-ready table from the raw JSONL event log."""

    df = load_events_dataframe(events_path)

    if df.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    dataset = df.copy()

    for column in DATASET_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = None

    for column in NUMERIC_COLUMNS:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    for column in TEXT_COLUMNS:
        dataset[column] = dataset[column].fillna("").astype(str)

    dataset["actual_outcome"] = dataset["actual_outcome"].replace("", "unknown")
    dataset["is_labeled"] = dataset["actual_outcome"].isin(LABELED_OUTCOME_VALUES)

    if labeled_only:
        dataset = dataset[dataset["is_labeled"]].copy()

    return dataset[DATASET_COLUMNS]


def export_dataset(
    events_path: str | Path,
    output_path: str | Path,
    labeled_only: bool = False,
) -> pd.DataFrame:
    """Export events JSONL into a CSV dataset and return the exported dataframe."""

    dataset = build_dataset(events_path=events_path, labeled_only=labeled_only)

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(target_path, index=False, encoding="utf-8")

    return dataset


def get_dataset_export_summary(dataset: pd.DataFrame) -> dict:
    if dataset.empty:
        return {
            "row_count": 0,
            "labeled_count": 0,
            "unlabeled_count": 0,
            "verdict_distribution": {},
            "actual_outcome_distribution": {},
        }

    labeled_count = int(dataset["is_labeled"].sum())

    return {
        "row_count": int(len(dataset)),
        "labeled_count": labeled_count,
        "unlabeled_count": int(len(dataset) - labeled_count),
        "verdict_distribution": dataset["verdict"].value_counts(dropna=False).to_dict(),
        "actual_outcome_distribution": dataset["actual_outcome"].value_counts(dropna=False).to_dict(),
    }
