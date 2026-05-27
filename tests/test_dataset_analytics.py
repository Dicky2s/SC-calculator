import pandas as pd

from sc_mining.dataset.analytics import (
    ANALYTICS_COLUMNS,
    FEATURE_SIGNAL_COLUMNS,
    FORMULA_OUTCOME_MATRIX_COLUMNS,
    OUTCOME_NUMERIC_SUMMARY_COLUMNS,
    build_basic_analytics_report,
    build_feature_signal_table,
    build_formula_diagnostics,
    build_formula_outcome_matrix,
    build_labeled_dataset,
    build_outcome_numeric_summary,
    enrich_with_analytics_labels,
)
from sc_mining.dataset.exporter import DATASET_COLUMNS


def dataset_row(
    event_id: str,
    *,
    verdict: str = "take",
    actual_outcome: str = "good",
    mass: float = 12600.0,
    resistance: float = 0.34,
    instability: float = 0.12,
    margin: float = 54.8,
    risk_score: float = 0.1,
) -> dict:
    return {
        "event_id": event_id,
        "session_id": "session-1",
        "timestamp": "2026-05-25T12:00:00+00:00",
        "source": "manual_ui",
        "build_id": "prospector_helix_rieger_focus_v1",
        "ship_type": "prospector",
        "mass": mass,
        "resistance": resistance,
        "instability": instability,
        "distance": 92.0,
        "beam_count": 1,
        "beam_slots": "main",
        "beam_power_sum": 65.0,
        "required_power": 22.4,
        "effective_power": 77.2,
        "margin": margin,
        "risk_score": risk_score,
        "verdict": verdict,
        "actual_outcome": actual_outcome,
        "is_labeled": actual_outcome != "unknown",
        "outcome_comment": "manual label",
    }


def test_build_labeled_dataset_filters_unknown_outcomes():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", actual_outcome="unknown"),
            dataset_row("event-2", actual_outcome="good"),
            dataset_row("event-3", actual_outcome="bad"),
        ],
        columns=DATASET_COLUMNS,
    )

    labeled = build_labeled_dataset(dataset)

    assert len(labeled) == 2
    assert set(labeled["event_id"]) == {"event-2", "event-3"}


def test_enrich_with_analytics_labels_classifies_formula_vs_outcome():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", verdict="take", actual_outcome="good"),
            dataset_row("event-2", verdict="take", actual_outcome="too_unstable"),
            dataset_row("event-3", verdict="need_more_power", actual_outcome="good"),
            dataset_row("event-4", verdict="risky", actual_outcome="bad"),
        ],
        columns=DATASET_COLUMNS,
    )

    enriched = enrich_with_analytics_labels(dataset)

    assert list(enriched.columns) == ANALYTICS_COLUMNS
    assert set(enriched["analytics_label"]) == {
        "correct_take",
        "dangerous_take",
        "missed_opportunity",
        "risky_bad",
    }


def test_build_formula_outcome_matrix_counts_pairs():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", verdict="take", actual_outcome="good"),
            dataset_row("event-2", verdict="take", actual_outcome="good"),
            dataset_row("event-3", verdict="take", actual_outcome="bad"),
            dataset_row("event-4", verdict="risky", actual_outcome="bad"),
        ],
        columns=DATASET_COLUMNS,
    )

    matrix = build_formula_outcome_matrix(dataset)

    assert list(matrix.columns) == FORMULA_OUTCOME_MATRIX_COLUMNS
    pair_counts = {
        (row["verdict"], row["actual_outcome"]): row["count"]
        for _, row in matrix.iterrows()
    }
    assert pair_counts[("take", "good")] == 2
    assert pair_counts[("take", "bad")] == 1
    assert pair_counts[("risky", "bad")] == 1


def test_build_formula_diagnostics_returns_sorted_diagnostic_rows():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", verdict="take", actual_outcome="too_slow"),
            dataset_row("event-2", verdict="skip", actual_outcome="good"),
        ],
        columns=DATASET_COLUMNS,
    )

    diagnostics = build_formula_diagnostics(dataset)

    assert set(diagnostics["analytics_label"]) == {"dangerous_take", "missed_opportunity"}


def test_build_feature_signal_table_compares_good_and_not_good_means():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", actual_outcome="good", margin=50.0, risk_score=0.1),
            dataset_row("event-2", actual_outcome="bad", margin=-10.0, risk_score=0.8),
        ],
        columns=DATASET_COLUMNS,
    )

    signal = build_feature_signal_table(dataset, features=["margin", "risk_score"])

    assert list(signal.columns) == FEATURE_SIGNAL_COLUMNS
    margin_row = signal[signal["feature"] == "margin"].iloc[0]
    risk_row = signal[signal["feature"] == "risk_score"].iloc[0]
    assert margin_row["mean_good"] == 50.0
    assert margin_row["mean_not_good"] == -10.0
    assert margin_row["difference_good_minus_not_good"] == 60.0
    assert risk_row["difference_good_minus_not_good"] == -0.7


def test_build_outcome_numeric_summary_groups_by_actual_outcome():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", actual_outcome="good", mass=10000.0),
            dataset_row("event-2", actual_outcome="good", mass=20000.0),
            dataset_row("event-3", actual_outcome="bad", mass=30000.0),
        ],
        columns=DATASET_COLUMNS,
    )

    summary = build_outcome_numeric_summary(dataset, features=["mass"])

    assert list(summary.columns) == OUTCOME_NUMERIC_SUMMARY_COLUMNS
    good_mass = summary[(summary["actual_outcome"] == "good") & (summary["feature"] == "mass")].iloc[0]
    assert good_mass["row_count"] == 2
    assert good_mass["mean"] == 15000.0
    assert good_mass["median"] == 15000.0


def test_build_basic_analytics_report_counts_diagnostic_groups():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", verdict="take", actual_outcome="good"),
            dataset_row("event-2", verdict="take", actual_outcome="bad"),
            dataset_row("event-3", verdict="need_more_power", actual_outcome="good"),
            dataset_row("event-4", verdict="risky", actual_outcome="too_unstable"),
            dataset_row("event-5", verdict="risky", actual_outcome="good"),
            dataset_row("event-6", verdict="skip", actual_outcome="bad"),
            dataset_row("event-7", verdict="take", actual_outcome="unknown"),
        ],
        columns=DATASET_COLUMNS,
    )

    report = build_basic_analytics_report(dataset)

    assert report["labeled_row_count"] == 6
    assert report["good_count"] == 3
    assert report["not_good_count"] == 3
    assert report["dangerous_take_count"] == 1
    assert report["missed_opportunity_count"] == 1
    assert report["risky_bad_count"] == 1
    assert report["risky_good_count"] == 1
    assert report["correct_take_count"] == 1
    assert report["correct_avoid_count"] == 1


def test_feature_signal_constant_values_do_not_emit_correlation_warning() -> None:
    import warnings

    dataset = pd.DataFrame(
        [
            {"actual_outcome": "good", "mass": 100.0},
            {"actual_outcome": "poor", "mass": 100.0},
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = build_feature_signal_table(dataset, features=["mass"])

    assert table.loc[0, "correlation_with_good"] is None
    assert not any("invalid value encountered in divide" in str(item.message) for item in caught)
