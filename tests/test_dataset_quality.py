import pandas as pd

from sc_mining.dataset.exporter import DATASET_COLUMNS
from sc_mining.dataset.quality import (
    build_quality_report,
    distribution_to_dataframe,
    quality_issues_to_dataframe,
)


def dataset_row(
    event_id: str,
    *,
    mass: float = 12600.0,
    resistance: float = 0.34,
    instability: float = 0.12,
    verdict: str = "take",
    actual_outcome: str = "good",
    ship_type: str = "prospector",
    build_id: str = "prospector_helix_rieger_focus_v1",
) -> dict:
    return {
        "event_id": event_id,
        "session_id": "session-1",
        "timestamp": "2026-05-25T12:00:00+00:00",
        "source": "manual_ui",
        "build_id": build_id,
        "ship_type": ship_type,
        "mass": mass,
        "resistance": resistance,
        "instability": instability,
        "distance": 30.0,
        "beam_count": 1,
        "beam_slots": "main",
        "beam_power_sum": 65.0,
        "required_power": 22.4,
        "effective_power": 77.2,
        "margin": 54.8,
        "risk_score": 0.1,
        "verdict": verdict,
        "actual_outcome": actual_outcome,
        "is_labeled": actual_outcome != "unknown",
        "outcome_comment": "manual label",
    }


def test_quality_report_fails_for_empty_dataset():
    dataset = pd.DataFrame(columns=DATASET_COLUMNS)

    report = build_quality_report(dataset)

    assert report["status"] == "fail"
    assert report["row_count"] == 0
    assert any(issue["check"] == "row_count" for issue in report["issues"])


def test_quality_report_fails_without_labeled_rows():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", actual_outcome="unknown"),
            dataset_row("event-2", actual_outcome="unknown"),
        ]
    )

    report = build_quality_report(dataset)

    assert report["status"] == "fail"
    assert report["labeled_count"] == 0
    assert report["unknown_outcome_count"] == 2
    assert any(issue["check"] == "labeled_rows" for issue in report["issues"])


def test_quality_report_warns_for_small_labeled_dataset():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", actual_outcome="good"),
            dataset_row("event-2", actual_outcome="bad", verdict="risky"),
        ]
    )

    report = build_quality_report(dataset, min_labeled_rows=30)

    assert report["status"] == "warn"
    assert report["labeled_count"] == 2
    assert any(issue["check"] == "labeled_rows" for issue in report["issues"])


def test_quality_report_detects_numeric_anomalies():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", mass=3_000_000.0, resistance=1.2, actual_outcome="good"),
            dataset_row("event-2", mass=-5.0, actual_outcome="bad"),
        ]
    )

    report = build_quality_report(dataset, min_labeled_rows=1)
    issue_checks = {issue["check"] for issue in report["issues"]}

    assert report["status"] == "fail"
    assert "numeric_max" in issue_checks
    assert "numeric_min" in issue_checks


def test_quality_report_detects_duplicate_event_ids():
    dataset = pd.DataFrame(
        [
            dataset_row("event-1", actual_outcome="good"),
            dataset_row("event-1", actual_outcome="bad"),
        ]
    )

    report = build_quality_report(dataset, min_labeled_rows=1)

    assert report["duplicate_event_id_count"] == 1
    assert any(issue["check"] == "duplicate_event_id" for issue in report["issues"])


def test_quality_report_can_be_ok_for_balanced_dataset():
    dataset = pd.DataFrame(
        [
            dataset_row(
                "event-1",
                actual_outcome="good",
                verdict="take",
                ship_type="prospector",
                build_id="prospector_helix_rieger_focus_v1",
            ),
            dataset_row(
                "event-2",
                actual_outcome="bad",
                verdict="risky",
                ship_type="mole",
                build_id="mole_mixed_v1",
            ),
        ]
    )

    report = build_quality_report(dataset, min_labeled_rows=1)

    assert report["status"] == "ok"
    assert report["labeled_ratio"] == 1.0
    assert report["verdict_distribution"] == {"take": 1, "risky": 1}
    assert report["actual_outcome_distribution"] == {"good": 1, "bad": 1}


def test_quality_report_fails_for_missing_required_columns():
    dataset = pd.DataFrame([{"event_id": "event-1"}])

    report = build_quality_report(dataset)

    assert report["status"] == "fail"
    assert any(issue["check"] == "required_column" for issue in report["issues"])


def test_quality_issue_helpers_return_dataframes():
    dataset = pd.DataFrame([dataset_row("event-1", actual_outcome="unknown")])
    report = build_quality_report(dataset)

    issues_df = quality_issues_to_dataframe(report)
    dist_df = distribution_to_dataframe(report["actual_outcome_distribution"], "actual_outcome")

    assert list(issues_df.columns) == ["severity", "check", "column", "count", "message"]
    assert list(dist_df.columns) == ["actual_outcome", "count"]
    assert dist_df.iloc[0]["actual_outcome"] == "unknown"
