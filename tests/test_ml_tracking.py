from dataclasses import dataclass

from sc_mining.ml.tracking import (
    append_training_run,
    build_training_run_record,
    latest_training_run_by_source,
    load_training_runs,
    summarize_training_runs,
)


@dataclass(frozen=True)
class DummyTrainingResult:
    model_version: str = "baseline_rf_v1"
    model_source: str = "manual_real_data"
    model_path: str = "models/model.joblib"
    report_path: str = "reports/report.json"
    rows_total: int = 10
    rows_used: int = 8
    train_rows: int = 6
    test_rows: int = 2
    accuracy: float = 0.75
    target_distribution: dict[str, int] = None
    feature_columns: list[str] = None

    def __post_init__(self):
        object.__setattr__(self, "target_distribution", self.target_distribution or {"0": 4, "1": 4})
        object.__setattr__(self, "feature_columns", self.feature_columns or ["mass", "resistance"])


def test_build_training_run_record_from_training_result():
    result = DummyTrainingResult()

    record = build_training_run_record(
        result,
        notes="manual smoke",
        run_id="run-1",
        created_at="2026-05-25T10:00:00+00:00",
    )

    assert record.run_id == "run-1"
    assert record.created_at == "2026-05-25T10:00:00+00:00"
    assert record.model_source == "manual_real_data"
    assert record.rows_used == 8
    assert record.accuracy == 0.75
    assert record.target_distribution == {"0": 4, "1": 4}
    assert record.notes == "manual smoke"


def test_append_and_load_training_runs(tmp_path):
    path = tmp_path / "training_runs.jsonl"

    append_training_run(DummyTrainingResult(), path=path, notes="first")
    append_training_run(
        DummyTrainingResult(model_source="synthetic_smoke_test", accuracy=0.91),
        path=path,
        notes="synthetic",
    )

    runs = load_training_runs(path)

    assert len(runs) == 2
    assert set(runs["model_source"]) == {"manual_real_data", "synthetic_smoke_test"}
    assert set(runs["notes"]) == {"first", "synthetic"}


def test_summarize_training_runs_returns_best_and_latest(tmp_path):
    path = tmp_path / "training_runs.jsonl"

    append_training_run(
        DummyTrainingResult(model_source="manual_real_data", accuracy=0.5),
        path=path,
        notes="manual",
    )
    append_training_run(
        DummyTrainingResult(model_source="synthetic_smoke_test", accuracy=0.9),
        path=path,
        notes="synthetic",
    )

    runs = load_training_runs(path)
    summary = summarize_training_runs(runs)

    assert summary["run_count"] == 2
    assert summary["model_source_count"] == 2
    assert summary["best_accuracy"] == 0.9
    assert summary["latest_run_id"] is not None
    assert summary["latest_model_source"] in {"manual_real_data", "synthetic_smoke_test"}


def test_latest_training_run_by_source_returns_latest_matching_run(tmp_path):
    path = tmp_path / "training_runs.jsonl"

    first = build_training_run_record(
        DummyTrainingResult(model_source="manual_real_data", accuracy=0.4),
        run_id="manual-old",
        created_at="2026-05-25T10:00:00+00:00",
    )
    second = build_training_run_record(
        DummyTrainingResult(model_source="manual_real_data", accuracy=0.7),
        run_id="manual-new",
        created_at="2026-05-25T11:00:00+00:00",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            __import__("json").dumps(first.__dict__),
            __import__("json").dumps(second.__dict__),
        ]),
        encoding="utf-8",
    )

    runs = load_training_runs(path)
    latest = latest_training_run_by_source(runs, "manual_real_data")

    assert latest is not None
    assert latest["run_id"] == "manual-new"
    assert latest["accuracy"] == 0.7


def test_load_training_runs_missing_file_returns_empty_schema(tmp_path):
    runs = load_training_runs(tmp_path / "missing.jsonl")

    assert runs.empty
    assert "run_id" in runs.columns
    assert "model_source" in runs.columns
