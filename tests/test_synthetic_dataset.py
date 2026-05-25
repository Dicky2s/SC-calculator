import pandas as pd
import pytest

from sc_mining.dataset.exporter import DATASET_COLUMNS
from sc_mining.dataset.synthetic import (
    export_synthetic_dataset,
    generate_synthetic_dataset,
    synthetic_summary,
)


def test_generate_synthetic_dataset_returns_expected_schema():
    dataset = generate_synthetic_dataset(row_count=20, seed=7)

    assert list(dataset.columns) == DATASET_COLUMNS
    assert len(dataset) == 20
    assert set(dataset["source"]) == {"synthetic"}
    assert dataset["is_labeled"].all()
    assert "unknown" not in set(dataset["actual_outcome"])
    assert "good" in set(dataset["actual_outcome"])
    assert len(set(dataset["actual_outcome"]) - {"good"}) >= 1


def test_generate_synthetic_dataset_is_deterministic_for_seed():
    first = generate_synthetic_dataset(row_count=12, seed=123)
    second = generate_synthetic_dataset(row_count=12, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_generate_synthetic_dataset_rejects_invalid_row_count():
    with pytest.raises(ValueError, match="row_count"):
        generate_synthetic_dataset(row_count=1)


def test_export_synthetic_dataset_writes_csv(tmp_path):
    output_path = tmp_path / "datasets" / "mining_events_synthetic.csv"

    dataset = export_synthetic_dataset(output_path=output_path, row_count=18, seed=5)
    loaded = pd.read_csv(output_path)

    assert output_path.exists()
    assert len(dataset) == 18
    assert len(loaded) == 18
    assert set(loaded["source"]) == {"synthetic"}


def test_synthetic_summary_marks_data_as_synthetic():
    dataset = generate_synthetic_dataset(row_count=10, seed=42)

    summary = synthetic_summary(dataset)

    assert summary["row_count"] == 10
    assert summary["labeled_count"] == 10
    assert summary["unlabeled_count"] == 0
    assert summary["source"] == "synthetic"
    assert "warning" in summary
