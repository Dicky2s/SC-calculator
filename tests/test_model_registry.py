from pathlib import Path

from sc_mining.ml.registry import (
    MANUAL_MODEL_PATH,
    MODEL_SOURCE_LEGACY_MANUAL,
    MODEL_SOURCE_MANUAL_REAL,
    MODEL_SOURCE_SYNTHETIC,
    SYNTHETIC_MODEL_PATH,
    default_model_artifact_specs,
    existing_model_artifacts,
    infer_model_source,
    is_gameplay_review_model,
    model_source_warning,
    spec_to_dict,
)


def test_default_registry_separates_manual_and_synthetic_artifacts():
    specs = default_model_artifact_specs()

    sources = {spec.model_source for spec in specs}
    paths = {spec.model_path for spec in specs}

    assert MODEL_SOURCE_MANUAL_REAL in sources
    assert MODEL_SOURCE_SYNTHETIC in sources
    assert MANUAL_MODEL_PATH in paths
    assert SYNTHETIC_MODEL_PATH in paths


def test_infer_model_source_uses_explicit_file_names():
    assert infer_model_source("models/mining_outcome_baseline_manual.joblib") == MODEL_SOURCE_MANUAL_REAL
    assert infer_model_source("models/mining_outcome_baseline_synthetic.joblib") == MODEL_SOURCE_SYNTHETIC
    assert infer_model_source("models/mining_outcome_baseline.joblib") == MODEL_SOURCE_LEGACY_MANUAL
    assert infer_model_source("models/other.joblib") == "unknown"


def test_model_source_warnings_protect_smoke_test_artifacts():
    assert model_source_warning(MODEL_SOURCE_MANUAL_REAL) is None
    assert "Synthetic" in model_source_warning(MODEL_SOURCE_SYNTHETIC)
    assert "Legacy" in model_source_warning(MODEL_SOURCE_LEGACY_MANUAL)
    assert "Unknown" in model_source_warning("unknown")


def test_is_gameplay_review_model_only_allows_manual_real_data():
    assert is_gameplay_review_model(MODEL_SOURCE_MANUAL_REAL) is True
    assert is_gameplay_review_model(MODEL_SOURCE_SYNTHETIC) is False
    assert is_gameplay_review_model(MODEL_SOURCE_LEGACY_MANUAL) is False


def test_existing_model_artifacts_filters_by_file_existence(tmp_path):
    existing_model = tmp_path / "models" / "manual.joblib"
    existing_report = tmp_path / "reports" / "manual.json"
    missing_model = tmp_path / "models" / "synthetic.joblib"
    missing_report = tmp_path / "reports" / "synthetic.json"

    existing_model.parent.mkdir(parents=True)
    existing_report.parent.mkdir(parents=True)
    existing_model.write_bytes(b"fake")
    existing_report.write_text("{}", encoding="utf-8")

    specs = default_model_artifact_specs()
    custom_specs = [
        specs[0].__class__(
            label="Existing",
            model_path=existing_model,
            report_path=existing_report,
            model_source=MODEL_SOURCE_MANUAL_REAL,
            usage="test",
            safe_for_gameplay_review=True,
        ),
        specs[1].__class__(
            label="Missing",
            model_path=missing_model,
            report_path=missing_report,
            model_source=MODEL_SOURCE_SYNTHETIC,
            usage="test",
            safe_for_gameplay_review=False,
        ),
    ]

    existing = existing_model_artifacts(custom_specs)

    assert len(existing) == 1
    assert existing[0].label == "Existing"


def test_spec_to_dict_serializes_paths_as_strings():
    spec = default_model_artifact_specs()[0]
    payload = spec_to_dict(spec)

    assert isinstance(payload["model_path"], str)
    assert isinstance(payload["report_path"], str)
    assert payload["model_source"] == MODEL_SOURCE_MANUAL_REAL
