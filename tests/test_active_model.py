from pathlib import Path

from sc_mining.ml.active_model import (
    active_selection_from_spec,
    build_active_model_status,
    clear_active_model_config,
    read_active_model_config,
    write_active_model_config,
)
from sc_mining.ml.registry import (
    MODEL_SOURCE_MANUAL_REAL,
    MODEL_SOURCE_SYNTHETIC,
    ModelArtifactSpec,
)


def make_spec(tmp_path: Path, source: str = MODEL_SOURCE_MANUAL_REAL) -> ModelArtifactSpec:
    model_path = tmp_path / "models" / f"model_{source}.joblib"
    report_path = tmp_path / "reports" / f"report_{source}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"fake model")
    report_path.write_text("{}", encoding="utf-8")
    return ModelArtifactSpec(
        label=f"Model {source}",
        model_path=model_path,
        report_path=report_path,
        model_source=source,
        usage="test",
        safe_for_gameplay_review=source == MODEL_SOURCE_MANUAL_REAL,
    )


def test_active_selection_from_spec_copies_registry_metadata(tmp_path):
    spec = make_spec(tmp_path, MODEL_SOURCE_MANUAL_REAL)

    selection = active_selection_from_spec(spec, selected_at="2026-05-25T10:00:00+00:00")

    assert selection.label == spec.label
    assert selection.model_path == str(spec.model_path)
    assert selection.report_path == str(spec.report_path)
    assert selection.model_source == MODEL_SOURCE_MANUAL_REAL
    assert selection.safe_for_gameplay_review is True
    assert selection.model_exists is True
    assert selection.report_exists is True


def test_write_and_read_active_model_config(tmp_path):
    spec = make_spec(tmp_path, MODEL_SOURCE_MANUAL_REAL)
    config_path = tmp_path / "models" / "active_model.json"

    written = write_active_model_config(spec, path=config_path)
    loaded = read_active_model_config(config_path)

    assert loaded is not None
    assert written.model_path == loaded.model_path
    assert loaded.model_source == MODEL_SOURCE_MANUAL_REAL
    assert loaded.model_exists is True
    assert loaded.report_exists is True


def test_clear_active_model_config_removes_file(tmp_path):
    spec = make_spec(tmp_path, MODEL_SOURCE_MANUAL_REAL)
    config_path = tmp_path / "models" / "active_model.json"
    write_active_model_config(spec, path=config_path)

    assert clear_active_model_config(config_path) is True
    assert read_active_model_config(config_path) is None
    assert clear_active_model_config(config_path) is False


def test_build_active_model_status_reports_missing_selection(tmp_path):
    status = build_active_model_status(path=tmp_path / "missing.json", specs=[])

    assert status["configured"] is False
    assert status["valid"] is False
    assert status["reason"] == "No active model selected."


def test_build_active_model_status_reports_selected_synthetic_model(tmp_path):
    spec = make_spec(tmp_path, MODEL_SOURCE_SYNTHETIC)
    config_path = tmp_path / "models" / "active_model.json"
    write_active_model_config(spec, path=config_path)

    status = build_active_model_status(path=config_path, specs=[spec])

    assert status["configured"] is True
    assert status["valid"] is True
    assert status["active_model"]["model_source"] == MODEL_SOURCE_SYNTHETIC
    assert "Synthetic" in status["active_model"]["warning"]


def test_build_active_model_status_detects_deleted_model(tmp_path):
    spec = make_spec(tmp_path, MODEL_SOURCE_MANUAL_REAL)
    config_path = tmp_path / "models" / "active_model.json"
    write_active_model_config(spec, path=config_path)
    spec.model_path.unlink()

    status = build_active_model_status(path=config_path, specs=[spec])

    assert status["configured"] is True
    assert status["valid"] is False
    assert "missing" in status["reason"]
