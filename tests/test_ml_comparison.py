import pandas as pd

from sc_mining.dataset.exporter import DATASET_COLUMNS
from sc_mining.domain.models import (
    BeamState,
    BuildProfile,
    CalculationInput,
    CalculationResult,
    HeadBuild,
    RockInput,
)
from sc_mining.ml.baseline import load_baseline_model, train_baseline_model
from sc_mining.ml.comparison import (
    MODEL_SOURCE_MANUAL,
    MODEL_SOURCE_SYNTHETIC,
    agreement_label,
    apply_formula_ml_comparison_to_dataset,
    build_comparison_export_dataframe,
    build_inference_dataset_row,
    cleanup_export_dataframe,
    compare_formula_with_model,
    comparison_actual_outcome_coverage,
    comparison_export_csv,
    confidence_band,
    formula_expected_outcome,
    infer_model_source,
    model_source_warning,
)


def make_dataset(rows=40):
    records = []
    for index in range(rows):
        is_good = index % 2 == 0
        records.append(
            {
                "event_id": f"event-{index}",
                "session_id": "session-1",
                "timestamp": "2026-05-25T12:00:00+00:00",
                "source": "test",
                "build_id": "prospector_helix_rieger_focus_v1",
                "ship_type": "prospector",
                "mass": 12000 + index * 100,
                "resistance": 0.18 if is_good else 0.55,
                "instability": 0.04 if is_good else 0.42,
                "distance": 80 + index % 20,
                "beam_count": 1,
                "beam_slots": "main",
                "beam_power_sum": 65 + index % 10,
                "required_power": 20 + index % 9,
                "effective_power": 80 if is_good else 25,
                "margin": 50 if is_good else -10,
                "risk_score": 0.15 if is_good else 0.85,
                "verdict": "take" if is_good else "need_more_power",
                "actual_outcome": "good" if is_good else "too_unstable",
                "is_labeled": True,
                "outcome_comment": "test row",
            }
        )
    return pd.DataFrame(records, columns=DATASET_COLUMNS)


def make_calc_input() -> CalculationInput:
    build = BuildProfile(
        build_id="prospector_helix_rieger_focus_v1",
        ship_type="prospector",
        heads=[HeadBuild(slot="main", head_id="helix_s1", modules=[])],
    )
    return CalculationInput(
        rock=RockInput(mass=15000, resistance=0.2, instability=0.04, distance=85),
        build=build,
        beams=[BeamState(slot="main", power_percent=65, active_modules=[])],
    )


def make_result(verdict="take") -> CalculationResult:
    return CalculationResult(
        required_power=22.0,
        effective_power=80.0,
        margin=58.0,
        risk_score=0.12,
        verdict=verdict,
        notes=[],
    )


def test_formula_expected_outcome_maps_take_to_good():
    assert formula_expected_outcome("take") == "good"
    assert formula_expected_outcome("risky") == "not_good"
    assert formula_expected_outcome("skip") == "not_good"
    assert formula_expected_outcome("need_more_power") == "not_good"


def test_confidence_band_labels_probability_ranges():
    assert confidence_band(0.9) == "high_good"
    assert confidence_band(0.6) == "weak_good"
    assert confidence_band(0.5) == "uncertain"
    assert confidence_band(0.4) == "weak_not_good"
    assert confidence_band(0.1) == "high_not_good"


def test_agreement_label_detects_disagreement_cases():
    assert agreement_label("take", "good") == "formula_and_ml_take"
    assert agreement_label("take", "not_good") == "ml_warns_against_formula_take"
    assert agreement_label("risky", "good") == "ml_sees_possible_opportunity"
    assert agreement_label("need_more_power", "not_good") == "formula_and_ml_avoid"


def test_build_inference_dataset_row_matches_feature_schema():
    row = build_inference_dataset_row(make_calc_input(), make_result())

    assert len(row) == 1
    assert row.iloc[0]["build_id"] == "prospector_helix_rieger_focus_v1"
    assert row.iloc[0]["ship_type"] == "prospector"
    assert row.iloc[0]["beam_count"] == 1
    assert row.iloc[0]["verdict"] == "take"
    assert row.iloc[0]["actual_outcome"] == "unknown"


def test_compare_formula_with_model_returns_prediction(tmp_path):
    dataset = make_dataset(rows=40)
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    train_baseline_model(
        dataset=dataset,
        model_path=model_path,
        report_path=report_path,
        min_labeled_rows=10,
        test_size=0.25,
        random_state=7,
    )

    comparison = compare_formula_with_model(
        calc_input=make_calc_input(),
        result=make_result("take"),
        model_path=model_path,
    )

    assert comparison.model_available is True
    assert comparison.ml_prediction in {"good", "not_good"}
    assert comparison.ml_good_probability is not None
    assert 0 <= comparison.ml_good_probability <= 1
    assert comparison.agreement_label is not None


def test_compare_formula_with_model_handles_missing_model(tmp_path):
    comparison = compare_formula_with_model(
        calc_input=make_calc_input(),
        result=make_result("take"),
        model_path=tmp_path / "missing.joblib",
    )

    assert comparison.model_available is False
    assert "Model file not found" in comparison.reason
    assert comparison.ml_prediction is None


def test_apply_formula_ml_comparison_to_dataset_adds_columns(tmp_path):
    dataset = make_dataset(rows=40)
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    train_baseline_model(
        dataset=dataset,
        model_path=model_path,
        report_path=report_path,
        min_labeled_rows=10,
        test_size=0.25,
        random_state=7,
    )
    model = load_baseline_model(model_path)

    compared = apply_formula_ml_comparison_to_dataset(dataset.head(6), model)

    assert len(compared) == 6
    assert "ml_prediction" in compared.columns
    assert "ml_good_probability" in compared.columns
    assert "formula_expected_outcome" in compared.columns
    assert "formula_ml_agreement" in compared.columns
    assert "ml_confidence_band" in compared.columns
    assert set(compared["ml_prediction"]).issubset({"good", "not_good"})


def test_infer_model_source_detects_synthetic_artifacts():
    assert infer_model_source("models/mining_outcome_baseline.joblib") == MODEL_SOURCE_MANUAL
    assert infer_model_source("models/mining_outcome_baseline_synthetic.joblib") == MODEL_SOURCE_SYNTHETIC
    assert "Synthetic" in model_source_warning(MODEL_SOURCE_SYNTHETIC)


def test_cleanup_export_dataframe_drops_unnamed_index_columns():
    raw = make_dataset(rows=4).copy()
    raw.insert(0, "Unnamed: 0", [0, 1, 2, 3])

    cleaned = cleanup_export_dataframe(raw)

    assert "Unnamed: 0" not in cleaned.columns
    assert "event_id" in cleaned.columns


def test_comparison_actual_outcome_coverage_counts_unknowns():
    dataset = make_dataset(rows=4)
    dataset.loc[0:1, "actual_outcome"] = "unknown"

    coverage = comparison_actual_outcome_coverage(dataset)

    assert coverage["row_count"] == 4
    assert coverage["known_outcome_count"] == 2
    assert coverage["unknown_outcome_count"] == 2
    assert coverage["known_outcome_ratio"] == 0.5


def test_build_comparison_export_dataframe_is_stable_and_clean(tmp_path):
    dataset = make_dataset(rows=40)
    dataset.insert(0, "Unnamed: 0", range(len(dataset)))
    model_path = tmp_path / "synthetic_model.joblib"
    report_path = tmp_path / "report.json"

    train_baseline_model(
        dataset=dataset,
        model_path=model_path,
        report_path=report_path,
        min_labeled_rows=10,
        test_size=0.25,
        random_state=7,
        model_source=MODEL_SOURCE_SYNTHETIC,
    )
    model = load_baseline_model(model_path)

    compared = apply_formula_ml_comparison_to_dataset(
        dataset.head(6),
        model,
        model_source=MODEL_SOURCE_SYNTHETIC,
    )
    export_frame = build_comparison_export_dataframe(compared)
    csv_payload = comparison_export_csv(compared)

    assert "Unnamed: 0" not in export_frame.columns
    assert "model_source" in export_frame.columns
    assert set(export_frame["model_source"]) == {MODEL_SOURCE_SYNTHETIC}
    assert not csv_payload.startswith(",")
    assert "Unnamed: 0" not in csv_payload
