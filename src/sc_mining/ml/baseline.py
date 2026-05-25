from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sc_mining.dataset.exporter import NUMERIC_COLUMNS
from sc_mining.storage.event_reader import LABELED_OUTCOME_VALUES


MODEL_VERSION = "baseline_rf_v1"
POSITIVE_OUTCOME = "good"
TARGET_COLUMN = "is_good_outcome"

FEATURE_COLUMNS = [
    "ship_type",
    "build_id",
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
    "verdict",
]

NUMERIC_FEATURES = [
    column for column in NUMERIC_COLUMNS if column in FEATURE_COLUMNS
]

CATEGORICAL_FEATURES = [
    "ship_type",
    "build_id",
    "verdict",
]

MIN_LABELED_ROWS_FOR_TRAINING = 30
MIN_CLASSES_FOR_TRAINING = 2
DEFAULT_TEST_SIZE = 0.30


@dataclass(frozen=True)
class TrainingReadiness:
    ready: bool
    reason: str
    labeled_rows: int
    class_distribution: dict[str, int]


@dataclass(frozen=True)
class BaselineTrainingResult:
    model_version: str
    model_path: str
    report_path: str
    rows_total: int
    rows_used: int
    train_rows: int
    test_rows: int
    accuracy: float
    target_distribution: dict[str, int]
    feature_columns: list[str]


def _normalize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    working = dataset.copy()

    for column in FEATURE_COLUMNS:
        if column not in working.columns:
            working[column] = None

    if "actual_outcome" not in working.columns:
        working["actual_outcome"] = "unknown"

    if "is_labeled" not in working.columns:
        working["is_labeled"] = working["actual_outcome"].isin(LABELED_OUTCOME_VALUES)

    for column in NUMERIC_FEATURES:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        working[column] = working[column].fillna("missing").astype(str)

    working["actual_outcome"] = working["actual_outcome"].fillna("unknown").astype(str)
    working["is_labeled"] = working["actual_outcome"].isin(LABELED_OUTCOME_VALUES)
    working[TARGET_COLUMN] = (working["actual_outcome"] == POSITIVE_OUTCOME).astype(int)

    return working


def prepare_training_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return labeled rows with ML features and a binary target.

    Target definition:
    - 1: actual_outcome == good
    - 0: any other labeled outcome
    """

    working = _normalize_dataset(dataset)
    labeled = working[working["is_labeled"]].copy()
    return labeled[FEATURE_COLUMNS + ["actual_outcome", TARGET_COLUMN]]


def check_training_readiness(
    dataset: pd.DataFrame,
    min_labeled_rows: int = MIN_LABELED_ROWS_FOR_TRAINING,
) -> TrainingReadiness:
    training_frame = prepare_training_frame(dataset)
    labeled_rows = int(len(training_frame))

    if training_frame.empty:
        return TrainingReadiness(
            ready=False,
            reason="No labeled rows. Set actual_outcome to values other than unknown first.",
            labeled_rows=0,
            class_distribution={},
        )

    class_distribution = {
        str(key): int(value)
        for key, value in training_frame["actual_outcome"].value_counts().to_dict().items()
    }

    binary_class_count = int(training_frame[TARGET_COLUMN].nunique())
    if labeled_rows < min_labeled_rows:
        return TrainingReadiness(
            ready=False,
            reason=(
                f"Only {labeled_rows} labeled rows. "
                f"Collect at least {min_labeled_rows} for a weak baseline."
            ),
            labeled_rows=labeled_rows,
            class_distribution=class_distribution,
        )

    if binary_class_count < MIN_CLASSES_FOR_TRAINING:
        return TrainingReadiness(
            ready=False,
            reason="Only one binary target class is present. Need both good and not-good labeled outcomes.",
            labeled_rows=labeled_rows,
            class_distribution=class_distribution,
        )

    return TrainingReadiness(
        ready=True,
        reason="Dataset is ready for baseline training.",
        labeled_rows=labeled_rows,
        class_distribution=class_distribution,
    )


def build_baseline_pipeline(random_state: int = 42) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _can_stratify(target: pd.Series) -> bool:
    value_counts = target.value_counts()
    return int(value_counts.min()) >= 2 and int(value_counts.size) >= 2


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def train_baseline_model(
    dataset: pd.DataFrame,
    model_path: str | Path,
    report_path: str | Path,
    min_labeled_rows: int = MIN_LABELED_ROWS_FOR_TRAINING,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
) -> BaselineTrainingResult:
    """Train a first binary baseline model: good vs not-good outcome."""

    readiness = check_training_readiness(dataset, min_labeled_rows=min_labeled_rows)
    if not readiness.ready:
        raise ValueError(readiness.reason)

    training_frame = prepare_training_frame(dataset)
    features = training_frame[FEATURE_COLUMNS]
    target = training_frame[TARGET_COLUMN]

    stratify = target if _can_stratify(target) else None

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = build_baseline_pipeline(random_state=random_state)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    accuracy = float(accuracy_score(y_test, predictions))
    labels = [0, 1]
    report = {
        "model_version": MODEL_VERSION,
        "target_definition": {
            "positive_class": "actual_outcome == good",
            "negative_class": "any other labeled actual_outcome",
        },
        "rows_total": int(len(dataset)),
        "rows_used": int(len(training_frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "accuracy": round(accuracy, 4),
        "target_distribution": {
            str(key): int(value)
            for key, value in target.value_counts().sort_index().to_dict().items()
        },
        "actual_outcome_distribution": readiness.class_distribution,
        "feature_columns": FEATURE_COLUMNS,
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=["not_good", "good"],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "test_predictions": [
            {
                "actual": int(actual),
                "predicted": int(predicted),
                "good_probability": round(float(probability), 4),
            }
            for actual, predicted, probability in zip(y_test.tolist(), predictions.tolist(), probabilities.tolist())
        ],
    }

    target_model_path = Path(model_path)
    target_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target_model_path)

    target_report_path = Path(report_path)
    _write_json(target_report_path, report)

    return BaselineTrainingResult(
        model_version=MODEL_VERSION,
        model_path=str(target_model_path),
        report_path=str(target_report_path),
        rows_total=int(len(dataset)),
        rows_used=int(len(training_frame)),
        train_rows=int(len(x_train)),
        test_rows=int(len(x_test)),
        accuracy=round(accuracy, 4),
        target_distribution=report["target_distribution"],
        feature_columns=FEATURE_COLUMNS,
    )


def load_baseline_model(model_path: str | Path) -> Pipeline:
    return joblib.load(model_path)


def predict_good_probability(model: Pipeline, dataset: pd.DataFrame) -> pd.DataFrame:
    """Attach good/not-good predictions to rows with the same feature schema."""

    working = _normalize_dataset(dataset)
    features = working[FEATURE_COLUMNS]

    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)

    output = working.copy()
    output["ml_good_probability"] = probabilities.round(4)
    output["ml_prediction"] = ["good" if value == 1 else "not_good" for value in predictions]
    output["ml_model_version"] = MODEL_VERSION

    return output


def result_to_dict(result: BaselineTrainingResult) -> dict[str, Any]:
    return asdict(result)
