from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc_mining.ml.real_run import RealMLRunConfig, real_ml_run_result_to_dict, run_real_ml_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the manual real-data ML pipeline: export dataset, validate, train, track, and optionally promote."
    )
    parser.add_argument("--events", default="data/sessions/manual_events.jsonl", help="Raw JSONL events path.")
    parser.add_argument("--dataset", default="data/datasets/mining_events.csv", help="Output CSV dataset path.")
    parser.add_argument("--model", default="models/mining_outcome_baseline_manual.joblib", help="Output manual model path.")
    parser.add_argument("--report", default="reports/baseline_model_report_manual.json", help="Output training report path.")
    parser.add_argument("--runs", default="reports/training_runs.jsonl", help="Training run history JSONL path.")
    parser.add_argument("--active-model", default="models/active_model.json", help="Active model pointer path.")
    parser.add_argument("--min-labeled", type=int, default=30, help="Minimum labeled rows required for training.")
    parser.add_argument("--min-test-rows", type=int, default=5, help="Minimum test rows required by promotion gate.")
    parser.add_argument("--min-accuracy", type=float, default=0.60, help="Minimum accuracy required by promotion gate.")
    parser.add_argument("--max-false-good-rate", type=float, default=0.25, help="Maximum false-good rate warning threshold.")
    parser.add_argument("--test-size", type=float, default=0.30, help="Test split size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split and model.")
    parser.add_argument("--no-train", action="store_true", help="Only export and validate; do not train.")
    parser.add_argument("--promote-if-passed", action="store_true", help="Write active_model.json if promotion gate passes.")
    parser.add_argument("--notes", default="CLI real ML run", help="Notes stored in training_runs.jsonl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RealMLRunConfig(
        events_path=Path(args.events),
        dataset_path=Path(args.dataset),
        model_path=Path(args.model),
        report_path=Path(args.report),
        training_runs_path=Path(args.runs),
        active_model_path=Path(args.active_model),
        min_labeled_rows=args.min_labeled,
        min_test_rows=args.min_test_rows,
        min_accuracy=args.min_accuracy,
        max_false_good_rate=args.max_false_good_rate,
        test_size=args.test_size,
        random_state=args.seed,
        train_if_ready=not args.no_train,
        promote_if_passed=args.promote_if_passed,
        notes=args.notes,
    )
    result = run_real_ml_pipeline(config)
    print(json.dumps(real_ml_run_result_to_dict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
