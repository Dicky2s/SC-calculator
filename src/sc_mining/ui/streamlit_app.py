from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.dataset.exporter import build_dataset, export_dataset, get_dataset_export_summary
from sc_mining.dataset.analytics import (
    build_basic_analytics_report,
    build_feature_signal_table,
    build_formula_diagnostics,
    build_formula_outcome_matrix,
    build_outcome_numeric_summary,
)
from sc_mining.dataset.quality import (
    build_quality_report,
    distribution_to_dataframe,
    quality_issues_to_dataframe,
)
from sc_mining.dataset.synthetic import export_synthetic_dataset, synthetic_summary
from sc_mining.ml.baseline import (
    MIN_LABELED_ROWS_FOR_TRAINING,
    check_training_readiness,
    load_baseline_model,
    result_to_dict,
    train_baseline_model,
)
from sc_mining.ml.comparison import (
    apply_formula_ml_comparison_to_dataset,
    build_comparison_export_dataframe,
    compare_formula_with_model,
    comparison_actual_outcome_coverage,
    comparison_export_csv,
    comparison_to_dict,
    infer_model_source,
    model_source_warning,
)
from sc_mining.domain.models import (
    BeamState,
    CalculationInput,
    OutcomeFeedback,
    RockInput,
)
from sc_mining.storage.event_logger import save_calculation_event
from sc_mining.storage.outcome_labeler import update_event_outcome
from sc_mining.storage.event_reader import get_events_summary, load_events_dataframe


CONFIG_DIR = Path("configs")
BUILDS_DIR = CONFIG_DIR / "builds"
EVENTS_PATH = Path("data") / "sessions" / "manual_events.jsonl"
DATASET_PATH = Path("data") / "datasets" / "mining_events.csv"
SYNTHETIC_DATASET_PATH = Path("data") / "datasets" / "mining_events_synthetic.csv"
MODEL_PATH = Path("models") / "mining_outcome_baseline.joblib"
MODEL_REPORT_PATH = Path("reports") / "baseline_model_report.json"
SYNTHETIC_MODEL_PATH = Path("models") / "mining_outcome_baseline_synthetic.joblib"
SYNTHETIC_MODEL_REPORT_PATH = Path("reports") / "baseline_model_report_synthetic.json"

OUTCOME_OPTIONS = {
    "unknown": "Unknown / not checked yet",
    "good": "Good — реально стоило брать",
    "bad": "Bad — по факту не стоило брать",
    "too_slow": "Too slow — слишком долго печётся",
    "too_unstable": "Too unstable — слишком нестабильно",
    "not_enough_power": "Not enough power — не хватило мощности",
    "overheated": "Overheated — перегрев / критический риск",
    "wrong_prediction": "Wrong prediction — калькулятор ошибся",
}


def list_build_files() -> list[Path]:
    return sorted(BUILDS_DIR.glob("*.yaml"))


def format_verdict(verdict: str) -> str:
    if verdict == "take":
        return "TAKE"
    if verdict == "risky":
        return "RISKY"
    if verdict == "skip":
        return "SKIP"
    if verdict == "need_more_power":
        return "NEED MORE POWER"
    return verdict.upper()


def default_session_id() -> str:
    return "manual_" + datetime.now().strftime("%Y_%m_%d")


def render_result_metrics(result) -> None:
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    metric_col1.metric("Verdict", format_verdict(result.verdict))
    metric_col2.metric("Required", result.required_power)
    metric_col3.metric("Effective", result.effective_power)
    metric_col4.metric("Margin", result.margin)
    metric_col5.metric("Risk", result.risk_score)


def render_outcome_form() -> OutcomeFeedback:
    st.subheader("Actual outcome")
    st.caption(
        "Optional manual label. Use it after checking the rock in-game. "
        "This is the future ML target."
    )

    outcome_key = st.selectbox(
        "Actual result",
        options=list(OUTCOME_OPTIONS.keys()),
        format_func=lambda key: OUTCOME_OPTIONS[key],
        index=0,
    )

    comment = st.text_area(
        "Outcome comment",
        placeholder="Example: fractured fine, but too slow for this value / unstable above 70% / worth taking with 2 beams",
        height=90,
    )

    return OutcomeFeedback(
        actual_outcome=outcome_key,
        comment=comment.strip(),
    )




def format_event_option(row: pd.Series) -> str:
    event_id = str(row.get("event_id", ""))
    short_id = event_id[:8] if event_id else "missing"
    timestamp = str(row.get("timestamp", ""))[:19]
    ship_type = row.get("ship_type", "")
    mass = row.get("mass", "")
    verdict = row.get("verdict", "")
    outcome = row.get("actual_outcome", "unknown")
    return f"{short_id} | {timestamp} | {ship_type} | mass={mass} | verdict={verdict} | outcome={outcome}"


def render_outcome_labeling_queue(df: pd.DataFrame) -> None:
    st.subheader("Outcome labeling queue")
    st.caption(
        "Update unknown outcomes after checking rocks in-game. "
        "These labels become the supervised ML target."
    )

    if df.empty:
        st.info("No events available for labeling.")
        return

    normalized_outcome = df["actual_outcome"].fillna("unknown").astype(str)
    unknown_count = int((normalized_outcome == "unknown").sum())
    labeled_count = int((normalized_outcome != "unknown").sum())

    queue_col1, queue_col2, queue_col3 = st.columns(3)
    queue_col1.metric("Unknown", unknown_count)
    queue_col2.metric("Labeled", labeled_count)
    queue_col3.metric("Total", len(df))

    show_unknown_only = st.checkbox(
        "Show unknown outcomes only",
        value=True,
        key="labeling_show_unknown_only",
    )

    queue = df.copy()
    if show_unknown_only:
        queue = queue[queue["actual_outcome"].fillna("unknown").astype(str) == "unknown"]

    if queue.empty:
        st.success("No unknown events in the current labeling queue.")
        return

    queue = queue.sort_values("timestamp", ascending=False, na_position="last")
    event_ids = queue["event_id"].astype(str).tolist()
    option_labels = {
        str(row["event_id"]): format_event_option(row)
        for _, row in queue.iterrows()
    }

    with st.form("outcome_labeling_form"):
        selected_event_id = st.selectbox(
            "Event to label",
            options=event_ids,
            format_func=lambda event_id: option_labels.get(str(event_id), str(event_id)),
        )

        selected_row = queue[queue["event_id"].astype(str) == str(selected_event_id)].iloc[0]

        preview_columns = [
            "event_id",
            "timestamp",
            "ship_type",
            "build_id",
            "mass",
            "resistance",
            "instability",
            "distance",
            "margin",
            "risk_score",
            "verdict",
            "actual_outcome",
            "outcome_comment",
        ]
        st.write("Selected event preview")
        st.dataframe(
            pd.DataFrame([selected_row[preview_columns].to_dict()]),
            width="stretch",
        )

        label_options = [key for key in OUTCOME_OPTIONS.keys() if key != "unknown"]
        new_outcome = st.selectbox(
            "New actual outcome",
            options=label_options,
            format_func=lambda key: OUTCOME_OPTIONS[key],
        )
        new_comment = st.text_area(
            "Label comment",
            value=str(selected_row.get("outcome_comment") or ""),
            placeholder="Example: в игре взялся нормально / был слишком нестабилен / мощности не хватило",
            height=90,
        )

        submitted = st.form_submit_button("Update outcome label")

    if submitted:
        try:
            update_result = update_event_outcome(
                path=EVENTS_PATH,
                event_id=str(selected_event_id),
                outcome=OutcomeFeedback(
                    actual_outcome=new_outcome,
                    comment=new_comment.strip(),
                ),
            )
        except Exception as exc:
            st.error(f"Could not update event outcome: {exc}")
            return

        st.success(
            f"Updated {update_result['event_id']}: "
            f"{update_result['previous_outcome']} → {update_result['actual_outcome']}"
        )
        st.rerun()

def render_basic_analytics_block(dataset: pd.DataFrame) -> None:
    st.subheader("Basic analytics")
    st.caption(
        "Inspects labeled outcomes and highlights where the rule-based formula agrees or disagrees with real results."
    )

    report = build_basic_analytics_report(dataset)

    if report["labeled_row_count"] == 0:
        st.info(
            "No labeled events yet. Add actual_outcome values other than unknown to enable analytics."
        )
        return

    analytics_col1, analytics_col2, analytics_col3, analytics_col4, analytics_col5, analytics_col6 = st.columns(6)
    analytics_col1.metric("Labeled", report["labeled_row_count"])
    analytics_col2.metric("Good", report["good_count"])
    analytics_col3.metric("Not good", report["not_good_count"])
    analytics_col4.metric("Dangerous take", report["dangerous_take_count"])
    analytics_col5.metric("Missed opportunity", report["missed_opportunity_count"])
    analytics_col6.metric("Risky bad", report["risky_bad_count"])

    matrix = build_formula_outcome_matrix(dataset)
    diagnostics = build_formula_diagnostics(dataset)
    feature_signal = build_feature_signal_table(dataset)
    outcome_numeric_summary = build_outcome_numeric_summary(
        dataset,
        features=[
            "mass",
            "resistance",
            "instability",
            "distance",
            "beam_power_sum",
            "margin",
            "risk_score",
        ],
    )

    st.write("Formula verdict vs actual outcome")
    if matrix.empty:
        st.info("No formula/outcome pairs available.")
    else:
        matrix_pivot = matrix.pivot_table(
            index="verdict",
            columns="actual_outcome",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        st.dataframe(matrix_pivot, width="stretch")

    st.write("Feature signal: good vs not-good outcomes")
    if feature_signal.empty:
        st.info("Not enough labeled data to compare feature signals.")
    else:
        st.dataframe(feature_signal, width="stretch")

    st.write("Numeric summary by actual outcome")
    if outcome_numeric_summary.empty:
        st.info("No numeric outcome summary available.")
    else:
        st.dataframe(outcome_numeric_summary, width="stretch")

    st.write("Formula diagnostics")
    if diagnostics.empty:
        st.info("No diagnostics available.")
    else:
        selected_labels = st.multiselect(
            "Diagnostic labels",
            options=sorted(diagnostics["analytics_label"].unique()),
            default=sorted(diagnostics["analytics_label"].unique()),
        )
        diagnostics_view = diagnostics
        if selected_labels:
            diagnostics_view = diagnostics_view[diagnostics_view["analytics_label"].isin(selected_labels)]
        st.dataframe(diagnostics_view, width="stretch")


def render_ml_baseline_block(
    dataset: pd.DataFrame,
    title: str = "Baseline ML model",
    model_path: Path = MODEL_PATH,
    report_path: Path = MODEL_REPORT_PATH,
    key_prefix: str = "manual",
    default_min_labeled_rows: int = MIN_LABELED_ROWS_FOR_TRAINING,
    model_source: str = "manual_baseline",
) -> None:
    st.subheader(title)
    st.caption(
        "Trains a first weak supervised model on labeled events. "
        "Target: good outcome vs not-good outcome. Use this as a baseline, not as final truth."
    )

    min_labeled_rows = st.number_input(
        "Minimum labeled rows for training",
        min_value=4,
        max_value=1000,
        value=default_min_labeled_rows,
        step=1,
        help="Keep 30+ for a meaningful weak baseline. Lower values are only for smoke testing the pipeline.",
        key=f"{key_prefix}_min_labeled_rows",
    )

    readiness = check_training_readiness(
        dataset,
        min_labeled_rows=int(min_labeled_rows),
    )

    ml_col1, ml_col2, ml_col3 = st.columns(3)
    ml_col1.metric("Ready", "YES" if readiness.ready else "NO")
    ml_col2.metric("Labeled rows", readiness.labeled_rows)
    ml_col3.metric("Target classes", len(readiness.class_distribution))

    st.write(f"Reason: {readiness.reason}")

    if readiness.class_distribution:
        st.write("Actual outcome class distribution")
        st.dataframe(
            distribution_to_dataframe(
                readiness.class_distribution,
                "actual_outcome",
            ),
            width="stretch",
        )

    st.write(f"Model output: `{model_path}`")
    st.write(f"Report output: `{report_path}`")

    train_clicked = st.button(
        "Train baseline model",
        disabled=not readiness.ready,
        key=f"{key_prefix}_train_baseline_model",
    )

    if train_clicked:
        try:
            training_result = train_baseline_model(
                dataset=dataset,
                model_path=model_path,
                report_path=report_path,
                min_labeled_rows=int(min_labeled_rows),
                model_source=model_source,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success(
            f"Trained {training_result.model_version}: "
            f"accuracy={training_result.accuracy}, rows_used={training_result.rows_used}"
        )
        st.json(result_to_dict(training_result))

    if report_path.exists():
        with st.expander("Last baseline report", expanded=False):
            st.json(report_path.read_text(encoding="utf-8"))



def available_model_paths() -> dict[str, Path]:
    options: dict[str, Path] = {}
    if MODEL_PATH.exists():
        options["Manual baseline model"] = MODEL_PATH
    if SYNTHETIC_MODEL_PATH.exists():
        options["Synthetic smoke-test model"] = SYNTHETIC_MODEL_PATH
    return options


def render_calculator_ml_comparison(calc_input: CalculationInput, result) -> None:
    st.subheader("Formula vs ML comparison")
    st.caption(
        "Compares the rule-based formula verdict with the latest trained baseline model. "
        "Use synthetic models only to smoke-test the integration."
    )

    model_options = available_model_paths()
    if not model_options:
        st.info(
            "No trained model found yet. Train a manual baseline model or the synthetic smoke-test model first."
        )
        return

    selected_label = st.selectbox(
        "Model for comparison",
        options=list(model_options.keys()),
        key="calculator_ml_model_selector",
    )
    selected_path = model_options[selected_label]

    selected_model_source = infer_model_source(selected_path)
    warning = model_source_warning(selected_model_source)
    if warning:
        st.warning(warning)

    comparison = compare_formula_with_model(
        calc_input=calc_input,
        result=result,
        model_path=selected_path,
        model_source=selected_model_source,
    )

    if not comparison.model_available:
        st.info(comparison.reason)
        return

    comp_col1, comp_col2, comp_col3, comp_col4, comp_col5 = st.columns(5)
    comp_col1.metric("Formula verdict", format_verdict(comparison.formula_verdict))
    comp_col2.metric("Formula expects", comparison.formula_expected_outcome)
    comp_col3.metric("ML prediction", comparison.ml_prediction)
    comp_col4.metric("ML good probability", comparison.ml_good_probability)
    comp_col5.metric("Model source", comparison.model_source)

    st.write(f"Agreement: `{comparison.agreement_label}`")
    st.write(f"Confidence band: `{comparison.confidence_band}`")
    st.caption(comparison.recommendation)

    with st.expander("Raw comparison payload", expanded=False):
        st.json(comparison_to_dict(comparison))


def render_dataset_ml_comparison_block(dataset: pd.DataFrame) -> None:
    st.subheader("Formula vs ML dataset comparison")
    st.caption(
        "Applies a trained baseline model to the current dataset and highlights rows where ML and the formula disagree."
    )

    model_options = available_model_paths()
    if not model_options:
        st.info("No trained model found yet. Train a baseline model before running dataset comparison.")
        return

    selected_label = st.selectbox(
        "Model for dataset comparison",
        options=list(model_options.keys()),
        key="dataset_ml_model_selector",
    )
    selected_path = model_options[selected_label]

    selected_model_source = infer_model_source(selected_path)
    warning = model_source_warning(selected_model_source)
    if warning:
        st.warning(warning)

    try:
        model = load_baseline_model(selected_path)
        compared = apply_formula_ml_comparison_to_dataset(
            dataset,
            model,
            model_source=selected_model_source,
        )
    except Exception as exc:  # defensive UI boundary; tests cover normal path
        st.error(f"Could not apply model: {exc}")
        return

    if compared.empty:
        st.info("No rows available for comparison.")
        return

    agreement_distribution = (
        compared.groupby("formula_ml_agreement", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    st.write("Agreement distribution")
    st.dataframe(agreement_distribution, width="stretch")

    coverage = comparison_actual_outcome_coverage(compared)
    coverage_col1, coverage_col2, coverage_col3, coverage_col4 = st.columns(4)
    coverage_col1.metric("Rows", coverage["row_count"])
    coverage_col2.metric("Known outcomes", coverage["known_outcome_count"])
    coverage_col3.metric("Unknown outcomes", coverage["unknown_outcome_count"])
    coverage_col4.metric("Known outcome ratio", coverage["known_outcome_ratio"])

    labels = sorted(compared["formula_ml_agreement"].dropna().unique())
    selected_labels = st.multiselect(
        "Agreement labels",
        options=labels,
        default=labels,
        key="dataset_ml_agreement_filter",
    )

    view = compared
    if selected_labels:
        view = view[view["formula_ml_agreement"].isin(selected_labels)]

    export_view = build_comparison_export_dataframe(view)
    st.dataframe(export_view, width="stretch")

    st.download_button(
        "Download clean comparison CSV",
        data=comparison_export_csv(view),
        file_name="formula_vs_ml_comparison.csv",
        mime="text/csv",
        help="Exports without the extra pandas/Streamlit index column such as Unnamed: 0.",
    )

def render_synthetic_dataset_block() -> pd.DataFrame:
    st.subheader("Synthetic smoke-test dataset")
    st.caption(
        "Generates labeled synthetic rows for validating the analytics/training pipeline. "
        "Do not treat synthetic metrics as real gameplay model quality."
    )

    synth_col1, synth_col2, synth_col3, synth_col4 = st.columns([1, 1, 1, 2])

    with synth_col1:
        row_count = st.number_input(
            "Synthetic rows",
            min_value=10,
            max_value=5000,
            value=100,
            step=10,
            key="synthetic_row_count",
        )

    with synth_col2:
        seed = st.number_input(
            "Seed",
            min_value=1,
            max_value=999999,
            value=42,
            step=1,
            key="synthetic_seed",
        )

    with synth_col3:
        good_ratio = st.slider(
            "Good ratio",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            key="synthetic_good_ratio",
        )

    with synth_col4:
        st.write(f"Output: `{SYNTHETIC_DATASET_PATH}`")

    generate_clicked = st.button("Generate synthetic CSV", key="generate_synthetic_csv")

    if generate_clicked:
        synthetic_dataset = export_synthetic_dataset(
            output_path=SYNTHETIC_DATASET_PATH,
            row_count=int(row_count),
            seed=int(seed),
            good_ratio=float(good_ratio),
        )
        st.success(
            f"Generated {len(synthetic_dataset)} synthetic rows at {SYNTHETIC_DATASET_PATH}"
        )
        st.json(synthetic_summary(synthetic_dataset))

    if not SYNTHETIC_DATASET_PATH.exists():
        st.info("No synthetic dataset yet. Generate it to smoke-test training without real labels.")
        return pd.DataFrame()

    synthetic_dataset = pd.read_csv(SYNTHETIC_DATASET_PATH)
    st.warning(
        "Synthetic data is only for pipeline smoke tests. Do not merge it into the real manual dataset."
    )

    synth_summary = synthetic_summary(synthetic_dataset)
    synth_summary_col1, synth_summary_col2, synth_summary_col3 = st.columns(3)
    synth_summary_col1.metric("Rows", synth_summary["row_count"])
    synth_summary_col2.metric("Labeled", synth_summary["labeled_count"])
    synth_summary_col3.metric("Unlabeled", synth_summary["unlabeled_count"])

    with st.expander("Synthetic outcome distribution", expanded=False):
        st.dataframe(
            distribution_to_dataframe(
                synth_summary["actual_outcome_distribution"],
                "actual_outcome",
            ),
            width="stretch",
        )

    return synthetic_dataset

def render_calculator_tab(heads, modules, build, session_id: str) -> None:
    st.subheader("Rock parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mass = st.number_input(
            "Mass",
            min_value=1.0,
            value=12600.0,
            step=100.0,
        )

    with col2:
        resistance = st.number_input(
            "Resistance",
            min_value=0.0,
            value=0.34,
            step=0.01,
            format="%.2f",
        )

    with col3:
        instability = st.number_input(
            "Instability",
            min_value=0.0,
            value=0.12,
            step=0.01,
            format="%.2f",
        )

    with col4:
        distance = st.number_input(
            "Distance",
            min_value=1.0,
            value=92.0,
            step=1.0,
        )

    st.subheader("Beam states")

    beams: list[BeamState] = []

    for head in build.heads:
        col_a, col_b = st.columns([1, 2])

        default_enabled = head.slot in {"main", "left"}

        with col_a:
            enabled = st.checkbox(
                f"Enable beam: {head.slot}",
                value=default_enabled,
            )

        with col_b:
            power = st.slider(
                f"Power %: {head.slot}",
                min_value=0,
                max_value=100,
                value=65,
                step=1,
                disabled=not enabled,
            )

        if enabled:
            beams.append(
                BeamState(
                    slot=head.slot,
                    power_percent=float(power),
                    active_modules=[],
                )
            )

    calc_input = CalculationInput(
        rock=RockInput(
            mass=mass,
            resistance=resistance,
            instability=instability,
            distance=distance,
        ),
        build=build,
        beams=beams,
    )

    result = calculate(calc_input, heads=heads, modules=modules)

    st.subheader("Result")
    render_result_metrics(result)

    render_calculator_ml_comparison(calc_input, result)

    outcome = render_outcome_form()

    st.subheader("Save event")

    col_save, col_path = st.columns([1, 3])

    with col_save:
        save_clicked = st.button("Save event", type="primary")

    with col_path:
        st.write(f"Output: `{EVENTS_PATH}`")

    if save_clicked:
        event = save_calculation_event(
            path=EVENTS_PATH,
            session_id=session_id,
            calc_input=calc_input,
            result=result,
            source="manual_ui",
            outcome=outcome,
        )

        st.success(f"Saved event: {event['event_id']}")

    st.subheader("Details")

    rows = [
        {"metric": "session_id", "value": str(session_id)},
        {"metric": "build_id", "value": str(build.build_id)},
        {"metric": "ship_type", "value": str(build.ship_type)},
        {"metric": "required_power", "value": str(result.required_power)},
        {"metric": "effective_power", "value": str(result.effective_power)},
        {"metric": "margin", "value": str(result.margin)},
        {"metric": "risk_score", "value": str(result.risk_score)},
        {"metric": "verdict", "value": str(result.verdict)},
        {"metric": "actual_outcome", "value": str(outcome.actual_outcome)},
    ]

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

    if result.notes:
        st.subheader("Notes")
        for note in result.notes:
            st.write(f"- {note}")


def render_saved_events_tab() -> None:
    st.subheader("Saved events")
    st.write(f"Source: `{EVENTS_PATH}`")

    df = load_events_dataframe(EVENTS_PATH)
    summary = get_events_summary(df)

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Events", summary["event_count"])
    metric_col2.metric("Labeled", summary["labeled_event_count"])
    metric_col3.metric("Sessions", summary["session_count"])
    metric_col4.metric("Builds", summary["build_count"])
    metric_col5.metric("Ships", summary["ship_count"])

    if df.empty:
        st.info("No saved events yet. Save a calculation from the Calculator tab first.")
        return

    filtered = df.copy()

    with st.expander("Filters", expanded=True):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            session_values = sorted(
                value for value in filtered["session_id"].dropna().unique()
            )
            selected_sessions = st.multiselect(
                "Session",
                options=session_values,
                default=session_values,
            )

        with filter_col2:
            ship_values = sorted(
                value for value in filtered["ship_type"].dropna().unique()
            )
            selected_ships = st.multiselect(
                "Ship",
                options=ship_values,
                default=ship_values,
            )

        with filter_col3:
            verdict_values = sorted(
                value for value in filtered["verdict"].dropna().unique()
            )
            selected_verdicts = st.multiselect(
                "Verdict",
                options=verdict_values,
                default=verdict_values,
            )

        with filter_col4:
            outcome_values = sorted(
                value for value in filtered["actual_outcome"].dropna().unique()
            )
            selected_outcomes = st.multiselect(
                "Actual outcome",
                options=outcome_values,
                default=outcome_values,
            )

    if selected_sessions:
        filtered = filtered[filtered["session_id"].isin(selected_sessions)]

    if selected_ships:
        filtered = filtered[filtered["ship_type"].isin(selected_ships)]

    if selected_verdicts:
        filtered = filtered[filtered["verdict"].isin(selected_verdicts)]

    if selected_outcomes:
        filtered = filtered[filtered["actual_outcome"].isin(selected_outcomes)]

    filtered = filtered.sort_values("timestamp", ascending=False, na_position="last")

    st.subheader("Filtered events")
    st.dataframe(filtered, width="stretch")

    render_outcome_labeling_queue(df)

    verdict_chart_data = (
        filtered.groupby("verdict", dropna=False)
        .size()
        .reset_index(name="count")
        .set_index("verdict")
    )

    st.subheader("Verdict distribution")
    st.bar_chart(verdict_chart_data)

    outcome_chart_data = (
        filtered.groupby("actual_outcome", dropna=False)
        .size()
        .reset_index(name="count")
        .set_index("actual_outcome")
    )

    st.subheader("Actual outcome distribution")
    st.bar_chart(outcome_chart_data)

    numeric_columns = [
        "mass",
        "resistance",
        "instability",
        "distance",
        "beam_power_sum",
        "required_power",
        "effective_power",
        "margin",
        "risk_score",
    ]

    st.subheader("Numeric summary")
    st.dataframe(
        filtered[numeric_columns].describe().round(3),
        width="stretch",
    )

    st.subheader("Dataset export")
    st.caption(
        "Export raw JSONL events into a flat CSV table for analytics and future ML training."
    )

    export_labeled_only = st.checkbox(
        "Export only labeled events",
        value=False,
        help="Use this when preparing a training dataset. Unknown outcomes are useful for logs, but not for supervised ML.",
    )

    export_col1, export_col2 = st.columns([1, 3])

    with export_col1:
        export_clicked = st.button("Export CSV")

    with export_col2:
        st.write(f"Output: `{DATASET_PATH}`")

    if export_clicked:
        exported_dataset = export_dataset(
            events_path=EVENTS_PATH,
            output_path=DATASET_PATH,
            labeled_only=export_labeled_only,
        )
        export_summary = get_dataset_export_summary(exported_dataset)

        st.success(
            f"Exported {export_summary['row_count']} rows "
            f"to {DATASET_PATH}"
        )
        st.json(export_summary)

    st.subheader("Dataset quality")
    st.caption(
        "Checks whether the exported dataset is usable for analytics and future supervised ML."
    )

    quality_dataset = build_dataset(EVENTS_PATH, labeled_only=False)
    quality_report = build_quality_report(quality_dataset)

    quality_col1, quality_col2, quality_col3, quality_col4, quality_col5 = st.columns(5)
    quality_col1.metric("Status", quality_report["status"].upper())
    quality_col2.metric("Rows", quality_report["row_count"])
    quality_col3.metric("Labeled", quality_report["labeled_count"])
    quality_col4.metric("Unknown", quality_report["unknown_outcome_count"])
    quality_col5.metric("Duplicates", quality_report["duplicate_event_id_count"])

    if quality_report["status"] == "ok":
        st.success("Dataset quality status: OK for basic analytics.")
    elif quality_report["status"] == "warn":
        st.warning("Dataset quality status: WARN. Good for inspection, not ready for reliable ML yet.")
    else:
        st.error("Dataset quality status: FAIL. Fix critical issues before ML training.")

    issues_df = quality_issues_to_dataframe(quality_report)
    st.write("Quality issues")
    if issues_df.empty:
        st.success("No quality issues found.")
    else:
        st.dataframe(issues_df, width="stretch")

    quality_view_col1, quality_view_col2, quality_view_col3 = st.columns(3)

    with quality_view_col1:
        st.write("Verdict distribution")
        st.dataframe(
            distribution_to_dataframe(
                quality_report["verdict_distribution"],
                "verdict",
            ),
            width="stretch",
        )

    with quality_view_col2:
        st.write("Actual outcome distribution")
        st.dataframe(
            distribution_to_dataframe(
                quality_report["actual_outcome_distribution"],
                "actual_outcome",
            ),
            width="stretch",
        )

    with quality_view_col3:
        st.write("Missing values")
        st.dataframe(
            distribution_to_dataframe(
                quality_report["missing_values"],
                "column",
            ),
            width="stretch",
        )

    synthetic_dataset = render_synthetic_dataset_block()

    render_basic_analytics_block(quality_dataset)
    render_ml_baseline_block(quality_dataset)
    render_dataset_ml_comparison_block(quality_dataset)

    if not synthetic_dataset.empty:
        with st.expander("Train on synthetic dataset for smoke test", expanded=False):
            render_ml_baseline_block(
                synthetic_dataset,
                title="Synthetic baseline smoke test",
                model_path=SYNTHETIC_MODEL_PATH,
                report_path=SYNTHETIC_MODEL_REPORT_PATH,
                key_prefix="synthetic",
                default_min_labeled_rows=30,
                model_source="synthetic_smoke_test",
            )


def main() -> None:
    st.set_page_config(
        page_title="SC Mining Assistant",
        layout="wide",
    )

    st.title("SC Mining Assistant")
    st.caption("Manual baseline calculator + event logger + outcome-labeled dataset viewer")

    heads = load_heads(CONFIG_DIR / "heads.yaml")
    modules = load_modules(CONFIG_DIR / "modules.yaml")

    build_files = list_build_files()
    if not build_files:
        st.error("No build YAML files found in configs/builds")
        return

    st.sidebar.subheader("Session")

    session_id = st.sidebar.text_input(
        "Session ID",
        value=default_session_id(),
    )

    build_file = st.sidebar.selectbox(
        "Build profile",
        build_files,
        format_func=lambda path: path.name,
    )

    build = load_build(build_file)

    st.sidebar.subheader("Current build")
    st.sidebar.write(f"Build ID: `{build.build_id}`")
    st.sidebar.write(f"Ship: `{build.ship_type}`")

    calculator_tab, saved_events_tab = st.tabs(["Calculator", "Saved events"])

    with calculator_tab:
        render_calculator_tab(
            heads=heads,
            modules=modules,
            build=build,
            session_id=session_id,
        )

    with saved_events_tab:
        render_saved_events_tab()


if __name__ == "__main__":
    main()
