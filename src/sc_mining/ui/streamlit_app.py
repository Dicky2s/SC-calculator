from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.dataset.exporter import build_dataset, export_dataset, get_dataset_export_summary
from sc_mining.dataset.quality import (
    build_quality_report,
    distribution_to_dataframe,
    quality_issues_to_dataframe,
)
from sc_mining.domain.models import (
    BeamState,
    CalculationInput,
    OutcomeFeedback,
    RockInput,
)
from sc_mining.storage.event_logger import save_calculation_event
from sc_mining.storage.event_reader import get_events_summary, load_events_dataframe


CONFIG_DIR = Path("configs")
BUILDS_DIR = CONFIG_DIR / "builds"
EVENTS_PATH = Path("data") / "sessions" / "manual_events.jsonl"
DATASET_PATH = Path("data") / "datasets" / "mining_events.csv"

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
