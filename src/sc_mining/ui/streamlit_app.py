from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import BeamState, CalculationInput, RockInput
from sc_mining.storage.event_logger import save_calculation_event
from sc_mining.storage.event_reader import get_events_summary, load_events_dataframe


CONFIG_DIR = Path("configs")
BUILDS_DIR = CONFIG_DIR / "builds"
EVENTS_PATH = Path("data") / "sessions" / "manual_events.jsonl"


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

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Events", summary["event_count"])
    metric_col2.metric("Sessions", summary["session_count"])
    metric_col3.metric("Builds", summary["build_count"])
    metric_col4.metric("Ships", summary["ship_count"])

    if df.empty:
        st.info("No saved events yet. Save a calculation from the Calculator tab first.")
        return

    filtered = df.copy()

    with st.expander("Filters", expanded=True):
        filter_col1, filter_col2, filter_col3 = st.columns(3)

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

    if selected_sessions:
        filtered = filtered[filtered["session_id"].isin(selected_sessions)]

    if selected_ships:
        filtered = filtered[filtered["ship_type"].isin(selected_ships)]

    if selected_verdicts:
        filtered = filtered[filtered["verdict"].isin(selected_verdicts)]

    filtered = filtered.sort_values("timestamp", ascending=False, na_position="last")

    st.subheader("Filtered events")
    st.dataframe(filtered, width="stretch")

    chart_data = (
        filtered.groupby("verdict", dropna=False)
        .size()
        .reset_index(name="count")
        .set_index("verdict")
    )

    st.subheader("Verdict distribution")
    st.bar_chart(chart_data)

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


def main() -> None:
    st.set_page_config(
        page_title="SC Mining Assistant",
        layout="wide",
    )

    st.title("SC Mining Assistant")
    st.caption("Manual baseline calculator + manual event logger + event dataset viewer")

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
