from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from sc_mining.ui.table_utils import make_arrow_safe_dataframe
import streamlit as st

from sc_mining.domain.calculator import calculate
from sc_mining.domain.recommendations import build_power_distance_recommendation
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
from sc_mining.ml.registry import (
    LEGACY_MANUAL_MODEL_PATH,
    LEGACY_MANUAL_MODEL_REPORT_PATH,
    MANUAL_MODEL_PATH,
    MANUAL_MODEL_REPORT_PATH,
    MODEL_SOURCE_MANUAL_REAL,
    MODEL_SOURCE_SYNTHETIC,
    SYNTHETIC_MODEL_PATH as REGISTRY_SYNTHETIC_MODEL_PATH,
    SYNTHETIC_MODEL_REPORT_PATH as REGISTRY_SYNTHETIC_MODEL_REPORT_PATH,
    default_model_artifact_specs,
    existing_model_artifacts,
    spec_to_dict,
)

from sc_mining.ml.tracking import (
    TRAINING_RUNS_PATH,
    append_training_run,
    load_training_runs,
    summarize_training_runs,
)
from sc_mining.ml.active_model import (
    ACTIVE_MODEL_CONFIG_PATH,
    build_active_model_status,
    clear_active_model_config,
    read_active_model_config,
    write_active_model_config,
)
from sc_mining.ml.prediction_logging import (
    build_prediction_log_dataframe,
    build_prediction_log_summary,
)
from sc_mining.ml.prediction_evaluation import (
    build_prediction_evaluation_dataframe,
    build_prediction_evaluation_matrix,
    build_prediction_evaluation_summary,
)
from sc_mining.ml.promotion import (
    PromotionCriteria,
    evaluate_model_promotion,
    promotion_decision_to_dict,
)
from sc_mining.ml.real_run import (
    RealMLRunConfig,
    real_ml_run_result_to_dict,
    result_to_dataframe as real_ml_run_result_to_dataframe,
    run_real_ml_pipeline,
)
from sc_mining.domain.models import (
    BeamState,
    CalculationInput,
    OutcomeFeedback,
    ResourceComponent,
    ResourceYieldFeedback,
    RefinedResourceOutput,
    RefineryFeedback,
    CalibrationFeedback,
    PowerDistanceObservation,
    RockInput,
    RunContext,
)
from sc_mining.storage.event_logger import save_calculation_event
from sc_mining.storage.outcome_labeler import update_event_outcome
from sc_mining.storage.refinery_labeler import update_event_refinery
from sc_mining.storage.calibration_labeler import update_event_calibration
from sc_mining.storage.event_reader import get_events_summary, load_events_dataframe
from sc_mining.storage.event_history import (
    build_event_detail_payload,
    build_event_timeline,
    get_event_by_id,
)


CONFIG_DIR = Path("configs")
BUILDS_DIR = CONFIG_DIR / "builds"
EVENTS_PATH = Path("data") / "sessions" / "manual_events.jsonl"
DATASET_PATH = Path("data") / "datasets" / "mining_events.csv"
SYNTHETIC_DATASET_PATH = Path("data") / "datasets" / "mining_events_synthetic.csv"
MODEL_PATH = MANUAL_MODEL_PATH
MODEL_REPORT_PATH = MANUAL_MODEL_REPORT_PATH
LEGACY_MODEL_PATH = LEGACY_MANUAL_MODEL_PATH
LEGACY_MODEL_REPORT_PATH = LEGACY_MANUAL_MODEL_REPORT_PATH
SYNTHETIC_MODEL_PATH = REGISTRY_SYNTHETIC_MODEL_PATH
SYNTHETIC_MODEL_REPORT_PATH = REGISTRY_SYNTHETIC_MODEL_REPORT_PATH

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

CALIBRATION_OBSERVATION_OPTIONS = {
    "unknown": "Unknown / not classified",
    "no_warmup": "No warm-up — beam did not heat the rock",
    "warmup": "Warm-up — starts heating / progress moves",
    "stable_hold": "Stable hold — comfortable to keep in range",
    "overpowered": "Overpowered — too much / overshoot risk",
    "too_unstable": "Too unstable — jumps too much",
    "too_slow": "Too slow — technically works but too slow",
}

RESOURCE_OPTIONS = [
    "unknown",
    "aluminum",
    "agricium",
    "bexalite",
    "beryl",
    "borase",
    "copper",
    "diamond",
    "gold",
    "hephaestanite",
    "iron",
    "laranite",
    "quantainium",
    "quartz",
    "taranite",
    "tin",
    "titanium",
    "tungsten",
    "other",
]


def list_build_files() -> list[Path]:
    return sorted(BUILDS_DIR.glob("*.yaml"))



def build_profile_label(build) -> str:
    return f"{build.build_id} ({len(build.heads)} beam slot{'s' if len(build.heads) != 1 else ''})"


def build_loadout_rows(build, modules: dict) -> list[dict]:
    rows: list[dict] = []
    for head in build.heads:
        rows.append(
            {
                "slot": head.slot,
                "head_id": head.head_id,
                "modules": ", ".join(head.modules) if head.modules else "—",
                "module_names": ", ".join(
                    modules[module_id].name if module_id in modules else module_id
                    for module_id in head.modules
                ) if head.modules else "—",
            }
        )
    return rows


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


def display_safe_dataframe(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(make_arrow_safe_dataframe(df), **kwargs)


def render_result_metrics(result) -> None:
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    metric_col1.metric("Verdict", format_verdict(result.verdict))
    metric_col2.metric("Required", result.required_power)
    metric_col3.metric("Effective", result.effective_power)
    metric_col4.metric("Margin", result.margin)
    metric_col5.metric("Risk", result.risk_score)


def render_outcome_form(key_prefix: str = "calculator") -> OutcomeFeedback:
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
        key=f"{key_prefix}_actual_outcome",
    )

    comment = st.text_area(
        "Outcome comment",
        placeholder="Example: fractured fine, but too slow for this value / unstable above 70% / worth taking with 2 beams",
        height=90,
        key=f"{key_prefix}_outcome_comment",
    )

    return OutcomeFeedback(
        actual_outcome=outcome_key,
        comment=comment.strip(),
    )


def _clean_resource_rows(rows: list[dict]) -> list[ResourceComponent]:
    resources: list[ResourceComponent] = []

    for row in rows:
        resource_name = str(row.get("resource_name", "unknown") or "unknown")
        resource_percent = row.get("resource_percent")
        raw_scu_estimate = row.get("raw_scu_estimate")
        comment = str(row.get("comment", "") or "")

        percent_missing = resource_percent in (None, "") or pd.isna(resource_percent)
        raw_scu_missing = raw_scu_estimate in (None, "") or pd.isna(raw_scu_estimate)

        if resource_name == "unknown" and percent_missing and raw_scu_missing and not comment.strip():
            continue

        resources.append(
            ResourceComponent(
                resource_name=resource_name,
                resource_percent=float(resource_percent) if not percent_missing else None,
                raw_scu_estimate=float(raw_scu_estimate) if not raw_scu_missing else None,
                comment=comment.strip(),
            )
        )

    return resources




def _fill_resource_scu_from_total(resources: list[ResourceComponent], total_scu_estimate: float | None) -> list[ResourceComponent]:
    """Fill per-resource raw SCU from total composition SCU when only percentages are known.

    The in-game scan often shows a total composition size in SCU plus percentages
    per resource. This helper preserves manually entered SCU values, but fills
    missing ones from total_scu_estimate * percent / 100.
    """
    if not total_scu_estimate or total_scu_estimate <= 0:
        return resources

    filled: list[ResourceComponent] = []
    for item in resources:
        if item.raw_scu_estimate is None and item.resource_percent is not None:
            filled.append(
                ResourceComponent(
                    resource_name=item.resource_name,
                    resource_percent=item.resource_percent,
                    raw_scu_estimate=round(total_scu_estimate * item.resource_percent / 100.0, 3),
                    comment=item.comment,
                )
            )
        else:
            filled.append(item)

    return filled


def _clean_refined_resource_rows(rows: list[dict]) -> list[RefinedResourceOutput]:
    refined_resources: list[RefinedResourceOutput] = []

    for row in rows:
        resource_name = str(row.get("resource_name", "unknown") or "unknown")
        refined_scu_actual = row.get("refined_scu_actual")
        sell_value_auec = row.get("sell_value_auec")
        comment = str(row.get("comment", "") or "")

        refined_missing = refined_scu_actual in (None, "") or pd.isna(refined_scu_actual)
        sell_missing = sell_value_auec in (None, "") or pd.isna(sell_value_auec)

        if resource_name == "unknown" and refined_missing and sell_missing and not comment.strip():
            continue

        refined_resources.append(
            RefinedResourceOutput(
                resource_name=resource_name,
                refined_scu_actual=float(refined_scu_actual) if not refined_missing else None,
                sell_value_auec=float(sell_value_auec) if not sell_missing else None,
                comment=comment.strip(),
            )
        )

    return refined_resources


def render_resource_yield_form(key_prefix: str = "calculator") -> ResourceYieldFeedback:
    st.subheader("Scan composition / resources")
    st.caption(
        "Capture the same things you see on the mining scan: total composition size in SCU and one or more resource rows. "
        "If you enter total SCU and percentages only, per-resource raw SCU is estimated automatically."
    )

    header_col1, header_col2 = st.columns(2)
    with header_col1:
        total_scu_estimate = st.number_input(
            "Composition total, SCU",
            min_value=0.0,
            value=0.0,
            step=0.1,
            key=f"{key_prefix}_total_scu_estimate",
            help="Value from the scan block, for example 'Composition 23.87 SCU'. This is stored as an extra analytics/ML feature.",
        )
    with header_col2:
        st.info("Tip: enter scan percentages first. Missing per-resource SCU values are derived from total SCU automatically.")

    default_resources = pd.DataFrame(
        [
            {"resource_name": "unknown", "resource_percent": 0.0, "raw_scu_estimate": 0.0, "comment": ""},
            {"resource_name": "unknown", "resource_percent": 0.0, "raw_scu_estimate": 0.0, "comment": ""},
            {"resource_name": "unknown", "resource_percent": 0.0, "raw_scu_estimate": 0.0, "comment": ""},
        ]
    )

    edited_resources = st.data_editor(
        default_resources,
        num_rows="dynamic",
        hide_index=True,
        key=f"{key_prefix}_resources_table",
        column_config={
            "resource_name": st.column_config.SelectboxColumn(
                "Resource",
                options=RESOURCE_OPTIONS,
                required=False,
            ),
            "resource_percent": st.column_config.NumberColumn(
                "Scan %",
                min_value=0.0,
                max_value=100.0,
                step=0.01,
            ),
            "raw_scu_estimate": st.column_config.NumberColumn(
                "Raw SCU",
                min_value=0.0,
                step=0.01,
                help="Optional. Leave empty/0 to derive from total composition SCU and scan %."
            ),
            "comment": st.column_config.TextColumn("Comment"),
        },
    )

    resources = _clean_resource_rows(edited_resources.to_dict("records"))
    total_scu_value = total_scu_estimate if total_scu_estimate > 0 else None
    resources = _fill_resource_scu_from_total(resources, total_scu_value)

    if resources:
        preview_rows = [
            {
                "resource_name": item.resource_name,
                "resource_percent": item.resource_percent,
                "raw_scu_estimate": item.raw_scu_estimate,
                "comment": item.comment,
            }
            for item in resources
        ]
        with st.expander("Derived resource preview", expanded=False):
            display_safe_dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)

    resource_comment = st.text_area(
        "Mining/resource comment",
        placeholder="Example: mixed rock, high-value part only / scan looked off / estimated SCU before refining",
        height=70,
        key=f"{key_prefix}_resource_comment",
    )

    yield_col1, yield_col2 = st.columns(2)
    with yield_col1:
        mining_time_seconds = st.number_input(
            "Mining time, sec",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key=f"{key_prefix}_mining_time_seconds",
        )
    with yield_col2:
        estimated_value_auec = st.number_input(
            "Estimated value before refinery, aUEC",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            key=f"{key_prefix}_estimated_value_auec",
        )

    primary_resource = "unknown"
    resource_percent = None
    raw_scu_estimate = None

    if resources:
        primary = max(resources, key=lambda item: item.resource_percent or 0.0)
        primary_resource = primary.resource_name
        resource_percent = primary.resource_percent
        raw_scu_estimate = primary.raw_scu_estimate

    return ResourceYieldFeedback(
        primary_resource=primary_resource,
        resource_percent=resource_percent,
        raw_scu_estimate=raw_scu_estimate,
        total_scu_estimate=total_scu_value,
        refined_scu_estimate=None,
        estimated_value_auec=estimated_value_auec if estimated_value_auec > 0 else None,
        mining_time_seconds=mining_time_seconds if mining_time_seconds > 0 else None,
        comment=resource_comment.strip(),
        resources=resources,
    )


def render_refinery_form(key_prefix: str = "calculator") -> RefineryFeedback:
    st.subheader("Refinery / future yield")
    st.caption(
        "Optional separate block. Fill it now if known, or leave it empty and extend/update later when refinery data is available."
    )

    with st.expander("Refinery result fields", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            refinery_method = st.selectbox(
                "Refinery method",
                options=["unknown", "cormack", "dinix", "electrostarolysis", "ferron", "gaskin", "pyrometric", "thermonatic", "xcr"],
                index=0,
                key=f"{key_prefix}_refinery_method",
            )
        with col2:
            refinery_location = st.text_input(
                "Refinery location",
                value="",
                key=f"{key_prefix}_refinery_location",
            )
        with col3:
            refinery_fee_auec = st.number_input(
                "Refinery fee, aUEC",
                min_value=0.0,
                value=0.0,
                step=500.0,
                key=f"{key_prefix}_refinery_fee_auec",
            )

        time_col1, time_col2 = st.columns(2)
        with time_col1:
            refinery_start_at = st.text_input(
                "Refinery start time",
                value="",
                placeholder="optional ISO/date text",
                key=f"{key_prefix}_refinery_start_at",
            )
        with time_col2:
            refinery_complete_at = st.text_input(
                "Refinery complete time",
                value="",
                placeholder="optional ISO/date text",
                key=f"{key_prefix}_refinery_complete_at",
            )

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            refined_scu_actual = st.number_input(
                "Actual refined SCU",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f"{key_prefix}_refined_scu_actual",
            )
        with result_col2:
            refined_value_auec = st.number_input(
                "Refined estimated value, aUEC",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                key=f"{key_prefix}_refined_value_auec",
            )
        with result_col3:
            sell_value_auec = st.number_input(
                "Actual sell value, aUEC",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                key=f"{key_prefix}_sell_value_auec",
            )

        st.write("Refined resources / sale result")
        default_refined_resources = pd.DataFrame(
            [
                {"resource_name": "unknown", "refined_scu_actual": 0.0, "sell_value_auec": 0.0, "comment": ""},
                {"resource_name": "unknown", "refined_scu_actual": 0.0, "sell_value_auec": 0.0, "comment": ""},
                {"resource_name": "unknown", "refined_scu_actual": 0.0, "sell_value_auec": 0.0, "comment": ""},
            ]
        )
        edited_refined_resources = st.data_editor(
            default_refined_resources,
            num_rows="dynamic",
            hide_index=True,
            key=f"{key_prefix}_refined_resources_table",
            column_config={
                "resource_name": st.column_config.SelectboxColumn(
                    "Resource",
                    options=RESOURCE_OPTIONS,
                    required=False,
                ),
                "refined_scu_actual": st.column_config.NumberColumn(
                    "Actual refined SCU",
                    min_value=0.0,
                    step=0.1,
                ),
                "sell_value_auec": st.column_config.NumberColumn(
                    "Sell value, aUEC",
                    min_value=0.0,
                    step=1000.0,
                ),
                "comment": st.column_config.TextColumn("Comment"),
            },
        )

        refinery_comment = st.text_area(
            "Refinery comment",
            height=70,
            key=f"{key_prefix}_refinery_comment",
        )

    refined_resources = _clean_refined_resource_rows(
        edited_refined_resources.to_dict("records")
    )

    return RefineryFeedback(
        refinery_method=refinery_method,
        refinery_location=refinery_location.strip(),
        refinery_start_at=refinery_start_at.strip(),
        refinery_complete_at=refinery_complete_at.strip(),
        refined_scu_actual=refined_scu_actual if refined_scu_actual > 0 else None,
        refined_value_auec=refined_value_auec if refined_value_auec > 0 else None,
        refinery_fee_auec=refinery_fee_auec if refinery_fee_auec > 0 else None,
        sell_value_auec=sell_value_auec if sell_value_auec > 0 else None,
        comment=refinery_comment.strip(),
        refined_resources=refined_resources,
    )



def _clean_calibration_observation_rows(rows: list[dict]) -> list[PowerDistanceObservation]:
    observations: list[PowerDistanceObservation] = []

    for row in rows:
        distance = row.get("distance")
        power_percent = row.get("power_percent")
        observation = str(row.get("observation", "unknown") or "unknown")
        beam_warmed = row.get("beam_warmed")
        held_stable = row.get("held_stable")
        comment = str(row.get("comment", "") or "")

        distance_missing = distance in (None, "") or pd.isna(distance)
        power_missing = power_percent in (None, "") or pd.isna(power_percent)
        empty_row = (
            distance_missing
            and power_missing
            and observation == "unknown"
            and str(beam_warmed) in {"", "None", "nan"}
            and str(held_stable) in {"", "None", "nan"}
            and not comment.strip()
        )
        if empty_row:
            continue

        if distance_missing or power_missing:
            continue

        warmed_value = None if pd.isna(beam_warmed) else bool(beam_warmed)
        stable_value = None if pd.isna(held_stable) else bool(held_stable)

        observations.append(
            PowerDistanceObservation(
                distance=float(distance),
                power_percent=float(power_percent),
                observation=observation,
                beam_warmed=warmed_value,
                held_stable=stable_value,
                comment=comment.strip(),
            )
        )

    return observations


def render_calibration_form(key_prefix: str = "calculator") -> CalibrationFeedback:
    st.subheader("Power/distance calibration")
    st.caption(
        "Optional: use this when the formula/helper does not match the game. "
        "These structured observations are more useful than free-text comments for formula calibration."
    )

    with st.expander("Add calibration observations", expanded=False):
        formula_issue_flag = st.checkbox(
            "Formula/helper looked wrong for this rock",
            value=False,
            key=f"{key_prefix}_formula_issue_flag",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            observed_distance = st.number_input(
                "Observed distance, m",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key=f"{key_prefix}_observed_distance",
            )
        with col2:
            observed_min_warmup_power_percent = st.number_input(
                "Observed min warm-up power %",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key=f"{key_prefix}_observed_min_warmup_power",
            )
        with col3:
            observed_stable_power_percent = st.number_input(
                "Observed stable hold power %",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key=f"{key_prefix}_observed_stable_power",
            )

        default_observations = pd.DataFrame(
            [
                {"distance": None, "power_percent": None, "observation": "unknown", "beam_warmed": False, "held_stable": False, "comment": ""},
                {"distance": None, "power_percent": None, "observation": "unknown", "beam_warmed": False, "held_stable": False, "comment": ""},
                {"distance": None, "power_percent": None, "observation": "unknown", "beam_warmed": False, "held_stable": False, "comment": ""},
            ]
        )

        edited_observations = st.data_editor(
            default_observations,
            num_rows="dynamic",
            hide_index=True,
            key=f"{key_prefix}_calibration_observations_table",
            column_config={
                "distance": st.column_config.NumberColumn("Distance, m", min_value=1.0, step=1.0),
                "power_percent": st.column_config.NumberColumn("Power %", min_value=20.0, max_value=100.0, step=1.0),
                "observation": st.column_config.SelectboxColumn(
                    "Observation",
                    options=list(CALIBRATION_OBSERVATION_OPTIONS.keys()),
                    required=False,
                ),
                "beam_warmed": st.column_config.CheckboxColumn("Warmed"),
                "held_stable": st.column_config.CheckboxColumn("Stable"),
                "comment": st.column_config.TextColumn("Comment"),
            },
        )

        calibration_comment = st.text_area(
            "Calibration comment",
            placeholder="Example: helper says 20% at 15m should warm up, but real warm-up starts around 78%; stable hold around 81%.",
            height=80,
            key=f"{key_prefix}_calibration_comment",
        )

    observations = _clean_calibration_observation_rows(edited_observations.to_dict("records"))

    return CalibrationFeedback(
        formula_issue_flag=formula_issue_flag,
        observed_min_warmup_power_percent=(
            observed_min_warmup_power_percent if observed_min_warmup_power_percent >= 20 else None
        ),
        observed_stable_power_percent=(
            observed_stable_power_percent if observed_stable_power_percent >= 20 else None
        ),
        observed_distance=observed_distance if observed_distance > 0 else None,
        comment=calibration_comment.strip(),
        observations=observations,
    )

def render_power_distance_helper(calc_input, heads, modules) -> None:
    st.subheader("Power / distance helper")
    st.caption(
        "Formula-based scan for the selected rock/build. It searches 10-120m and 20-100% power and intentionally ignores the current beam power slider, because the helper itself recommends power."
    )

    recommendation = build_power_distance_recommendation(calc_input, heads=heads, modules=modules)

    if recommendation.scanned_count == 0:
        st.info(recommendation.note)
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Minimum warm-up**")
        if recommendation.minimum_warmup is None:
            st.warning("No safe warm-up pair found in scan range.")
        else:
            candidate = recommendation.minimum_warmup
            st.metric("Scan distance, m", f"{candidate.distance:.0f} m")
            st.metric("Power", f"{candidate.power_percent:.0f}%")
            st.write(
                f"margin={candidate.margin:.2f}, risk={candidate.risk_score:.3f}, verdict={candidate.verdict}"
            )

    with col2:
        st.markdown("**Recommended stable hold**")
        if recommendation.stable_hold is None:
            st.warning("No stable hold pair found in scan range.")
        else:
            candidate = recommendation.stable_hold
            st.metric("Scan distance, m", f"{candidate.distance:.0f} m")
            st.metric("Power", f"{candidate.power_percent:.0f}%")
            st.write(
                f"margin={candidate.margin:.2f}, margin_ratio={candidate.margin_ratio:.3f}, risk={candidate.risk_score:.3f}"
            )

    with st.expander("Recommendation details", expanded=False):
        st.json({
            "minimum_warmup": recommendation.minimum_warmup.__dict__ if recommendation.minimum_warmup else None,
            "stable_hold": recommendation.stable_hold.__dict__ if recommendation.stable_hold else None,
            "scanned_count": recommendation.scanned_count,
            "note": recommendation.note,
        })

def render_run_context_sidebar() -> RunContext:
    """Return default run metadata without adding extra UI controls.

    Build/ship selection is the main user-facing control. Run metadata stays in the
    event schema for backward compatibility and future analysis, but the UI no
    longer asks for crew/operator fields during regular capture.
    """

    return RunContext()


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
        display_safe_dataframe(
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


def _safe_json_rows(value: object) -> list[dict]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if value is None or value == "":
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [row for row in parsed if isinstance(row, dict)]


def _refinery_queue_missing_mask(df: pd.DataFrame) -> pd.Series:
    method = df["refinery_method"].fillna("unknown").astype(str)
    refined_count = pd.to_numeric(df["refined_resource_count"], errors="coerce").fillna(0)
    total_refined = pd.to_numeric(df["total_refined_scu_actual"], errors="coerce").fillna(0)
    sell_value = pd.to_numeric(df["sell_value_auec"], errors="coerce").fillna(0)
    total_sell_value = pd.to_numeric(df["total_resource_sell_value_auec"], errors="coerce").fillna(0)
    return (
        method.eq("unknown")
        & refined_count.eq(0)
        & total_refined.eq(0)
        & sell_value.eq(0)
        & total_sell_value.eq(0)
    )


def _build_refinery_editor_defaults(selected_row: pd.Series) -> pd.DataFrame:
    existing_rows = _safe_json_rows(selected_row.get("refined_resources_json"))
    if not existing_rows:
        source_rows = _safe_json_rows(selected_row.get("resources_json"))
        existing_rows = [
            {
                "resource_name": row.get("resource_name", "unknown"),
                "refined_scu_actual": 0.0,
                "sell_value_auec": 0.0,
                "comment": "",
            }
            for row in source_rows
            if str(row.get("resource_name", "unknown") or "unknown") != "unknown"
        ]

    if not existing_rows:
        existing_rows = [
            {"resource_name": "unknown", "refined_scu_actual": 0.0, "sell_value_auec": 0.0, "comment": ""}
        ]

    rows = []
    for row in existing_rows:
        rows.append(
            {
                "resource_name": row.get("resource_name", "unknown") or "unknown",
                "refined_scu_actual": row.get("refined_scu_actual") or 0.0,
                "sell_value_auec": row.get("sell_value_auec") or 0.0,
                "comment": row.get("comment", "") or "",
            }
        )
    return pd.DataFrame(rows)


def render_refinery_update_queue(df: pd.DataFrame) -> None:
    st.subheader("Refinery outcome queue")
    st.caption(
        "Add final refinery/sale results to an existing mining event after the job completes. "
        "This keeps initial rock capture fast and lets yield/profit data arrive later."
    )

    if df.empty:
        st.info("No events available for refinery updates.")
        return

    missing_mask = _refinery_queue_missing_mask(df)
    missing_count = int(missing_mask.sum())
    updated_count = int((~missing_mask).sum())

    ref_col1, ref_col2, ref_col3 = st.columns(3)
    ref_col1.metric("Missing refinery", missing_count)
    ref_col2.metric("Updated refinery", updated_count)
    ref_col3.metric("Total", len(df))

    show_missing_only = st.checkbox(
        "Show events without refinery result only",
        value=True,
        key="refinery_show_missing_only",
    )

    queue = df.copy()
    if show_missing_only:
        queue = queue[missing_mask]

    if queue.empty:
        st.success("No events waiting for refinery result in the current queue.")
        return

    queue = queue.sort_values("timestamp", ascending=False, na_position="last")
    event_ids = queue["event_id"].astype(str).tolist()
    option_labels = {
        str(row["event_id"]): format_event_option(row)
        for _, row in queue.iterrows()
    }

    with st.form("refinery_update_form"):
        selected_event_id = st.selectbox(
            "Event to update",
            options=event_ids,
            format_func=lambda event_id: option_labels.get(str(event_id), str(event_id)),
            key="refinery_update_event_id",
        )
        selected_row = queue[queue["event_id"].astype(str) == str(selected_event_id)].iloc[0]

        preview_columns = [
            "event_id",
            "timestamp",
            "ship_type",
            "build_id",
            "mass",
            "resource_names",
            "resources_json",
            "refinery_method",
            "refined_scu_actual",
            "sell_value_auec",
        ]
        st.write("Selected event preview")
        display_safe_dataframe(
            pd.DataFrame([selected_row[preview_columns].to_dict()]),
            width="stretch",
        )

        method_options = ["unknown", "cormack", "dinix", "electrostarolysis", "ferron", "gaskin", "pyrometric", "thermonatic", "xcr"]
        current_method = str(selected_row.get("refinery_method") or "unknown")
        method_index = method_options.index(current_method) if current_method in method_options else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            refinery_method = st.selectbox(
                "Refinery method",
                options=method_options,
                index=method_index,
                key="refinery_update_method",
            )
        with col2:
            refinery_location = st.text_input(
                "Refinery location",
                value=str(selected_row.get("refinery_location") or ""),
                key="refinery_update_location",
            )
        with col3:
            refinery_fee_auec = st.number_input(
                "Refinery fee, aUEC",
                min_value=0.0,
                value=float(selected_row.get("refinery_fee_auec") or 0.0),
                step=500.0,
                key="refinery_update_fee",
            )

        time_col1, time_col2 = st.columns(2)
        with time_col1:
            refinery_start_at = st.text_input(
                "Refinery start time",
                value=str(selected_row.get("refinery_start_at") or ""),
                key="refinery_update_start_at",
            )
        with time_col2:
            refinery_complete_at = st.text_input(
                "Refinery complete time",
                value=str(selected_row.get("refinery_complete_at") or ""),
                key="refinery_update_complete_at",
            )

        total_col1, total_col2, total_col3 = st.columns(3)
        with total_col1:
            refined_scu_actual = st.number_input(
                "Total actual refined SCU",
                min_value=0.0,
                value=float(selected_row.get("total_refined_scu_actual") or selected_row.get("refined_scu_actual") or 0.0),
                step=0.1,
                key="refinery_update_refined_scu_actual",
            )
        with total_col2:
            refined_value_auec = st.number_input(
                "Refined estimated value, aUEC",
                min_value=0.0,
                value=float(selected_row.get("refined_value_auec") or 0.0),
                step=1000.0,
                key="refinery_update_refined_value",
            )
        with total_col3:
            sell_value_auec = st.number_input(
                "Total actual sell value, aUEC",
                min_value=0.0,
                value=float(selected_row.get("total_resource_sell_value_auec") or selected_row.get("sell_value_auec") or 0.0),
                step=1000.0,
                key="refinery_update_sell_value",
            )

        st.write("Per-resource refinery output")
        edited_refined_resources = st.data_editor(
            _build_refinery_editor_defaults(selected_row),
            num_rows="dynamic",
            hide_index=True,
            key="refinery_update_refined_resources_table",
            column_config={
                "resource_name": st.column_config.SelectboxColumn(
                    "Resource",
                    options=RESOURCE_OPTIONS,
                    required=False,
                ),
                "refined_scu_actual": st.column_config.NumberColumn(
                    "Actual refined SCU",
                    min_value=0.0,
                    step=0.1,
                ),
                "sell_value_auec": st.column_config.NumberColumn(
                    "Sell value, aUEC",
                    min_value=0.0,
                    step=1000.0,
                ),
                "comment": st.column_config.TextColumn("Comment"),
            },
        )

        refinery_comment = st.text_area(
            "Refinery comment",
            value=str(selected_row.get("refinery_comment") or ""),
            height=80,
            key="refinery_update_comment",
        )

        submitted = st.form_submit_button("Update refinery result")

    if submitted:
        try:
            update_result = update_event_refinery(
                path=EVENTS_PATH,
                event_id=str(selected_event_id),
                refinery=RefineryFeedback(
                    refinery_method=refinery_method,
                    refinery_location=refinery_location.strip(),
                    refinery_start_at=refinery_start_at.strip(),
                    refinery_complete_at=refinery_complete_at.strip(),
                    refined_scu_actual=refined_scu_actual if refined_scu_actual > 0 else None,
                    refined_value_auec=refined_value_auec if refined_value_auec > 0 else None,
                    refinery_fee_auec=refinery_fee_auec if refinery_fee_auec > 0 else None,
                    sell_value_auec=sell_value_auec if sell_value_auec > 0 else None,
                    comment=refinery_comment.strip(),
                    refined_resources=_clean_refined_resource_rows(
                        edited_refined_resources.to_dict("records")
                    ),
                ),
            )
        except Exception as exc:
            st.error(f"Could not update refinery result: {exc}")
            return

        st.success(
            f"Updated refinery result for {update_result['event_id']}. "
            f"has_refinery_result={update_result['has_refinery_result']}"
        )
        st.rerun()



def _calibration_queue_missing_mask(df: pd.DataFrame) -> pd.Series:
    attempt_count = pd.to_numeric(df.get("calibration_attempt_count", 0), errors="coerce").fillna(0)
    formula_issue = df.get("formula_issue_flag", False).fillna(False).astype(bool)
    comment = df.get("calibration_comment", "").fillna("").astype(str).str.strip()
    min_warmup = pd.to_numeric(df.get("observed_min_warmup_power_percent", 0), errors="coerce").fillna(0)
    stable_power = pd.to_numeric(df.get("observed_stable_power_percent", 0), errors="coerce").fillna(0)
    has_calibration = (
        (attempt_count > 0)
        | formula_issue
        | (comment != "")
        | (min_warmup > 0)
        | (stable_power > 0)
    )
    return ~has_calibration


def _build_calibration_editor_defaults(selected_row: pd.Series) -> pd.DataFrame:
    existing = _safe_json_rows(selected_row.get("calibration_attempts_json"))
    if existing:
        return pd.DataFrame(existing)

    return pd.DataFrame(
        [
            {"distance": None, "power_percent": None, "observation": "unknown", "beam_warmed": False, "held_stable": False, "comment": ""},
            {"distance": None, "power_percent": None, "observation": "unknown", "beam_warmed": False, "held_stable": False, "comment": ""},
            {"distance": None, "power_percent": None, "observation": "unknown", "beam_warmed": False, "held_stable": False, "comment": ""},
        ]
    )


def render_calibration_update_queue(df: pd.DataFrame) -> None:
    st.subheader("Power/distance calibration queue")
    st.caption(
        "Convert free-text notes like '20% at 15m did not warm up; stable at 81%' into structured observations. "
        "This is for correcting the formula/helper, not for training the good/not-good model directly."
    )

    if df.empty:
        st.info("No events available for calibration updates.")
        return

    missing_mask = _calibration_queue_missing_mask(df)
    cal_col1, cal_col2, cal_col3 = st.columns(3)
    cal_col1.metric("Missing calibration", int(missing_mask.sum()))
    cal_col2.metric("With calibration", int((~missing_mask).sum()))
    cal_col3.metric("Total", len(df))

    show_missing_only = st.checkbox(
        "Show events without calibration only",
        value=False,
        key="calibration_show_missing_only",
    )

    queue = df.copy()
    if show_missing_only:
        queue = queue[missing_mask]

    if queue.empty:
        st.success("No events waiting for calibration in the current queue.")
        return

    queue = queue.sort_values("timestamp", ascending=False, na_position="last")
    event_ids = queue["event_id"].astype(str).tolist()
    option_labels = {
        str(row["event_id"]): format_event_option(row)
        for _, row in queue.iterrows()
    }

    with st.form("calibration_update_form"):
        selected_event_id = st.selectbox(
            "Event to update",
            options=event_ids,
            format_func=lambda event_id: option_labels.get(str(event_id), str(event_id)),
            key="calibration_update_event_id",
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
            "beam_power_sum",
            "verdict",
            "outcome_comment",
        ]
        st.write("Selected event preview")
        display_safe_dataframe(
            pd.DataFrame([selected_row[preview_columns].to_dict()]),
            width="stretch",
        )

        formula_issue_flag = st.checkbox(
            "Formula/helper looked wrong for this event",
            value=bool(selected_row.get("formula_issue_flag") or False),
            key="calibration_update_formula_issue_flag",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            observed_distance = st.number_input(
                "Observed distance, m",
                min_value=0.0,
                value=float(selected_row.get("observed_distance") or 0.0),
                step=1.0,
                key="calibration_update_observed_distance",
            )
        with col2:
            observed_min_warmup_power_percent = st.number_input(
                "Observed min warm-up power %",
                min_value=0.0,
                max_value=100.0,
                value=float(selected_row.get("observed_min_warmup_power_percent") or 0.0),
                step=1.0,
                key="calibration_update_min_warmup_power",
            )
        with col3:
            observed_stable_power_percent = st.number_input(
                "Observed stable hold power %",
                min_value=0.0,
                max_value=100.0,
                value=float(selected_row.get("observed_stable_power_percent") or 0.0),
                step=1.0,
                key="calibration_update_stable_power",
            )

        st.write("Power/distance observations")
        edited_observations = st.data_editor(
            _build_calibration_editor_defaults(selected_row),
            num_rows="dynamic",
            hide_index=True,
            key="calibration_update_observations_table",
            column_config={
                "distance": st.column_config.NumberColumn("Distance, m", min_value=1.0, step=1.0),
                "power_percent": st.column_config.NumberColumn("Power %", min_value=20.0, max_value=100.0, step=1.0),
                "observation": st.column_config.SelectboxColumn(
                    "Observation",
                    options=list(CALIBRATION_OBSERVATION_OPTIONS.keys()),
                    required=False,
                ),
                "beam_warmed": st.column_config.CheckboxColumn("Warmed"),
                "held_stable": st.column_config.CheckboxColumn("Stable"),
                "comment": st.column_config.TextColumn("Comment"),
            },
        )

        calibration_comment = st.text_area(
            "Calibration comment",
            value=str(selected_row.get("calibration_comment") or ""),
            height=90,
            key="calibration_update_comment",
        )

        submitted = st.form_submit_button("Update calibration observations")

    if submitted:
        try:
            update_result = update_event_calibration(
                path=EVENTS_PATH,
                event_id=str(selected_event_id),
                calibration=CalibrationFeedback(
                    formula_issue_flag=formula_issue_flag,
                    observed_min_warmup_power_percent=(
                        observed_min_warmup_power_percent if observed_min_warmup_power_percent >= 20 else None
                    ),
                    observed_stable_power_percent=(
                        observed_stable_power_percent if observed_stable_power_percent >= 20 else None
                    ),
                    observed_distance=observed_distance if observed_distance > 0 else None,
                    comment=calibration_comment.strip(),
                    observations=_clean_calibration_observation_rows(
                        edited_observations.to_dict("records")
                    ),
                ),
            )
        except Exception as exc:
            st.error(f"Could not update calibration observations: {exc}")
            return

        st.success(
            f"Updated calibration observations for {update_result['event_id']}. "
            f"has_calibration_observations={update_result['has_calibration_observations']}"
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
        display_safe_dataframe(matrix_pivot, width="stretch")

    st.write("Feature signal: good vs not-good outcomes")
    if feature_signal.empty:
        st.info("Not enough labeled data to compare feature signals.")
    else:
        display_safe_dataframe(feature_signal, width="stretch")

    st.write("Numeric summary by actual outcome")
    if outcome_numeric_summary.empty:
        st.info("No numeric outcome summary available.")
    else:
        display_safe_dataframe(outcome_numeric_summary, width="stretch")

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
        display_safe_dataframe(diagnostics_view, width="stretch")


def render_ml_baseline_block(
    dataset: pd.DataFrame,
    title: str = "Baseline ML model",
    model_path: Path = MODEL_PATH,
    report_path: Path = MODEL_REPORT_PATH,
    key_prefix: str = "manual",
    default_min_labeled_rows: int = MIN_LABELED_ROWS_FOR_TRAINING,
    model_source: str = MODEL_SOURCE_MANUAL_REAL,
) -> None:
    st.subheader(title)
    st.caption(
        "Trains a first weak supervised model on labeled events. "
        "Target: good outcome vs not-good outcome. Use this as a baseline, not as final truth."
    )

    st.write(f"Model source: `{model_source}`")
    source_warning = model_source_warning(model_source)
    if source_warning:
        st.warning(source_warning)
    elif model_source == MODEL_SOURCE_MANUAL_REAL:
        st.info(
            "Manual real-data model path. Use this for gameplay review only after enough real labeled events are collected."
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
        display_safe_dataframe(
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
        run_record = append_training_run(
            training_result=training_result,
            path=TRAINING_RUNS_PATH,
            notes=f"UI training run: {title}",
        )
        st.write(f"Training run logged: `{run_record.run_id}`")
        st.write(f"Run history: `{TRAINING_RUNS_PATH}`")
        st.json(result_to_dict(training_result))

    if report_path.exists():
        with st.expander("Last baseline report", expanded=False):
            st.json(report_path.read_text(encoding="utf-8"))



def available_model_paths() -> dict[str, Path]:
    options: dict[str, Path] = {}
    for spec in existing_model_artifacts():
        options[f"{spec.label} [{spec.model_source}]"] = spec.model_path
    return options


def active_model_select_index(options: dict[str, Path]) -> int:
    active = read_active_model_config(ACTIVE_MODEL_CONFIG_PATH)
    if active is None:
        return 0

    active_path = Path(active.model_path)
    for index, model_path in enumerate(options.values()):
        if Path(model_path) == active_path:
            return index
    return 0


def render_model_artifact_separation_block() -> None:
    st.subheader("Model artifact separation")
    st.caption(
        "Keeps real/manual and synthetic model artifacts separate. "
        "Synthetic models are for smoke tests only; manual real-data models are the only candidates for gameplay review."
    )

    rows = []
    for spec in default_model_artifact_specs():
        payload = spec_to_dict(spec)
        payload["exists"] = spec.model_path.exists()
        payload["report_exists"] = spec.report_path.exists()
        rows.append(payload)

    registry_df = pd.DataFrame(rows)
    display_safe_dataframe(registry_df, width="stretch")

    if SYNTHETIC_MODEL_PATH.exists() and not MODEL_PATH.exists():
        st.warning(
            "Only the synthetic smoke-test model exists. This validates the pipeline but should not be used as gameplay advice."
        )
    elif MODEL_PATH.exists():
        st.success("Manual real-data model artifact exists. Check the report quality before trusting it.")
    else:
        st.info("No manual real-data model yet. Label real outcomes, then train the manual baseline.")



def render_active_model_selection_block() -> None:
    st.subheader("Active model selection")
    st.caption(
        "Selects which existing model artifact is used by default for Calculator and dataset inference. "
        "This writes a small registry pointer, not a new model."
    )

    status = build_active_model_status(ACTIVE_MODEL_CONFIG_PATH)
    active_payload = status.get("active_model")

    active_col1, active_col2, active_col3, active_col4 = st.columns(4)
    active_col1.metric("Configured", "YES" if status["configured"] else "NO")
    active_col2.metric("Valid", "YES" if status["valid"] else "NO")
    active_col3.metric("Available models", status["available_count"])
    active_col4.metric(
        "Active source",
        active_payload["model_source"] if active_payload else "—",
    )

    st.write(f"Active config: `{ACTIVE_MODEL_CONFIG_PATH}`")
    st.write(f"Reason: {status['reason']}")

    if active_payload:
        if active_payload.get("warning"):
            st.warning(active_payload["warning"])
        elif active_payload.get("safe_for_gameplay_review"):
            st.success("Active model is a manual real-data artifact. Review report quality before using its output.")

        with st.expander("Active model config", expanded=False):
            st.json(active_payload)

    specs = existing_model_artifacts()
    if not specs:
        st.info("No existing model artifacts. Train manual or synthetic baseline first.")
        return

    option_labels = [f"{spec.label} [{spec.model_source}]" for spec in specs]
    label_to_spec = dict(zip(option_labels, specs))

    selected_label = st.selectbox(
        "Model artifact to activate",
        options=option_labels,
        key="active_model_selector",
    )
    selected_spec = label_to_spec[selected_label]

    selected_warning = model_source_warning(selected_spec.model_source)
    if selected_warning:
        st.warning(selected_warning)

    action_col1, action_col2 = st.columns([1, 3])
    with action_col1:
        set_clicked = st.button("Set active model", key="set_active_model")
    with action_col2:
        clear_clicked = st.button(
            "Clear active model",
            key="clear_active_model",
            disabled=not status["configured"],
        )

    if set_clicked:
        selection = write_active_model_config(selected_spec, ACTIVE_MODEL_CONFIG_PATH)
        st.success(f"Active model set: {selection.label} [{selection.model_source}]")
        st.rerun()

    if clear_clicked:
        clear_active_model_config(ACTIVE_MODEL_CONFIG_PATH)
        st.success("Active model selection cleared.")
        st.rerun()


def render_model_promotion_gate_block(events: pd.DataFrame) -> None:
    st.subheader("Model promotion gate")
    st.caption(
        "Reviews whether the manual real-data model is safe to promote as the active inference model. "
        "This is a guardrail, not a proof that the model is correct."
    )

    criteria_col1, criteria_col2, criteria_col3, criteria_col4 = st.columns(4)
    with criteria_col1:
        min_rows = st.number_input(
            "Min training rows",
            min_value=4,
            max_value=1000,
            value=30,
            step=1,
            key="promotion_min_rows",
        )
    with criteria_col2:
        min_test_rows = st.number_input(
            "Min test rows",
            min_value=1,
            max_value=500,
            value=5,
            step=1,
            key="promotion_min_test_rows",
        )
    with criteria_col3:
        min_accuracy = st.slider(
            "Min accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.01,
            key="promotion_min_accuracy",
        )
    with criteria_col4:
        max_false_good_rate = st.slider(
            "Max false-good rate",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            key="promotion_max_false_good_rate",
            help="False-good means the ML snapshot predicted good but actual_outcome later became not-good.",
        )

    prediction_eval_summary = build_prediction_evaluation_summary(events)
    criteria = PromotionCriteria(
        min_rows_used=int(min_rows),
        min_test_rows=int(min_test_rows),
        min_accuracy=float(min_accuracy),
        max_false_good_rate=float(max_false_good_rate),
    )
    decision = evaluate_model_promotion(
        model_path=MODEL_PATH,
        report_path=MODEL_REPORT_PATH,
        criteria=criteria,
        prediction_evaluation_summary=prediction_eval_summary,
    )

    promo_col1, promo_col2, promo_col3, promo_col4, promo_col5 = st.columns(5)
    promo_col1.metric("Gate status", decision.status.upper())
    promo_col2.metric("Can promote", "YES" if decision.can_promote else "NO")
    promo_col3.metric("Rows used", decision.metrics.get("rows_used", 0))
    promo_col4.metric("Accuracy", decision.metrics.get("accuracy", "—"))
    promo_col5.metric("False-good rate", decision.metrics.get("false_good_rate", "—"))

    st.write(f"Candidate model: `{MODEL_PATH}`")
    st.write(f"Candidate report: `{MODEL_REPORT_PATH}`")

    if decision.status == "pass":
        st.success("Promotion gate passed. Manual real-data model can be selected as the active model candidate.")
    elif decision.status == "warn":
        st.warning("Promotion gate passed with warnings. Review warnings before activating the model.")
    else:
        st.error("Promotion gate failed. Do not promote this model yet.")

    if decision.reasons:
        st.write("Blocking reasons")
        display_safe_dataframe(pd.DataFrame({"reason": decision.reasons}), width="stretch")

    if decision.warnings:
        st.write("Warnings")
        display_safe_dataframe(pd.DataFrame({"warning": decision.warnings}), width="stretch")

    with st.expander("Promotion decision payload", expanded=False):
        st.json(promotion_decision_to_dict(decision))

    manual_spec = next(
        spec for spec in default_model_artifact_specs()
        if spec.model_source == MODEL_SOURCE_MANUAL_REAL
    )
    promote_clicked = st.button(
        "Promote manual model to active",
        disabled=not decision.can_promote,
        key="promote_manual_model_to_active",
        help="Writes models/active_model.json to point inference at the manual real-data model.",
    )

    if promote_clicked:
        selection = write_active_model_config(manual_spec, ACTIVE_MODEL_CONFIG_PATH)
        st.success(f"Promoted active model: {selection.label} [{selection.model_source}]")
        st.rerun()



def render_real_ml_run_starter_block(events: pd.DataFrame, dataset: pd.DataFrame) -> None:
    st.subheader("Real ML run starter")
    st.caption(
        "One controlled run for the manual real-data model: export dataset, validate, train, log the run, "
        "check promotion criteria, and optionally update active_model.json. Synthetic data is not used here."
    )

    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    with control_col1:
        min_labeled_rows = st.number_input(
            "Run min labeled rows",
            min_value=4,
            max_value=2000,
            value=30,
            step=1,
            key="real_run_min_labeled_rows",
        )
    with control_col2:
        min_test_rows = st.number_input(
            "Run min test rows",
            min_value=1,
            max_value=500,
            value=5,
            step=1,
            key="real_run_min_test_rows",
        )
    with control_col3:
        min_accuracy = st.slider(
            "Run min accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.01,
            key="real_run_min_accuracy",
        )
    with control_col4:
        max_false_good_rate = st.slider(
            "Run max false-good rate",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            key="real_run_max_false_good_rate",
        )

    option_col1, option_col2 = st.columns(2)
    with option_col1:
        promote_if_passed = st.checkbox(
            "Promote if gate passes",
            value=False,
            key="real_run_promote_if_passed",
            help="When enabled, writes models/active_model.json only if the manual model passes promotion gate checks.",
        )
    with option_col2:
        train_if_ready = st.checkbox(
            "Train if ready",
            value=True,
            key="real_run_train_if_ready",
            help="Disable this to run export/readiness/promotion checks without training.",
        )

    readiness = check_training_readiness(dataset, min_labeled_rows=int(min_labeled_rows))
    prediction_eval_summary = build_prediction_evaluation_summary(events)

    run_col1, run_col2, run_col3, run_col4 = st.columns(4)
    run_col1.metric("Rows", len(dataset))
    run_col2.metric("Labeled", readiness.labeled_rows)
    run_col3.metric("Ready", "YES" if readiness.ready else "NO")
    run_col4.metric("Evaluable predictions", prediction_eval_summary["evaluable_prediction_count"])

    st.write(f"Dataset output: `{DATASET_PATH}`")
    st.write(f"Manual model output: `{MODEL_PATH}`")
    st.write(f"Manual report output: `{MODEL_REPORT_PATH}`")
    st.write(f"Training run log: `{TRAINING_RUNS_PATH}`")

    if not readiness.ready:
        st.warning(readiness.reason)

    run_clicked = st.button(
        "Run real ML pipeline",
        key="run_real_ml_pipeline",
        help="Exports the latest events, validates the dataset, trains the manual real-data model when ready, logs the run, and evaluates promotion.",
    )

    if not run_clicked:
        return

    config = RealMLRunConfig(
        events_path=EVENTS_PATH,
        dataset_path=DATASET_PATH,
        model_path=MODEL_PATH,
        report_path=MODEL_REPORT_PATH,
        training_runs_path=TRAINING_RUNS_PATH,
        active_model_path=ACTIVE_MODEL_CONFIG_PATH,
        min_labeled_rows=int(min_labeled_rows),
        min_test_rows=int(min_test_rows),
        min_accuracy=float(min_accuracy),
        max_false_good_rate=float(max_false_good_rate),
        train_if_ready=bool(train_if_ready),
        promote_if_passed=bool(promote_if_passed),
        notes="UI real ML run starter",
    )
    result = run_real_ml_pipeline(config)

    if result.promoted:
        st.success("Real ML run completed and manual model was promoted to active.")
    elif result.trained:
        st.success("Real ML run completed and manual model was trained.")
    elif result.training_ready:
        st.info("Real ML run completed. Dataset is ready, but training was disabled or promotion did not run.")
    else:
        st.warning("Real ML run completed, but dataset is not ready for training yet.")

    display_safe_dataframe(real_ml_run_result_to_dataframe(result), width="stretch")

    if result.reasons:
        st.write("Blocking reasons")
        display_safe_dataframe(pd.DataFrame({"reason": result.reasons}), width="stretch")

    if result.warnings:
        st.write("Warnings")
        display_safe_dataframe(pd.DataFrame({"warning": result.warnings}), width="stretch")

    with st.expander("Full real ML run payload", expanded=False):
        st.json(real_ml_run_result_to_dict(result))

def render_training_run_history_block() -> None:
    st.subheader("Training run history")
    st.caption(
        "Lightweight experiment tracking. Every UI training action appends one JSONL run record."
    )

    runs = load_training_runs(TRAINING_RUNS_PATH)
    summary = summarize_training_runs(runs)

    history_col1, history_col2, history_col3, history_col4 = st.columns(4)
    history_col1.metric("Runs", summary["run_count"])
    history_col2.metric("Sources", summary["model_source_count"])
    history_col3.metric("Best accuracy", summary["best_accuracy"] if summary["best_accuracy"] is not None else "—")
    history_col4.metric("Latest source", summary["latest_model_source"] or "—")

    st.write(f"Run log: `{TRAINING_RUNS_PATH}`")

    if runs.empty:
        st.info("No training runs logged yet. Train a manual or synthetic baseline model first.")
        return

    view = runs.sort_values("created_at", ascending=False, na_position="last")
    source_values = sorted(view["model_source"].dropna().astype(str).unique())
    selected_sources = st.multiselect(
        "Model sources",
        options=source_values,
        default=source_values,
        key="training_history_sources",
    )

    if selected_sources:
        view = view[view["model_source"].astype(str).isin(selected_sources)]

    display_columns = [
        "created_at",
        "model_source",
        "model_version",
        "rows_used",
        "train_rows",
        "test_rows",
        "accuracy",
        "model_path",
        "report_path",
        "run_id",
        "notes",
    ]
    display_safe_dataframe(view[display_columns], width="stretch")

    csv_payload = view[display_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download training runs CSV",
        data=csv_payload,
        file_name="training_runs.csv",
        mime="text/csv",
    )

def render_calculator_ml_comparison(calc_input: CalculationInput, result, key_prefix: str = "calculator"):
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
        return None

    selected_label = st.selectbox(
        "Model for comparison",
        options=list(model_options.keys()),
        index=active_model_select_index(model_options),
        key=f"{key_prefix}_ml_model_selector",
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
        return None

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

    return comparison


def render_prediction_logging_block(events: pd.DataFrame) -> None:
    st.subheader("Prediction logging")
    st.caption(
        "Shows Formula vs ML prediction snapshots saved inside manual events. "
        "This is the inference audit log for later comparison against real outcomes."
    )

    summary = build_prediction_log_summary(events)
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    pred_col1.metric("Events", summary["event_count"])
    pred_col2.metric("Logged predictions", summary["prediction_count"])
    pred_col3.metric("Prediction coverage", summary["prediction_coverage_ratio"])

    prediction_log = build_prediction_log_dataframe(events)
    if prediction_log.empty:
        st.info("No logged ML prediction snapshots yet. Save an event from Calculator while a model is available.")
        return

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.write("Model source distribution")
        display_safe_dataframe(
            pd.DataFrame(
                summary["model_source_distribution"].items(),
                columns=["model_source", "count"],
            ),
            width="stretch",
        )

    with dist_col2:
        st.write("Formula vs ML agreement distribution")
        display_safe_dataframe(
            pd.DataFrame(
                summary["agreement_distribution"].items(),
                columns=["agreement", "count"],
            ),
            width="stretch",
        )

    display_safe_dataframe(prediction_log, width="stretch")

    st.download_button(
        "Download prediction log CSV",
        data=prediction_log.to_csv(index=False).encode("utf-8"),
        file_name="prediction_log.csv",
        mime="text/csv",
    )



def render_prediction_evaluation_block(events: pd.DataFrame) -> None:
    st.subheader("Prediction outcome evaluation")
    st.caption(
        "Evaluates saved ML prediction snapshots after actual_outcome labels are added. "
        "This measures historical inference quality, not retrained model quality."
    )

    summary = build_prediction_evaluation_summary(events)
    eval_col1, eval_col2, eval_col3, eval_col4, eval_col5 = st.columns(5)
    eval_col1.metric("Logged predictions", summary["logged_prediction_count"])
    eval_col2.metric("Evaluable", summary["evaluable_prediction_count"])
    eval_col3.metric("Accuracy", summary["accuracy"] if summary["accuracy"] is not None else "—")
    eval_col4.metric("False good", summary["false_good_count"])
    eval_col5.metric("False not-good", summary["false_not_good_count"])

    evaluated = build_prediction_evaluation_dataframe(events)
    if evaluated.empty:
        st.info(
            "No evaluable predictions yet. Save events with ML snapshots and later label actual_outcome values."
        )
        return

    if summary["false_good_count"] > 0:
        st.warning(
            "False-good cases exist: the ML snapshot predicted good, but the actual outcome was not-good. "
            "These are high-priority review cases."
        )

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.write("Prediction error distribution")
        display_safe_dataframe(
            pd.DataFrame(
                summary["error_type_distribution"].items(),
                columns=["error_type", "count"],
            ),
            width="stretch",
        )

    with dist_col2:
        st.write("Evaluated model source distribution")
        display_safe_dataframe(
            pd.DataFrame(
                summary["model_source_distribution"].items(),
                columns=["model_source", "count"],
            ),
            width="stretch",
        )

    matrix = build_prediction_evaluation_matrix(events)
    st.write("Actual target vs ML prediction")
    if matrix.empty:
        st.info("No evaluation matrix available yet.")
    else:
        matrix_pivot = matrix.pivot_table(
            index="actual_target",
            columns="ml_prediction",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        display_safe_dataframe(matrix_pivot, width="stretch")

    error_types = sorted(evaluated["error_type"].dropna().astype(str).unique())
    selected_error_types = st.multiselect(
        "Prediction error types",
        options=error_types,
        default=error_types,
        key="prediction_evaluation_error_types",
    )

    view = evaluated
    if selected_error_types:
        view = view[view["error_type"].isin(selected_error_types)]

    display_safe_dataframe(view, width="stretch")

    st.download_button(
        "Download prediction evaluation CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="prediction_evaluation.csv",
        mime="text/csv",
    )

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
        index=active_model_select_index(model_options),
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
    display_safe_dataframe(agreement_distribution, width="stretch")

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
    display_safe_dataframe(export_view, width="stretch")

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
        display_safe_dataframe(
            distribution_to_dataframe(
                synth_summary["actual_outcome_distribution"],
                "actual_outcome",
            ),
            width="stretch",
        )

    return synthetic_dataset

def render_calculator_tab(
    heads,
    modules,
    build,
    session_id: str,
    run_context: RunContext,
    source: str = "manual_ui",
    key_prefix: str = "calculator",
    page_title: str = "Scan capture",
) -> None:
    st.subheader(page_title)
    st.caption("Enter values in the same shape as the in-game scan: Mass, Resistance, Instability, Distance, and Composition SCU below.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mass = st.number_input(
            "Mass",
            min_value=1.0,
            value=12600.0,
            step=100.0,
            key=f"{key_prefix}_mass",
        )

    with col2:
        resistance = st.number_input(
            "Resistance (%)",
            min_value=0.0,
            value=0.34,
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_resistance",
        )

    with col3:
        instability = st.number_input(
            "Instability",
            min_value=0.0,
            value=0.12,
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_instability",
        )

    with col4:
        distance = st.number_input(
            "Scan distance, m",
            min_value=1.0,
            value=92.0,
            step=1.0,
            key=f"{key_prefix}_distance",
        )

    st.subheader("Beam states")

    beams: list[BeamState] = []

    for head in build.heads:
        col_a, col_b, col_c = st.columns([1, 2, 2])

        default_enabled = head.slot in {"main", "left"}
        installed_active_modules = [
            module_id
            for module_id in head.modules
            if module_id in modules and modules[module_id].type == "active"
        ]

        with col_a:
            enabled = st.checkbox(
                f"Enable beam: {head.slot}",
                value=default_enabled,
                key=f"{key_prefix}_{head.slot}_enabled",
            )

        with col_b:
            power = st.slider(
                f"Power %: {head.slot}",
                min_value=20,
                max_value=100,
                value=20,
                step=1,
                disabled=not enabled,
                key=f"{key_prefix}_{head.slot}_power",
                help="Mining laser power starts at 20%; distance changes delivered effective power.",
            )

        with col_c:
            active_modules = st.multiselect(
                f"Active modules: {head.slot}",
                options=installed_active_modules,
                default=[],
                format_func=lambda module_id: modules[module_id].name,
                disabled=not enabled or not installed_active_modules,
                key=f"{key_prefix}_{head.slot}_active_modules",
            )

        if enabled:
            beams.append(
                BeamState(
                    slot=head.slot,
                    power_percent=float(power),
                    active_modules=list(active_modules),
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

    render_power_distance_helper(calc_input, heads=heads, modules=modules)

    ml_comparison = render_calculator_ml_comparison(calc_input, result, key_prefix=key_prefix)

    outcome = render_outcome_form(key_prefix=key_prefix)
    resource_yield = render_resource_yield_form(key_prefix=key_prefix)
    refinery = render_refinery_form(key_prefix=key_prefix)
    calibration = render_calibration_form(key_prefix=key_prefix)

    st.subheader("Save event")

    col_save, col_path = st.columns([1, 3])

    with col_save:
        save_clicked = st.button("Save event", type="primary", key=f"{key_prefix}_save_event")

    with col_path:
        st.write(f"Output: `{EVENTS_PATH}`")

    if save_clicked:
        event = save_calculation_event(
            path=EVENTS_PATH,
            session_id=session_id,
            calc_input=calc_input,
            result=result,
            source=source,
            outcome=outcome,
            resource_yield=resource_yield,
            refinery=refinery,
            calibration=calibration,
            run_context=run_context,
            ml_prediction_snapshot=(
                comparison_to_dict(ml_comparison)
                if ml_comparison is not None and ml_comparison.model_available
                else None
            ),
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
        {"metric": "primary_resource", "value": str(resource_yield.primary_resource)},
        {"metric": "resource_percent", "value": str(resource_yield.resource_percent)},
        {"metric": "resource_count", "value": str(len(resource_yield.resources))},
        {"metric": "refinery_method", "value": str(refinery.refinery_method)},
        {"metric": "refined_scu_actual", "value": str(refinery.refined_scu_actual)},
        {"metric": "calibration_attempts", "value": str(len(calibration.observations))},
        {"metric": "formula_issue_flag", "value": str(calibration.formula_issue_flag)},
        {
            "metric": "ml_prediction_logged_on_save",
            "value": str(ml_comparison is not None and ml_comparison.model_available),
        },
    ]

    df = pd.DataFrame(rows)
    display_safe_dataframe(df, width="stretch")

    if result.notes:
        st.subheader("Notes")
        for note in result.notes:
            st.write(f"- {note}")



def _records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _show_key_value_payload(payload: dict, title: str) -> None:
    st.markdown(f"**{title}**")
    if not payload:
        st.info("No data in this section.")
        return
    rows = [{"field": key, "value": value} for key, value in payload.items() if key != "notes"]
    display_safe_dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_event_history_viewer(df: pd.DataFrame) -> None:
    st.subheader("Event detail / history view")
    st.caption(
        "Open one event as a readable card instead of one wide table row. "
        "This shows lifecycle steps, rock, formula, ML snapshot, resources, refinery and calibration data."
    )

    if df.empty:
        st.info("No events available for detail view.")
        return

    queue = df.sort_values("timestamp", ascending=False, na_position="last").copy()
    event_ids = queue["event_id"].astype(str).tolist()
    option_labels = {
        str(row["event_id"]): format_event_option(row)
        for _, row in queue.iterrows()
    }

    selected_event_id = st.selectbox(
        "Event",
        options=event_ids,
        format_func=lambda event_id: option_labels.get(str(event_id), str(event_id)),
        key="event_detail_selected_event_id",
    )

    raw_event = get_event_by_id(EVENTS_PATH, str(selected_event_id))
    if raw_event is None:
        st.error(f"Event not found in {EVENTS_PATH}: {selected_event_id}")
        return

    payload = build_event_detail_payload(raw_event)
    summary = payload["summary"]
    result = payload["result"]
    rock = payload["rock"]
    ml_prediction = payload["ml_prediction"]
    outcome = payload["outcome"]
    resources = payload["resources"]
    refinery = payload["refinery"]
    calibration = payload["calibration"]

    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)
    summary_col1.metric("Ship", summary.get("ship_type") or "—")
    summary_col2.metric("Verdict", str(summary.get("verdict") or "—"))
    summary_col3.metric("Outcome", str(summary.get("actual_outcome") or "unknown"))
    summary_col4.metric("Mass", rock.get("mass", "—"))
    summary_col5.metric("Scan distance, m", rock.get("distance", "—"))

    st.write(f"Event ID: `{summary.get('event_id')}`")
    st.write(f"Timestamp: `{summary.get('timestamp')}` | Session: `{summary.get('session_id')}` | Source: `{summary.get('source')}`")
    st.write(f"Build: `{summary.get('build_id')}`")

    timeline = build_event_timeline(payload)
    st.write("Lifecycle")
    if timeline:
        display_safe_dataframe(pd.DataFrame(timeline), width="stretch", hide_index=True)
    else:
        st.info("No lifecycle timestamps available.")

    tab_summary, tab_formula, tab_resources, tab_refinery, tab_calibration, tab_raw = st.tabs(
        ["Summary", "Formula / ML", "Resources", "Refinery", "Calibration", "Raw JSON"]
    )

    with tab_summary:
        col_left, col_right = st.columns(2)
        with col_left:
            _show_key_value_payload(rock, "Rock")
            st.write("Beams")
            beams_df = _records_to_dataframe(payload.get("beams", []))
            if beams_df.empty:
                st.info("No beam data.")
            else:
                display_safe_dataframe(beams_df, width="stretch", hide_index=True)
        with col_right:
            _show_key_value_payload(outcome, "Outcome")
            _show_key_value_payload(payload.get("labeling", {}).get("outcome", {}), "Outcome labeling metadata")

    with tab_formula:
        col_left, col_right = st.columns(2)
        with col_left:
            _show_key_value_payload(result, "Formula result")
            notes = result.get("notes") or []
            if notes:
                st.write("Formula notes")
                for note in notes:
                    st.write(f"- {note}")
        with col_right:
            _show_key_value_payload(ml_prediction, "ML prediction snapshot")
            if ml_prediction.get("model_source") == "synthetic_smoke_test":
                st.warning("This event used a synthetic smoke-test model. Treat it as pipeline validation only.")

    with tab_resources:
        resource_rows = resources.get("resources") or []
        resource_summary = {key: value for key, value in resources.items() if key != "resources"}
        _show_key_value_payload(resource_summary, "Resource summary")
        st.write("Resources in rock")
        resources_df = _records_to_dataframe(resource_rows)
        if resources_df.empty:
            st.info("No resource rows captured.")
        else:
            display_safe_dataframe(resources_df, width="stretch", hide_index=True)

    with tab_refinery:
        refined_rows = refinery.get("refined_resources") or []
        refinery_summary = {key: value for key, value in refinery.items() if key != "refined_resources"}
        _show_key_value_payload(refinery_summary, "Refinery summary")
        _show_key_value_payload(payload.get("labeling", {}).get("refinery", {}), "Refinery update metadata")
        st.write("Refined resources")
        refined_df = _records_to_dataframe(refined_rows)
        if refined_df.empty:
            st.info("No refined resource rows captured yet.")
        else:
            display_safe_dataframe(refined_df, width="stretch", hide_index=True)

    with tab_calibration:
        observation_rows = calibration.get("observations") or []
        calibration_summary = {key: value for key, value in calibration.items() if key != "observations"}
        _show_key_value_payload(calibration_summary, "Calibration summary")
        _show_key_value_payload(payload.get("labeling", {}).get("calibration", {}), "Calibration update metadata")
        st.write("Power/distance observations")
        observations_df = _records_to_dataframe(observation_rows)
        if observations_df.empty:
            st.info("No structured power/distance observations yet.")
        else:
            display_safe_dataframe(observations_df, width="stretch", hide_index=True)

    with tab_raw:
        st.json(raw_event)


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
        filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)

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

        with filter_col5:
            resource_values = sorted(
                value for value in filtered["primary_resource"].fillna("unknown").unique()
            )
            selected_resources = st.multiselect(
                "Resource",
                options=resource_values,
                default=resource_values,
            )

    if selected_sessions:
        filtered = filtered[filtered["session_id"].isin(selected_sessions)]

    if selected_ships:
        filtered = filtered[filtered["ship_type"].isin(selected_ships)]

    if selected_verdicts:
        filtered = filtered[filtered["verdict"].isin(selected_verdicts)]

    if selected_outcomes:
        filtered = filtered[filtered["actual_outcome"].isin(selected_outcomes)]


    if selected_resources:
        filtered = filtered[filtered["primary_resource"].fillna("unknown").isin(selected_resources)]

    filtered = filtered.sort_values("timestamp", ascending=False, na_position="last")

    st.subheader("Filtered events")
    display_safe_dataframe(filtered, width="stretch")

    render_event_history_viewer(filtered)

    render_outcome_labeling_queue(df)
    render_refinery_update_queue(df)
    render_calibration_update_queue(df)
    render_prediction_logging_block(df)
    render_prediction_evaluation_block(df)

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
        "resource_percent",
        "raw_scu_estimate",
        "refined_scu_estimate",
        "estimated_value_auec",
        "mining_time_seconds",
        "resource_count",
        "total_resource_percent",
        "refined_scu_actual",
        "refined_value_auec",
        "refinery_fee_auec",
        "sell_value_auec",
        "refined_resource_count",
        "total_refined_scu_actual",
        "total_resource_sell_value_auec",
    ]

    st.subheader("Numeric summary")
    display_safe_dataframe(
        filtered[numeric_columns].describe().round(3),
        width="stretch",
    )

    st.subheader("Resource / yield summary")
    resource_distribution = (
        filtered.groupby("primary_resource", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    display_safe_dataframe(resource_distribution, width="stretch")

    yield_columns = [
        "primary_resource",
        "resource_names",
        "resource_count",
        "total_resource_percent",
        "resource_percent",
        "raw_scu_estimate",
        "estimated_value_auec",
        "mining_time_seconds",
        "resources_json",
        "refinery_method",
        "refinery_location",
        "refined_scu_actual",
        "refined_value_auec",
        "refinery_fee_auec",
        "sell_value_auec",
        "refined_resource_count",
        "total_refined_scu_actual",
        "total_resource_sell_value_auec",
    ]
    display_safe_dataframe(filtered[yield_columns], width="stretch")

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
        display_safe_dataframe(issues_df, width="stretch")

    quality_view_col1, quality_view_col2, quality_view_col3 = st.columns(3)

    with quality_view_col1:
        st.write("Verdict distribution")
        display_safe_dataframe(
            distribution_to_dataframe(
                quality_report["verdict_distribution"],
                "verdict",
            ),
            width="stretch",
        )

    with quality_view_col2:
        st.write("Actual outcome distribution")
        display_safe_dataframe(
            distribution_to_dataframe(
                quality_report["actual_outcome_distribution"],
                "actual_outcome",
            ),
            width="stretch",
        )

    with quality_view_col3:
        st.write("Missing values")
        display_safe_dataframe(
            distribution_to_dataframe(
                quality_report["missing_values"],
                "column",
            ),
            width="stretch",
        )

    synthetic_dataset = render_synthetic_dataset_block()

    render_basic_analytics_block(quality_dataset)
    render_real_ml_run_starter_block(df, quality_dataset)
    render_model_artifact_separation_block()
    render_active_model_selection_block()
    render_model_promotion_gate_block(df)
    render_training_run_history_block()
    render_ml_baseline_block(
        quality_dataset,
        title="Manual real-data baseline model",
        model_path=MODEL_PATH,
        report_path=MODEL_REPORT_PATH,
        key_prefix="manual_real",
        default_min_labeled_rows=MIN_LABELED_ROWS_FOR_TRAINING,
        model_source=MODEL_SOURCE_MANUAL_REAL,
    )
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
                model_source=MODEL_SOURCE_SYNTHETIC,
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

    build_by_path = {path: load_build(path) for path in build_files}

    st.sidebar.subheader("Session")

    session_id = st.sidebar.text_input(
        "Session ID",
        value=default_session_id(),
    )

    run_context = render_run_context_sidebar()

    ship_types = sorted({build.ship_type for build in build_by_path.values()})
    selected_ship = st.sidebar.selectbox(
        "Ship",
        ship_types,
        key="selected_ship_type",
    )

    filtered_build_files = [
        path
        for path in build_files
        if build_by_path[path].ship_type == selected_ship
    ]

    build_file = st.sidebar.selectbox(
        "Build profile",
        filtered_build_files,
        format_func=lambda path: build_profile_label(build_by_path[path]),
    )

    build = build_by_path[build_file]

    st.sidebar.subheader("Current build")
    st.sidebar.write(f"Build ID: `{build.build_id}`")
    st.sidebar.write(f"Ship: `{build.ship_type}`")
    st.sidebar.write(f"File: `{build_file.name}`")
    st.sidebar.caption("To change ship/loadout, select Ship and then Build profile.")

    with st.sidebar.expander("Selected loadout", expanded=True):
        display_safe_dataframe(
            pd.DataFrame(build_loadout_rows(build, modules)),
            width="stretch",
            hide_index=True,
        )

    calculator_tab, saved_events_tab = st.tabs([
        "General calculator",
        "Saved events",
    ])

    with calculator_tab:
        st.caption(
            "One capture page for every ship/build. Choose Prospector, MOLE, Golem, "
            "or any future build from the Build profile selector in the sidebar."
        )
        render_calculator_tab(
            heads=heads,
            modules=modules,
            build=build,
            session_id=session_id,
            run_context=run_context,
            source="manual_ui",
            key_prefix="calculator",
            page_title="Rock parameters",
        )

    with saved_events_tab:
        render_saved_events_tab()


if __name__ == "__main__":
    main()
