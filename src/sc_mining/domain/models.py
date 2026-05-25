from typing import Literal

from pydantic import BaseModel, Field


ModuleType = Literal["passive", "active"]
Verdict = Literal["take", "risky", "skip", "need_more_power"]
OutcomeLabel = Literal[
    "unknown",
    "good",
    "bad",
    "too_slow",
    "too_unstable",
    "not_enough_power",
    "overheated",
    "wrong_prediction",
]
PowerObservationLabel = Literal[
    "unknown",
    "no_warmup",
    "warmup",
    "stable_hold",
    "overpowered",
    "too_unstable",
    "too_slow",
]


class RockInput(BaseModel):
    mass: float = Field(gt=0)
    resistance: float = Field(ge=0, le=1)
    instability: float = Field(ge=0, le=1)
    distance: float = Field(gt=0)


class HeadConfig(BaseModel):
    name: str
    size: int
    base_power: float
    stability_modifier: float = 1.0
    optimal_window_modifier: float = 1.0


class ModuleConfig(BaseModel):
    name: str
    type: ModuleType
    power_modifier: float = 1.0
    resistance_modifier: float = 1.0
    instability_modifier: float = 1.0
    optimal_window_modifier: float = 1.0


class HeadBuild(BaseModel):
    slot: str
    head_id: str
    modules: list[str] = Field(default_factory=list)


class BuildProfile(BaseModel):
    build_id: str
    ship_type: str
    heads: list[HeadBuild]


class BeamState(BaseModel):
    slot: str
    power_percent: float = Field(ge=20, le=100)
    active_modules: list[str] = Field(default_factory=list)


class CalculationInput(BaseModel):
    rock: RockInput
    build: BuildProfile
    beams: list[BeamState]


class CalculationResult(BaseModel):
    required_power: float
    effective_power: float
    margin: float
    risk_score: float
    verdict: Verdict
    notes: list[str] = Field(default_factory=list)


class OutcomeFeedback(BaseModel):
    actual_outcome: OutcomeLabel = "unknown"
    comment: str = ""


class ResourceComponent(BaseModel):
    """One resource entry in a mixed rock.

    This is intentionally simple for fast capture: resource name + percent +
    optional raw SCU estimate. Refinery results are stored separately because they
    can be known later than the mining event.
    """

    resource_name: str = "unknown"
    resource_percent: float | None = Field(default=None, ge=0, le=100)
    raw_scu_estimate: float | None = Field(default=None, ge=0)
    comment: str = ""


class ResourceYieldFeedback(BaseModel):
    # Backward-compatible summary fields used by older analytics/ML tables.
    primary_resource: str = "unknown"
    resource_percent: float | None = Field(default=None, ge=0, le=100)
    raw_scu_estimate: float | None = Field(default=None, ge=0)
    # Optional total ore size from the in-game scan readout (Composition XX SCU).
    total_scu_estimate: float | None = Field(default=None, ge=0)
    refined_scu_estimate: float | None = Field(default=None, ge=0)
    estimated_value_auec: float | None = Field(default=None, ge=0)
    mining_time_seconds: float | None = Field(default=None, ge=0)
    comment: str = ""

    # New multi-resource capture.
    resources: list[ResourceComponent] = Field(default_factory=list)


class RefinedResourceOutput(BaseModel):
    """One post-refinery resource output row.

    This is filled later, after the refinery job or sale is known. It lets one
    mining event keep both the original mixed-rock composition and the actual
    output/sale result per resource.
    """

    resource_name: str = "unknown"
    refined_scu_actual: float | None = Field(default=None, ge=0)
    sell_value_auec: float | None = Field(default=None, ge=0)
    comment: str = ""


class RefineryFeedback(BaseModel):
    """Optional refinery outcome block.

    It is separated from the mining resource table because refinery data may be
    filled after the rock is mined and after the refinery job completes.
    """

    refinery_method: str = "unknown"
    refinery_location: str = ""
    refinery_start_at: str = ""
    refinery_complete_at: str = ""
    refined_scu_actual: float | None = Field(default=None, ge=0)
    refined_value_auec: float | None = Field(default=None, ge=0)
    refinery_fee_auec: float | None = Field(default=None, ge=0)
    sell_value_auec: float | None = Field(default=None, ge=0)
    comment: str = ""
    refined_resources: list[RefinedResourceOutput] = Field(default_factory=list)


class PowerDistanceObservation(BaseModel):
    """One real in-game observation for power/distance calibration.

    Use this when the formula says one thing, but the game behaves differently:
    for example, 20% at 15m does not warm up, while 81% at 15m holds stable.
    These rows turn free-text comments into structured calibration data.
    """

    distance: float = Field(gt=0)
    power_percent: float = Field(ge=20, le=100)
    observation: PowerObservationLabel = "unknown"
    beam_warmed: bool | None = None
    held_stable: bool | None = None
    comment: str = ""


class CalibrationFeedback(BaseModel):
    """Optional real-game calibration block for formula tuning.

    This is not a gameplay outcome label. It captures what power/distance actually
    did in-game so the rule-based formula can be corrected later.
    """

    formula_issue_flag: bool = False
    observed_min_warmup_power_percent: float | None = Field(default=None, ge=20, le=100)
    observed_stable_power_percent: float | None = Field(default=None, ge=20, le=100)
    observed_distance: float | None = Field(default=None, gt=0)
    comment: str = ""
    observations: list[PowerDistanceObservation] = Field(default_factory=list)


class RunContext(BaseModel):
    operator_name: str = ""
    crew_size: int = Field(default=1, ge=1, le=10)
    run_tag: str = ""
