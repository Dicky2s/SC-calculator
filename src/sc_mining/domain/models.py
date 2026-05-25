from typing import Literal

from pydantic import BaseModel, Field


ModuleType = Literal["passive", "active"]
Verdict = Literal["take", "risky", "skip", "need_more_power"]


class RockInput(BaseModel):
    mass: float = Field(gt=0)
    resistance: float = Field(ge=0)
    instability: float = Field(ge=0)
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
    power_percent: float = Field(ge=0, le=100)
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