"""Pydantic schemas shared across the API."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class GraphPoint(BaseModel):
    t: float
    value: float


class DiagnosticGraph(BaseModel):
    id: str
    title: str
    description: str
    x_label: str
    y_label: str
    unit: str = ""
    series: list[GraphPoint]
    inference: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Literal["normal", "info", "warning", "critical"] = "info"
    suggestions: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class SeparationResult(BaseModel):
    sample_rate: int
    duration_s: float
    hss_rms: float
    lss_rms: float
    hss_snr_db: float
    lss_snr_db: float
    heartbeat_bpm: float | None = None
    breath_rate_bpm: float | None = None
    dominant_diagnosis: str
    diagnosis_confidence: float


class AnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    created_at: str
    separation: SeparationResult
    graphs: list[DiagnosticGraph]
    summary: str
    lifestyle: list[str]
    disclaimer: str


class ModelInfo(BaseModel):
    id: str
    name: str
    task: str
    source: str
    license: str
    commercial_ok: bool
    parameters: str | None = None
    modality: str
    notes: str


class AgentRequest(BaseModel):
    question: str
    analysis_id: str | None = None
    context: dict[str, Any] | None = None


class AgentResponse(BaseModel):
    answer: str
    provider: str
    model: str
    used_fallback: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    providers_available: list[str]
    models_available: list[str]
