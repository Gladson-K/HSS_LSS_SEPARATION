"""HTTP API routes."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.core import ai_agent, analysis, model_registry
from app.core.schemas import (
    AgentRequest,
    AgentResponse,
    AnalysisResponse,
    DiagnosticGraph,
    HealthResponse,
    ModelInfo,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from app.ml import foundation_models

    return HealthResponse(
        status="ok",
        version=get_settings().version,
        providers_available=ai_agent.available_providers(),
        models_available=foundation_models.available_backends(),
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models(permissive_only: bool = False) -> list[ModelInfo]:
    return model_registry.permissive_models() if permissive_only else model_registry.list_models()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    settings = get_settings()
    name = file.filename or "recording.wav"
    if not name.lower().endswith(settings.allowed_audio_suffixes):
        raise HTTPException(status_code=415, detail="Unsupported audio format")
    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        return analysis.run_analysis(raw, filename=name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str) -> AnalysisResponse:
    resp = analysis.get_analysis(analysis_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return resp


@router.get("/diagnostics/{analysis_id}/graphs", response_model=list[DiagnosticGraph])
def get_graphs(analysis_id: str) -> list[DiagnosticGraph]:
    resp = analysis.get_analysis(analysis_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return resp.graphs


@router.get("/diagnostics/{analysis_id}/graphs/{index}", response_model=DiagnosticGraph)
def get_graph(analysis_id: str, index: int) -> DiagnosticGraph:
    resp = analysis.get_analysis(analysis_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if index < 0 or index >= len(resp.graphs):
        raise HTTPException(status_code=404, detail="Graph index out of range")
    return resp.graphs[index]


@router.get("/agent/providers")
def agent_providers() -> dict[str, list[str]]:
    return {"providers": ai_agent.available_providers()}


@router.post("/agent", response_model=AgentResponse)
def agent(req: AgentRequest) -> AgentResponse:
    context = req.context or {}
    if req.analysis_id:
        resp = analysis.get_analysis(req.analysis_id)
        if resp is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        context = {
            "separation": resp.separation.model_dump(),
            "graphs": [g.model_dump() for g in resp.graphs],
            "summary": resp.summary,
            "lifestyle": resp.lifestyle,
        }
    return ai_agent.ask(req.question, context)
