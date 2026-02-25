from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.services.metrics import metrics_service
from src.api.services.model_registry import model_registry
from src.api.services.prediction import prediction_service
from src.config import METRICS_PATH

router = APIRouter(prefix="/models", tags=["models"])


class RegisterProfileRequest(BaseModel):
    name: str
    path: str
    description: str = ""


@router.get("")
async def list_models() -> dict:
    """List all available model profiles and the currently active one."""
    return model_registry.list_profiles()


@router.get("/active")
async def get_active_model() -> dict:
    """Return info about the currently active model profile."""
    active = model_registry.get_active()
    all_profiles = model_registry.list_profiles()
    info = all_profiles["profiles"].get(active, {})
    return {
        "active": active,
        "info": info,
        "model_loaded": prediction_service.is_loaded,
    }


@router.put("/active/{profile_name}")
async def switch_model(profile_name: str) -> dict:
    """Hot-swap the active model. Resets inference metrics on success."""
    try:
        path = model_registry.get_profile_path(profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not (path / "pipeline.joblib").exists():
        raise HTTPException(
            status_code=400,
            detail=f"No pipeline.joblib found at {path}. Has this model been trained?",
        )

    try:
        prediction_service.load(model_path=path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    model_registry.set_active(profile_name)
    metrics_service.reset()

    return {"status": "switched", "active": profile_name, "path": str(path)}


@router.get("/active/metrics")
async def get_training_metrics() -> dict:
    """Return training-time metrics (AUC, F1, etc.) for the active model."""
    active = model_registry.get_active()
    try:
        path = model_registry.get_profile_path(active)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    metrics_file = path / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"metrics.json not found for profile '{active}'. Has the model been trained?",
        )

    with open(metrics_file) as f:
        return json.load(f)


@router.post("")
async def register_model(request: RegisterProfileRequest) -> dict:
    """Register a new model profile pointing to an artifact directory."""
    path = Path(request.path)
    if not (path / "pipeline.joblib").exists():
        raise HTTPException(
            status_code=400,
            detail=f"No pipeline.joblib found at {request.path}",
        )

    model_registry.register(request.name, request.path, request.description)
    return {"status": "registered", "name": request.name, "path": request.path}
