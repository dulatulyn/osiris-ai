from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.response import HealthResponse
from src.api.services.prediction import prediction_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if prediction_service.is_loaded else "degraded",
        model_loaded=prediction_service.is_loaded,
    )
