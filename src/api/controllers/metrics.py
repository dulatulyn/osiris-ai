from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.response import InferenceMetricsResponse
from src.api.services.metrics import metrics_service

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=InferenceMetricsResponse)
async def get_metrics() -> InferenceMetricsResponse:
    """Return live inference statistics since server start (or last model switch)."""
    data = metrics_service.get_metrics()
    return InferenceMetricsResponse(**data)
