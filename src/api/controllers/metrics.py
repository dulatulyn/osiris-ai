from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.response import MetricsResponse
from src.api.services.metrics import metrics_service
from src.api.services.prediction import prediction_service

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    dataset_path: Optional[str] = Query(None, description="Path to dataset CSV for evaluation"),
) -> MetricsResponse:
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        result = metrics_service.evaluate(dataset_path=dataset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found, provide a valid dataset_path")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return MetricsResponse(**result)
