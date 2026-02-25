from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.request import ApplicationRequest
from src.api.schemas.response import PredictionResponse
from src.api.services.prediction import prediction_service

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(application: ApplicationRequest) -> PredictionResponse:
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    result = prediction_service.predict(application.model_dump())
    return PredictionResponse(**result)
