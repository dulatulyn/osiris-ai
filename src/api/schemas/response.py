from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    fraud_score: int = Field(..., ge=0, le=100)
    risk_level: str
    probability: float


class InferenceMetricsResponse(BaseModel):
    total_predictions: int
    fraud_predictions: int
    legit_predictions: int
    fraud_rate: float
    avg_fraud_score: float
    score_distribution: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelProfileInfo(BaseModel):
    path: str
    description: str = ""
    created_at: str = ""


class ModelsListResponse(BaseModel):
    active: str
    profiles: dict[str, ModelProfileInfo]


class ActiveModelResponse(BaseModel):
    active: str
    info: dict
    model_loaded: bool
