from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    fraud_score: int = Field(..., ge=0, le=100)
    risk_level: str
    probability: float


class ConfusionMatrixResponse(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int


class MetricsResponse(BaseModel):
    auc_roc: float
    auc_pr: float
    log_loss: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    mcc: float
    confusion_matrix: ConfusionMatrixResponse
    total_samples: int
    fraud_samples: int
    legit_samples: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
