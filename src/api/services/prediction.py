from __future__ import annotations

import joblib
import numpy as np
import torch

from src.config import ENCODERS_PATH, FEATURE_META_PATH, MODEL_WEIGHTS_PATH, SCALER_PATH
from src.training.model import FraudDetector
from src.training.preprocessing import preprocess_single


class PredictionService:
    def __init__(self):
        self._model: FraudDetector | None = None
        self._scaler = None
        self._encoders = None
        self._feature_cols: list[str] | None = None
        self._input_dim: int = 0
        self._device = torch.device("cpu")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        meta = joblib.load(FEATURE_META_PATH)
        self._feature_cols = meta["feature_cols"]
        self._input_dim = meta["input_dim"]

        self._scaler = joblib.load(SCALER_PATH)
        self._encoders = joblib.load(ENCODERS_PATH)

        self._model = FraudDetector(input_dim=self._input_dim, feature_cols=self._feature_cols)
        state = torch.load(MODEL_WEIGHTS_PATH, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, application_data: dict) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        X = preprocess_single(
            application_data,
            scaler=self._scaler,
            encoders=self._encoders,
            feature_cols=self._feature_cols,
        )

        tensor = torch.tensor(X, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            logit = self._model(tensor)
            probability = torch.sigmoid(logit).item()

        fraud_score = int(round(probability * 100))
        fraud_score = max(0, min(100, fraud_score))

        risk_level = self._classify_risk(fraud_score)

        return {
            "fraud_score": fraud_score,
            "risk_level": risk_level,
            "probability": round(probability, 6),
        }

    @staticmethod
    def _classify_risk(score: int) -> str:
        if score <= 20:
            return "low"
        if score <= 50:
            return "medium"
        if score <= 80:
            return "high"
        return "critical"


prediction_service = PredictionService()
