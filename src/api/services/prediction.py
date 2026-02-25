from __future__ import annotations

import threading
from pathlib import Path

import joblib
import pandas as pd

from src.config import DROP_COLUMNS, FEATURE_META_PATH, PIPELINE_PATH


class PredictionService:
    def __init__(self):
        self._pipeline = None
        self._feature_cols: list[str] | None = None
        self._threshold: float = 0.5
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self, model_path: Path | None = None) -> None:
        """Load pipeline.joblib and feature_meta.joblib from the given path."""
        pipeline_path = (model_path / "pipeline.joblib") if model_path else PIPELINE_PATH
        meta_path = (model_path / "feature_meta.joblib") if model_path else FEATURE_META_PATH

        pipeline = joblib.load(pipeline_path)
        meta = joblib.load(meta_path)

        with self._lock:
            self._pipeline = pipeline
            self._feature_cols = meta["feature_cols"]
            self._threshold = float(meta.get("threshold", 0.5))

    def predict(self, application_data: dict) -> dict:
        with self._lock:
            if not self.is_loaded:
                raise RuntimeError("Model is not loaded")

            df = pd.DataFrame([application_data])
            drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
            df = df.drop(columns=drop_cols, errors="ignore")

            for col in self._feature_cols:
                if col not in df.columns:
                    df[col] = 0

            df = df[self._feature_cols]

            prob = float(self._pipeline.predict_proba(df)[0, 1])

        fraud_score = max(0, min(100, int(round(prob * 100))))
        risk_level = self._classify_risk(fraud_score)

        return {
            "fraud_score": fraud_score,
            "risk_level": risk_level,
            "probability": round(prob, 6),
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
