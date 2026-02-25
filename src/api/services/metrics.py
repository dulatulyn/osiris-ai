from __future__ import annotations

import json
from pathlib import Path

from src.config import METRICS_PATH


class MetricsService:
    def evaluate(self, dataset_path: str | None = None) -> dict:
        """
        Loads precomputed metrics from training (metrics.json).
        
        Args:
            dataset_path: Ignored. Maintained for backwards compatibility with controller signature.
            
        Returns:
            Dictionary containing metrics like auc_roc, log_loss, etc.
        """
        if not METRICS_PATH.exists():
            raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}. Has the model been trained?")

        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        # Standardizing output if some fields are missing from legacy trainer outputs
        if "total_samples" not in metrics:
            metrics["total_samples"] = 0
            metrics["fraud_samples"] = 0
            metrics["legit_samples"] = 0

        return metrics


metrics_service = MetricsService()
