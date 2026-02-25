from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class _InferenceStats:
    total: int = 0
    fraud: int = 0
    legit: int = 0
    score_sum: float = 0.0
    buckets: dict = field(
        default_factory=lambda: {"0-20": 0, "21-50": 0, "51-80": 0, "81-100": 0}
    )

    def record(self, fraud_score: int, risk_level: str) -> None:
        self.total += 1
        is_high_risk = risk_level in ("high", "critical")
        if is_high_risk:
            self.fraud += 1
        else:
            self.legit += 1
        self.score_sum += fraud_score

        if fraud_score <= 20:
            self.buckets["0-20"] += 1
        elif fraud_score <= 50:
            self.buckets["21-50"] += 1
        elif fraud_score <= 80:
            self.buckets["51-80"] += 1
        else:
            self.buckets["81-100"] += 1

    def to_dict(self) -> dict:
        return {
            "total_predictions": self.total,
            "fraud_predictions": self.fraud,
            "legit_predictions": self.legit,
            "fraud_rate": round(self.fraud / max(self.total, 1), 4),
            "avg_fraud_score": round(self.score_sum / max(self.total, 1), 2),
            "score_distribution": dict(self.buckets),
        }

    def reset(self) -> None:
        self.total = 0
        self.fraud = 0
        self.legit = 0
        self.score_sum = 0.0
        self.buckets = {"0-20": 0, "21-50": 0, "51-80": 0, "81-100": 0}


class MetricsService:
    def __init__(self):
        self._stats = _InferenceStats()
        self._lock = threading.Lock()

    def record(self, fraud_score: int, risk_level: str) -> None:
        with self._lock:
            self._stats.record(fraud_score, risk_level)

    def get_metrics(self) -> dict:
        with self._lock:
            return self._stats.to_dict()

    def reset(self) -> None:
        with self._lock:
            self._stats.reset()


metrics_service = MetricsService()
