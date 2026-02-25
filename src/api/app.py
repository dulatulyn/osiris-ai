from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.controllers.health import router as health_router
from src.api.controllers.metrics import router as metrics_router
from src.api.controllers.predict import router as predict_router
from src.api.services.prediction import prediction_service
from src.config import MODEL_WEIGHTS_PATH

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_WEIGHTS_PATH.exists():
        prediction_service.load()
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model weights not found at %s, server starting without model", MODEL_WEIGHTS_PATH)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Osiris Fraud Detection API",
        description="Financial fraud detection scoring API for loan applications",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(predict_router)
    app.include_router(health_router)
    app.include_router(metrics_router)

    return app
