from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.controllers.health import router as health_router
from src.api.controllers.metrics import router as metrics_router
from src.api.controllers.models import router as models_router
from src.api.controllers.predict import router as predict_router
from src.api.services.model_registry import model_registry
from src.api.services.prediction import prediction_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        active = model_registry.get_active()
        path = model_registry.get_profile_path(active)
        prediction_service.load(model_path=path)
        logger.info("Model loaded: profile=%s path=%s", active, path)
    except Exception as exc:
        logger.warning("Could not load model on startup: %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Osiris Fraud Detection API",
        description="Financial fraud detection scoring API for loan applications",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.include_router(predict_router)
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(models_router)

    return app
