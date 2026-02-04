"""FastAPI application for building power efficiency GNN.

Provides REST endpoints for data ingestion and power prediction.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch

from fastapi.responses import StreamingResponse
from io import StringIO

from .routes import ingest, predict
from .routes.ui import router as ui_router
from .routes.buildings import router as buildings_router
from .routes.integrations import router as integrations_router
from .websocket import router as ws_router
from ..auth.routes import router as auth_router
from ..db.session import engine, Base
from ..ingestion.csv_parser import generate_template_csv

logger = logging.getLogger(__name__)

# Global state
_model = None
_graph_builder = None
_feature_store = None


def get_model():
    """Get the loaded model."""
    return _model


def get_graph_builder():
    """Get the graph builder instance."""
    return _graph_builder


def get_feature_store():
    """Get the feature store instance."""
    return _feature_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    global _model, _graph_builder, _feature_store

    logger.info("Starting Building Power GNN API...")

    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")

    # Initialize components
    from ..graph import GraphBuilder, FeatureStore

    _graph_builder = GraphBuilder()
    _feature_store = FeatureStore(window_size=168)

    # Load model if checkpoint exists
    checkpoint_path = Path("checkpoints/best_model.pt")
    if checkpoint_path.exists():
        try:
            from ..model import BuildingPowerGNN
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Get node feature dims from checkpoint or use defaults
            node_feature_dims = {
                "hvac": 8,
                "lighting": 6,
                "sensor": 5,
                "room": 7,
                "meter": 4,
                "weatherstation": 3,
            }

            _model = BuildingPowerGNN(node_feature_dims=node_feature_dims)
            _model.load_state_dict(checkpoint["model_state_dict"])
            _model.eval()
            logger.info(f"Loaded model from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")

    logger.info("API startup complete")

    yield

    # Cleanup
    logger.info("Shutting down Building Power GNN API...")
    _model = None
    _graph_builder = None
    _feature_store = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Building Power Efficiency GNN API",
        description="GNN-based predictive power efficiency analysis for buildings",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(buildings_router, prefix="/api/v1/buildings", tags=["Buildings"])
    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingestion"])
    app.include_router(predict.router, prefix="/api/v1/predict", tags=["Prediction"])
    app.include_router(integrations_router, prefix="/api/v1/integrations", tags=["Integrations"])
    app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])
    app.include_router(ui_router, tags=["UI"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": _model is not None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    @app.get("/api/v1/templates/{data_type}")
    async def download_template(data_type: str):
        """Download a CSV template for data upload."""
        if data_type not in ["nodes", "edges", "readings"]:
            raise HTTPException(status_code=400, detail="Invalid data type")

        content = generate_template_csv(data_type)
        return StreamingResponse(
            StringIO(content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={data_type}_template.csv"}
        )

    return app


# Create default app instance
app = create_app()
