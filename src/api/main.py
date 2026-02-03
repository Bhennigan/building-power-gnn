"""FastAPI application for building power efficiency GNN.

Provides REST endpoints for data ingestion and power prediction.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch

from .routes import ingest, predict
from .websocket import router as ws_router

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
    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingestion"])
    app.include_router(predict.router, prefix="/api/v1/predict", tags=["Prediction"])
    app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": _model is not None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "Building Power Efficiency GNN API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create default app instance
app = create_app()
