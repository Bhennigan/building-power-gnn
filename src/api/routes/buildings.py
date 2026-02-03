"""Building API routes for CRUD and data upload."""

from typing import Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...db import get_db, crud
from ...auth.dependencies import get_current_user, TenantContext, get_tenant_context
from ...db.models import User
from ...ingestion.csv_parser import CSVParser

logger = logging.getLogger(__name__)

router = APIRouter()


class BuildingCreate(BaseModel):
    """Create building request."""
    name: str = Field(..., min_length=1, max_length=200)
    building_type: Optional[str] = None
    address: Optional[str] = None
    total_area_sqft: Optional[float] = None
    floors: Optional[int] = None


class BuildingResponse(BaseModel):
    """Building response."""
    id: str
    name: str
    building_type: Optional[str]
    address: Optional[str]
    total_area_sqft: Optional[float]
    floors: Optional[int]
    node_count: int = 0
    edge_count: int = 0

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    """Upload response."""
    nodes_added: int = 0
    edges_added: int = 0
    readings_added: int = 0
    errors: list[str] = []
    warnings: list[str] = []


@router.post("", response_model=BuildingResponse)
async def create_building(
    request: BuildingCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new building."""
    building = crud.create_building(
        db,
        tenant_id=user.tenant_id,
        name=request.name,
        building_type=request.building_type,
        address=request.address,
        total_area_sqft=request.total_area_sqft,
        floors=request.floors
    )

    return BuildingResponse(
        id=building.id,
        name=building.name,
        building_type=building.building_type,
        address=building.address,
        total_area_sqft=building.total_area_sqft,
        floors=building.floors
    )


@router.get("", response_model=list[BuildingResponse])
async def list_buildings(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all buildings for current tenant."""
    buildings = crud.get_buildings(db, user.tenant_id)
    result = []
    for b in buildings:
        stats = crud.get_building_stats(db, b.id)
        result.append(BuildingResponse(
            id=b.id,
            name=b.name,
            building_type=b.building_type,
            address=b.address,
            total_area_sqft=b.total_area_sqft,
            floors=b.floors,
            node_count=stats["total_nodes"],
            edge_count=stats["total_edges"]
        ))
    return result


@router.get("/{building_id}", response_model=BuildingResponse)
async def get_building(
    building_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get building details."""
    building = crud.get_building(db, building_id)
    if not building or building.tenant_id != user.tenant_id:
        raise HTTPException(status_code=404, detail="Building not found")

    stats = crud.get_building_stats(db, building_id)
    return BuildingResponse(
        id=building.id,
        name=building.name,
        building_type=building.building_type,
        address=building.address,
        total_area_sqft=building.total_area_sqft,
        floors=building.floors,
        node_count=stats["total_nodes"],
        edge_count=stats["total_edges"]
    )


@router.delete("/{building_id}")
async def delete_building(
    building_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a building."""
    building = crud.get_building(db, building_id)
    if not building or building.tenant_id != user.tenant_id:
        raise HTTPException(status_code=404, detail="Building not found")

    crud.delete_building(db, building_id)
    return {"message": "Building deleted"}


@router.post("/{building_id}/upload", response_model=UploadResponse)
async def upload_building_data(
    building_id: str,
    file: UploadFile = File(...),
    data_type: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload CSV/Excel data to a building."""
    # Verify building access
    building = crud.get_building(db, building_id)
    if not building or building.tenant_id != user.tenant_id:
        raise HTTPException(status_code=404, detail="Building not found")

    # Read file content
    content = await file.read()
    filename = file.filename or "upload.csv"

    # Parse file
    parser = CSVParser()
    result = parser.parse_file(content, filename, data_type)

    response = UploadResponse(
        errors=result.errors,
        warnings=result.warnings
    )

    if not result.is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Parse errors: {'; '.join(result.errors)}"
        )

    # Store parsed data
    try:
        if not result.nodes_df.empty:
            count = crud.bulk_create_nodes(db, building_id, result.nodes_df)
            response.nodes_added = count
            logger.info(f"Added {count} nodes to building {building_id}")

        if not result.edges_df.empty:
            count = crud.bulk_create_edges(db, building_id, result.edges_df)
            response.edges_added = count
            logger.info(f"Added {count} edges to building {building_id}")

        if not result.readings_df.empty:
            count = crud.bulk_create_readings(db, building_id, result.readings_df)
            response.readings_added = count
            logger.info(f"Added {count} readings to building {building_id}")

    except Exception as e:
        logger.exception(f"Error storing data for building {building_id}")
        raise HTTPException(status_code=500, detail=f"Failed to store data: {str(e)}")

    return response


@router.get("/{building_id}/stats")
async def get_building_stats(
    building_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get building statistics."""
    building = crud.get_building(db, building_id)
    if not building or building.tenant_id != user.tenant_id:
        raise HTTPException(status_code=404, detail="Building not found")

    return crud.get_building_stats(db, building_id)


@router.post("/{building_id}/train")
async def train_building_model(
    building_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start model training for a building."""
    building = crud.get_building(db, building_id)
    if not building or building.tenant_id != user.tenant_id:
        raise HTTPException(status_code=404, detail="Building not found")

    # Check if there's enough data
    stats = crud.get_building_stats(db, building_id)
    if stats["total_nodes"] < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 nodes to train a model"
        )

    # Create model record
    model = crud.create_trained_model(
        db,
        tenant_id=user.tenant_id,
        building_id=building_id,
        name=f"Model for {building.name}"
    )

    # Start training in background
    background_tasks.add_task(
        _train_model_task,
        model_id=model.id,
        building_id=building_id,
        tenant_id=user.tenant_id
    )

    return {
        "message": "Training started",
        "model_id": model.id,
        "status": "training"
    }


async def _train_model_task(model_id: str, building_id: str, tenant_id: str):
    """Background task to train a model."""
    from ...db.session import SessionLocal
    import time

    db = SessionLocal()
    try:
        # Update status to training
        crud.update_model_status(db, model_id, "training")

        # Get building data
        nodes = crud.get_nodes(db, building_id)
        edges = crud.get_edges(db, building_id)

        logger.info(f"Training model {model_id} with {len(nodes)} nodes, {len(edges)} edges")

        # Simulate training (in production, this would use the actual GNN training)
        # For MVP, we'll just simulate the training process
        time.sleep(5)

        # Mark as ready
        crud.update_model_status(
            db,
            model_id,
            "ready",
            model_path=f"models/{tenant_id}/{model_id}.pt",
            metrics={"accuracy": 0.85, "loss": 0.15}
        )

        logger.info(f"Model {model_id} training complete")

    except Exception as e:
        logger.exception(f"Training failed for model {model_id}")
        crud.update_model_status(db, model_id, "failed")
    finally:
        db.close()
