"""Data ingestion API endpoints.

Handles XML bulk upload and form-based node/edge creation.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime
import tempfile
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd

from ...ingestion import (
    XMLParser,
    XMLValidationError,
    NodeCreate,
    EdgeCreate,
    TimeSeriesReading,
    TimeSeriesBatch,
    GraphValidator,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for demo (use database in production)
_nodes: dict[str, dict] = {}
_edges: list[dict] = []
_timeseries: list[dict] = []


class IngestionResponse(BaseModel):
    """Response for ingestion operations."""
    success: bool
    message: str
    nodes_added: int = 0
    edges_added: int = 0
    readings_added: int = 0
    warnings: list[str] = Field(default_factory=list)


class GraphStats(BaseModel):
    """Current graph statistics."""
    total_nodes: int
    total_edges: int
    total_readings: int
    nodes_by_type: dict[str, int]
    edges_by_type: dict[str, int]


@router.post("/xml", response_model=IngestionResponse)
async def upload_xml(
    file: UploadFile = File(...),
    validate: bool = True,
    background_tasks: BackgroundTasks = None,
):
    """Upload and process XML building graph file.

    Args:
        file: XML file to upload.
        validate: Whether to validate against XSD schema.

    Returns:
        Ingestion result summary.
    """
    if not file.filename.endswith(".xml"):
        raise HTTPException(status_code=400, detail="File must be XML format")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        parser = XMLParser()

        # Validate if requested
        if validate:
            is_valid, errors = parser.validate(tmp_path)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail={"message": "XML validation failed", "errors": errors}
                )

        # Parse
        result = parser.parse(tmp_path, validate=False, show_progress=False)

        # Validate graph integrity
        validator = GraphValidator()
        validation_result = validator.validate_full_graph(
            result.nodes_df,
            result.edges_df,
            result.timeseries_df,
        )

        # Store data
        nodes_added = 0
        for _, row in result.nodes_df.iterrows():
            node_id = row["node_id"]
            _nodes[node_id] = row.to_dict()
            nodes_added += 1

        edges_added = 0
        for _, row in result.edges_df.iterrows():
            _edges.append(row.to_dict())
            edges_added += 1

        readings_added = 0
        for _, row in result.timeseries_df.iterrows():
            _timeseries.append(row.to_dict())
            readings_added += 1

        return IngestionResponse(
            success=True,
            message=f"Successfully processed {file.filename}",
            nodes_added=nodes_added,
            edges_added=edges_added,
            readings_added=readings_added,
            warnings=validation_result.warnings,
        )

    except XMLValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/node", response_model=IngestionResponse)
async def create_node(node: NodeCreate):
    """Create a single node via form submission.

    Args:
        node: Node data from form.

    Returns:
        Ingestion result.
    """
    if node.node_id in _nodes:
        raise HTTPException(
            status_code=409,
            detail=f"Node {node.node_id} already exists"
        )

    _nodes[node.node_id] = node.model_dump()

    logger.info(f"Created node: {node.node_id} (type: {node.node_type})")

    return IngestionResponse(
        success=True,
        message=f"Created node {node.node_id}",
        nodes_added=1,
    )


@router.put("/node/{node_id}", response_model=IngestionResponse)
async def update_node(node_id: str, node: NodeCreate):
    """Update an existing node.

    Args:
        node_id: ID of node to update.
        node: Updated node data.

    Returns:
        Ingestion result.
    """
    if node_id not in _nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    if node.node_id != node_id:
        raise HTTPException(
            status_code=400,
            detail="Node ID in path must match body"
        )

    _nodes[node_id] = node.model_dump()

    return IngestionResponse(
        success=True,
        message=f"Updated node {node_id}",
        nodes_added=0,
    )


@router.delete("/node/{node_id}")
async def delete_node(node_id: str):
    """Delete a node and its connected edges.

    Args:
        node_id: ID of node to delete.

    Returns:
        Deletion result.
    """
    if node_id not in _nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    del _nodes[node_id]

    # Remove connected edges
    global _edges
    initial_edges = len(_edges)
    _edges = [e for e in _edges if e["source"] != node_id and e["target"] != node_id]
    edges_removed = initial_edges - len(_edges)

    return {
        "success": True,
        "message": f"Deleted node {node_id}",
        "edges_removed": edges_removed,
    }


@router.post("/edge", response_model=IngestionResponse)
async def create_edge(edge: EdgeCreate):
    """Create a single edge via form submission.

    Args:
        edge: Edge data from form.

    Returns:
        Ingestion result.
    """
    # Validate node references
    if edge.source not in _nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Source node {edge.source} does not exist"
        )
    if edge.target not in _nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Target node {edge.target} does not exist"
        )

    _edges.append(edge.model_dump())

    logger.info(f"Created edge: {edge.source} --{edge.edge_type}--> {edge.target}")

    return IngestionResponse(
        success=True,
        message=f"Created edge {edge.source} -> {edge.target}",
        edges_added=1,
    )


@router.post("/timeseries", response_model=IngestionResponse)
async def add_timeseries(reading: TimeSeriesReading):
    """Add a single time series reading.

    Args:
        reading: Time series reading data.

    Returns:
        Ingestion result.
    """
    if reading.node_id not in _nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Node {reading.node_id} does not exist"
        )

    _timeseries.append(reading.model_dump())

    return IngestionResponse(
        success=True,
        message="Added reading",
        readings_added=1,
    )


@router.post("/timeseries/batch", response_model=IngestionResponse)
async def add_timeseries_batch(batch: TimeSeriesBatch):
    """Add multiple time series readings in batch.

    Args:
        batch: Batch of time series readings.

    Returns:
        Ingestion result.
    """
    # Validate all node references
    invalid_nodes = set()
    for reading in batch.readings:
        if reading.node_id not in _nodes:
            invalid_nodes.add(reading.node_id)

    if invalid_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid node references: {list(invalid_nodes)}"
        )

    for reading in batch.readings:
        _timeseries.append(reading.model_dump())

    return IngestionResponse(
        success=True,
        message=f"Added {len(batch.readings)} readings",
        readings_added=len(batch.readings),
    )


@router.get("/stats", response_model=GraphStats)
async def get_stats():
    """Get current graph statistics.

    Returns:
        Graph statistics summary.
    """
    nodes_by_type: dict[str, int] = {}
    for node in _nodes.values():
        node_type = node.get("node_type", "unknown")
        if hasattr(node_type, "value"):
            node_type = node_type.value
        nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1

    edges_by_type: dict[str, int] = {}
    for edge in _edges:
        edge_type = edge.get("edge_type", "unknown")
        if hasattr(edge_type, "value"):
            edge_type = edge_type.value
        edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1

    return GraphStats(
        total_nodes=len(_nodes),
        total_edges=len(_edges),
        total_readings=len(_timeseries),
        nodes_by_type=nodes_by_type,
        edges_by_type=edges_by_type,
    )


@router.get("/nodes")
async def list_nodes(
    node_type: Optional[str] = None,
    zone: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List nodes with optional filtering.

    Args:
        node_type: Filter by node type.
        zone: Filter by zone.
        limit: Maximum results to return.
        offset: Offset for pagination.

    Returns:
        List of nodes.
    """
    nodes = list(_nodes.values())

    # Filter
    if node_type:
        nodes = [n for n in nodes if str(n.get("node_type", "")).lower() == node_type.lower()]
    if zone:
        nodes = [n for n in nodes if n.get("zone") == zone]

    # Paginate
    total = len(nodes)
    nodes = nodes[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "nodes": nodes,
    }


@router.get("/edges")
async def list_edges(
    edge_type: Optional[str] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List edges with optional filtering.

    Args:
        edge_type: Filter by edge type.
        source: Filter by source node.
        target: Filter by target node.
        limit: Maximum results to return.
        offset: Offset for pagination.

    Returns:
        List of edges.
    """
    edges = _edges.copy()

    # Filter
    if edge_type:
        edges = [e for e in edges if str(e.get("edge_type", "")).lower() == edge_type.lower()]
    if source:
        edges = [e for e in edges if e.get("source") == source]
    if target:
        edges = [e for e in edges if e.get("target") == target]

    # Paginate
    total = len(edges)
    edges = edges[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "edges": edges,
    }


@router.delete("/clear")
async def clear_all():
    """Clear all stored data (use with caution).

    Returns:
        Confirmation of data cleared.
    """
    global _nodes, _edges, _timeseries
    counts = {
        "nodes_cleared": len(_nodes),
        "edges_cleared": len(_edges),
        "readings_cleared": len(_timeseries),
    }

    _nodes = {}
    _edges = []
    _timeseries = []

    logger.warning("All data cleared")

    return {"success": True, "message": "All data cleared", **counts}


def get_current_data():
    """Get current in-memory data for graph building."""
    return {
        "nodes": _nodes,
        "edges": _edges,
        "timeseries": _timeseries,
    }
