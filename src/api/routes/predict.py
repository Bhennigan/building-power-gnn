"""Prediction API endpoints.

Provides power consumption prediction and anomaly detection.
"""

from typing import Optional
from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import torch
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictionRequest(BaseModel):
    """Request for power prediction."""
    node_ids: list[str] = Field(..., min_length=1, description="Node IDs to predict for")
    horizon_hours: int = Field(24, ge=1, le=168, description="Prediction horizon in hours")
    include_confidence: bool = Field(True, description="Include confidence intervals")


class NodePrediction(BaseModel):
    """Prediction result for a single node."""
    node_id: str
    node_type: str
    predicted_power_kwh: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    anomaly_score: float = 0.0
    is_anomaly: bool = False


class PredictionResponse(BaseModel):
    """Response containing predictions."""
    timestamp: datetime
    horizon_hours: int
    predictions: list[NodePrediction]
    total_predicted_power_kwh: float
    model_version: str = "1.0.0"


class AnomalyReport(BaseModel):
    """Anomaly detection report."""
    node_id: str
    node_type: str
    anomaly_score: float
    severity: str  # 'low', 'medium', 'high'
    description: str
    detected_at: datetime


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    building_id: Optional[str] = None
    node_types: Optional[list[str]] = None
    horizon_hours: int = Field(24, ge=1, le=168)


@router.post("/power", response_model=PredictionResponse)
async def predict_power(request: PredictionRequest):
    """Predict power consumption for specified nodes.

    Args:
        request: Prediction request with node IDs and horizon.

    Returns:
        Power consumption predictions.
    """
    from ..main import get_model, get_graph_builder, get_feature_store
    from .ingest import get_current_data

    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train or load a model first."
        )

    # Get current data
    data = get_current_data()
    nodes = data["nodes"]

    # Validate node IDs
    invalid_ids = [nid for nid in request.node_ids if nid not in nodes]
    if invalid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid node IDs: {invalid_ids}"
        )

    predictions = []
    total_power = 0.0

    try:
        # Build graph from current data
        graph_builder = get_graph_builder()
        feature_store = get_feature_store()

        nodes_df = pd.DataFrame(list(nodes.values()))
        edges_df = pd.DataFrame(data["edges"])

        graph_data = graph_builder.build(nodes_df, edges_df)

        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(graph_data)

        # Extract predictions for requested nodes
        for node_id in request.node_ids:
            node_info = nodes[node_id]
            node_type = str(node_info.get("node_type", "")).lower()

            # Get prediction from model output
            pred_power = 0.0
            anomaly_score = 0.0

            if node_type in outputs:
                node_outputs = outputs[node_type]
                # Find index of this node
                idx = graph_builder.get_node_index(node_type.upper(), node_id)
                if idx is not None and "power_pred" in node_outputs:
                    pred_power = node_outputs["power_pred"][idx].item()
                    if "anomaly_score" in node_outputs:
                        anomaly_score = node_outputs["anomaly_score"][idx].item()

            # Scale by horizon
            pred_power = pred_power * request.horizon_hours

            # Confidence intervals (simplified)
            confidence_lower = None
            confidence_upper = None
            if request.include_confidence:
                std_estimate = abs(pred_power) * 0.1  # 10% uncertainty
                confidence_lower = pred_power - 1.96 * std_estimate
                confidence_upper = pred_power + 1.96 * std_estimate

            predictions.append(NodePrediction(
                node_id=node_id,
                node_type=node_type,
                predicted_power_kwh=round(pred_power, 2),
                confidence_lower=round(confidence_lower, 2) if confidence_lower else None,
                confidence_upper=round(confidence_upper, 2) if confidence_upper else None,
                anomaly_score=round(anomaly_score, 4),
                is_anomaly=anomaly_score > 0.5,
            ))

            total_power += pred_power

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        timestamp=datetime.utcnow(),
        horizon_hours=request.horizon_hours,
        predictions=predictions,
        total_predicted_power_kwh=round(total_power, 2),
    )


@router.post("/batch", response_model=PredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict power consumption for all or filtered nodes.

    Args:
        request: Batch prediction request with optional filters.

    Returns:
        Power consumption predictions for all matching nodes.
    """
    from .ingest import get_current_data

    data = get_current_data()
    nodes = data["nodes"]

    # Filter nodes
    node_ids = []
    for node_id, node in nodes.items():
        if request.node_types:
            node_type = str(node.get("node_type", ""))
            if node_type not in request.node_types:
                continue
        node_ids.append(node_id)

    if not node_ids:
        raise HTTPException(
            status_code=400,
            detail="No nodes match the specified filters"
        )

    # Use the main prediction endpoint
    pred_request = PredictionRequest(
        node_ids=node_ids,
        horizon_hours=request.horizon_hours,
        include_confidence=True,
    )

    return await predict_power(pred_request)


@router.get("/anomalies", response_model=list[AnomalyReport])
async def detect_anomalies(
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Anomaly score threshold"),
    node_type: Optional[str] = None,
):
    """Detect anomalies in current building data.

    Args:
        threshold: Minimum anomaly score to report.
        node_type: Filter by node type.

    Returns:
        List of detected anomalies.
    """
    from ..main import get_model, get_graph_builder
    from .ingest import get_current_data

    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    data = get_current_data()
    nodes = data["nodes"]

    if not nodes:
        return []

    try:
        graph_builder = get_graph_builder()
        nodes_df = pd.DataFrame(list(nodes.values()))
        edges_df = pd.DataFrame(data["edges"])

        graph_data = graph_builder.build(nodes_df, edges_df)

        model.eval()
        with torch.no_grad():
            outputs = model(graph_data)

        anomalies = []
        now = datetime.utcnow()

        for node_id, node_info in nodes.items():
            nt = str(node_info.get("node_type", "")).lower()

            if node_type and nt != node_type.lower():
                continue

            if nt in outputs and "anomaly_score" in outputs[nt]:
                idx = graph_builder.get_node_index(nt.upper(), node_id)
                if idx is not None:
                    score = outputs[nt]["anomaly_score"][idx].item()

                    if score >= threshold:
                        # Determine severity
                        if score >= 0.8:
                            severity = "high"
                        elif score >= 0.6:
                            severity = "medium"
                        else:
                            severity = "low"

                        anomalies.append(AnomalyReport(
                            node_id=node_id,
                            node_type=nt,
                            anomaly_score=round(score, 4),
                            severity=severity,
                            description=f"Unusual pattern detected in {nt} node",
                            detected_at=now,
                        ))

        # Sort by score descending
        anomalies.sort(key=lambda x: x.anomaly_score, reverse=True)

        return anomalies

    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@router.get("/efficiency")
async def get_efficiency_score(
    node_id: Optional[str] = None,
    zone: Optional[str] = None,
):
    """Calculate power efficiency score for nodes or zones.

    Args:
        node_id: Specific node to evaluate.
        zone: Zone to evaluate.

    Returns:
        Efficiency metrics.
    """
    from .ingest import get_current_data

    data = get_current_data()
    nodes = data["nodes"]
    timeseries = data["timeseries"]

    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes in graph")

    # Filter nodes
    target_nodes = []
    if node_id:
        if node_id not in nodes:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        target_nodes = [nodes[node_id]]
    elif zone:
        target_nodes = [n for n in nodes.values() if n.get("zone") == zone]
    else:
        target_nodes = list(nodes.values())

    if not target_nodes:
        raise HTTPException(status_code=400, detail="No matching nodes found")

    # Calculate efficiency metrics
    hvac_nodes = [n for n in target_nodes if str(n.get("node_type", "")).upper() == "HVAC"]

    total_capacity = sum(float(n.get("capacity_kw", 0) or 0) for n in hvac_nodes)
    avg_efficiency = 0.0
    if hvac_nodes:
        efficiencies = [float(n.get("attr_efficiency_rating", 0.85) or 0.85) for n in hvac_nodes]
        avg_efficiency = sum(efficiencies) / len(efficiencies)

    # Get recent power consumption from time series
    recent_power = 0.0
    if timeseries:
        node_ids = {n.get("node_id") for n in target_nodes}
        recent = [t for t in timeseries if t.get("node_id") in node_ids]
        if recent:
            recent_power = sum(float(t.get("value", 0)) for t in recent[-24:])  # Last 24 readings

    # Calculate efficiency score (simplified)
    efficiency_score = avg_efficiency * 100
    if total_capacity > 0 and recent_power > 0:
        utilization = recent_power / (total_capacity * 24)  # Assuming hourly readings
        efficiency_score = min(100, efficiency_score * (1 - abs(utilization - 0.7) / 0.7))

    return {
        "efficiency_score": round(efficiency_score, 1),
        "average_equipment_efficiency": round(avg_efficiency, 3),
        "total_hvac_capacity_kw": round(total_capacity, 2),
        "recent_power_consumption_kwh": round(recent_power, 2),
        "nodes_evaluated": len(target_nodes),
        "hvac_units": len(hvac_nodes),
        "recommendations": _get_efficiency_recommendations(efficiency_score, avg_efficiency),
    }


def _get_efficiency_recommendations(score: float, efficiency: float) -> list[str]:
    """Generate efficiency improvement recommendations."""
    recommendations = []

    if score < 60:
        recommendations.append("Consider scheduling an energy audit")
    if efficiency < 0.8:
        recommendations.append("Upgrade HVAC equipment to higher efficiency models")
    if score < 80:
        recommendations.append("Implement smart scheduling for HVAC systems")
        recommendations.append("Review zone temperature setpoints")

    if not recommendations:
        recommendations.append("System operating efficiently - maintain current settings")

    return recommendations


@router.get("/forecast")
async def get_power_forecast(
    hours: int = Query(24, ge=1, le=168, description="Forecast horizon"),
    resolution: str = Query("hourly", description="Forecast resolution"),
):
    """Get hourly power consumption forecast.

    Args:
        hours: Number of hours to forecast.
        resolution: Forecast resolution ('hourly' or 'daily').

    Returns:
        Time-series forecast data.
    """
    from .ingest import get_current_data

    data = get_current_data()
    nodes = data["nodes"]

    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes in graph")

    # Generate forecast (simplified - would use model in production)
    now = datetime.utcnow()
    forecast = []

    base_power = len(nodes) * 10  # Base estimate

    for h in range(hours):
        timestamp = now + timedelta(hours=h)
        hour_of_day = timestamp.hour

        # Simple diurnal pattern
        if 8 <= hour_of_day <= 18:
            multiplier = 1.3  # Business hours
        elif 6 <= hour_of_day <= 22:
            multiplier = 1.0  # Active hours
        else:
            multiplier = 0.6  # Night

        predicted = base_power * multiplier

        forecast.append({
            "timestamp": timestamp.isoformat(),
            "predicted_power_kwh": round(predicted, 2),
            "confidence_lower": round(predicted * 0.8, 2),
            "confidence_upper": round(predicted * 1.2, 2),
        })

    # Aggregate if daily resolution requested
    if resolution == "daily":
        daily_forecast = []
        for i in range(0, len(forecast), 24):
            day_data = forecast[i:i + 24]
            if day_data:
                daily_forecast.append({
                    "date": day_data[0]["timestamp"][:10],
                    "predicted_power_kwh": round(sum(d["predicted_power_kwh"] for d in day_data), 2),
                    "confidence_lower": round(sum(d["confidence_lower"] for d in day_data), 2),
                    "confidence_upper": round(sum(d["confidence_upper"] for d in day_data), 2),
                })
        forecast = daily_forecast

    return {
        "forecast": forecast,
        "horizon_hours": hours,
        "resolution": resolution,
        "generated_at": now.isoformat(),
    }
