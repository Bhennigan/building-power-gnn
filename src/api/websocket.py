"""WebSocket endpoints for real-time graph updates.

Provides live graph preview and streaming predictions.
"""

from typing import Optional
from datetime import datetime
import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: dict[str, list[WebSocket]] = {
            "graph": [],
            "predictions": [],
            "anomalies": [],
        }

    async def connect(self, websocket: WebSocket, channel: str = "graph"):
        """Accept and register a new connection."""
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)
        logger.info(f"WebSocket connected to channel: {channel}")

    def disconnect(self, websocket: WebSocket, channel: str = "graph"):
        """Remove a connection."""
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)
        logger.info(f"WebSocket disconnected from channel: {channel}")

    async def broadcast(self, message: dict, channel: str = "graph"):
        """Broadcast message to all connections on a channel."""
        if channel not in self.active_connections:
            return

        dead_connections = []
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(conn, channel)

    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")


manager = ConnectionManager()


@router.websocket("/graph")
async def graph_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time graph updates.

    Receives:
        - add_node: Add a new node
        - add_edge: Add a new edge
        - remove_node: Remove a node
        - get_graph: Request current graph state

    Broadcasts:
        - graph_update: Graph structure changed
        - node_added: New node added
        - edge_added: New edge added
    """
    await manager.connect(websocket, "graph")

    try:
        # Send initial graph state
        from .routes.ingest import get_current_data
        data = get_current_data()
        await manager.send_personal(websocket, {
            "type": "graph_state",
            "data": {
                "nodes": list(data["nodes"].values()),
                "edges": data["edges"],
            },
            "timestamp": datetime.utcnow().isoformat(),
        })

        while True:
            # Receive and process messages
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)

            msg_type = message.get("type")
            payload = message.get("data", {})

            if msg_type == "add_node":
                await _handle_add_node(payload)
                await manager.broadcast({
                    "type": "node_added",
                    "data": payload,
                    "timestamp": datetime.utcnow().isoformat(),
                }, "graph")

            elif msg_type == "add_edge":
                await _handle_add_edge(payload)
                await manager.broadcast({
                    "type": "edge_added",
                    "data": payload,
                    "timestamp": datetime.utcnow().isoformat(),
                }, "graph")

            elif msg_type == "remove_node":
                node_id = payload.get("node_id")
                await _handle_remove_node(node_id)
                await manager.broadcast({
                    "type": "node_removed",
                    "data": {"node_id": node_id},
                    "timestamp": datetime.utcnow().isoformat(),
                }, "graph")

            elif msg_type == "get_graph":
                data = get_current_data()
                await manager.send_personal(websocket, {
                    "type": "graph_state",
                    "data": {
                        "nodes": list(data["nodes"].values()),
                        "edges": data["edges"],
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                })

            elif msg_type == "ping":
                await manager.send_personal(websocket, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, "graph")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, "graph")


@router.websocket("/predictions")
async def predictions_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming predictions.

    Sends periodic prediction updates for monitored nodes.
    """
    await manager.connect(websocket, "predictions")

    try:
        # Configuration from client
        raw_config = await websocket.receive_text()
        config = json.loads(raw_config)

        node_ids = config.get("node_ids", [])
        interval_seconds = config.get("interval", 60)

        await manager.send_personal(websocket, {
            "type": "config_accepted",
            "data": {"node_ids": node_ids, "interval": interval_seconds},
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Stream predictions at configured interval
        while True:
            predictions = await _get_predictions_for_nodes(node_ids)

            await manager.send_personal(websocket, {
                "type": "predictions",
                "data": predictions,
                "timestamp": datetime.utcnow().isoformat(),
            })

            await asyncio.sleep(interval_seconds)

    except WebSocketDisconnect:
        manager.disconnect(websocket, "predictions")
    except Exception as e:
        logger.error(f"Predictions WebSocket error: {e}")
        manager.disconnect(websocket, "predictions")


@router.websocket("/anomalies")
async def anomalies_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time anomaly alerts.

    Sends alerts when anomalies are detected.
    """
    await manager.connect(websocket, "anomalies")

    try:
        # Configuration
        raw_config = await websocket.receive_text()
        config = json.loads(raw_config)

        threshold = config.get("threshold", 0.5)
        check_interval = config.get("interval", 30)

        await manager.send_personal(websocket, {
            "type": "config_accepted",
            "data": {"threshold": threshold, "interval": check_interval},
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Monitor for anomalies
        while True:
            anomalies = await _detect_anomalies(threshold)

            if anomalies:
                await manager.send_personal(websocket, {
                    "type": "anomaly_alert",
                    "data": anomalies,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            await asyncio.sleep(check_interval)

    except WebSocketDisconnect:
        manager.disconnect(websocket, "anomalies")
    except Exception as e:
        logger.error(f"Anomalies WebSocket error: {e}")
        manager.disconnect(websocket, "anomalies")


async def _handle_add_node(data: dict) -> None:
    """Handle adding a node via WebSocket."""
    from .routes.ingest import _nodes

    node_id = data.get("node_id")
    if node_id:
        _nodes[node_id] = data
        logger.info(f"Added node via WebSocket: {node_id}")


async def _handle_add_edge(data: dict) -> None:
    """Handle adding an edge via WebSocket."""
    from .routes.ingest import _edges, _nodes

    source = data.get("source")
    target = data.get("target")

    if source in _nodes and target in _nodes:
        _edges.append(data)
        logger.info(f"Added edge via WebSocket: {source} -> {target}")


async def _handle_remove_node(node_id: str) -> None:
    """Handle removing a node via WebSocket."""
    from .routes.ingest import _nodes, _edges

    if node_id in _nodes:
        del _nodes[node_id]
        # Remove connected edges
        _edges[:] = [e for e in _edges if e.get("source") != node_id and e.get("target") != node_id]
        logger.info(f"Removed node via WebSocket: {node_id}")


async def _get_predictions_for_nodes(node_ids: list[str]) -> list[dict]:
    """Get predictions for specified nodes."""
    from .routes.ingest import get_current_data

    if not node_ids:
        return []

    data = get_current_data()
    nodes = data["nodes"]

    predictions = []
    for node_id in node_ids:
        if node_id in nodes:
            # Simplified prediction (would use model in production)
            node = nodes[node_id]
            base_power = float(node.get("capacity_kw", 10) or 10)

            predictions.append({
                "node_id": node_id,
                "node_type": str(node.get("node_type", "unknown")),
                "predicted_power_kwh": round(base_power * 0.7, 2),
                "anomaly_score": 0.1,
            })

    return predictions


async def _detect_anomalies(threshold: float) -> list[dict]:
    """Detect anomalies above threshold."""
    from .routes.ingest import get_current_data

    data = get_current_data()
    nodes = data["nodes"]

    # Simplified anomaly detection
    anomalies = []
    for node_id, node in nodes.items():
        # Random anomaly simulation (would use model in production)
        import random
        score = random.random() * 0.3  # Usually low

        if score >= threshold:
            anomalies.append({
                "node_id": node_id,
                "node_type": str(node.get("node_type", "unknown")),
                "anomaly_score": round(score, 4),
                "severity": "high" if score > 0.8 else "medium" if score > 0.6 else "low",
            })

    return anomalies


async def broadcast_graph_update(update_type: str, data: dict):
    """Broadcast graph update to all connected clients.

    Call this when graph changes occur via REST API.
    """
    await manager.broadcast({
        "type": update_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }, "graph")
