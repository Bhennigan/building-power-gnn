"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and info endpoints."""

    def test_health_check(self, client):
        """Health endpoint should return status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestIngestionEndpoints:
    """Tests for data ingestion endpoints."""

    def test_create_node(self, client):
        """Should create a new node."""
        node_data = {
            "node_id": "test_hvac_1",
            "node_type": "HVAC",
            "zone": "floor_1",
            "capacity_kw": 50.0,
        }
        response = client.post("/api/v1/ingest/node", json=node_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["nodes_added"] == 1

    def test_create_node_duplicate(self, client):
        """Should reject duplicate node ID."""
        node_data = {
            "node_id": "test_hvac_dup",
            "node_type": "HVAC",
        }
        # Create first node
        client.post("/api/v1/ingest/node", json=node_data)
        # Try to create duplicate
        response = client.post("/api/v1/ingest/node", json=node_data)
        assert response.status_code == 409

    def test_create_edge(self, client):
        """Should create an edge between nodes."""
        # Create nodes first
        client.post("/api/v1/ingest/node", json={
            "node_id": "edge_test_hvac",
            "node_type": "HVAC",
        })
        client.post("/api/v1/ingest/node", json={
            "node_id": "edge_test_room",
            "node_type": "Room",
        })

        # Create edge
        edge_data = {
            "source": "edge_test_hvac",
            "target": "edge_test_room",
            "edge_type": "serves",
        }
        response = client.post("/api/v1/ingest/edge", json=edge_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_create_edge_invalid_source(self, client):
        """Should reject edge with invalid source."""
        edge_data = {
            "source": "nonexistent_node",
            "target": "edge_test_room",
            "edge_type": "serves",
        }
        response = client.post("/api/v1/ingest/edge", json=edge_data)
        assert response.status_code == 400

    def test_add_timeseries(self, client):
        """Should add time series reading."""
        # Create node first
        client.post("/api/v1/ingest/node", json={
            "node_id": "ts_test_sensor",
            "node_type": "Sensor",
            "subtype": "temperature",
        })

        # Add reading
        reading_data = {
            "node_id": "ts_test_sensor",
            "timestamp": "2024-01-01T08:00:00",
            "value": 72.5,
            "metric": "temperature",
        }
        response = client.post("/api/v1/ingest/timeseries", json=reading_data)
        assert response.status_code == 200

    def test_get_stats(self, client):
        """Should return graph statistics."""
        response = client.get("/api/v1/ingest/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
        assert "total_edges" in data

    def test_list_nodes(self, client):
        """Should list nodes with pagination."""
        response = client.get("/api/v1/ingest/nodes?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "total" in data

    def test_list_edges(self, client):
        """Should list edges with filtering."""
        response = client.get("/api/v1/ingest/edges?edge_type=serves")
        assert response.status_code == 200
        data = response.json()
        assert "edges" in data

    def test_delete_node(self, client):
        """Should delete node and connected edges."""
        # Create node
        client.post("/api/v1/ingest/node", json={
            "node_id": "delete_test_node",
            "node_type": "HVAC",
        })

        # Delete node
        response = client.delete("/api/v1/ingest/node/delete_test_node")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_efficiency_score(self, client):
        """Should calculate efficiency score."""
        # Create some nodes first
        client.post("/api/v1/ingest/node", json={
            "node_id": "eff_test_hvac",
            "node_type": "HVAC",
            "capacity_kw": 50,
        })

        response = client.get("/api/v1/predict/efficiency")
        assert response.status_code == 200
        data = response.json()
        assert "efficiency_score" in data
        assert "recommendations" in data

    def test_forecast(self, client):
        """Should generate power forecast."""
        # Create a node first
        client.post("/api/v1/ingest/node", json={
            "node_id": "forecast_test_hvac",
            "node_type": "HVAC",
            "capacity_kw": 50,
        })

        response = client.get("/api/v1/predict/forecast?hours=24")
        assert response.status_code == 200
        data = response.json()
        assert "forecast" in data
        assert len(data["forecast"]) > 0


class TestValidation:
    """Tests for input validation."""

    def test_invalid_node_type(self, client):
        """Should reject invalid node type."""
        node_data = {
            "node_id": "invalid_type_node",
            "node_type": "InvalidType",
        }
        response = client.post("/api/v1/ingest/node", json=node_data)
        assert response.status_code == 422

    def test_invalid_edge_type(self, client):
        """Should reject invalid edge type."""
        # Create nodes
        client.post("/api/v1/ingest/node", json={"node_id": "val_hvac", "node_type": "HVAC"})
        client.post("/api/v1/ingest/node", json={"node_id": "val_room", "node_type": "Room"})

        edge_data = {
            "source": "val_hvac",
            "target": "val_room",
            "edge_type": "invalid_type",
        }
        response = client.post("/api/v1/ingest/edge", json=edge_data)
        assert response.status_code == 422

    def test_self_loop_rejected(self, client):
        """Should reject self-loop edges."""
        client.post("/api/v1/ingest/node", json={"node_id": "self_loop_node", "node_type": "HVAC"})

        edge_data = {
            "source": "self_loop_node",
            "target": "self_loop_node",
            "edge_type": "serves",
        }
        response = client.post("/api/v1/ingest/edge", json=edge_data)
        assert response.status_code == 422

    def test_invalid_node_id_format(self, client):
        """Should reject invalid node ID format."""
        node_data = {
            "node_id": "invalid node id with spaces",
            "node_type": "HVAC",
        }
        response = client.post("/api/v1/ingest/node", json=node_data)
        assert response.status_code == 422
