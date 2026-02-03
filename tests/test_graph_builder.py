"""Tests for graph construction and feature encoding."""

import pytest
import torch
import pandas as pd
import numpy as np

from src.graph import GraphBuilder, IncrementalGraphBuilder, NodeFeatureEncoder


@pytest.fixture
def sample_nodes_df():
    """Create sample nodes DataFrame."""
    return pd.DataFrame([
        {"node_id": "hvac_1", "node_type": "HVAC", "zone": "floor_1", "capacity_kw": 50, "attr_efficiency_rating": "0.92"},
        {"node_id": "hvac_2", "node_type": "HVAC", "zone": "floor_2", "capacity_kw": 75, "attr_efficiency_rating": "0.88"},
        {"node_id": "sensor_1", "node_type": "Sensor", "subtype": "temperature", "zone": "floor_1"},
        {"node_id": "room_101", "node_type": "Room", "zone": "floor_1", "area_sqft": 500, "occupancy_max": 20},
        {"node_id": "room_102", "node_type": "Room", "zone": "floor_1", "area_sqft": 600, "occupancy_max": 25},
        {"node_id": "meter_1", "node_type": "Meter", "subtype": "electrical"},
    ])


@pytest.fixture
def sample_edges_df():
    """Create sample edges DataFrame."""
    return pd.DataFrame([
        {"source": "hvac_1", "target": "room_101", "edge_type": "serves", "weight": 1.0, "bidirectional": False},
        {"source": "hvac_2", "target": "room_102", "edge_type": "serves", "weight": 1.0, "bidirectional": False},
        {"source": "sensor_1", "target": "room_101", "edge_type": "monitors", "weight": 1.0, "bidirectional": False},
        {"source": "meter_1", "target": "hvac_1", "edge_type": "feeds", "weight": 1.0, "bidirectional": False},
        {"source": "room_101", "target": "room_102", "edge_type": "adjacent", "weight": 1.0, "bidirectional": True},
    ])


@pytest.fixture
def sample_timeseries_df():
    """Create sample time series DataFrame."""
    return pd.DataFrame([
        {"node_id": "sensor_1", "timestamp": pd.Timestamp("2024-01-01 08:00:00"), "value": 72.5, "metric": "temperature"},
        {"node_id": "sensor_1", "timestamp": pd.Timestamp("2024-01-01 09:00:00"), "value": 73.0, "metric": "temperature"},
        {"node_id": "hvac_1", "timestamp": pd.Timestamp("2024-01-01 08:00:00"), "value": 25.5, "metric": "power"},
    ])


class TestNodeFeatureEncoder:
    """Tests for NodeFeatureEncoder."""

    def test_fit_learns_mappings(self, sample_nodes_df):
        """Encoder should learn categorical mappings from data."""
        encoder = NodeFeatureEncoder()
        encoder.fit(sample_nodes_df)

        assert len(encoder._zone_map) > 0
        assert "floor_1" in encoder._zone_map

    def test_encode_hvac(self, sample_nodes_df):
        """Encoder should produce correct HVAC features."""
        encoder = NodeFeatureEncoder()
        encoder.fit(sample_nodes_df)

        hvac_row = sample_nodes_df[sample_nodes_df["node_id"] == "hvac_1"].iloc[0]
        features = encoder.encode_hvac(hvac_row)

        assert features.shape == (8,)
        assert features[0] == 50.0  # capacity_kw

    def test_encode_room(self, sample_nodes_df):
        """Encoder should produce correct Room features."""
        encoder = NodeFeatureEncoder()
        encoder.fit(sample_nodes_df)

        room_row = sample_nodes_df[sample_nodes_df["node_id"] == "room_101"].iloc[0]
        features = encoder.encode_room(room_row)

        assert features.shape == (7,)
        assert features[0] == 500.0  # area_sqft

    def test_encode_sensor(self, sample_nodes_df):
        """Encoder should produce correct Sensor features."""
        encoder = NodeFeatureEncoder()
        encoder.fit(sample_nodes_df)

        sensor_row = sample_nodes_df[sample_nodes_df["node_id"] == "sensor_1"].iloc[0]
        features = encoder.encode_sensor(sensor_row)

        assert features.shape == (5,)


class TestGraphBuilder:
    """Tests for GraphBuilder."""

    def test_build_creates_heterodata(self, sample_nodes_df, sample_edges_df):
        """Builder should create HeteroData object."""
        builder = GraphBuilder()
        graph = builder.build(sample_nodes_df, sample_edges_df)

        assert graph is not None
        assert len(graph.node_types) > 0

    def test_build_includes_all_node_types(self, sample_nodes_df, sample_edges_df):
        """Built graph should include all node types."""
        builder = GraphBuilder()
        graph = builder.build(sample_nodes_df, sample_edges_df)

        assert "hvac" in graph.node_types
        assert "room" in graph.node_types
        assert "sensor" in graph.node_types
        assert "meter" in graph.node_types

    def test_build_creates_node_features(self, sample_nodes_df, sample_edges_df):
        """Built graph should have node features."""
        builder = GraphBuilder()
        graph = builder.build(sample_nodes_df, sample_edges_df)

        assert hasattr(graph["hvac"], "x")
        assert graph["hvac"].x.shape[0] == 2  # 2 HVAC nodes

    def test_build_creates_edges(self, sample_nodes_df, sample_edges_df):
        """Built graph should have edge indices."""
        builder = GraphBuilder()
        graph = builder.build(sample_nodes_df, sample_edges_df)

        assert len(graph.edge_types) > 0

    def test_build_with_timeseries(self, sample_nodes_df, sample_edges_df, sample_timeseries_df):
        """Built graph should include temporal features."""
        builder = GraphBuilder()
        graph = builder.build(sample_nodes_df, sample_edges_df, sample_timeseries_df)

        # Check that temporal features are added
        assert graph is not None

    def test_get_node_index(self, sample_nodes_df, sample_edges_df):
        """Builder should track node indices correctly."""
        builder = GraphBuilder()
        builder.build(sample_nodes_df, sample_edges_df)

        idx = builder.get_node_index("HVAC", "hvac_1")
        assert idx is not None
        assert idx == 0 or idx == 1  # Should be one of the two HVAC nodes

    def test_normalize_features(self, sample_nodes_df, sample_edges_df):
        """Features should be normalized when requested."""
        builder = GraphBuilder()
        graph = builder.build(sample_nodes_df, sample_edges_df, normalize_features=True)

        # Check that features have approximately zero mean
        for node_type in graph.node_types:
            if hasattr(graph[node_type], "x"):
                x = graph[node_type].x
                assert x.mean().abs() < 1.0  # Approximately normalized


class TestIncrementalGraphBuilder:
    """Tests for IncrementalGraphBuilder."""

    def test_add_node(self):
        """Should add nodes incrementally."""
        builder = IncrementalGraphBuilder()
        builder.add_node("hvac_1", "HVAC", capacity_kw=50)

        assert builder.get_node_count() == 1

    def test_add_edge(self):
        """Should add edges incrementally."""
        builder = IncrementalGraphBuilder()
        builder.add_node("hvac_1", "HVAC", capacity_kw=50)
        builder.add_node("room_1", "Room", area_sqft=500)
        builder.add_edge("hvac_1", "room_1", "serves")

        assert builder.get_edge_count() == 1

    def test_remove_node(self):
        """Should remove nodes and their edges."""
        builder = IncrementalGraphBuilder()
        builder.add_node("hvac_1", "HVAC", capacity_kw=50)
        builder.add_node("room_1", "Room", area_sqft=500)
        builder.add_edge("hvac_1", "room_1", "serves")

        result = builder.remove_node("hvac_1")

        assert result is True
        assert builder.get_node_count() == 1
        assert builder.get_edge_count() == 0

    def test_build(self):
        """Should build HeteroData from current state."""
        builder = IncrementalGraphBuilder()
        builder.add_node("hvac_1", "HVAC", capacity_kw=50, zone="floor_1")
        builder.add_node("room_1", "Room", area_sqft=500, zone="floor_1")
        builder.add_edge("hvac_1", "room_1", "serves")

        graph = builder.build()

        assert graph is not None
        assert "hvac" in graph.node_types
        assert "room" in graph.node_types

    def test_clear(self):
        """Should clear all data."""
        builder = IncrementalGraphBuilder()
        builder.add_node("hvac_1", "HVAC")
        builder.add_node("room_1", "Room")
        builder.add_edge("hvac_1", "room_1", "serves")

        builder.clear()

        assert builder.get_node_count() == 0
        assert builder.get_edge_count() == 0
