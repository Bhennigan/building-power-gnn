"""Graph builder module for constructing PyTorch Geometric HeteroData objects.

Converts parsed building data into heterogeneous graph structures suitable for GNN training.
"""

from typing import Optional
import logging

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class NodeFeatureEncoder:
    """Encodes node attributes into feature tensors."""

    # Default feature dimensions per node type
    FEATURE_DIMS = {
        "HVAC": 8,
        "Lighting": 6,
        "Sensor": 5,
        "Room": 7,
        "Meter": 4,
        "WeatherStation": 3,
    }

    def __init__(self):
        """Initialize encoder with mappings for categorical features."""
        self._zone_map: dict[str, int] = {}
        self._floor_map: dict[str, int] = {}
        self._subtype_map: dict[str, int] = {}

    def fit(self, nodes_df: pd.DataFrame) -> "NodeFeatureEncoder":
        """Learn categorical mappings from data.

        Args:
            nodes_df: DataFrame with node data.

        Returns:
            Self for method chaining.
        """
        if "zone" in nodes_df.columns:
            zones = nodes_df["zone"].dropna().unique()
            self._zone_map = {z: i for i, z in enumerate(zones)}

        if "floor" in nodes_df.columns:
            floors = nodes_df["floor"].dropna().unique()
            self._floor_map = {f: i for i, f in enumerate(floors)}

        if "subtype" in nodes_df.columns:
            subtypes = nodes_df["subtype"].dropna().unique()
            self._subtype_map = {s: i for i, s in enumerate(subtypes)}

        return self

    def encode_hvac(self, row: pd.Series) -> torch.Tensor:
        """Encode HVAC node features."""
        features = [
            float(row.get("capacity_kw", 0) or 0),
            float(row.get("attr_efficiency_rating", 0.9) or 0.9),
            float(row.get("attr_age_years", 0) or 0),
            self._zone_map.get(row.get("zone"), 0),
            self._floor_map.get(row.get("floor"), 0),
            1.0 if row.get("attr_variable_speed") == "true" else 0.0,
            float(row.get("attr_cop", 3.0) or 3.0),  # Coefficient of performance
            1.0,  # Bias term
        ]
        return torch.tensor(features, dtype=torch.float32)

    def encode_lighting(self, row: pd.Series) -> torch.Tensor:
        """Encode Lighting node features."""
        lighting_type_map = {"LED": 0, "fluorescent": 1, "incandescent": 2, "halogen": 3}
        features = [
            row.get("wattage", 0) or 0,
            lighting_type_map.get(row.get("attr_type", "LED"), 0),
            1.0 if row.get("attr_occupancy_sensor") == "true" else 0.0,
            1.0 if row.get("attr_dimmable") == "true" else 0.0,
            self._zone_map.get(row.get("zone"), 0),
            1.0,  # Bias term
        ]
        return torch.tensor(features, dtype=torch.float32)

    def encode_sensor(self, row: pd.Series) -> torch.Tensor:
        """Encode Sensor node features."""
        features = [
            self._subtype_map.get(row.get("subtype"), 0),
            float(row.get("attr_accuracy", 0.1) or 0.1),
            float(row.get("attr_sample_rate_hz", 1.0) or 1.0),
            self._zone_map.get(row.get("zone"), 0),
            1.0,  # Bias term
        ]
        return torch.tensor(features, dtype=torch.float32)

    def encode_room(self, row: pd.Series) -> torch.Tensor:
        """Encode Room node features."""
        features = [
            row.get("area_sqft", 0) or 0,
            row.get("occupancy_max", 0) or 0,
            self._zone_map.get(row.get("zone"), 0),
            self._floor_map.get(row.get("floor"), 0),
            float(row.get("attr_ceiling_height", 10.0) or 10.0),
            float(row.get("attr_window_ratio", 0.2) or 0.2),
            1.0,  # Bias term
        ]
        return torch.tensor(features, dtype=torch.float32)

    def encode_meter(self, row: pd.Series) -> torch.Tensor:
        """Encode Meter node features."""
        meter_type_map = {"electrical": 0, "gas": 1, "water": 2}
        features = [
            meter_type_map.get(row.get("subtype", "electrical"), 0),
            float(row.get("attr_resolution_minutes", 15) or 15),
            float(row.get("attr_max_load_kw", 1000) or 1000),
            1.0,  # Bias term
        ]
        return torch.tensor(features, dtype=torch.float32)

    def encode_weather_station(self, row: pd.Series) -> torch.Tensor:
        """Encode WeatherStation node features."""
        features = [
            float(row.get("attr_latitude", 0) or 0),
            float(row.get("attr_longitude", 0) or 0),
            1.0,  # Bias term
        ]
        return torch.tensor(features, dtype=torch.float32)

    def encode_node(self, node_type: str, row: pd.Series) -> torch.Tensor:
        """Encode a single node based on its type."""
        encoders = {
            "HVAC": self.encode_hvac,
            "Lighting": self.encode_lighting,
            "Sensor": self.encode_sensor,
            "Room": self.encode_room,
            "Meter": self.encode_meter,
            "WeatherStation": self.encode_weather_station,
        }
        encoder = encoders.get(node_type)
        if encoder is None:
            raise ValueError(f"Unknown node type: {node_type}")
        return encoder(row)


class GraphBuilder:
    """Builds PyTorch Geometric HeteroData from parsed building data."""

    def __init__(self):
        """Initialize graph builder."""
        self.encoder = NodeFeatureEncoder()
        self._node_id_to_idx: dict[str, dict[str, int]] = {}

    def build(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        timeseries_df: Optional[pd.DataFrame] = None,
        normalize_features: bool = True
    ) -> HeteroData:
        """Build HeteroData graph from DataFrames.

        Args:
            nodes_df: DataFrame with node data.
            edges_df: DataFrame with edge data.
            timeseries_df: Optional DataFrame with time series data.
            normalize_features: Whether to normalize numerical features.

        Returns:
            PyTorch Geometric HeteroData object.
        """
        # Fit encoder to data
        self.encoder.fit(nodes_df)

        data = HeteroData()

        # Build node features by type
        self._build_nodes(data, nodes_df, normalize_features)

        # Build edges
        self._build_edges(data, edges_df)

        # Add time series features if available
        if timeseries_df is not None and not timeseries_df.empty:
            self._add_temporal_features(data, nodes_df, timeseries_df)

        logger.info(f"Built graph: {data}")
        return data

    def _build_nodes(
        self,
        data: HeteroData,
        nodes_df: pd.DataFrame,
        normalize: bool
    ) -> None:
        """Build node features for each node type."""
        self._node_id_to_idx = {}

        for node_type in nodes_df["node_type"].unique():
            type_nodes = nodes_df[nodes_df["node_type"] == node_type]

            # Create index mapping for this type
            self._node_id_to_idx[node_type] = {
                node_id: idx for idx, node_id in enumerate(type_nodes["node_id"])
            }

            # Encode features
            features = []
            for _, row in type_nodes.iterrows():
                feat = self.encoder.encode_node(node_type, row)
                features.append(feat)

            if features:
                x = torch.stack(features)
                if normalize:
                    x = self._normalize_features(x)
                data[node_type.lower()].x = x
                data[node_type.lower()].node_ids = list(type_nodes["node_id"])

        logger.info(f"Built nodes for types: {list(self._node_id_to_idx.keys())}")

    def _build_edges(self, data: HeteroData, edges_df: pd.DataFrame) -> None:
        """Build edge indices for each edge type."""
        # Group edges by (source_type, edge_type, target_type)
        edge_groups: dict[tuple, list[tuple[int, int]]] = {}

        for _, edge in edges_df.iterrows():
            source_id = edge["source"]
            target_id = edge["target"]
            edge_type = edge["edge_type"]

            # Find source and target node types
            source_type = self._find_node_type(source_id)
            target_type = self._find_node_type(target_id)

            if source_type is None or target_type is None:
                logger.warning(f"Skipping edge {source_id} -> {target_id}: node not found")
                continue

            # Get indices
            source_idx = self._node_id_to_idx[source_type][source_id]
            target_idx = self._node_id_to_idx[target_type][target_id]

            # Create edge key
            edge_key = (source_type.lower(), edge_type, target_type.lower())
            if edge_key not in edge_groups:
                edge_groups[edge_key] = []
            edge_groups[edge_key].append((source_idx, target_idx))

            # Handle bidirectional edges
            if edge.get("bidirectional", False):
                reverse_key = (target_type.lower(), f"{edge_type}_rev", source_type.lower())
                if reverse_key not in edge_groups:
                    edge_groups[reverse_key] = []
                edge_groups[reverse_key].append((target_idx, source_idx))

        # Add edges to HeteroData
        for edge_key, edges in edge_groups.items():
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_key].edge_index = edge_index
                logger.info(f"Added {len(edges)} edges of type {edge_key}")

    def _find_node_type(self, node_id: str) -> Optional[str]:
        """Find the type of a node by its ID."""
        for node_type, id_map in self._node_id_to_idx.items():
            if node_id in id_map:
                return node_type
        return None

    def _add_temporal_features(
        self,
        data: HeteroData,
        nodes_df: pd.DataFrame,
        timeseries_df: pd.DataFrame
    ) -> None:
        """Add aggregated temporal features to nodes."""
        # Group time series by node
        ts_stats = timeseries_df.groupby("node_id").agg({
            "value": ["mean", "std", "min", "max", "count"]
        }).reset_index()
        ts_stats.columns = ["node_id", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_count"]

        # Merge with nodes
        nodes_with_ts = nodes_df.merge(ts_stats, on="node_id", how="left")

        # Add temporal features to each node type
        for node_type in nodes_with_ts["node_type"].unique():
            type_key = node_type.lower()
            if type_key not in data.node_types:
                continue

            type_nodes = nodes_with_ts[nodes_with_ts["node_type"] == node_type]
            temporal_features = torch.tensor(
                type_nodes[["ts_mean", "ts_std", "ts_min", "ts_max", "ts_count"]].fillna(0).values,
                dtype=torch.float32
            )
            data[type_key].temporal_features = temporal_features

    @staticmethod
    def _normalize_features(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize features to zero mean and unit variance."""
        if x.size(0) <= 1:
            # Can't normalize with single sample, just center
            return x - x.mean(dim=0, keepdim=True)
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False) + eps
        return (x - mean) / std

    def get_node_index(self, node_type: str, node_id: str) -> Optional[int]:
        """Get the index of a node in the graph.

        Args:
            node_type: Type of the node.
            node_id: ID of the node.

        Returns:
            Index in the node tensor, or None if not found.
        """
        type_map = self._node_id_to_idx.get(node_type)
        if type_map is None:
            return None
        return type_map.get(node_id)


class IncrementalGraphBuilder:
    """Builds graphs incrementally for real-time updates."""

    def __init__(self):
        """Initialize incremental builder."""
        self._nodes: dict[str, dict] = {}  # node_id -> node_data
        self._edges: list[dict] = []
        self._builder = GraphBuilder()

    def add_node(self, node_id: str, node_type: str, **attributes) -> None:
        """Add or update a node."""
        self._nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            **attributes
        }

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its edges."""
        if node_id not in self._nodes:
            return False

        del self._nodes[node_id]
        self._edges = [e for e in self._edges if e["source"] != node_id and e["target"] != node_id]
        return True

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float = 1.0,
        bidirectional: bool = False
    ) -> None:
        """Add an edge."""
        self._edges.append({
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "weight": weight,
            "bidirectional": bidirectional
        })

    def build(self, timeseries_df: Optional[pd.DataFrame] = None) -> HeteroData:
        """Build graph from current state."""
        nodes_df = pd.DataFrame(list(self._nodes.values()))
        edges_df = pd.DataFrame(self._edges)
        return self._builder.build(nodes_df, edges_df, timeseries_df)

    def get_node_count(self) -> int:
        """Get current node count."""
        return len(self._nodes)

    def get_edge_count(self) -> int:
        """Get current edge count."""
        return len(self._edges)

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
