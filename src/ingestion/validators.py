"""Validation utilities for building graph data.

Provides schema validation, data integrity checks, and error reporting.
"""

from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
import re

from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd


class NodeType(str, Enum):
    """Valid node types for building graphs."""
    HVAC = "HVAC"
    LIGHTING = "Lighting"
    SENSOR = "Sensor"
    ROOM = "Room"
    METER = "Meter"
    WEATHER_STATION = "WeatherStation"


class SensorSubtype(str, Enum):
    """Valid sensor subtypes."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    OCCUPANCY = "occupancy"
    CO2 = "co2"
    LIGHT = "light"
    POWER = "power"


class EdgeType(str, Enum):
    """Valid edge types for building graphs."""
    SERVES = "serves"
    MONITORS = "monitors"
    FEEDS = "feeds"
    ADJACENT = "adjacent"
    CONTROLS = "controls"


# Pydantic models for API validation

class NodeCreate(BaseModel):
    """Schema for creating a new node via API."""
    node_id: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    node_type: NodeType
    subtype: Optional[str] = None
    zone: Optional[str] = None
    floor: Optional[str] = None
    area_sqft: Optional[float] = Field(None, gt=0)
    capacity_kw: Optional[float] = Field(None, gt=0)
    wattage: Optional[float] = Field(None, gt=0)
    occupancy_max: Optional[int] = Field(None, gt=0)
    attributes: dict[str, str] = Field(default_factory=dict)

    @field_validator("subtype")
    @classmethod
    def validate_sensor_subtype(cls, v, info):
        """Validate sensor subtype if node is a sensor."""
        if info.data.get("node_type") == NodeType.SENSOR and v:
            try:
                SensorSubtype(v)
            except ValueError:
                valid = [s.value for s in SensorSubtype]
                raise ValueError(f"Invalid sensor subtype. Must be one of: {valid}")
        return v


class EdgeCreate(BaseModel):
    """Schema for creating a new edge via API."""
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    edge_type: EdgeType
    weight: float = Field(1.0, gt=0)
    bidirectional: bool = False

    @model_validator(mode="after")
    def validate_no_self_loop(self):
        """Ensure edge doesn't connect node to itself."""
        if self.source == self.target:
            raise ValueError("Self-loops are not allowed")
        return self


class TimeSeriesReading(BaseModel):
    """Schema for a single time series reading."""
    node_id: str = Field(..., min_length=1)
    timestamp: str  # ISO format datetime
    value: float
    metric: str = "default"
    unit: Optional[str] = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        """Validate ISO format timestamp."""
        import re
        iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        if not re.match(iso_pattern, v):
            raise ValueError("Timestamp must be in ISO format (YYYY-MM-DDTHH:MM:SS)")
        return v


class TimeSeriesBatch(BaseModel):
    """Schema for batch time series upload."""
    readings: list[TimeSeriesReading] = Field(..., min_length=1)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
        return self


class GraphValidator:
    """Validates building graph data integrity."""

    # Edge type constraints: source_type -> target_type
    EDGE_CONSTRAINTS = {
        EdgeType.SERVES: [(NodeType.HVAC, NodeType.ROOM), (NodeType.LIGHTING, NodeType.ROOM)],
        EdgeType.MONITORS: [(NodeType.SENSOR, NodeType.ROOM), (NodeType.SENSOR, NodeType.HVAC)],
        EdgeType.FEEDS: [(NodeType.METER, NodeType.HVAC), (NodeType.METER, NodeType.LIGHTING)],
        EdgeType.ADJACENT: [(NodeType.ROOM, NodeType.ROOM)],
        EdgeType.CONTROLS: [(NodeType.SENSOR, NodeType.HVAC), (NodeType.SENSOR, NodeType.LIGHTING)],
    }

    def validate_nodes(self, nodes_df: pd.DataFrame) -> ValidationResult:
        """Validate nodes DataFrame."""
        result = ValidationResult()

        if nodes_df.empty:
            result.add_error("Nodes DataFrame is empty")
            return result

        # Check required columns
        required = ["node_id", "node_type"]
        for col in required:
            if col not in nodes_df.columns:
                result.add_error(f"Missing required column: {col}")

        if not result.is_valid:
            return result

        # Check for duplicate IDs
        duplicates = nodes_df[nodes_df["node_id"].duplicated()]["node_id"].tolist()
        if duplicates:
            result.add_error(f"Duplicate node IDs: {duplicates}")

        # Validate node types
        valid_types = [t.value for t in NodeType]
        invalid_types = nodes_df[~nodes_df["node_type"].isin(valid_types)]["node_type"].unique()
        if len(invalid_types) > 0:
            result.add_error(f"Invalid node types: {list(invalid_types)}")

        # Type-specific validation
        hvac_nodes = nodes_df[nodes_df["node_type"] == NodeType.HVAC.value]
        if not hvac_nodes.empty and "capacity_kw" in hvac_nodes.columns:
            missing_capacity = hvac_nodes[hvac_nodes["capacity_kw"].isna()]["node_id"].tolist()
            if missing_capacity:
                result.add_warning(f"HVAC nodes missing capacity_kw: {missing_capacity}")

        room_nodes = nodes_df[nodes_df["node_type"] == NodeType.ROOM.value]
        if not room_nodes.empty and "area_sqft" in room_nodes.columns:
            missing_area = room_nodes[room_nodes["area_sqft"].isna()]["node_id"].tolist()
            if missing_area:
                result.add_warning(f"Room nodes missing area_sqft: {missing_area}")

        return result

    def validate_edges(
        self,
        edges_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        check_constraints: bool = True
    ) -> ValidationResult:
        """Validate edges DataFrame."""
        result = ValidationResult()

        if edges_df.empty:
            result.add_warning("Edges DataFrame is empty")
            return result

        # Check required columns
        required = ["source", "target", "edge_type"]
        for col in required:
            if col not in edges_df.columns:
                result.add_error(f"Missing required column: {col}")

        if not result.is_valid:
            return result

        # Check node references exist
        node_ids = set(nodes_df["node_id"].tolist()) if not nodes_df.empty else set()

        missing_sources = set(edges_df["source"]) - node_ids
        if missing_sources:
            result.add_error(f"Edge sources reference non-existent nodes: {missing_sources}")

        missing_targets = set(edges_df["target"]) - node_ids
        if missing_targets:
            result.add_error(f"Edge targets reference non-existent nodes: {missing_targets}")

        # Validate edge types
        valid_types = [t.value for t in EdgeType]
        invalid_types = edges_df[~edges_df["edge_type"].isin(valid_types)]["edge_type"].unique()
        if len(invalid_types) > 0:
            result.add_error(f"Invalid edge types: {list(invalid_types)}")

        # Check self-loops
        self_loops = edges_df[edges_df["source"] == edges_df["target"]]
        if not self_loops.empty:
            result.add_error(f"Self-loops detected: {self_loops[['source', 'target']].values.tolist()}")

        # Check edge type constraints
        if check_constraints and not nodes_df.empty:
            node_type_map = dict(zip(nodes_df["node_id"], nodes_df["node_type"]))
            for _, edge in edges_df.iterrows():
                edge_type = EdgeType(edge["edge_type"])
                source_type = node_type_map.get(edge["source"])
                target_type = node_type_map.get(edge["target"])

                if source_type and target_type:
                    allowed = self.EDGE_CONSTRAINTS.get(edge_type, [])
                    try:
                        source_enum = NodeType(source_type)
                        target_enum = NodeType(target_type)
                        if allowed and (source_enum, target_enum) not in allowed:
                            result.add_warning(
                                f"Edge type '{edge_type.value}' typically doesn't connect "
                                f"'{source_type}' to '{target_type}'"
                            )
                    except ValueError:
                        pass  # Already reported as invalid type

        return result

    def validate_timeseries(
        self,
        timeseries_df: pd.DataFrame,
        nodes_df: pd.DataFrame
    ) -> ValidationResult:
        """Validate time series DataFrame."""
        result = ValidationResult()

        if timeseries_df.empty:
            return result  # Empty time series is okay

        # Check required columns
        required = ["node_id", "timestamp", "value"]
        for col in required:
            if col not in timeseries_df.columns:
                result.add_error(f"Missing required column: {col}")

        if not result.is_valid:
            return result

        # Check node references
        node_ids = set(nodes_df["node_id"].tolist()) if not nodes_df.empty else set()
        ts_nodes = set(timeseries_df["node_id"].unique())
        missing_nodes = ts_nodes - node_ids
        if missing_nodes:
            result.add_error(f"Time series reference non-existent nodes: {missing_nodes}")

        # Check for NaN values
        nan_values = timeseries_df[timeseries_df["value"].isna()]
        if not nan_values.empty:
            result.add_warning(f"Found {len(nan_values)} readings with NaN values")

        # Check timestamp ordering per node
        for node_id in timeseries_df["node_id"].unique():
            node_ts = timeseries_df[timeseries_df["node_id"] == node_id]
            if not node_ts["timestamp"].is_monotonic_increasing:
                result.add_warning(f"Timestamps for node '{node_id}' are not monotonically increasing")

        return result

    def validate_full_graph(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        timeseries_df: Optional[pd.DataFrame] = None
    ) -> ValidationResult:
        """Run all validations on a complete graph."""
        result = ValidationResult()

        result.merge(self.validate_nodes(nodes_df))
        result.merge(self.validate_edges(edges_df, nodes_df))

        if timeseries_df is not None:
            result.merge(self.validate_timeseries(timeseries_df, nodes_df))

        # Graph connectivity check
        if not nodes_df.empty and not edges_df.empty:
            connected_nodes = set(edges_df["source"]) | set(edges_df["target"])
            all_nodes = set(nodes_df["node_id"])
            isolated = all_nodes - connected_nodes
            if isolated:
                result.add_warning(f"Isolated nodes (no edges): {isolated}")

        return result
