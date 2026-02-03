"""Feature store for managing time-series data.

Provides efficient storage and retrieval of temporal features for GNN training.
"""

from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

import torch
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TemporalWindow:
    """Represents a sliding window of time-series data for a node."""

    def __init__(self, window_size: int = 168, feature_dim: int = 1):
        """Initialize temporal window.

        Args:
            window_size: Number of time steps to keep (default 168 = 1 week hourly).
            feature_dim: Dimension of features at each time step.
        """
        self.window_size = window_size
        self.feature_dim = feature_dim
        self._buffer = np.zeros((window_size, feature_dim), dtype=np.float32)
        self._timestamps: list[datetime] = []
        self._write_idx = 0
        self._is_full = False

    def add(self, timestamp: datetime, values: np.ndarray) -> None:
        """Add a new reading to the window.

        Args:
            timestamp: Timestamp of the reading.
            values: Feature values (shape: feature_dim,).
        """
        self._buffer[self._write_idx] = values
        self._timestamps.append(timestamp)
        if len(self._timestamps) > self.window_size:
            self._timestamps.pop(0)

        self._write_idx = (self._write_idx + 1) % self.window_size
        if self._write_idx == 0:
            self._is_full = True

    def get_tensor(self) -> torch.Tensor:
        """Get the window data as a tensor.

        Returns:
            Tensor of shape (window_size, feature_dim) with oldest first.
        """
        if self._is_full:
            # Reorder so oldest is first
            ordered = np.concatenate([
                self._buffer[self._write_idx:],
                self._buffer[:self._write_idx]
            ], axis=0)
        else:
            # Not full yet, pad with zeros
            ordered = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
            valid_len = min(self._write_idx, self.window_size)
            ordered[-valid_len:] = self._buffer[:valid_len]

        return torch.tensor(ordered, dtype=torch.float32)

    def get_latest(self, n: int = 1) -> np.ndarray:
        """Get the n most recent readings."""
        if self._write_idx == 0 and not self._is_full:
            return np.zeros((n, self.feature_dim), dtype=np.float32)

        n = min(n, self._write_idx if not self._is_full else self.window_size)
        if self._write_idx >= n:
            return self._buffer[self._write_idx - n:self._write_idx]
        else:
            # Wrap around
            return np.concatenate([
                self._buffer[-(n - self._write_idx):],
                self._buffer[:self._write_idx]
            ], axis=0)

    @property
    def length(self) -> int:
        """Number of readings in the window."""
        return self.window_size if self._is_full else self._write_idx


class FeatureStore:
    """Manages temporal features for all nodes in a building graph."""

    def __init__(
        self,
        window_size: int = 168,
        default_feature_dim: int = 1,
        aggregation_interval: timedelta = timedelta(hours=1)
    ):
        """Initialize feature store.

        Args:
            window_size: Number of time steps per node window.
            default_feature_dim: Default feature dimension.
            aggregation_interval: Time interval for aggregating readings.
        """
        self.window_size = window_size
        self.default_feature_dim = default_feature_dim
        self.aggregation_interval = aggregation_interval

        self._windows: dict[str, dict[str, TemporalWindow]] = defaultdict(dict)
        self._last_update: dict[str, datetime] = {}
        self._pending_readings: dict[str, list] = defaultdict(list)

    def register_node(
        self,
        node_id: str,
        metrics: list[str],
        feature_dim: Optional[int] = None
    ) -> None:
        """Register a node with its expected metrics.

        Args:
            node_id: ID of the node.
            metrics: List of metric names to track.
            feature_dim: Feature dimension (defaults to number of metrics).
        """
        feature_dim = feature_dim or len(metrics)
        for metric in metrics:
            self._windows[node_id][metric] = TemporalWindow(
                window_size=self.window_size,
                feature_dim=1  # One dimension per metric
            )
        logger.debug(f"Registered node {node_id} with metrics: {metrics}")

    def add_reading(
        self,
        node_id: str,
        metric: str,
        timestamp: datetime,
        value: float
    ) -> None:
        """Add a single reading.

        Args:
            node_id: ID of the node.
            metric: Name of the metric.
            timestamp: Timestamp of the reading.
            value: Value of the reading.
        """
        # Auto-register if needed
        if node_id not in self._windows or metric not in self._windows[node_id]:
            self._windows[node_id][metric] = TemporalWindow(
                window_size=self.window_size,
                feature_dim=1
            )

        self._windows[node_id][metric].add(timestamp, np.array([value]))
        self._last_update[node_id] = timestamp

    def add_readings_batch(self, readings_df: pd.DataFrame) -> int:
        """Add multiple readings from a DataFrame.

        Args:
            readings_df: DataFrame with columns [node_id, timestamp, value, metric].

        Returns:
            Number of readings added.
        """
        count = 0
        for _, row in readings_df.iterrows():
            self.add_reading(
                node_id=row["node_id"],
                metric=row.get("metric", "default"),
                timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime) else datetime.fromisoformat(str(row["timestamp"])),
                value=float(row["value"])
            )
            count += 1

        logger.info(f"Added {count} readings to feature store")
        return count

    def get_node_features(
        self,
        node_id: str,
        metrics: Optional[list[str]] = None
    ) -> torch.Tensor:
        """Get temporal features for a node.

        Args:
            node_id: ID of the node.
            metrics: Specific metrics to include (all if None).

        Returns:
            Tensor of shape (window_size, num_metrics).
        """
        if node_id not in self._windows:
            return torch.zeros(self.window_size, self.default_feature_dim)

        node_windows = self._windows[node_id]
        if metrics is None:
            metrics = list(node_windows.keys())

        tensors = []
        for metric in metrics:
            if metric in node_windows:
                tensors.append(node_windows[metric].get_tensor())
            else:
                tensors.append(torch.zeros(self.window_size, 1))

        if not tensors:
            return torch.zeros(self.window_size, self.default_feature_dim)

        return torch.cat(tensors, dim=1)

    def get_batch_features(
        self,
        node_ids: list[str],
        metrics: Optional[list[str]] = None
    ) -> torch.Tensor:
        """Get temporal features for multiple nodes.

        Args:
            node_ids: List of node IDs.
            metrics: Specific metrics to include.

        Returns:
            Tensor of shape (num_nodes, window_size, num_metrics).
        """
        features = [self.get_node_features(nid, metrics) for nid in node_ids]
        return torch.stack(features, dim=0)

    def get_latest_values(
        self,
        node_id: str,
        n: int = 1
    ) -> dict[str, np.ndarray]:
        """Get the n most recent values for a node.

        Args:
            node_id: ID of the node.
            n: Number of recent values to retrieve.

        Returns:
            Dictionary mapping metric name to values array.
        """
        if node_id not in self._windows:
            return {}

        return {
            metric: window.get_latest(n)
            for metric, window in self._windows[node_id].items()
        }

    def compute_statistics(self, node_id: str) -> dict[str, dict]:
        """Compute statistics for a node's time series.

        Args:
            node_id: ID of the node.

        Returns:
            Dictionary with statistics per metric.
        """
        if node_id not in self._windows:
            return {}

        stats = {}
        for metric, window in self._windows[node_id].items():
            tensor = window.get_tensor()
            stats[metric] = {
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "length": window.length,
            }
        return stats

    def get_all_node_ids(self) -> list[str]:
        """Get all registered node IDs."""
        return list(self._windows.keys())

    def get_node_metrics(self, node_id: str) -> list[str]:
        """Get all metrics for a node."""
        return list(self._windows.get(node_id, {}).keys())

    def clear_node(self, node_id: str) -> None:
        """Clear all data for a node."""
        if node_id in self._windows:
            del self._windows[node_id]
        if node_id in self._last_update:
            del self._last_update[node_id]

    def clear_all(self) -> None:
        """Clear all stored data."""
        self._windows.clear()
        self._last_update.clear()
        self._pending_readings.clear()

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all data to a DataFrame for persistence."""
        records = []
        for node_id, metrics in self._windows.items():
            for metric, window in metrics.items():
                tensor = window.get_tensor()
                for i, timestamp in enumerate(window._timestamps):
                    records.append({
                        "node_id": node_id,
                        "metric": metric,
                        "timestamp": timestamp,
                        "value": tensor[i].item() if tensor.ndim > 1 else tensor[i].item()
                    })
        return pd.DataFrame(records)
