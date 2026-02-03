"""Graph transformation and augmentation utilities.

Provides data augmentation and preprocessing for GNN training.
"""

from typing import Optional
import random

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
import numpy as np


class AddSelfLoops(BaseTransform):
    """Add self-loops to all node types."""

    def __init__(self, edge_types: Optional[list[str]] = None):
        """Initialize transform.

        Args:
            edge_types: Specific edge type names for self-loops.
                       If None, creates 'self_loop' edges.
        """
        self.edge_types = edge_types

    def forward(self, data: HeteroData) -> HeteroData:
        """Add self-loops to the graph."""
        for node_type in data.node_types:
            num_nodes = data[node_type].x.size(0)
            self_loop_indices = torch.arange(num_nodes, dtype=torch.long)
            edge_index = torch.stack([self_loop_indices, self_loop_indices], dim=0)

            edge_type_name = (node_type, "self_loop", node_type)
            data[edge_type_name].edge_index = edge_index

        return data


class FeatureNoise(BaseTransform):
    """Add Gaussian noise to node features for augmentation."""

    def __init__(
        self,
        noise_std: float = 0.01,
        node_types: Optional[list[str]] = None
    ):
        """Initialize transform.

        Args:
            noise_std: Standard deviation of Gaussian noise.
            node_types: Node types to apply noise to (all if None).
        """
        self.noise_std = noise_std
        self.node_types = node_types

    def forward(self, data: HeteroData) -> HeteroData:
        """Add noise to features."""
        types = self.node_types or data.node_types

        for node_type in types:
            if node_type in data.node_types and hasattr(data[node_type], "x"):
                x = data[node_type].x
                noise = torch.randn_like(x) * self.noise_std
                data[node_type].x = x + noise

        return data


class EdgeDropout(BaseTransform):
    """Randomly drop edges for augmentation."""

    def __init__(
        self,
        dropout_rate: float = 0.1,
        edge_types: Optional[list[tuple[str, str, str]]] = None
    ):
        """Initialize transform.

        Args:
            dropout_rate: Probability of dropping each edge.
            edge_types: Edge types to apply dropout to (all if None).
        """
        self.dropout_rate = dropout_rate
        self.edge_types = edge_types

    def forward(self, data: HeteroData) -> HeteroData:
        """Apply edge dropout."""
        types = self.edge_types or data.edge_types

        for edge_type in types:
            if edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
                num_edges = edge_index.size(1)

                # Create mask
                mask = torch.rand(num_edges) > self.dropout_rate
                data[edge_type].edge_index = edge_index[:, mask]

        return data


class TemporalMask(BaseTransform):
    """Mask temporal features for self-supervised learning."""

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_value: float = 0.0
    ):
        """Initialize transform.

        Args:
            mask_ratio: Ratio of temporal features to mask.
            mask_value: Value to use for masked positions.
        """
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    def forward(self, data: HeteroData) -> HeteroData:
        """Apply temporal masking."""
        for node_type in data.node_types:
            if hasattr(data[node_type], "temporal_features"):
                features = data[node_type].temporal_features
                mask = torch.rand(features.shape) < self.mask_ratio

                # Store original for loss computation
                data[node_type].temporal_features_original = features.clone()
                data[node_type].temporal_mask = mask

                # Apply mask
                masked_features = features.clone()
                masked_features[mask] = self.mask_value
                data[node_type].temporal_features = masked_features

        return data


class NormalizeFeatures(BaseTransform):
    """Normalize node features to zero mean and unit variance."""

    def __init__(
        self,
        node_types: Optional[list[str]] = None,
        eps: float = 1e-8
    ):
        """Initialize transform.

        Args:
            node_types: Node types to normalize (all if None).
            eps: Small value for numerical stability.
        """
        self.node_types = node_types
        self.eps = eps
        self._stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def fit(self, data: HeteroData) -> "NormalizeFeatures":
        """Compute normalization statistics from data."""
        types = self.node_types or data.node_types

        for node_type in types:
            if node_type in data.node_types and hasattr(data[node_type], "x"):
                x = data[node_type].x
                mean = x.mean(dim=0)
                std = x.std(dim=0) + self.eps
                self._stats[node_type] = (mean, std)

        return self

    def forward(self, data: HeteroData) -> HeteroData:
        """Apply normalization."""
        for node_type, (mean, std) in self._stats.items():
            if node_type in data.node_types and hasattr(data[node_type], "x"):
                data[node_type].x = (data[node_type].x - mean) / std

        return data


class AddNodeDegree(BaseTransform):
    """Add node degree as a feature."""

    def __init__(self, in_degree: bool = True, out_degree: bool = True):
        """Initialize transform.

        Args:
            in_degree: Whether to add in-degree.
            out_degree: Whether to add out-degree.
        """
        self.in_degree = in_degree
        self.out_degree = out_degree

    def forward(self, data: HeteroData) -> HeteroData:
        """Add degree features."""
        # Compute degrees for each node type
        degrees = {nt: {"in": 0, "out": 0} for nt in data.node_types}

        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index

            if self.out_degree:
                src_degree = torch.bincount(
                    edge_index[0],
                    minlength=data[src_type].x.size(0)
                ).float()
                if isinstance(degrees[src_type]["out"], int):
                    degrees[src_type]["out"] = src_degree
                else:
                    degrees[src_type]["out"] = degrees[src_type]["out"] + src_degree

            if self.in_degree:
                dst_degree = torch.bincount(
                    edge_index[1],
                    minlength=data[dst_type].x.size(0)
                ).float()
                if isinstance(degrees[dst_type]["in"], int):
                    degrees[dst_type]["in"] = dst_degree
                else:
                    degrees[dst_type]["in"] = degrees[dst_type]["in"] + dst_degree

        # Add degree features to nodes
        for node_type in data.node_types:
            degree_features = []
            if self.in_degree:
                in_deg = degrees[node_type]["in"]
                if isinstance(in_deg, int):
                    in_deg = torch.zeros(data[node_type].x.size(0))
                degree_features.append(in_deg.unsqueeze(1))

            if self.out_degree:
                out_deg = degrees[node_type]["out"]
                if isinstance(out_deg, int):
                    out_deg = torch.zeros(data[node_type].x.size(0))
                degree_features.append(out_deg.unsqueeze(1))

            if degree_features:
                degree_tensor = torch.cat(degree_features, dim=1)
                data[node_type].degree_features = degree_tensor

        return data


class ComposeTransforms:
    """Compose multiple transforms into a single transform."""

    def __init__(self, transforms: list[BaseTransform]):
        """Initialize with list of transforms."""
        self.transforms = transforms

    def __call__(self, data: HeteroData) -> HeteroData:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            data = transform(data)
        return data


def create_train_transform(
    noise_std: float = 0.01,
    edge_dropout: float = 0.1
) -> ComposeTransforms:
    """Create a standard training transform pipeline."""
    return ComposeTransforms([
        AddSelfLoops(),
        FeatureNoise(noise_std=noise_std),
        EdgeDropout(dropout_rate=edge_dropout),
        AddNodeDegree(),
    ])


def create_eval_transform() -> ComposeTransforms:
    """Create a standard evaluation transform pipeline."""
    return ComposeTransforms([
        AddSelfLoops(),
        AddNodeDegree(),
    ])
