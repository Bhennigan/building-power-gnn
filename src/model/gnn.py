"""GNN architecture for building power efficiency prediction.

Implements heterogeneous graph neural networks with temporal encoding.
"""

from typing import Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    HeteroConv,
    SAGEConv,
    GATConv,
    Linear,
    BatchNorm,
    LayerNorm,
)

logger = logging.getLogger(__name__)


class HeteroGNNBlock(nn.Module):
    """A single heterogeneous GNN layer with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_types: list[tuple[str, str, str]],
        conv_type: str = "sage",
        heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize GNN block.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            edge_types: List of edge type tuples (src, rel, dst).
            conv_type: Type of convolution ('sage' or 'gat').
            heads: Number of attention heads (for GAT).
            dropout: Dropout rate.
        """
        super().__init__()

        # Build convolution dictionary
        conv_dict = {}
        for edge_type in edge_types:
            if conv_type == "sage":
                conv_dict[edge_type] = SAGEConv(
                    (-1, -1),  # Use lazy initialization for heterogeneous graphs
                    out_channels,
                    aggr="mean",
                )
            elif conv_type == "gat":
                conv_dict[edge_type] = GATConv(
                    (-1, -1),  # Use lazy initialization for heterogeneous graphs
                    out_channels // heads,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                )
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

        self.conv = HeteroConv(conv_dict, aggr="sum")
        self.norm = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

        # Get unique node types from edge types
        node_types = set()
        for src, _, dst in edge_types:
            node_types.add(src)
            node_types.add(dst)

        for node_type in node_types:
            self.norm[node_type] = LayerNorm(out_channels)

        # Residual projection if dimensions differ
        self.residual = nn.ModuleDict()
        if in_channels != out_channels:
            for node_type in node_types:
                self.residual[node_type] = Linear(in_channels, out_channels)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the GNN block.

        Args:
            x_dict: Dictionary mapping node types to feature tensors.
            edge_index_dict: Dictionary mapping edge types to edge indices.

        Returns:
            Updated node features dictionary.
        """
        # Store input for residual
        x_residual = {}
        for node_type, x in x_dict.items():
            if node_type in self.residual:
                x_residual[node_type] = self.residual[node_type](x)
            else:
                x_residual[node_type] = x

        # Message passing
        out_dict = self.conv(x_dict, edge_index_dict)

        # Apply normalization, residual, and activation
        result = {}
        for node_type in x_dict.keys():
            # Use message passing output if available, otherwise use residual
            if node_type in out_dict and out_dict[node_type] is not None:
                x = out_dict[node_type]
                if node_type in self.norm:
                    x = self.norm[node_type](x)
                if node_type in x_residual:
                    x = x + x_residual[node_type]
                x = F.relu(x)
                x = self.dropout(x)
            else:
                # No incoming messages, use projected input
                x = x_residual[node_type]
                x = F.relu(x)
                x = self.dropout(x)
            result[node_type] = x

        return result


class TemporalEncoder(nn.Module):
    """LSTM-based temporal feature encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        """Initialize temporal encoder.

        Args:
            input_dim: Input feature dimension per time step.
            hidden_dim: Hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Encoded tensor of shape (batch, output_dim).
        """
        _, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]

        return h_n


class BuildingPowerGNN(nn.Module):
    """Main GNN model for building power efficiency prediction.

    Architecture:
    1. Node-type specific input projection
    2. Heterogeneous graph convolution layers
    3. Temporal encoding for time-series features
    4. Global pooling and prediction head
    """

    # Default edge types for building graphs
    DEFAULT_EDGE_TYPES = [
        ("hvac", "serves", "room"),
        ("sensor", "monitors", "room"),
        ("meter", "feeds", "hvac"),
        ("room", "adjacent", "room"),
        ("sensor", "controls", "hvac"),
        ("lighting", "serves", "room"),
        ("meter", "feeds", "lighting"),
    ]

    def __init__(
        self,
        node_feature_dims: dict[str, int],
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        temporal_dim: int = 64,
        temporal_input_dim: int = 5,
        num_temporal_layers: int = 2,
        conv_type: str = "sage",
        heads: int = 4,
        dropout: float = 0.1,
        edge_types: Optional[list[tuple[str, str, str]]] = None,
        output_dim: int = 1,
    ):
        """Initialize the model.

        Args:
            node_feature_dims: Dict mapping node types to feature dimensions.
            hidden_dim: Hidden dimension for GNN layers.
            num_gnn_layers: Number of GNN layers.
            temporal_dim: Dimension of temporal encoding.
            temporal_input_dim: Input dimension for temporal features.
            num_temporal_layers: Number of LSTM layers.
            conv_type: Type of graph convolution.
            heads: Number of attention heads.
            dropout: Dropout rate.
            edge_types: Edge types in the graph.
            output_dim: Output dimension (1 for power prediction).
        """
        super().__init__()

        self.node_types = list(node_feature_dims.keys())
        self.edge_types = edge_types or self.DEFAULT_EDGE_TYPES
        self.hidden_dim = hidden_dim

        # Filter edge types to only include those with registered node types
        valid_edge_types = [
            et for et in self.edge_types
            if et[0] in self.node_types and et[2] in self.node_types
        ]
        self.valid_edge_types = valid_edge_types

        # Input projections for each node type
        self.input_projections = nn.ModuleDict({
            node_type: nn.Sequential(
                Linear(dim, hidden_dim),
                LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for node_type, dim in node_feature_dims.items()
        })

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            HeteroGNNBlock(
                hidden_dim,
                hidden_dim,
                valid_edge_types,
                conv_type=conv_type,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(num_gnn_layers)
        ])

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_input_dim,
            hidden_dim=temporal_dim,
            num_layers=num_temporal_layers,
            dropout=dropout,
        )

        # Prediction head
        combined_dim = hidden_dim + self.temporal_encoder.output_dim
        self.predictor = nn.Sequential(
            Linear(combined_dim, hidden_dim),
            LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, output_dim),
        )

        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        data: HeteroData,
        temporal_features: Optional[dict[str, torch.Tensor]] = None,
        target_nodes: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            data: PyG HeteroData object.
            temporal_features: Dict mapping node types to temporal tensors
                              of shape (num_nodes, seq_len, temporal_dim).
            target_nodes: Dict mapping node types to target node indices.
                         If None, returns predictions for all nodes.

        Returns:
            Dictionary with 'power_pred' and 'anomaly_score' tensors.
        """
        # Project input features
        x_dict = {}
        for node_type in self.node_types:
            if node_type in data.node_types and hasattr(data[node_type], "x"):
                x_dict[node_type] = self.input_projections[node_type](data[node_type].x)

        # Build edge index dict
        edge_index_dict = {}
        for edge_type in self.valid_edge_types:
            if edge_type in data.edge_types:
                edge_index_dict[edge_type] = data[edge_type].edge_index

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x_dict = gnn_layer(x_dict, edge_index_dict)

        # Encode temporal features and combine
        outputs = {}
        for node_type, x in x_dict.items():
            if temporal_features is not None and node_type in temporal_features:
                temporal_encoded = self.temporal_encoder(temporal_features[node_type])
                combined = torch.cat([x, temporal_encoded], dim=-1)
            else:
                # Pad with zeros if no temporal features
                padding = torch.zeros(x.size(0), self.temporal_encoder.output_dim, device=x.device)
                combined = torch.cat([x, padding], dim=-1)

            # Select target nodes if specified
            if target_nodes is not None and node_type in target_nodes:
                combined = combined[target_nodes[node_type]]

            # Predictions
            power_pred = self.predictor(combined)
            anomaly_score = self.anomaly_head(combined)

            outputs[node_type] = {
                "power_pred": power_pred,
                "anomaly_score": anomaly_score,
                "embeddings": combined,
            }

        return outputs

    def get_embeddings(
        self,
        data: HeteroData,
        temporal_features: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Get node embeddings without prediction.

        Args:
            data: PyG HeteroData object.
            temporal_features: Optional temporal features.

        Returns:
            Dictionary mapping node types to embedding tensors.
        """
        with torch.no_grad():
            outputs = self.forward(data, temporal_features)
            return {nt: out["embeddings"] for nt, out in outputs.items()}


class BuildingPowerGNNLite(nn.Module):
    """Lightweight version of BuildingPowerGNN for faster inference."""

    def __init__(
        self,
        node_feature_dims: dict[str, int],
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        dropout: float = 0.1,
        edge_types: Optional[list[tuple[str, str, str]]] = None,
    ):
        """Initialize lightweight model."""
        super().__init__()

        self.node_types = list(node_feature_dims.keys())
        self.edge_types = edge_types or BuildingPowerGNN.DEFAULT_EDGE_TYPES
        self.valid_edge_types = [
            et for et in self.edge_types
            if et[0] in self.node_types and et[2] in self.node_types
        ]

        # Simple input projections
        self.input_projections = nn.ModuleDict({
            node_type: Linear(dim, hidden_dim)
            for node_type, dim in node_feature_dims.items()
        })

        # Simplified GNN layers using direct SAGEConv
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            conv_dict = {
                et: SAGEConv((-1, -1), hidden_dim)  # Lazy init for heterogeneous
                for et in self.valid_edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.dropout = nn.Dropout(dropout)

        # Simple predictor
        self.predictor = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Forward pass."""
        x_dict = {
            nt: self.input_projections[nt](data[nt].x)
            for nt in self.node_types
            if nt in data.node_types and hasattr(data[nt], "x")
        }

        edge_index_dict = {
            et: data[et].edge_index
            for et in self.valid_edge_types
            if et in data.edge_types
        }

        for conv in self.convs:
            # Store input for fallback
            x_input = x_dict.copy()
            out_dict = conv(x_dict, edge_index_dict)
            # Handle None values from HeteroConv
            x_dict = {}
            for nt in x_input.keys():
                if nt in out_dict and out_dict[nt] is not None:
                    x_dict[nt] = F.relu(self.dropout(out_dict[nt]))
                else:
                    x_dict[nt] = F.relu(self.dropout(x_input[nt]))

        return {
            nt: {"power_pred": self.predictor(x)}
            for nt, x in x_dict.items()
        }
