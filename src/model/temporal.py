"""Temporal encoding modules for time-series features.

Provides various architectures for encoding temporal patterns in building data.
"""

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.

        Args:
            d_model: Dimension of the model.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    """Learnable time embedding based on hour, day, and month."""

    def __init__(self, embed_dim: int):
        """Initialize time embedding.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()

        # Hour of day (0-23)
        self.hour_embed = nn.Embedding(24, embed_dim // 4)
        # Day of week (0-6)
        self.day_embed = nn.Embedding(7, embed_dim // 4)
        # Month (0-11)
        self.month_embed = nn.Embedding(12, embed_dim // 4)
        # Holiday flag
        self.holiday_embed = nn.Embedding(2, embed_dim // 4)

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hours: torch.Tensor,
        days: torch.Tensor,
        months: torch.Tensor,
        holidays: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute time embedding.

        Args:
            hours: Hour of day tensor (batch, seq_len).
            days: Day of week tensor (batch, seq_len).
            months: Month tensor (batch, seq_len).
            holidays: Holiday flag tensor (batch, seq_len).

        Returns:
            Time embedding tensor (batch, seq_len, embed_dim).
        """
        h = self.hour_embed(hours)
        d = self.day_embed(days)
        m = self.month_embed(months)

        if holidays is None:
            holidays = torch.zeros_like(hours)
        hol = self.holiday_embed(holidays)

        combined = torch.cat([h, d, m, hol], dim=-1)
        return self.proj(combined)


class TransformerTemporalEncoder(nn.Module):
    """Transformer-based temporal encoder."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 168,
    ):
        """Initialize transformer encoder.

        Args:
            input_dim: Input feature dimension.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            dim_feedforward: Feedforward dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_dim = d_model

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode temporal sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).
            mask: Optional attention mask.

        Returns:
            Encoded tensor of shape (batch, d_model).
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use [CLS] token equivalent (first position) or mean pooling
        return x.mean(dim=1)


class ConvTemporalEncoder(nn.Module):
    """1D CNN-based temporal encoder for efficient processing."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kernel_sizes: list[int] = [3, 5, 7],
        dropout: float = 0.1,
    ):
        """Initialize convolutional encoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension per kernel.
            kernel_sizes: List of kernel sizes for multi-scale patterns.
            dropout: Dropout rate.
        """
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])

        self.output_dim = hidden_dim * len(kernel_sizes)
        self.fc = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Encoded tensor of shape (batch, output_dim).
        """
        # Transpose for conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)
            # Global max pooling
            out = out.max(dim=2)[0]
            conv_outputs.append(out)

        # Concatenate multi-scale features
        combined = torch.cat(conv_outputs, dim=-1)
        return self.fc(combined)


class GRUTemporalEncoder(nn.Module):
    """GRU-based temporal encoder (lighter than LSTM)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        """Initialize GRU encoder."""
        super().__init__()

        self.gru = nn.GRU(
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
        _, h_n = self.gru(x)

        if self.gru.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]

        return h_n


class TemporalFusion(nn.Module):
    """Fuses graph embeddings with temporal features."""

    def __init__(
        self,
        graph_dim: int,
        temporal_dim: int,
        output_dim: int,
        fusion_type: str = "concat",
        dropout: float = 0.1,
    ):
        """Initialize fusion module.

        Args:
            graph_dim: Dimension of graph embeddings.
            temporal_dim: Dimension of temporal embeddings.
            output_dim: Output dimension.
            fusion_type: Type of fusion ('concat', 'add', 'gate').
            dropout: Dropout rate.
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.dropout = nn.Dropout(dropout)

        if fusion_type == "concat":
            self.proj = nn.Linear(graph_dim + temporal_dim, output_dim)
        elif fusion_type == "add":
            self.graph_proj = nn.Linear(graph_dim, output_dim)
            self.temporal_proj = nn.Linear(temporal_dim, output_dim)
        elif fusion_type == "gate":
            self.graph_proj = nn.Linear(graph_dim, output_dim)
            self.temporal_proj = nn.Linear(temporal_dim, output_dim)
            self.gate = nn.Sequential(
                nn.Linear(graph_dim + temporal_dim, output_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        graph_emb: torch.Tensor,
        temporal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse graph and temporal embeddings.

        Args:
            graph_emb: Graph embedding (batch, graph_dim).
            temporal_emb: Temporal embedding (batch, temporal_dim).

        Returns:
            Fused embedding (batch, output_dim).
        """
        if self.fusion_type == "concat":
            combined = torch.cat([graph_emb, temporal_emb], dim=-1)
            out = self.proj(combined)
        elif self.fusion_type == "add":
            out = self.graph_proj(graph_emb) + self.temporal_proj(temporal_emb)
        elif self.fusion_type == "gate":
            g = self.graph_proj(graph_emb)
            t = self.temporal_proj(temporal_emb)
            combined = torch.cat([graph_emb, temporal_emb], dim=-1)
            gate = self.gate(combined)
            out = gate * g + (1 - gate) * t

        out = self.dropout(out)
        return self.norm(out)


class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal encoder capturing patterns at different resolutions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        scales: list[int] = [1, 4, 24],  # hourly, 4-hour, daily
        dropout: float = 0.1,
    ):
        """Initialize multi-scale encoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension per scale.
            scales: Downsampling factors for each scale.
            dropout: Dropout rate.
        """
        super().__init__()

        self.scales = scales
        self.encoders = nn.ModuleList([
            GRUTemporalEncoder(
                input_dim,
                hidden_dim,
                num_layers=1,
                dropout=dropout,
                bidirectional=True,
            )
            for _ in scales
        ])

        self.output_dim = hidden_dim * 2 * len(scales)  # bidirectional
        self.fusion = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode at multiple scales.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Multi-scale encoded tensor.
        """
        outputs = []

        for scale, encoder in zip(self.scales, self.encoders):
            if scale == 1:
                scaled_x = x
            else:
                # Downsample by averaging
                seq_len = x.size(1)
                new_len = seq_len // scale
                if new_len > 0:
                    scaled_x = x[:, :new_len * scale].view(
                        x.size(0), new_len, scale, x.size(2)
                    ).mean(dim=2)
                else:
                    scaled_x = x.mean(dim=1, keepdim=True)

            out = encoder(scaled_x)
            outputs.append(out)

        combined = torch.cat(outputs, dim=-1)
        return self.fusion(combined)
