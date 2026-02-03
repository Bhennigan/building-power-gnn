"""Tests for GNN model architecture and forward pass."""

import pytest
import torch
from torch_geometric.data import HeteroData

from src.model import BuildingPowerGNN, BuildingPowerGNNLite, TemporalEncoder


@pytest.fixture
def node_feature_dims():
    """Default node feature dimensions."""
    return {
        "hvac": 8,
        "room": 7,
        "sensor": 5,
        "meter": 4,
    }


@pytest.fixture
def sample_hetero_data():
    """Create sample HeteroData for testing."""
    data = HeteroData()

    # Add node features
    data["hvac"].x = torch.randn(3, 8)  # 3 HVAC nodes
    data["room"].x = torch.randn(5, 7)  # 5 rooms
    data["sensor"].x = torch.randn(4, 5)  # 4 sensors
    data["meter"].x = torch.randn(2, 4)  # 2 meters

    # Add edges
    data["hvac", "serves", "room"].edge_index = torch.tensor([
        [0, 1, 2],
        [0, 1, 2],
    ])
    data["sensor", "monitors", "room"].edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    data["meter", "feeds", "hvac"].edge_index = torch.tensor([
        [0, 0, 1],
        [0, 1, 2],
    ])

    return data


@pytest.fixture
def sample_temporal_features():
    """Create sample temporal features."""
    return {
        "hvac": torch.randn(3, 168, 5),  # 3 nodes, 168 time steps, 5 features
        "room": torch.randn(5, 168, 5),
        "sensor": torch.randn(4, 168, 5),
        "meter": torch.randn(2, 168, 5),
    }


class TestTemporalEncoder:
    """Tests for TemporalEncoder."""

    def test_lstm_encoder(self):
        """LSTM encoder should produce correct output shape."""
        encoder = TemporalEncoder(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
            bidirectional=True,
        )

        x = torch.randn(8, 168, 5)  # batch=8, seq=168, features=5
        output = encoder(x)

        assert output.shape == (8, 128)  # 64 * 2 for bidirectional

    def test_lstm_encoder_unidirectional(self):
        """Unidirectional LSTM should produce smaller output."""
        encoder = TemporalEncoder(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
            bidirectional=False,
        )

        x = torch.randn(8, 168, 5)
        output = encoder(x)

        assert output.shape == (8, 64)


class TestBuildingPowerGNN:
    """Tests for BuildingPowerGNN model."""

    def test_model_initialization(self, node_feature_dims):
        """Model should initialize with correct parameters."""
        model = BuildingPowerGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=128,
            num_gnn_layers=3,
        )

        assert model is not None
        assert len(model.gnn_layers) == 3

    def test_forward_pass(self, node_feature_dims, sample_hetero_data):
        """Forward pass should produce outputs for each node type."""
        model = BuildingPowerGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=64,
            num_gnn_layers=2,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(sample_hetero_data)

        assert isinstance(outputs, dict)
        assert "hvac" in outputs
        assert "room" in outputs

    def test_output_contains_predictions(self, node_feature_dims, sample_hetero_data):
        """Outputs should contain power predictions."""
        model = BuildingPowerGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=64,
            num_gnn_layers=2,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(sample_hetero_data)

        for node_type, out in outputs.items():
            assert "power_pred" in out
            assert "anomaly_score" in out
            assert "embeddings" in out

    def test_output_shapes(self, node_feature_dims, sample_hetero_data):
        """Output shapes should match node counts."""
        model = BuildingPowerGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=64,
            num_gnn_layers=2,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(sample_hetero_data)

        # HVAC: 3 nodes
        assert outputs["hvac"]["power_pred"].shape[0] == 3
        # Room: 5 nodes
        assert outputs["room"]["power_pred"].shape[0] == 5

    def test_forward_with_temporal(
        self, node_feature_dims, sample_hetero_data, sample_temporal_features
    ):
        """Forward pass should work with temporal features."""
        model = BuildingPowerGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=64,
            num_gnn_layers=2,
            temporal_dim=32,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(sample_hetero_data, sample_temporal_features)

        assert "hvac" in outputs
        assert outputs["hvac"]["power_pred"] is not None

    def test_get_embeddings(self, node_feature_dims, sample_hetero_data):
        """get_embeddings should return node embeddings."""
        model = BuildingPowerGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=64,
            num_gnn_layers=2,
        )

        embeddings = model.get_embeddings(sample_hetero_data)

        assert isinstance(embeddings, dict)
        for node_type, emb in embeddings.items():
            assert isinstance(emb, torch.Tensor)

    def test_model_training_mode(self, node_feature_dims, sample_hetero_data):
        """Model should work in training mode with gradients."""
        model = BuildingPowerGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=64,
            num_gnn_layers=2,
        )

        model.train()
        outputs = model(sample_hetero_data)

        # Check that gradients can be computed
        loss = outputs["hvac"]["power_pred"].mean()
        loss.backward()

        # Check that parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad


class TestBuildingPowerGNNLite:
    """Tests for lightweight model variant."""

    def test_lite_model_initialization(self, node_feature_dims):
        """Lite model should initialize correctly."""
        model = BuildingPowerGNNLite(
            node_feature_dims=node_feature_dims,
            hidden_dim=64,
            num_gnn_layers=2,
        )

        assert model is not None

    def test_lite_forward_pass(self, node_feature_dims, sample_hetero_data):
        """Lite model should produce predictions."""
        model = BuildingPowerGNNLite(
            node_feature_dims=node_feature_dims,
            hidden_dim=64,
            num_gnn_layers=2,
        )

        model.eval()
        with torch.no_grad():
            outputs = model(sample_hetero_data)

        assert "hvac" in outputs
        assert "power_pred" in outputs["hvac"]

    def test_lite_model_smaller(self, node_feature_dims):
        """Lite model should have fewer layers than full model."""
        full_model = BuildingPowerGNN(
            node_feature_dims=node_feature_dims,
            hidden_dim=128,
            num_gnn_layers=3,
        )
        lite_model = BuildingPowerGNNLite(
            node_feature_dims=node_feature_dims,
            hidden_dim=64,
            num_gnn_layers=2,
        )

        # Compare architecture properties
        assert full_model.hidden_dim == 128
        assert len(full_model.gnn_layers) == 3
        assert len(lite_model.convs) == 2
