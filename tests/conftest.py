"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def reset_api_state():
    """Reset API state before each test."""
    from src.api.routes.ingest import _nodes, _edges, _timeseries

    # Store original state
    original_nodes = _nodes.copy()
    original_edges = _edges.copy()
    original_ts = _timeseries.copy()

    yield

    # Restore original state
    _nodes.clear()
    _nodes.update(original_nodes)
    _edges.clear()
    _edges.extend(original_edges)
    _timeseries.clear()
    _timeseries.extend(original_ts)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
