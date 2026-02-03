"""Tests for XML parsing and validation."""

import tempfile
from pathlib import Path

import pytest
import pandas as pd

from src.ingestion import XMLParser, XMLValidationError, ParsedBuildingGraph


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<BuildingGraph version="1.0" building_id="test_building_001">
    <Metadata>
        <BuildingName>Test Office Building</BuildingName>
        <Location latitude="40.7128" longitude="-74.0060" timezone="America/New_York"/>
        <TotalArea>50000</TotalArea>
        <Floors>5</Floors>
    </Metadata>
    <Nodes>
        <Node id="hvac_1" type="HVAC" zone="floor_1" capacity_kw="50">
            <Attributes>
                <Attribute name="efficiency_rating" value="0.92"/>
                <Attribute name="age_years" value="3"/>
            </Attributes>
        </Node>
        <Node id="sensor_temp_1" type="Sensor" subtype="temperature" zone="floor_1"/>
        <Node id="room_101" type="Room" area_sqft="500" zone="floor_1">
            <Attributes>
                <Attribute name="occupancy_max" value="20"/>
            </Attributes>
        </Node>
        <Node id="meter_main" type="Meter" subtype="electrical"/>
    </Nodes>
    <Edges>
        <Edge source="hvac_1" target="room_101" type="serves"/>
        <Edge source="sensor_temp_1" target="room_101" type="monitors"/>
        <Edge source="meter_main" target="hvac_1" type="feeds"/>
    </Edges>
    <TimeSeries>
        <Reading node="sensor_temp_1" timestamp="2024-01-01T08:00:00" value="72.5" metric="temperature"/>
        <Reading node="sensor_temp_1" timestamp="2024-01-01T09:00:00" value="73.0" metric="temperature"/>
        <Reading node="hvac_1" timestamp="2024-01-01T08:00:00" value="25.5" metric="power"/>
    </TimeSeries>
</BuildingGraph>
"""

INVALID_XML = """<?xml version="1.0" encoding="UTF-8"?>
<BuildingGraph version="1.0" building_id="test_building">
    <Nodes>
        <Node id="invalid_node" type="InvalidType"/>
    </Nodes>
    <Edges>
    </Edges>
</BuildingGraph>
"""


@pytest.fixture
def sample_xml_file():
    """Create a temporary XML file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(SAMPLE_XML)
        f.flush()
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def invalid_xml_file():
    """Create a temporary XML file with invalid data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(INVALID_XML)
        f.flush()
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def parser():
    """Create an XML parser instance."""
    return XMLParser()


class TestXMLValidation:
    """Tests for XML validation."""

    def test_valid_xml_passes_validation(self, parser, sample_xml_file):
        """Valid XML should pass schema validation."""
        is_valid, errors = parser.validate(sample_xml_file)
        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_xml_fails_validation(self, parser, invalid_xml_file):
        """Invalid XML should fail schema validation."""
        is_valid, errors = parser.validate(invalid_xml_file)
        assert is_valid is False
        assert len(errors) > 0

    def test_nonexistent_file_raises_error(self, parser):
        """Validation of nonexistent file should handle gracefully."""
        is_valid, errors = parser.validate(Path("/nonexistent/file.xml"))
        assert is_valid is False


class TestXMLParsing:
    """Tests for XML parsing."""

    def test_parse_building_id(self, parser, sample_xml_file):
        """Parser should extract building ID."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        assert result.building_id == "test_building_001"

    def test_parse_metadata(self, parser, sample_xml_file):
        """Parser should extract metadata."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        assert result.metadata["building_name"] == "Test Office Building"
        assert result.metadata["latitude"] == 40.7128
        assert result.metadata["floors"] == 5

    def test_parse_nodes(self, parser, sample_xml_file):
        """Parser should extract all nodes."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        assert len(result.nodes_df) == 4
        assert "hvac_1" in result.nodes_df["node_id"].values
        assert "sensor_temp_1" in result.nodes_df["node_id"].values
        assert "room_101" in result.nodes_df["node_id"].values

    def test_parse_node_types(self, parser, sample_xml_file):
        """Parser should correctly identify node types."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        hvac_row = result.nodes_df[result.nodes_df["node_id"] == "hvac_1"].iloc[0]
        assert hvac_row["node_type"] == "HVAC"
        assert hvac_row["capacity_kw"] == 50.0

    def test_parse_custom_attributes(self, parser, sample_xml_file):
        """Parser should extract custom attributes."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        hvac_row = result.nodes_df[result.nodes_df["node_id"] == "hvac_1"].iloc[0]
        assert hvac_row["attr_efficiency_rating"] == "0.92"

    def test_parse_edges(self, parser, sample_xml_file):
        """Parser should extract all edges."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        assert len(result.edges_df) == 3
        serves_edges = result.edges_df[result.edges_df["edge_type"] == "serves"]
        assert len(serves_edges) == 1

    def test_parse_timeseries(self, parser, sample_xml_file):
        """Parser should extract time series data."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        assert len(result.timeseries_df) == 3
        temp_readings = result.timeseries_df[result.timeseries_df["metric"] == "temperature"]
        assert len(temp_readings) == 2

    def test_parse_with_validation(self, parser, sample_xml_file):
        """Parsing with validation should work for valid XML."""
        result = parser.parse(sample_xml_file, validate=True, show_progress=False)
        assert isinstance(result, ParsedBuildingGraph)

    def test_parse_invalid_raises_error(self, parser, invalid_xml_file):
        """Parsing invalid XML with validation should raise error."""
        with pytest.raises(XMLValidationError):
            parser.parse(invalid_xml_file, validate=True, show_progress=False)


class TestParsedBuildingGraph:
    """Tests for ParsedBuildingGraph dataclass."""

    def test_node_count_property(self, parser, sample_xml_file):
        """node_count property should return correct count."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        assert result.node_count == 4

    def test_edge_count_property(self, parser, sample_xml_file):
        """edge_count property should return correct count."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        assert result.edge_count == 3

    def test_reading_count_property(self, parser, sample_xml_file):
        """reading_count property should return correct count."""
        result = parser.parse(sample_xml_file, validate=False, show_progress=False)
        assert result.reading_count == 3
