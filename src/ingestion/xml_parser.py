"""XML bulk ingestion module for building graph data.

Provides XSD validation, parsing, and transformation to Pandas DataFrames.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from lxml import etree
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schemas" / "building_graph.xsd"


@dataclass
class ParsedBuildingGraph:
    """Container for parsed building graph data."""

    building_id: str
    metadata: dict = field(default_factory=dict)
    nodes_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    edges_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    timeseries_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def node_count(self) -> int:
        return len(self.nodes_df)

    @property
    def edge_count(self) -> int:
        return len(self.edges_df)

    @property
    def reading_count(self) -> int:
        return len(self.timeseries_df)


class XMLValidationError(Exception):
    """Raised when XML validation fails."""
    pass


class XMLParser:
    """Parser for building graph XML files with XSD validation."""

    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize parser with XSD schema.

        Args:
            schema_path: Path to XSD schema file. Uses default if not provided.
        """
        self.schema_path = schema_path or SCHEMA_PATH
        self._schema = None
        self._load_schema()

    def _load_schema(self) -> None:
        """Load and compile XSD schema."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        schema_doc = etree.parse(str(self.schema_path))
        self._schema = etree.XMLSchema(schema_doc)
        logger.info(f"Loaded schema from {self.schema_path}")

    def validate(self, xml_path: Path) -> tuple[bool, list[str]]:
        """Validate XML file against schema.

        Args:
            xml_path: Path to XML file to validate.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        try:
            doc = etree.parse(str(xml_path))
            is_valid = self._schema.validate(doc)
            errors = [str(e) for e in self._schema.error_log]
            return is_valid, errors
        except etree.XMLSyntaxError as e:
            return False, [f"XML syntax error: {e}"]
        except OSError as e:
            return False, [f"File error: {e}"]

    def parse(
        self,
        xml_path: Path,
        validate: bool = True,
        show_progress: bool = True
    ) -> ParsedBuildingGraph:
        """Parse XML file into structured data.

        Args:
            xml_path: Path to XML file.
            validate: Whether to validate against schema first.
            show_progress: Whether to show progress bars.

        Returns:
            ParsedBuildingGraph with nodes, edges, and time series data.

        Raises:
            XMLValidationError: If validation fails.
        """
        xml_path = Path(xml_path)

        if validate:
            is_valid, errors = self.validate(xml_path)
            if not is_valid:
                raise XMLValidationError(
                    f"Validation failed for {xml_path}:\n" + "\n".join(errors)
                )

        tree = etree.parse(str(xml_path))
        root = tree.getroot()

        building_id = root.get("building_id")
        logger.info(f"Parsing building graph: {building_id}")

        # Parse metadata
        metadata = self._parse_metadata(root)

        # Parse nodes
        nodes_df = self._parse_nodes(root, show_progress)

        # Parse edges
        edges_df = self._parse_edges(root, show_progress)

        # Parse time series
        timeseries_df = self._parse_timeseries(root, show_progress)

        return ParsedBuildingGraph(
            building_id=building_id,
            metadata=metadata,
            nodes_df=nodes_df,
            edges_df=edges_df,
            timeseries_df=timeseries_df
        )

    def _parse_metadata(self, root: etree._Element) -> dict:
        """Extract metadata from XML root."""
        metadata = {"version": root.get("version", "1.0")}

        meta_elem = root.find("Metadata")
        if meta_elem is not None:
            if (name := meta_elem.find("BuildingName")) is not None:
                metadata["building_name"] = name.text

            if (loc := meta_elem.find("Location")) is not None:
                metadata["latitude"] = float(loc.get("latitude", 0))
                metadata["longitude"] = float(loc.get("longitude", 0))
                metadata["timezone"] = loc.get("timezone", "UTC")

            if (area := meta_elem.find("TotalArea")) is not None:
                metadata["total_area"] = float(area.text)

            if (floors := meta_elem.find("Floors")) is not None:
                metadata["floors"] = int(floors.text)

        return metadata

    def _parse_nodes(
        self,
        root: etree._Element,
        show_progress: bool
    ) -> pd.DataFrame:
        """Parse all nodes into DataFrame."""
        nodes_elem = root.find("Nodes")
        if nodes_elem is None:
            return pd.DataFrame()

        node_elements = nodes_elem.findall("Node")
        records = []

        iterator = tqdm(node_elements, desc="Parsing nodes") if show_progress else node_elements

        for node in iterator:
            record = {
                "node_id": node.get("id"),
                "node_type": node.get("type"),
                "subtype": node.get("subtype"),
                "zone": node.get("zone"),
                "floor": node.get("floor"),
                "area_sqft": self._safe_float(node.get("area_sqft")),
                "capacity_kw": self._safe_float(node.get("capacity_kw")),
                "wattage": self._safe_float(node.get("wattage")),
                "occupancy_max": self._safe_int(node.get("occupancy_max")),
            }

            # Parse custom attributes
            attrs_elem = node.find("Attributes")
            if attrs_elem is not None:
                for attr in attrs_elem.findall("Attribute"):
                    attr_name = attr.get("name")
                    attr_value = attr.get("value")
                    record[f"attr_{attr_name}"] = attr_value

            records.append(record)

        df = pd.DataFrame(records)
        df = df.set_index("node_id", drop=False)
        logger.info(f"Parsed {len(df)} nodes")
        return df

    def _parse_edges(
        self,
        root: etree._Element,
        show_progress: bool
    ) -> pd.DataFrame:
        """Parse all edges into DataFrame."""
        edges_elem = root.find("Edges")
        if edges_elem is None:
            return pd.DataFrame()

        edge_elements = edges_elem.findall("Edge")
        records = []

        iterator = tqdm(edge_elements, desc="Parsing edges") if show_progress else edge_elements

        for edge in iterator:
            record = {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "edge_type": edge.get("type"),
                "weight": self._safe_float(edge.get("weight"), default=1.0),
                "bidirectional": edge.get("bidirectional", "false").lower() == "true"
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Parsed {len(df)} edges")
        return df

    def _parse_timeseries(
        self,
        root: etree._Element,
        show_progress: bool
    ) -> pd.DataFrame:
        """Parse time series readings into DataFrame."""
        ts_elem = root.find("TimeSeries")
        if ts_elem is None:
            return pd.DataFrame()

        reading_elements = ts_elem.findall("Reading")
        records = []

        iterator = tqdm(reading_elements, desc="Parsing time series") if show_progress else reading_elements

        for reading in iterator:
            record = {
                "node_id": reading.get("node"),
                "timestamp": datetime.fromisoformat(reading.get("timestamp")),
                "value": float(reading.get("value")),
                "metric": reading.get("metric", "default"),
                "unit": reading.get("unit")
            }
            records.append(record)

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(["node_id", "timestamp"])
        logger.info(f"Parsed {len(df)} time series readings")
        return df

    @staticmethod
    def _safe_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
        """Safely convert string to float."""
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _safe_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
        """Safely convert string to int."""
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default


class BatchXMLProcessor:
    """Process multiple XML files in batch with progress tracking."""

    def __init__(self, parser: Optional[XMLParser] = None):
        """Initialize batch processor.

        Args:
            parser: XMLParser instance to use. Creates new one if not provided.
        """
        self.parser = parser or XMLParser()

    def process_directory(
        self,
        directory: Path,
        pattern: str = "*.xml",
        validate: bool = True,
        continue_on_error: bool = True
    ) -> tuple[list[ParsedBuildingGraph], list[tuple[Path, str]]]:
        """Process all XML files in a directory.

        Args:
            directory: Directory containing XML files.
            pattern: Glob pattern for XML files.
            validate: Whether to validate each file.
            continue_on_error: Whether to continue processing on errors.

        Returns:
            Tuple of (successful results, list of (path, error) for failures)
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))

        results = []
        errors = []

        for xml_file in tqdm(files, desc="Processing XML files"):
            try:
                result = self.parser.parse(xml_file, validate=validate, show_progress=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {xml_file}: {e}")
                errors.append((xml_file, str(e)))
                if not continue_on_error:
                    raise

        logger.info(f"Processed {len(results)} files successfully, {len(errors)} failures")
        return results, errors

    def merge_results(self, results: list[ParsedBuildingGraph]) -> ParsedBuildingGraph:
        """Merge multiple parsed graphs into one.

        Args:
            results: List of parsed building graphs.

        Returns:
            Single merged ParsedBuildingGraph.
        """
        if not results:
            return ParsedBuildingGraph(building_id="merged")

        nodes_dfs = [r.nodes_df for r in results if not r.nodes_df.empty]
        edges_dfs = [r.edges_df for r in results if not r.edges_df.empty]
        ts_dfs = [r.timeseries_df for r in results if not r.timeseries_df.empty]

        return ParsedBuildingGraph(
            building_id="merged",
            metadata={"source_count": len(results)},
            nodes_df=pd.concat(nodes_dfs, ignore_index=True) if nodes_dfs else pd.DataFrame(),
            edges_df=pd.concat(edges_dfs, ignore_index=True) if edges_dfs else pd.DataFrame(),
            timeseries_df=pd.concat(ts_dfs, ignore_index=True) if ts_dfs else pd.DataFrame()
        )
