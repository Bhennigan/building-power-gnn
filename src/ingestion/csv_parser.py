"""CSV and Excel file parsing for data ingestion."""

from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field
from io import BytesIO
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParsedCSVData:
    """Container for parsed CSV/Excel data."""
    nodes_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    edges_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    readings_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


# Expected columns for each data type
NODE_REQUIRED_COLUMNS = {"node_id", "node_type"}
NODE_OPTIONAL_COLUMNS = {
    "subtype", "zone", "floor", "capacity_kw", "wattage",
    "area_sqft", "occupancy_max"
}

EDGE_REQUIRED_COLUMNS = {"source", "target", "edge_type"}
EDGE_OPTIONAL_COLUMNS = {"weight", "bidirectional"}

READING_REQUIRED_COLUMNS = {"node_id", "timestamp", "value"}
READING_OPTIONAL_COLUMNS = {"metric", "unit"}

VALID_NODE_TYPES = {"HVAC", "Lighting", "Sensor", "Room", "Meter", "WeatherStation"}
VALID_EDGE_TYPES = {"serves", "monitors", "feeds", "adjacent", "controls"}


class CSVParser:
    """Parser for CSV and Excel files."""

    def parse_file(
        self,
        file_content: bytes,
        filename: str,
        data_type: Optional[str] = None
    ) -> ParsedCSVData:
        """Parse a CSV or Excel file.

        Args:
            file_content: Raw file bytes.
            filename: Original filename (used for extension detection).
            data_type: Type of data ('nodes', 'edges', 'readings', or None for auto-detect).

        Returns:
            ParsedCSVData with parsed DataFrames.
        """
        result = ParsedCSVData()

        try:
            # Read file based on extension
            df = self._read_file(file_content, filename)

            if df.empty:
                result.errors.append("File is empty")
                return result

            # Normalize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # Auto-detect data type if not specified
            if data_type is None:
                data_type = self._detect_data_type(df)

            if data_type is None:
                result.errors.append(
                    "Could not detect data type. Required columns: "
                    f"Nodes: {NODE_REQUIRED_COLUMNS}, "
                    f"Edges: {EDGE_REQUIRED_COLUMNS}, "
                    f"Readings: {READING_REQUIRED_COLUMNS}"
                )
                return result

            # Parse based on type
            if data_type == "nodes":
                result.nodes_df, errors, warnings = self._parse_nodes(df)
                result.errors.extend(errors)
                result.warnings.extend(warnings)

            elif data_type == "edges":
                result.edges_df, errors, warnings = self._parse_edges(df)
                result.errors.extend(errors)
                result.warnings.extend(warnings)

            elif data_type == "readings":
                result.readings_df, errors, warnings = self._parse_readings(df)
                result.errors.extend(errors)
                result.warnings.extend(warnings)

            else:
                result.errors.append(f"Unknown data type: {data_type}")

        except Exception as e:
            result.errors.append(f"Failed to parse file: {str(e)}")
            logger.exception(f"Error parsing {filename}")

        return result

    def parse_multi_sheet_excel(self, file_content: bytes) -> ParsedCSVData:
        """Parse an Excel file with multiple sheets (nodes, edges, readings)."""
        result = ParsedCSVData()

        try:
            excel_file = pd.ExcelFile(BytesIO(file_content))
            sheet_names = [s.lower() for s in excel_file.sheet_names]

            # Try to find and parse each sheet
            for sheet_name in excel_file.sheet_names:
                lower_name = sheet_name.lower()
                df = pd.read_excel(excel_file, sheet_name=sheet_name)

                if df.empty:
                    continue

                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

                if "node" in lower_name:
                    result.nodes_df, errors, warnings = self._parse_nodes(df)
                    result.errors.extend(errors)
                    result.warnings.extend(warnings)

                elif "edge" in lower_name or "connection" in lower_name:
                    result.edges_df, errors, warnings = self._parse_edges(df)
                    result.errors.extend(errors)
                    result.warnings.extend(warnings)

                elif "reading" in lower_name or "time" in lower_name or "sensor" in lower_name:
                    result.readings_df, errors, warnings = self._parse_readings(df)
                    result.errors.extend(errors)
                    result.warnings.extend(warnings)

        except Exception as e:
            result.errors.append(f"Failed to parse Excel file: {str(e)}")
            logger.exception("Error parsing Excel file")

        return result

    def _read_file(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """Read file content into DataFrame."""
        buffer = BytesIO(file_content)
        ext = Path(filename).suffix.lower()

        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(buffer)
        elif ext == ".csv":
            # Try different encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    buffer.seek(0)
                    return pd.read_csv(buffer, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode CSV file")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _detect_data_type(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect data type from columns."""
        columns = set(df.columns)

        if NODE_REQUIRED_COLUMNS.issubset(columns):
            return "nodes"
        elif EDGE_REQUIRED_COLUMNS.issubset(columns):
            return "edges"
        elif READING_REQUIRED_COLUMNS.issubset(columns):
            return "readings"

        return None

    def _parse_nodes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str], list[str]]:
        """Parse nodes DataFrame."""
        errors = []
        warnings = []

        # Check required columns
        missing = NODE_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            errors.append(f"Missing required columns for nodes: {missing}")
            return pd.DataFrame(), errors, warnings

        # Validate node types
        invalid_types = set(df["node_type"].unique()) - VALID_NODE_TYPES
        if invalid_types:
            warnings.append(
                f"Non-standard node types found: {invalid_types}. "
                f"Valid types: {VALID_NODE_TYPES}"
            )

        # Check for duplicates
        duplicates = df[df["node_id"].duplicated()]["node_id"].tolist()
        if duplicates:
            errors.append(f"Duplicate node IDs: {duplicates}")

        # Clean data
        df = df.copy()
        df["node_id"] = df["node_id"].astype(str).str.strip()
        df["node_type"] = df["node_type"].astype(str).str.strip()

        # Convert numeric columns
        for col in ["capacity_kw", "wattage", "area_sqft"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "occupancy_max" in df.columns:
            df["occupancy_max"] = pd.to_numeric(df["occupancy_max"], errors="coerce").astype("Int64")

        logger.info(f"Parsed {len(df)} nodes")
        return df, errors, warnings

    def _parse_edges(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str], list[str]]:
        """Parse edges DataFrame."""
        errors = []
        warnings = []

        # Check required columns
        missing = EDGE_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            errors.append(f"Missing required columns for edges: {missing}")
            return pd.DataFrame(), errors, warnings

        # Validate edge types
        invalid_types = set(df["edge_type"].unique()) - VALID_EDGE_TYPES
        if invalid_types:
            warnings.append(
                f"Non-standard edge types found: {invalid_types}. "
                f"Valid types: {VALID_EDGE_TYPES}"
            )

        # Check for self-loops
        self_loops = df[df["source"] == df["target"]]
        if not self_loops.empty:
            errors.append(f"Self-loops found: {self_loops[['source', 'target']].values.tolist()}")

        # Clean data
        df = df.copy()
        df["source"] = df["source"].astype(str).str.strip()
        df["target"] = df["target"].astype(str).str.strip()
        df["edge_type"] = df["edge_type"].astype(str).str.strip()

        if "weight" not in df.columns:
            df["weight"] = 1.0
        else:
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)

        if "bidirectional" not in df.columns:
            df["bidirectional"] = False
        else:
            df["bidirectional"] = df["bidirectional"].astype(bool)

        logger.info(f"Parsed {len(df)} edges")
        return df, errors, warnings

    def _parse_readings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str], list[str]]:
        """Parse readings DataFrame."""
        errors = []
        warnings = []

        # Check required columns
        missing = READING_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            errors.append(f"Missing required columns for readings: {missing}")
            return pd.DataFrame(), errors, warnings

        # Clean data
        df = df.copy()
        df["node_id"] = df["node_id"].astype(str).str.strip()

        # Parse timestamps
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            errors.append(f"Could not parse timestamps: {e}")
            return pd.DataFrame(), errors, warnings

        # Parse values
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        nan_values = df["value"].isna().sum()
        if nan_values > 0:
            warnings.append(f"{nan_values} readings have invalid values and will be skipped")
            df = df.dropna(subset=["value"])

        if "metric" not in df.columns:
            df["metric"] = "default"

        logger.info(f"Parsed {len(df)} readings")
        return df, errors, warnings


def generate_template_csv(data_type: str) -> str:
    """Generate a template CSV for a data type."""
    if data_type == "nodes":
        return """node_id,node_type,subtype,zone,floor,capacity_kw,wattage,area_sqft,occupancy_max
hvac_floor1,HVAC,,floor_1,1,100,,,
hvac_floor2,HVAC,,floor_2,2,75,,,
sensor_temp_1,Sensor,temperature,floor_1,1,,,,
sensor_occ_1,Sensor,occupancy,floor_1,1,,,,
room_101,Room,,floor_1,1,,,500,25
room_102,Room,,floor_1,1,,,600,30
lighting_1,Lighting,LED,floor_1,1,,5000,,
meter_main,Meter,electrical,,,,,,
"""
    elif data_type == "edges":
        return """source,target,edge_type,weight,bidirectional
hvac_floor1,room_101,serves,1.0,false
hvac_floor1,room_102,serves,1.0,false
sensor_temp_1,room_101,monitors,1.0,false
sensor_occ_1,room_101,monitors,1.0,false
sensor_temp_1,hvac_floor1,controls,1.0,false
meter_main,hvac_floor1,feeds,1.0,false
lighting_1,room_101,serves,1.0,false
room_101,room_102,adjacent,1.0,true
"""
    elif data_type == "readings":
        return """node_id,timestamp,value,metric,unit
sensor_temp_1,2024-01-15T08:00:00,68.5,temperature,F
sensor_temp_1,2024-01-15T09:00:00,70.2,temperature,F
sensor_temp_1,2024-01-15T10:00:00,71.8,temperature,F
sensor_occ_1,2024-01-15T08:00:00,15,occupancy,
sensor_occ_1,2024-01-15T09:00:00,35,occupancy,
hvac_floor1,2024-01-15T08:00:00,45.2,power,kW
hvac_floor1,2024-01-15T09:00:00,62.5,power,kW
"""
    else:
        raise ValueError(f"Unknown data type: {data_type}")
