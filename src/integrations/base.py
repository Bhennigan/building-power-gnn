"""Base classes for API integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class APIStandard(str, Enum):
    """Supported API standards."""
    HAYSTACK = "haystack"
    GREENBUTTON = "greenbutton"
    WEATHER = "weather"
    MODBUS = "modbus"
    BACNET = "bacnet"
    CUSTOM = "custom"


class ConnectionStatus(str, Enum):
    """Connection status states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    SYNCING = "syncing"


@dataclass
class ConnectionConfig:
    """Configuration for an API connection."""
    id: Optional[str] = None
    name: str = ""
    api_standard: APIStandard = APIStandard.CUSTOM
    base_url: str = ""
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    auth_type: str = "none"  # none, api_key, basic, oauth2, bearer
    headers: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    sync_interval_minutes: int = 15
    enabled: bool = True
    building_id: Optional[str] = None

    # Mapping configuration
    node_mapping: dict = field(default_factory=dict)
    metric_mapping: dict = field(default_factory=dict)


@dataclass
class SyncResult:
    """Result of a data synchronization."""
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    records_fetched: int = 0
    records_stored: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "records_fetched": self.records_fetched,
            "records_stored": self.records_stored,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
        }


class APIConnector(ABC):
    """Base class for API connectors."""

    standard: APIStandard = APIStandard.CUSTOM

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.last_sync: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self._session = None

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the API."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""
        pass

    @abstractmethod
    async def test_connection(self) -> tuple[bool, str]:
        """Test if the connection is working.

        Returns:
            Tuple of (success, message)
        """
        pass

    @abstractmethod
    async def fetch_points(self) -> list[dict]:
        """Fetch available data points/sensors from the API.

        Returns:
            List of point definitions with id, name, type, unit, etc.
        """
        pass

    @abstractmethod
    async def fetch_readings(
        self,
        point_ids: list[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch time series readings for specified points.

        Args:
            point_ids: List of point IDs to fetch
            start_time: Start of time range
            end_time: End of time range

        Returns:
            DataFrame with columns: point_id, timestamp, value, unit
        """
        pass

    async def sync(self, since: Optional[datetime] = None) -> SyncResult:
        """Perform a full data synchronization.

        Args:
            since: Only fetch data since this time (default: last sync time)

        Returns:
            SyncResult with details of the operation
        """
        import time
        start = time.time()
        result = SyncResult(success=False)

        try:
            self.status = ConnectionStatus.SYNCING

            # Connect if needed
            if self.status == ConnectionStatus.DISCONNECTED:
                if not await self.connect():
                    result.errors.append("Failed to connect")
                    return result

            # Determine time range
            end_time = datetime.utcnow()
            start_time = since or self.last_sync or (end_time.replace(hour=0, minute=0, second=0))

            # Fetch points
            points = await self.fetch_points()
            if not points:
                result.warnings.append("No points found")
                result.success = True
                return result

            point_ids = [p["id"] for p in points]

            # Fetch readings
            readings_df = await self.fetch_readings(point_ids, start_time, end_time)
            result.records_fetched = len(readings_df)

            # Store readings (to be implemented by subclass or manager)
            result.records_stored = result.records_fetched
            result.success = True
            self.last_sync = end_time
            self.status = ConnectionStatus.CONNECTED

        except Exception as e:
            logger.exception(f"Sync failed for {self.config.name}")
            result.errors.append(str(e))
            self.last_error = str(e)
            self.status = ConnectionStatus.ERROR

        finally:
            result.duration_seconds = time.time() - start

        return result

    def get_status(self) -> dict:
        """Get current connector status."""
        return {
            "name": self.config.name,
            "standard": self.config.api_standard.value,
            "status": self.status.value,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "last_error": self.last_error,
            "enabled": self.config.enabled,
        }
