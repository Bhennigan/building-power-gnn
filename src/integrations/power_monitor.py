"""Base class for power monitoring device integrations.

Supports real-time energy monitoring devices like:
- Emporia Vue
- Sense Energy Monitor
- IoTaWatt
- Shelly EM
- Generic MQTT/REST power monitors
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import logging

import pandas as pd

from .base import APIConnector, APIStandard, ConnectionConfig, ConnectionStatus, SyncResult

logger = logging.getLogger(__name__)


class PowerMonitorProvider(str, Enum):
    """Supported power monitoring providers."""
    EMPORIA = "emporia"
    SENSE = "sense"
    IOTAWATT = "iotawatt"
    SHELLY = "shelly"
    GENERIC_REST = "generic_rest"
    GENERIC_MQTT = "generic_mqtt"


class MeasurementScale(str, Enum):
    """Time scale for measurements."""
    SECOND = "1S"
    MINUTE = "1MIN"
    FIFTEEN_MINUTES = "15MIN"
    HOUR = "1H"
    DAY = "1D"
    WEEK = "1W"
    MONTH = "1MON"
    YEAR = "1Y"


@dataclass
class PowerChannel:
    """Represents a power monitoring channel/circuit."""
    id: str
    name: str
    channel_number: int
    device_id: str
    channel_type: str = "circuit"  # circuit, main, solar, battery
    multiplier: float = 1.0
    parent_channel_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PowerReading:
    """A single power reading."""
    channel_id: str
    timestamp: datetime
    watts: float
    kwh: Optional[float] = None
    voltage: Optional[float] = None
    current: Optional[float] = None
    power_factor: Optional[float] = None


class PowerMonitorConnector(APIConnector):
    """Base connector for power monitoring devices.

    Extend this class to add support for specific power monitoring
    devices and services. Implementations must provide:
    - Authentication mechanism
    - Device/channel discovery
    - Real-time and historical data retrieval
    """

    standard = APIStandard.POWER_MONITOR

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.provider = config.params.get("provider", PowerMonitorProvider.GENERIC_REST)
        self.channels: list[PowerChannel] = []
        self._device_id: Optional[str] = None

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the power monitor service.

        Returns:
            True if authentication successful
        """
        pass

    @abstractmethod
    async def discover_devices(self) -> list[dict]:
        """Discover available monitoring devices.

        Returns:
            List of device info dictionaries with at least:
            - id: Device identifier
            - name: Human-readable name
            - model: Device model
            - channels: Number of monitoring channels
        """
        pass

    @abstractmethod
    async def discover_channels(self, device_id: str) -> list[PowerChannel]:
        """Discover monitoring channels for a device.

        Args:
            device_id: The device to query

        Returns:
            List of PowerChannel objects
        """
        pass

    @abstractmethod
    async def get_live_reading(self, channel_ids: Optional[list[str]] = None) -> list[PowerReading]:
        """Get current/live power readings.

        Args:
            channel_ids: Specific channels to read, or None for all

        Returns:
            List of current PowerReading objects
        """
        pass

    @abstractmethod
    async def get_historical_readings(
        self,
        channel_ids: list[str],
        start_time: datetime,
        end_time: datetime,
        scale: MeasurementScale = MeasurementScale.MINUTE
    ) -> pd.DataFrame:
        """Get historical power readings.

        Args:
            channel_ids: Channels to query
            start_time: Start of time range
            end_time: End of time range
            scale: Time resolution for data

        Returns:
            DataFrame with columns: channel_id, timestamp, watts, kwh
        """
        pass

    async def fetch_points(self) -> list[dict]:
        """Return discovered channels as points."""
        if not self.channels:
            if self._device_id:
                self.channels = await self.discover_channels(self._device_id)

        return [
            {
                "id": ch.id,
                "name": ch.name,
                "type": "power",
                "subtype": ch.channel_type,
                "unit": "W",
                "device_id": ch.device_id,
                "channel_number": ch.channel_number,
                "metadata": ch.metadata,
            }
            for ch in self.channels
        ]

    async def fetch_readings(
        self,
        point_ids: list[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch historical readings for specified channels."""
        return await self.get_historical_readings(
            channel_ids=point_ids,
            start_time=start_time,
            end_time=end_time,
            scale=MeasurementScale.MINUTE
        )

    async def sync(self, since: Optional[datetime] = None) -> SyncResult:
        """Sync power data from the monitoring device."""
        start_time = datetime.now()
        errors = []
        warnings = []
        records_fetched = 0
        records_stored = 0

        try:
            if not self._client and not await self.connect():
                return SyncResult(
                    success=False,
                    timestamp=datetime.now(),
                    records_fetched=0,
                    records_stored=0,
                    errors=[self.last_error or "Failed to connect"],
                    warnings=[],
                    duration_seconds=(datetime.now() - start_time).total_seconds()
                )

            # Determine time range
            if since:
                query_start = since
            elif self.last_sync:
                query_start = self.last_sync
            else:
                # Default to last 24 hours
                query_start = datetime.now() - timedelta(hours=24)

            query_end = datetime.now()

            # Get all channel IDs
            points = await self.fetch_points()
            channel_ids = [p["id"] for p in points]

            if not channel_ids:
                warnings.append("No channels discovered")
                return SyncResult(
                    success=True,
                    timestamp=datetime.now(),
                    records_fetched=0,
                    records_stored=0,
                    errors=errors,
                    warnings=warnings,
                    duration_seconds=(datetime.now() - start_time).total_seconds()
                )

            # Fetch historical data
            df = await self.get_historical_readings(
                channel_ids=channel_ids,
                start_time=query_start,
                end_time=query_end,
                scale=MeasurementScale.MINUTE
            )

            records_fetched = len(df)

            # Store readings (implement in subclass or use data store)
            if not df.empty:
                records_stored = len(df)
                logger.info(f"Synced {records_stored} power readings")

            self.last_sync = datetime.now()
            self.last_error = None

            return SyncResult(
                success=True,
                timestamp=datetime.now(),
                records_fetched=records_fetched,
                records_stored=records_stored,
                errors=errors,
                warnings=warnings,
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            logger.exception("Power monitor sync failed")
            self.last_error = str(e)
            errors.append(str(e))

            return SyncResult(
                success=False,
                timestamp=datetime.now(),
                records_fetched=records_fetched,
                records_stored=records_stored,
                errors=errors,
                warnings=warnings,
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )
