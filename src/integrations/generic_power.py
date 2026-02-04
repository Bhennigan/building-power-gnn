"""Generic power monitor connector for REST/MQTT-based devices.

Provides a flexible connector that can work with various power monitoring
devices that expose data via REST API or MQTT. Configurable endpoints
and data mapping for maximum compatibility.

Supports:
- IoTaWatt
- Shelly EM/Pro
- Home Assistant energy sensors
- Any REST API returning power data
"""

from datetime import datetime, timedelta
from typing import Optional, Any
import logging

import httpx
import pandas as pd

from .base import ConnectionConfig, ConnectionStatus
from .power_monitor import (
    PowerMonitorConnector,
    PowerMonitorProvider,
    PowerChannel,
    PowerReading,
    MeasurementScale,
)

logger = logging.getLogger(__name__)


class GenericPowerConnector(PowerMonitorConnector):
    """Generic connector for REST-based power monitoring devices.

    Configurable via params:
    - base_url: Base URL of the device/API
    - endpoints: Dict mapping operation to endpoint path
      - devices: Endpoint to list devices
      - channels: Endpoint to list channels (use {device_id} placeholder)
      - live: Endpoint for live readings
      - history: Endpoint for historical data
    - data_mapping: Dict mapping response fields to standard fields
      - channel_id, name, watts, kwh, timestamp, etc.
    - auth_type: none, api_key, basic, bearer
    """

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.provider = config.params.get("provider", PowerMonitorProvider.GENERIC_REST)
        self._client: Optional[httpx.AsyncClient] = None

        # Configurable endpoints
        self.endpoints = config.params.get("endpoints", {
            "devices": "/devices",
            "channels": "/devices/{device_id}/channels",
            "live": "/readings/live",
            "history": "/readings/history",
        })

        # Data field mapping
        self.data_mapping = config.params.get("data_mapping", {
            "device_id": "id",
            "device_name": "name",
            "channel_id": "id",
            "channel_name": "name",
            "channel_number": "channel",
            "watts": "watts",
            "kwh": "kwh",
            "timestamp": "timestamp",
            "voltage": "voltage",
            "current": "current",
        })

        # Response parsing config
        self.response_config = config.params.get("response_config", {
            "devices_key": "devices",
            "channels_key": "channels",
            "readings_key": "readings",
            "data_key": "data",
        })

    async def connect(self) -> bool:
        """Initialize HTTP client connection."""
        try:
            self.status = ConnectionStatus.CONNECTING

            headers = {"Accept": "application/json"}

            # Add authentication headers
            if self.config.auth_type == "api_key" and self.config.api_key:
                headers["X-API-Key"] = self.config.api_key
            elif self.config.auth_type == "bearer" and self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            auth = None
            if self.config.auth_type == "basic" and self.config.username:
                auth = httpx.BasicAuth(
                    self.config.username,
                    self.config.password or ""
                )

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                auth=auth,
                timeout=30.0,
            )

            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to generic power monitor at {self.config.base_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.last_error = str(e)
            self.status = ConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from the device."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self.status = ConnectionStatus.DISCONNECTED

    async def authenticate(self) -> bool:
        """Authentication handled in connect() via headers."""
        return True

    def _extract_field(self, data: dict, field_path: str, default: Any = None) -> Any:
        """Extract a field from nested dict using dot notation."""
        parts = field_path.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                value = value[idx] if idx < len(value) else None
            else:
                return default
            if value is None:
                return default
        return value

    async def discover_devices(self) -> list[dict]:
        """Discover devices via REST API."""
        if not self._client:
            await self.connect()

        try:
            endpoint = self.endpoints.get("devices", "/devices")
            response = await self._client.get(endpoint)

            if response.status_code != 200:
                logger.warning(f"Device discovery failed: {response.status_code}")
                return []

            data = response.json()

            # Extract devices list from response
            devices_key = self.response_config.get("devices_key", "devices")
            devices_data = self._extract_field(data, devices_key, [])

            if isinstance(devices_data, dict):
                devices_data = [devices_data]

            devices = []
            for d in devices_data:
                device = {
                    "id": self._extract_field(d, self.data_mapping.get("device_id", "id"), ""),
                    "name": self._extract_field(d, self.data_mapping.get("device_name", "name"), "Unknown"),
                    "model": self._extract_field(d, "model", "Generic"),
                    "channels": self._extract_field(d, "channels", 1),
                }
                devices.append(device)

            return devices

        except Exception as e:
            logger.exception("Failed to discover devices")
            return []

    async def discover_channels(self, device_id: str) -> list[PowerChannel]:
        """Discover channels for a device."""
        if not self._client:
            await self.connect()

        try:
            endpoint = self.endpoints.get("channels", "/devices/{device_id}/channels")
            endpoint = endpoint.format(device_id=device_id)

            response = await self._client.get(endpoint)

            if response.status_code != 200:
                logger.warning(f"Channel discovery failed: {response.status_code}")
                return []

            data = response.json()

            # Extract channels list
            channels_key = self.response_config.get("channels_key", "channels")
            channels_data = self._extract_field(data, channels_key, [])

            if isinstance(channels_data, dict):
                channels_data = [channels_data]

            channels = []
            for i, ch in enumerate(channels_data):
                channel_id = self._extract_field(
                    ch, self.data_mapping.get("channel_id", "id"),
                    f"{device_id}_{i}"
                )
                channel = PowerChannel(
                    id=str(channel_id),
                    name=self._extract_field(ch, self.data_mapping.get("channel_name", "name"), f"Channel {i}"),
                    channel_number=self._extract_field(ch, self.data_mapping.get("channel_number", "channel"), i),
                    device_id=device_id,
                    channel_type=self._extract_field(ch, "type", "circuit"),
                )
                channels.append(channel)

            self.channels = channels
            return channels

        except Exception as e:
            logger.exception("Failed to discover channels")
            return []

    async def get_live_reading(self, channel_ids: Optional[list[str]] = None) -> list[PowerReading]:
        """Get live power readings."""
        if not self._client:
            await self.connect()

        try:
            endpoint = self.endpoints.get("live", "/readings/live")
            params = {}
            if channel_ids:
                params["channels"] = ",".join(channel_ids)

            response = await self._client.get(endpoint, params=params)

            if response.status_code != 200:
                logger.warning(f"Live reading failed: {response.status_code}")
                return []

            data = response.json()

            # Extract readings
            readings_key = self.response_config.get("readings_key", "readings")
            readings_data = self._extract_field(data, readings_key, [])

            if isinstance(readings_data, dict):
                readings_data = [readings_data]

            readings = []
            timestamp = datetime.now()

            for r in readings_data:
                channel_id = str(self._extract_field(r, self.data_mapping.get("channel_id", "id"), ""))

                if channel_ids and channel_id not in channel_ids:
                    continue

                watts = self._extract_field(r, self.data_mapping.get("watts", "watts"), 0)
                kwh = self._extract_field(r, self.data_mapping.get("kwh", "kwh"))
                voltage = self._extract_field(r, self.data_mapping.get("voltage", "voltage"))
                current = self._extract_field(r, self.data_mapping.get("current", "current"))

                readings.append(PowerReading(
                    channel_id=channel_id,
                    timestamp=timestamp,
                    watts=float(watts) if watts else 0,
                    kwh=float(kwh) if kwh else None,
                    voltage=float(voltage) if voltage else None,
                    current=float(current) if current else None,
                ))

            return readings

        except Exception as e:
            logger.exception("Failed to get live readings")
            return []

    async def get_historical_readings(
        self,
        channel_ids: list[str],
        start_time: datetime,
        end_time: datetime,
        scale: MeasurementScale = MeasurementScale.MINUTE
    ) -> pd.DataFrame:
        """Get historical power readings."""
        if not self._client:
            await self.connect()

        all_readings = []

        try:
            endpoint = self.endpoints.get("history", "/readings/history")
            params = {
                "channels": ",".join(channel_ids),
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "scale": scale.value,
            }

            response = await self._client.get(endpoint, params=params)

            if response.status_code != 200:
                logger.warning(f"Historical reading failed: {response.status_code}")
                return pd.DataFrame(columns=["channel_id", "timestamp", "watts", "kwh"])

            data = response.json()

            # Extract data
            data_key = self.response_config.get("data_key", "data")
            readings_data = self._extract_field(data, data_key, [])

            if isinstance(readings_data, dict):
                readings_data = [readings_data]

            for r in readings_data:
                channel_id = str(self._extract_field(r, self.data_mapping.get("channel_id", "channel_id"), ""))
                ts_raw = self._extract_field(r, self.data_mapping.get("timestamp", "timestamp"), "")
                watts = self._extract_field(r, self.data_mapping.get("watts", "watts"), 0)
                kwh = self._extract_field(r, self.data_mapping.get("kwh", "kwh"))

                if channel_id and ts_raw:
                    try:
                        if isinstance(ts_raw, (int, float)):
                            ts = datetime.fromtimestamp(ts_raw)
                        else:
                            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))

                        all_readings.append({
                            "channel_id": channel_id,
                            "timestamp": ts,
                            "watts": float(watts) if watts else 0,
                            "kwh": float(kwh) if kwh else None,
                        })
                    except (ValueError, TypeError):
                        continue

            if not all_readings:
                return pd.DataFrame(columns=["channel_id", "timestamp", "watts", "kwh"])

            df = pd.DataFrame(all_readings)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        except Exception as e:
            logger.exception("Failed to get historical readings")
            return pd.DataFrame(columns=["channel_id", "timestamp", "watts", "kwh"])

    async def test_connection(self) -> tuple[bool, str]:
        """Test the connection."""
        try:
            if not self._client:
                if not await self.connect():
                    return False, self.last_error or "Failed to connect"

            # Try to get devices or a simple health check
            devices = await self.discover_devices()

            if devices:
                return True, f"Connected. Found {len(devices)} device(s)"
            else:
                # Try a simple GET to base URL
                response = await self._client.get("/")
                if response.status_code < 500:
                    return True, "Connected to power monitor API"
                return False, f"Server returned {response.status_code}"

        except Exception as e:
            return False, str(e)


# Preset configurations for popular devices

IOTAWATT_CONFIG = {
    "endpoints": {
        "devices": "/status",
        "channels": "/status",
        "live": "/status",
        "history": "/query",
    },
    "data_mapping": {
        "channel_id": "name",
        "channel_name": "name",
        "watts": "Watts",
        "kwh": "Wh",
    },
    "response_config": {
        "devices_key": "",
        "channels_key": "inputs",
        "readings_key": "inputs",
    },
}

SHELLY_EM_CONFIG = {
    "endpoints": {
        "devices": "/shelly",
        "channels": "/status",
        "live": "/status",
        "history": "/emeter/{channel}/em_data",
    },
    "data_mapping": {
        "channel_id": "id",
        "watts": "power",
        "kwh": "total",
        "voltage": "voltage",
        "current": "current",
    },
    "response_config": {
        "devices_key": "",
        "channels_key": "emeters",
        "readings_key": "emeters",
    },
}

HOME_ASSISTANT_CONFIG = {
    "endpoints": {
        "devices": "/api/states",
        "channels": "/api/states",
        "live": "/api/states/{entity_id}",
        "history": "/api/history/period/{start}",
    },
    "data_mapping": {
        "channel_id": "entity_id",
        "channel_name": "attributes.friendly_name",
        "watts": "state",
        "timestamp": "last_changed",
    },
    "response_config": {
        "devices_key": "",
        "channels_key": "",
        "readings_key": "",
    },
}


def get_preset_config(provider: str) -> dict:
    """Get preset configuration for a known provider."""
    presets = {
        "iotawatt": IOTAWATT_CONFIG,
        "shelly": SHELLY_EM_CONFIG,
        "home_assistant": HOME_ASSISTANT_CONFIG,
    }
    return presets.get(provider.lower(), {})
