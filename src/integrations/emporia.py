"""Emporia Energy Monitor connector.

Connects to Emporia Vue energy monitors for real-time and historical
power consumption data. Supports:
- Emporia Vue (Gen 1 & 2)
- Emporia Vue Utility Connect
- Emporia Smart Plugs

Authentication uses Emporia's AWS Cognito backend.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import logging
import hashlib
import hmac
import base64

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

# Emporia API endpoints
EMPORIA_API_BASE = "https://api.emporiaenergy.com"
EMPORIA_COGNITO_REGION = "us-east-2"
EMPORIA_COGNITO_CLIENT_ID = "4qte47jbstod8apnfic0bunmrq"
EMPORIA_COGNITO_POOL_ID = "us-east-2_ghlOXVLi1"


class EmporiaConnector(PowerMonitorConnector):
    """Connector for Emporia energy monitoring devices.

    Requires Emporia account credentials (email/password).
    Retrieves real-time and historical power data from Emporia Vue devices.
    """

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.provider = PowerMonitorProvider.EMPORIA
        self._client: Optional[httpx.AsyncClient] = None
        self._id_token: Optional[str] = None
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._customer_gid: Optional[int] = None
        self._devices: list[dict] = []

        # Credentials from config
        self._email = config.username or config.params.get("email", "")
        self._password = config.password or config.params.get("password", "")

    async def connect(self) -> bool:
        """Initialize connection and authenticate."""
        try:
            self.status = ConnectionStatus.CONNECTING

            if not self._email or not self._password:
                self.last_error = "Email and password required for Emporia"
                self.status = ConnectionStatus.ERROR
                return False

            self._client = httpx.AsyncClient(
                base_url=EMPORIA_API_BASE,
                timeout=30.0,
            )

            # Authenticate
            if not await self.authenticate():
                return False

            # Discover devices
            self._devices = await self.discover_devices()
            if self._devices:
                self._device_id = str(self._devices[0].get("deviceGid", ""))

            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Emporia with {len(self._devices)} device(s)")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Emporia: {e}")
            self.last_error = str(e)
            self.status = ConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from Emporia."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._id_token = None
        self._access_token = None
        self.status = ConnectionStatus.DISCONNECTED

    async def authenticate(self) -> bool:
        """Authenticate with Emporia via AWS Cognito."""
        try:
            # Use Cognito SRP authentication
            auth_url = f"https://cognito-idp.{EMPORIA_COGNITO_REGION}.amazonaws.com/"

            # InitiateAuth request
            auth_payload = {
                "AuthFlow": "USER_PASSWORD_AUTH",
                "ClientId": EMPORIA_COGNITO_CLIENT_ID,
                "AuthParameters": {
                    "USERNAME": self._email,
                    "PASSWORD": self._password,
                }
            }

            async with httpx.AsyncClient(timeout=30.0) as cognito_client:
                response = await cognito_client.post(
                    auth_url,
                    json=auth_payload,
                    headers={
                        "Content-Type": "application/x-amz-json-1.1",
                        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
                    }
                )

                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("message", "Authentication failed")
                    self.last_error = f"Emporia auth failed: {error_msg}"
                    self.status = ConnectionStatus.ERROR
                    logger.error(f"Emporia authentication failed: {response.text}")
                    return False

                auth_result = response.json()
                result = auth_result.get("AuthenticationResult", {})

                self._id_token = result.get("IdToken")
                self._access_token = result.get("AccessToken")
                self._refresh_token = result.get("RefreshToken")

                # Token expires in ExpiresIn seconds
                expires_in = result.get("ExpiresIn", 3600)
                self._token_expiry = datetime.now() + timedelta(seconds=expires_in)

                if not self._id_token:
                    self.last_error = "No ID token received"
                    self.status = ConnectionStatus.ERROR
                    return False

                logger.info("Successfully authenticated with Emporia")
                return True

        except Exception as e:
            logger.exception("Emporia authentication error")
            self.last_error = str(e)
            self.status = ConnectionStatus.ERROR
            return False

    async def _ensure_authenticated(self) -> bool:
        """Ensure we have valid authentication tokens."""
        if not self._id_token:
            return await self.authenticate()

        # Check if token is expired or about to expire
        if self._token_expiry and datetime.now() >= self._token_expiry - timedelta(minutes=5):
            return await self._refresh_auth()

        return True

    async def _refresh_auth(self) -> bool:
        """Refresh authentication tokens."""
        if not self._refresh_token:
            return await self.authenticate()

        try:
            auth_url = f"https://cognito-idp.{EMPORIA_COGNITO_REGION}.amazonaws.com/"

            refresh_payload = {
                "AuthFlow": "REFRESH_TOKEN_AUTH",
                "ClientId": EMPORIA_COGNITO_CLIENT_ID,
                "AuthParameters": {
                    "REFRESH_TOKEN": self._refresh_token,
                }
            }

            async with httpx.AsyncClient(timeout=30.0) as cognito_client:
                response = await cognito_client.post(
                    auth_url,
                    json=refresh_payload,
                    headers={
                        "Content-Type": "application/x-amz-json-1.1",
                        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
                    }
                )

                if response.status_code != 200:
                    # Refresh failed, try full auth
                    return await self.authenticate()

                result = response.json().get("AuthenticationResult", {})
                self._id_token = result.get("IdToken")
                self._access_token = result.get("AccessToken")

                expires_in = result.get("ExpiresIn", 3600)
                self._token_expiry = datetime.now() + timedelta(seconds=expires_in)

                return True

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return await self.authenticate()

    def _auth_headers(self) -> dict:
        """Get headers with authentication token."""
        return {
            "authtoken": self._id_token or "",
            "Accept": "application/json",
        }

    async def discover_devices(self) -> list[dict]:
        """Discover Emporia devices associated with the account."""
        if not await self._ensure_authenticated():
            return []

        try:
            response = await self._client.get(
                "/customers/devices",
                headers=self._auth_headers(),
            )

            if response.status_code != 200:
                logger.error(f"Failed to get devices: {response.status_code}")
                return []

            data = response.json()

            # Extract customer GID for later API calls
            if data.get("customerGid"):
                self._customer_gid = data["customerGid"]

            devices = data.get("devices", [])
            self._devices = devices

            logger.info(f"Discovered {len(devices)} Emporia device(s)")
            return devices

        except Exception as e:
            logger.exception("Failed to discover Emporia devices")
            return []

    async def discover_channels(self, device_id: str) -> list[PowerChannel]:
        """Discover monitoring channels for an Emporia device."""
        if not await self._ensure_authenticated():
            return []

        channels = []

        try:
            # Find device in cached list
            device = None
            for d in self._devices:
                if str(d.get("deviceGid", "")) == device_id:
                    device = d
                    break

            if not device:
                # Refresh device list
                self._devices = await self.discover_devices()
                for d in self._devices:
                    if str(d.get("deviceGid", "")) == device_id:
                        device = d
                        break

            if not device:
                logger.warning(f"Device {device_id} not found")
                return []

            # Extract channels from device
            device_channels = device.get("channels", [])

            for ch in device_channels:
                channel_num = ch.get("channelNum", 0)
                channel_id = f"{device_id}_{channel_num}"

                # Determine channel type
                channel_type = "circuit"
                if channel_num in [1, 2, 3]:
                    channel_type = "main"  # Main phases
                elif ch.get("channelFlowDirection") == "Solar":
                    channel_type = "solar"

                power_channel = PowerChannel(
                    id=channel_id,
                    name=ch.get("name", f"Channel {channel_num}"),
                    channel_number=channel_num,
                    device_id=device_id,
                    channel_type=channel_type,
                    multiplier=ch.get("channelMultiplier", 1.0),
                    metadata={
                        "type": ch.get("type", ""),
                        "subType": ch.get("subType", ""),
                        "flowDirection": ch.get("channelFlowDirection", ""),
                    }
                )
                channels.append(power_channel)

            self.channels = channels
            logger.info(f"Discovered {len(channels)} channels for device {device_id}")
            return channels

        except Exception as e:
            logger.exception("Failed to discover channels")
            return []

    async def get_live_reading(self, channel_ids: Optional[list[str]] = None) -> list[PowerReading]:
        """Get current power readings from Emporia."""
        if not await self._ensure_authenticated():
            return []

        readings = []

        try:
            # Get instant readings for all devices
            instant_url = "/AppAPI"
            params = {
                "apiMethod": "getInstantDeviceListUsage",
                "deviceGids": ",".join(str(d.get("deviceGid", "")) for d in self._devices),
                "instant": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            response = await self._client.get(
                instant_url,
                params=params,
                headers=self._auth_headers(),
            )

            if response.status_code != 200:
                logger.error(f"Failed to get live readings: {response.status_code}")
                return []

            data = response.json()
            timestamp = datetime.now()

            # Parse device usage data
            for device_data in data.get("deviceListUsages", {}).get("devices", []):
                device_gid = str(device_data.get("deviceGid", ""))

                for channel_usage in device_data.get("channelUsages", []):
                    channel_num = channel_usage.get("channelNum", 0)
                    channel_id = f"{device_gid}_{channel_num}"

                    # Filter by requested channels
                    if channel_ids and channel_id not in channel_ids:
                        continue

                    watts = channel_usage.get("usage", 0)
                    if watts is not None:
                        readings.append(PowerReading(
                            channel_id=channel_id,
                            timestamp=timestamp,
                            watts=watts * 1000,  # Convert kW to W
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
        """Get historical power readings from Emporia."""
        if not await self._ensure_authenticated():
            return pd.DataFrame(columns=["channel_id", "timestamp", "watts", "kwh"])

        all_readings = []

        try:
            # Map scale to Emporia API scale
            scale_map = {
                MeasurementScale.SECOND: "1S",
                MeasurementScale.MINUTE: "1MIN",
                MeasurementScale.FIFTEEN_MINUTES: "15MIN",
                MeasurementScale.HOUR: "1H",
                MeasurementScale.DAY: "1D",
                MeasurementScale.WEEK: "1W",
                MeasurementScale.MONTH: "1MON",
                MeasurementScale.YEAR: "1Y",
            }
            api_scale = scale_map.get(scale, "1MIN")

            # Group channels by device
            device_channels: dict[str, list[int]] = {}
            for ch_id in channel_ids:
                parts = ch_id.split("_")
                if len(parts) >= 2:
                    device_gid = parts[0]
                    channel_num = int(parts[1])
                    if device_gid not in device_channels:
                        device_channels[device_gid] = []
                    device_channels[device_gid].append(channel_num)

            # Fetch data for each device
            for device_gid, channels in device_channels.items():
                chart_url = "/AppAPI"
                params = {
                    "apiMethod": "getChartUsage",
                    "deviceGid": device_gid,
                    "channel": ",".join(str(c) for c in channels),
                    "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "scale": api_scale,
                    "energyUnit": "KilowattHours",
                }

                response = await self._client.get(
                    chart_url,
                    params=params,
                    headers=self._auth_headers(),
                )

                if response.status_code != 200:
                    logger.warning(f"Failed to get history for device {device_gid}")
                    continue

                data = response.json()

                # Parse time series data
                for ch_num, ch_data in zip(channels, data.get("usageList", [])):
                    channel_id = f"{device_gid}_{ch_num}"

                    for usage_point in ch_data:
                        ts_str = usage_point.get("time", "")
                        kwh = usage_point.get("usage", 0)

                        if ts_str and kwh is not None:
                            try:
                                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                # Estimate watts from kWh based on scale
                                scale_hours = {
                                    "1S": 1/3600,
                                    "1MIN": 1/60,
                                    "15MIN": 0.25,
                                    "1H": 1,
                                    "1D": 24,
                                }.get(api_scale, 1)
                                watts = kwh / scale_hours * 1000 if scale_hours > 0 else 0

                                all_readings.append({
                                    "channel_id": channel_id,
                                    "timestamp": ts,
                                    "watts": watts,
                                    "kwh": kwh,
                                })
                            except ValueError:
                                continue

            if not all_readings:
                return pd.DataFrame(columns=["channel_id", "timestamp", "watts", "kwh"])

            df = pd.DataFrame(all_readings)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            logger.info(f"Fetched {len(df)} historical readings from Emporia")
            return df

        except Exception as e:
            logger.exception("Failed to get historical readings")
            return pd.DataFrame(columns=["channel_id", "timestamp", "watts", "kwh"])

    async def test_connection(self) -> tuple[bool, str]:
        """Test the Emporia connection."""
        try:
            if not self._client:
                if not await self.connect():
                    return False, self.last_error or "Failed to connect"

            if not await self._ensure_authenticated():
                return False, self.last_error or "Authentication failed"

            # Try to get devices
            devices = await self.discover_devices()

            if devices:
                device_names = [d.get("locationProperties", {}).get("deviceName", "Unknown") for d in devices]
                return True, f"Connected to Emporia. Devices: {', '.join(device_names)}"
            else:
                return True, "Connected to Emporia (no devices found)"

        except Exception as e:
            return False, str(e)

    async def get_device_status(self, device_id: Optional[str] = None) -> Optional[dict]:
        """Get status information for a device."""
        if not await self._ensure_authenticated():
            return None

        try:
            target_id = device_id or self._device_id
            if not target_id:
                return None

            # Find device in cache
            for device in self._devices:
                if str(device.get("deviceGid", "")) == target_id:
                    return {
                        "device_id": target_id,
                        "name": device.get("locationProperties", {}).get("deviceName", "Unknown"),
                        "model": device.get("model", "Unknown"),
                        "firmware": device.get("firmware", "Unknown"),
                        "channels": len(device.get("channels", [])),
                        "connected": device.get("deviceConnected", False),
                        "solar": device.get("solar", False),
                        "battery": device.get("battery", False),
                    }

            return None

        except Exception as e:
            logger.error(f"Failed to get device status: {e}")
            return None
