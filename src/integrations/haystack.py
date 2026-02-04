"""Project Haystack API connector.

Project Haystack is an open source initiative to standardize semantic data models
and web services for building equipment and operational data.

Reference: https://project-haystack.org/
"""

from datetime import datetime
from typing import Optional
import logging

import httpx
import pandas as pd

from .base import APIConnector, APIStandard, ConnectionConfig, ConnectionStatus

logger = logging.getLogger(__name__)


class HaystackConnector(APIConnector):
    """Connector for Project Haystack REST API."""

    standard = APIStandard.HAYSTACK

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._auth_token: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to Haystack server."""
        try:
            self.status = ConnectionStatus.CONNECTING

            headers = {"Accept": "application/json", **self.config.headers}

            # Handle authentication
            if self.config.auth_type == "bearer" and self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            elif self.config.auth_type == "api_key" and self.config.api_key:
                headers["X-API-Key"] = self.config.api_key

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=30.0,
            )

            # Authenticate if using SCRAM (Haystack 4.0)
            if self.config.auth_type == "scram" and self.config.username:
                await self._authenticate_scram()

            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Haystack server: {self.config.base_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Haystack: {e}")
            self.last_error = str(e)
            self.status = ConnectionStatus.ERROR
            return False

    async def _authenticate_scram(self) -> None:
        """Authenticate using SCRAM mechanism (Haystack 4.0)."""
        # SCRAM authentication handshake
        # Step 1: Hello
        response = await self._client.get(
            "/about",
            headers={"Authorization": f"HELLO username={self.config.username}"}
        )

        if response.status_code == 401:
            # Parse challenge and complete authentication
            # This is a simplified version - full SCRAM would need proper implementation
            auth_header = response.headers.get("WWW-Authenticate", "")
            logger.debug(f"SCRAM challenge: {auth_header}")
            # For now, fall back to basic auth if SCRAM fails
            if self.config.password:
                import base64
                creds = base64.b64encode(
                    f"{self.config.username}:{self.config.password}".encode()
                ).decode()
                self._client.headers["Authorization"] = f"Basic {creds}"

    async def disconnect(self) -> None:
        """Disconnect from server."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self.status = ConnectionStatus.DISCONNECTED
        logger.info(f"Disconnected from Haystack server: {self.config.base_url}")

    async def test_connection(self) -> tuple[bool, str]:
        """Test connection by calling the about endpoint."""
        try:
            if not self._client:
                await self.connect()

            response = await self._client.get("/about")

            if response.status_code == 200:
                data = response.json()
                vendor = data.get("vendorName", "Unknown")
                version = data.get("haystackVersion", "Unknown")
                return True, f"Connected to {vendor} (Haystack {version})"
            else:
                return False, f"Server returned status {response.status_code}"

        except Exception as e:
            return False, str(e)

    async def fetch_points(self) -> list[dict]:
        """Fetch all points (sensors/equipment) from the server."""
        try:
            if not self._client:
                await self.connect()

            # Query for points with sensor or equip markers
            filter_query = "point or sensor or equip"
            response = await self._client.get(
                "/read",
                params={"filter": filter_query}
            )

            if response.status_code != 200:
                logger.error(f"Failed to fetch points: {response.status_code}")
                return []

            data = response.json()
            rows = data.get("rows", [])

            points = []
            for row in rows:
                point = {
                    "id": row.get("id", "").replace("@", ""),
                    "name": row.get("dis", row.get("navName", "")),
                    "type": self._determine_point_type(row),
                    "unit": row.get("unit", ""),
                    "kind": row.get("kind", ""),
                    "tags": [k for k, v in row.items() if v == "m:"],  # Marker tags
                }
                points.append(point)

            logger.info(f"Fetched {len(points)} points from Haystack")
            return points

        except Exception as e:
            logger.exception("Error fetching points from Haystack")
            self.last_error = str(e)
            return []

    def _determine_point_type(self, row: dict) -> str:
        """Determine point type from Haystack tags."""
        if row.get("temp"):
            return "temperature"
        elif row.get("humidity"):
            return "humidity"
        elif row.get("power") or row.get("elec"):
            return "power"
        elif row.get("energy"):
            return "energy"
        elif row.get("occupancy") or row.get("occ"):
            return "occupancy"
        elif row.get("co2"):
            return "co2"
        elif row.get("flow"):
            return "flow"
        elif row.get("pressure"):
            return "pressure"
        elif row.get("sp") or row.get("setpoint"):
            return "setpoint"
        elif row.get("cmd") or row.get("command"):
            return "command"
        else:
            return "unknown"

    async def fetch_readings(
        self,
        point_ids: list[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch historical readings for points."""
        try:
            if not self._client:
                await self.connect()

            all_readings = []

            # Haystack hisRead takes one point at a time typically
            for point_id in point_ids:
                try:
                    # Format times as Haystack datetime strings
                    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
                    end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")

                    response = await self._client.get(
                        "/hisRead",
                        params={
                            "id": f"@{point_id}",
                            "range": f"{start_str},{end_str}"
                        }
                    )

                    if response.status_code != 200:
                        continue

                    data = response.json()
                    rows = data.get("rows", [])

                    for row in rows:
                        ts = row.get("ts", "")
                        val = row.get("val")

                        # Parse Haystack number format (e.g., "72.5°F")
                        if isinstance(val, str) and val:
                            import re
                            match = re.match(r"^([\d.-]+)", val)
                            if match:
                                val = float(match.group(1))

                        if val is not None:
                            all_readings.append({
                                "point_id": point_id,
                                "timestamp": ts,
                                "value": val,
                                "unit": row.get("unit", ""),
                            })

                except Exception as e:
                    logger.warning(f"Error fetching readings for {point_id}: {e}")
                    continue

            if not all_readings:
                return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

            df = pd.DataFrame(all_readings)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            logger.info(f"Fetched {len(df)} readings from Haystack")
            return df

        except Exception as e:
            logger.exception("Error fetching readings from Haystack")
            self.last_error = str(e)
            return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

    async def write_point(self, point_id: str, value: float, level: int = 16) -> bool:
        """Write a value to a writable point.

        Args:
            point_id: The point ID to write to
            value: The value to write
            level: Priority level (1-17, default 16)

        Returns:
            True if successful
        """
        try:
            if not self._client:
                await self.connect()

            response = await self._client.post(
                "/pointWrite",
                json={
                    "id": f"@{point_id}",
                    "level": level,
                    "val": value,
                }
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Error writing to point {point_id}: {e}")
            return False
