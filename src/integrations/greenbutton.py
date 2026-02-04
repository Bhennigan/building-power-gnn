"""Green Button API connector.

Green Button is a standard for utility energy data based on the
Energy Services Provider Interface (ESPI) standard.

Reference: https://www.greenbuttondata.org/
"""

from datetime import datetime
from typing import Optional
import logging
from xml.etree import ElementTree as ET

import httpx
import pandas as pd

from .base import APIConnector, APIStandard, ConnectionConfig, ConnectionStatus

logger = logging.getLogger(__name__)

# Green Button/ESPI namespaces
ATOM_NS = "http://www.w3.org/2005/Atom"
ESPI_NS = "http://naesb.org/espi"


class GreenButtonConnector(APIConnector):
    """Connector for Green Button (ESPI) API."""

    standard = APIStandard.GREENBUTTON

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._subscription_id: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to Green Button Data Custodian."""
        try:
            self.status = ConnectionStatus.CONNECTING

            headers = {
                "Accept": "application/atom+xml",
                **self.config.headers
            }

            # Handle OAuth2 bearer token
            if self.config.auth_type == "bearer" and self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            elif self.config.auth_type == "oauth2":
                # Would need OAuth2 flow implementation
                pass

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=30.0,
            )

            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to Green Button API: {self.config.base_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Green Button: {e}")
            self.last_error = str(e)
            self.status = ConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from server."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self.status = ConnectionStatus.DISCONNECTED

    async def test_connection(self) -> tuple[bool, str]:
        """Test connection by fetching service status."""
        try:
            if not self._client:
                await self.connect()

            # Try to fetch the application information
            response = await self._client.get("/espi/1_1/resource/ApplicationInformation")

            if response.status_code == 200:
                return True, "Connected to Green Button Data Custodian"
            elif response.status_code == 401:
                return False, "Authentication failed - check API credentials"
            else:
                return False, f"Server returned status {response.status_code}"

        except Exception as e:
            return False, str(e)

    async def fetch_points(self) -> list[dict]:
        """Fetch usage points (meters) from the subscription."""
        try:
            if not self._client:
                await self.connect()

            # Fetch usage points
            response = await self._client.get("/espi/1_1/resource/UsagePoint")

            if response.status_code != 200:
                logger.error(f"Failed to fetch usage points: {response.status_code}")
                return []

            # Parse ATOM/ESPI XML
            root = ET.fromstring(response.content)
            points = []

            for entry in root.findall(f".//{{{ATOM_NS}}}entry"):
                content = entry.find(f"{{{ATOM_NS}}}content")
                if content is None:
                    continue

                usage_point = content.find(f".//{{{ESPI_NS}}}UsagePoint")
                if usage_point is None:
                    continue

                # Extract usage point details
                point_id = entry.find(f"{{{ATOM_NS}}}id")
                title = entry.find(f"{{{ATOM_NS}}}title")
                service_kind = usage_point.find(f"{{{ESPI_NS}}}ServiceCategory/{{{ESPI_NS}}}kind")

                kind_map = {
                    "0": "electricity",
                    "1": "gas",
                    "2": "water",
                    "3": "time",
                    "4": "heat",
                    "5": "refuse",
                    "6": "sewerage",
                    "7": "rates",
                    "8": "tvLicense",
                    "9": "internet",
                }

                service_type = "unknown"
                if service_kind is not None and service_kind.text:
                    service_type = kind_map.get(service_kind.text, "unknown")

                points.append({
                    "id": point_id.text if point_id is not None else "",
                    "name": title.text if title is not None else "Usage Point",
                    "type": service_type,
                    "unit": "kWh" if service_type == "electricity" else "",
                    "kind": "meter",
                })

            logger.info(f"Fetched {len(points)} usage points from Green Button")
            return points

        except Exception as e:
            logger.exception("Error fetching usage points from Green Button")
            self.last_error = str(e)
            return []

    async def fetch_readings(
        self,
        point_ids: list[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch interval readings for usage points."""
        try:
            if not self._client:
                await self.connect()

            all_readings = []

            for point_id in point_ids:
                try:
                    # Fetch meter readings
                    url = f"/espi/1_1/resource/UsagePoint/{point_id}/MeterReading"
                    response = await self._client.get(url)

                    if response.status_code != 200:
                        continue

                    root = ET.fromstring(response.content)

                    # Find interval blocks
                    for block in root.findall(f".//{{{ESPI_NS}}}IntervalBlock"):
                        for reading in block.findall(f"{{{ESPI_NS}}}IntervalReading"):
                            time_period = reading.find(f"{{{ESPI_NS}}}timePeriod")
                            value_elem = reading.find(f"{{{ESPI_NS}}}value")

                            if time_period is None or value_elem is None:
                                continue

                            start_elem = time_period.find(f"{{{ESPI_NS}}}start")
                            if start_elem is None:
                                continue

                            # ESPI uses Unix timestamps
                            timestamp = datetime.fromtimestamp(int(start_elem.text))

                            # Filter by time range
                            if timestamp < start_time or timestamp > end_time:
                                continue

                            # Value is in base units (Wh for electricity)
                            value = float(value_elem.text) / 1000  # Convert to kWh

                            all_readings.append({
                                "point_id": point_id,
                                "timestamp": timestamp,
                                "value": value,
                                "unit": "kWh",
                            })

                except Exception as e:
                    logger.warning(f"Error fetching readings for {point_id}: {e}")
                    continue

            if not all_readings:
                return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

            df = pd.DataFrame(all_readings)
            df = df.sort_values(["point_id", "timestamp"])
            logger.info(f"Fetched {len(df)} readings from Green Button")
            return df

        except Exception as e:
            logger.exception("Error fetching readings from Green Button")
            self.last_error = str(e)
            return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

    async def fetch_usage_summary(self, point_id: str) -> Optional[dict]:
        """Fetch usage summary for a usage point."""
        try:
            if not self._client:
                await self.connect()

            url = f"/espi/1_1/resource/UsagePoint/{point_id}/UsageSummary"
            response = await self._client.get(url)

            if response.status_code != 200:
                return None

            root = ET.fromstring(response.content)
            summary = root.find(f".//{{{ESPI_NS}}}UsageSummary")

            if summary is None:
                return None

            billing_period = summary.find(f"{{{ESPI_NS}}}billingPeriod")
            overall_consumption = summary.find(f"{{{ESPI_NS}}}overallConsumptionLastPeriod")

            result = {
                "point_id": point_id,
            }

            if billing_period is not None:
                start = billing_period.find(f"{{{ESPI_NS}}}start")
                duration = billing_period.find(f"{{{ESPI_NS}}}duration")
                if start is not None:
                    result["billing_start"] = datetime.fromtimestamp(int(start.text))
                if duration is not None:
                    result["billing_duration_seconds"] = int(duration.text)

            if overall_consumption is not None:
                power_of_ten = int(overall_consumption.find(f"{{{ESPI_NS}}}powerOfTenMultiplier").text or 0)
                value = int(overall_consumption.find(f"{{{ESPI_NS}}}value").text or 0)
                result["total_consumption_kwh"] = value * (10 ** power_of_ten) / 1000

            return result

        except Exception as e:
            logger.error(f"Error fetching usage summary: {e}")
            return None
