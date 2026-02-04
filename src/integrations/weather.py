"""Weather API connectors for external conditions data.

Supports multiple weather data providers:
- OpenWeatherMap
- Weather.gov (NOAA)
- Open-Meteo (free, no API key required)
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

import httpx
import pandas as pd

from .base import APIConnector, APIStandard, ConnectionConfig, ConnectionStatus

logger = logging.getLogger(__name__)


class WeatherProvider:
    """Weather provider identifiers."""
    OPENWEATHERMAP = "openweathermap"
    NOAA = "noaa"
    OPEN_METEO = "open_meteo"


class WeatherConnector(APIConnector):
    """Connector for weather data APIs."""

    standard = APIStandard.WEATHER

    # Provider-specific base URLs
    PROVIDER_URLS = {
        WeatherProvider.OPENWEATHERMAP: "https://api.openweathermap.org/data/2.5",
        WeatherProvider.NOAA: "https://api.weather.gov",
        WeatherProvider.OPEN_METEO: "https://api.open-meteo.com/v1",
    }

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self.provider = config.params.get("provider", WeatherProvider.OPEN_METEO)
        # Ensure proper float conversion
        self.latitude = self._to_float(config.params.get("latitude"), 0.0)
        self.longitude = self._to_float(config.params.get("longitude"), 0.0)
        self.address = config.params.get("address", "")
        self._geocoded = False

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        """Safely convert value to float."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    async def _ensure_coordinates(self) -> bool:
        """Ensure we have valid coordinates, geocoding address if needed."""
        # If we already have valid coordinates, we're good
        if self.latitude != 0.0 and self.longitude != 0.0:
            return True

        # If we have an address, try to geocode it
        if self.address and not self._geocoded:
            from .geocoding import geocode_address
            coords = await geocode_address(self.address)
            self._geocoded = True
            if coords:
                self.latitude, self.longitude = coords
                # Update config params for persistence
                self.config.params["latitude"] = self.latitude
                self.config.params["longitude"] = self.longitude
                return True

        return self.latitude != 0.0 and self.longitude != 0.0

    async def connect(self) -> bool:
        """Initialize HTTP client."""
        try:
            self.status = ConnectionStatus.CONNECTING

            # Ensure we have coordinates (geocode address if needed)
            if not await self._ensure_coordinates():
                self.last_error = "No valid coordinates or address provided"
                self.status = ConnectionStatus.ERROR
                return False

            base_url = self.config.base_url or self.PROVIDER_URLS.get(
                self.provider,
                self.PROVIDER_URLS[WeatherProvider.OPEN_METEO]
            )

            headers = {"Accept": "application/json", **self.config.headers}

            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=30.0,
            )

            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Connected to weather API: {base_url} for ({self.latitude}, {self.longitude})")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to weather API: {e}")
            self.last_error = str(e)
            self.status = ConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from API."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self.status = ConnectionStatus.DISCONNECTED

    async def test_connection(self) -> tuple[bool, str]:
        """Test connection by fetching current weather."""
        try:
            # Ensure we have coordinates
            if not await self._ensure_coordinates():
                return False, "No valid coordinates. Please provide latitude/longitude or a valid address."

            if not self._client:
                if not await self.connect():
                    return False, self.last_error or "Failed to connect"

            if self.provider == WeatherProvider.OPEN_METEO:
                response = await self._client.get(
                    "/forecast",
                    params={
                        "latitude": self.latitude,
                        "longitude": self.longitude,
                        "current_weather": "true",
                    }
                )
            elif self.provider == WeatherProvider.OPENWEATHERMAP:
                response = await self._client.get(
                    "/weather",
                    params={
                        "lat": self.latitude,
                        "lon": self.longitude,
                        "appid": self.config.api_key,
                        "units": "metric",
                    }
                )
            elif self.provider == WeatherProvider.NOAA:
                response = await self._client.get(
                    f"/points/{self.latitude},{self.longitude}"
                )
            else:
                return False, f"Unknown provider: {self.provider}"

            if response.status_code == 200:
                return True, f"Connected to {self.provider} weather API"
            else:
                return False, f"Server returned status {response.status_code}"

        except Exception as e:
            return False, str(e)

    async def fetch_points(self) -> list[dict]:
        """Return available weather metrics as points."""
        # Weather APIs don't have discrete points like BMS
        # Return standard weather metrics
        return [
            {"id": "temperature", "name": "Temperature", "type": "temperature", "unit": "°C"},
            {"id": "humidity", "name": "Relative Humidity", "type": "humidity", "unit": "%"},
            {"id": "pressure", "name": "Barometric Pressure", "type": "pressure", "unit": "hPa"},
            {"id": "wind_speed", "name": "Wind Speed", "type": "wind", "unit": "m/s"},
            {"id": "wind_direction", "name": "Wind Direction", "type": "wind", "unit": "°"},
            {"id": "cloud_cover", "name": "Cloud Cover", "type": "solar", "unit": "%"},
            {"id": "precipitation", "name": "Precipitation", "type": "precipitation", "unit": "mm"},
            {"id": "solar_radiation", "name": "Solar Radiation", "type": "solar", "unit": "W/m²"},
        ]

    async def fetch_readings(
        self,
        point_ids: list[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch historical weather readings."""
        try:
            if not self._client:
                await self.connect()

            if self.provider == WeatherProvider.OPEN_METEO:
                return await self._fetch_open_meteo(point_ids, start_time, end_time)
            elif self.provider == WeatherProvider.OPENWEATHERMAP:
                return await self._fetch_openweathermap(point_ids, start_time, end_time)
            else:
                logger.warning(f"Historical data not supported for {self.provider}")
                return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

        except Exception as e:
            logger.exception("Error fetching weather readings")
            self.last_error = str(e)
            return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

    async def _fetch_open_meteo(
        self,
        point_ids: list[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch from Open-Meteo API (free, no API key)."""
        # Map our point IDs to Open-Meteo parameters
        param_map = {
            "temperature": "temperature_2m",
            "humidity": "relativehumidity_2m",
            "pressure": "surface_pressure",
            "wind_speed": "windspeed_10m",
            "wind_direction": "winddirection_10m",
            "cloud_cover": "cloudcover",
            "precipitation": "precipitation",
            "solar_radiation": "shortwave_radiation",
        }

        hourly_params = [param_map[p] for p in point_ids if p in param_map]

        if not hourly_params:
            return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

        response = await self._client.get(
            "/forecast",
            params={
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ",".join(hourly_params),
                "start_date": start_time.strftime("%Y-%m-%d"),
                "end_date": end_time.strftime("%Y-%m-%d"),
                "timezone": "UTC",
            }
        )

        if response.status_code != 200:
            logger.error(f"Open-Meteo API error: {response.status_code}")
            return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

        data = response.json()
        hourly = data.get("hourly", {})
        timestamps = hourly.get("time", [])

        all_readings = []
        reverse_map = {v: k for k, v in param_map.items()}
        points = await self.fetch_points()
        unit_map = {p["id"]: p["unit"] for p in points}

        for param, values in hourly.items():
            if param == "time":
                continue

            point_id = reverse_map.get(param)
            if not point_id or point_id not in point_ids:
                continue

            for ts, val in zip(timestamps, values):
                if val is not None:
                    all_readings.append({
                        "point_id": point_id,
                        "timestamp": ts,
                        "value": val,
                        "unit": unit_map.get(point_id, ""),
                    })

        if not all_readings:
            return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

        df = pd.DataFrame(all_readings)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logger.info(f"Fetched {len(df)} weather readings from Open-Meteo")
        return df

    async def _fetch_openweathermap(
        self,
        point_ids: list[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch from OpenWeatherMap API (requires API key)."""
        if not self.config.api_key:
            logger.error("OpenWeatherMap requires an API key")
            return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

        # OpenWeatherMap historical data requires paid subscription
        # Using the free 5-day forecast instead if within range
        all_readings = []

        response = await self._client.get(
            "/forecast",
            params={
                "lat": self.latitude,
                "lon": self.longitude,
                "appid": self.config.api_key,
                "units": "metric",
            }
        )

        if response.status_code != 200:
            logger.error(f"OpenWeatherMap API error: {response.status_code}")
            return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

        data = response.json()
        points = await self.fetch_points()
        unit_map = {p["id"]: p["unit"] for p in points}

        for item in data.get("list", []):
            timestamp = datetime.fromtimestamp(item["dt"])

            if timestamp < start_time or timestamp > end_time:
                continue

            main = item.get("main", {})
            wind = item.get("wind", {})
            clouds = item.get("clouds", {})
            rain = item.get("rain", {})

            readings_map = {
                "temperature": main.get("temp"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "wind_speed": wind.get("speed"),
                "wind_direction": wind.get("deg"),
                "cloud_cover": clouds.get("all"),
                "precipitation": rain.get("3h", 0),
            }

            for point_id, value in readings_map.items():
                if point_id in point_ids and value is not None:
                    all_readings.append({
                        "point_id": point_id,
                        "timestamp": timestamp,
                        "value": value,
                        "unit": unit_map.get(point_id, ""),
                    })

        if not all_readings:
            return pd.DataFrame(columns=["point_id", "timestamp", "value", "unit"])

        df = pd.DataFrame(all_readings)
        logger.info(f"Fetched {len(df)} weather readings from OpenWeatherMap")
        return df

    async def get_current_weather(self) -> Optional[dict]:
        """Get current weather conditions."""
        try:
            if not self._client:
                await self.connect()

            if self.provider == WeatherProvider.OPEN_METEO:
                response = await self._client.get(
                    "/forecast",
                    params={
                        "latitude": self.latitude,
                        "longitude": self.longitude,
                        "current_weather": "true",
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    current = data.get("current_weather", {})
                    return {
                        "timestamp": current.get("time"),
                        "temperature": current.get("temperature"),
                        "wind_speed": current.get("windspeed"),
                        "wind_direction": current.get("winddirection"),
                        "weather_code": current.get("weathercode"),
                    }

            elif self.provider == WeatherProvider.OPENWEATHERMAP:
                response = await self._client.get(
                    "/weather",
                    params={
                        "lat": self.latitude,
                        "lon": self.longitude,
                        "appid": self.config.api_key,
                        "units": "metric",
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "timestamp": datetime.fromtimestamp(data["dt"]).isoformat(),
                        "temperature": data["main"]["temp"],
                        "humidity": data["main"]["humidity"],
                        "pressure": data["main"]["pressure"],
                        "wind_speed": data["wind"]["speed"],
                        "wind_direction": data["wind"].get("deg"),
                        "description": data["weather"][0]["description"],
                    }

            return None

        except Exception as e:
            logger.error(f"Error fetching current weather: {e}")
            return None
