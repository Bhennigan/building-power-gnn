"""Geocoding service for converting addresses to coordinates.

Uses free Nominatim (OpenStreetMap) API for geocoding.
"""

from typing import Optional, Tuple
import logging

import httpx

logger = logging.getLogger(__name__)

# Nominatim API (free, no API key required)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


async def geocode_address(address: str) -> Optional[Tuple[float, float]]:
    """Convert an address to latitude/longitude coordinates.

    Args:
        address: Street address to geocode

    Returns:
        Tuple of (latitude, longitude) or None if not found
    """
    if not address or not address.strip():
        return None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                NOMINATIM_URL,
                params={
                    "q": address,
                    "format": "json",
                    "limit": 1,
                },
                headers={
                    "User-Agent": "PowerGraphAI/1.0 (Building Energy Management)"
                }
            )

            if response.status_code != 200:
                logger.warning(f"Geocoding failed with status {response.status_code}")
                return None

            results = response.json()

            if not results:
                logger.warning(f"No geocoding results for address: {address}")
                return None

            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])

            logger.info(f"Geocoded '{address}' to ({lat}, {lon})")
            return (lat, lon)

    except Exception as e:
        logger.error(f"Geocoding error for '{address}': {e}")
        return None


async def reverse_geocode(latitude: float, longitude: float) -> Optional[str]:
    """Convert coordinates to an address.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Address string or None if not found
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "format": "json",
                },
                headers={
                    "User-Agent": "PowerGraphAI/1.0 (Building Energy Management)"
                }
            )

            if response.status_code != 200:
                return None

            data = response.json()
            return data.get("display_name")

    except Exception as e:
        logger.error(f"Reverse geocoding error: {e}")
        return None
