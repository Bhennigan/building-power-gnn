"""API routes for managing external integrations."""

from datetime import datetime
from typing import Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...db import get_db, crud
from ...auth.dependencies import get_current_user
from ...db.models import User
from ...integrations import (
    IntegrationManager,
    ConnectionConfig,
    APIStandard,
    get_integration_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class IntegrationCreate(BaseModel):
    """Request to create a new integration."""
    name: str = Field(..., min_length=1, max_length=100)
    api_standard: str = Field(..., description="API standard: haystack, greenbutton, weather")
    base_url: str = Field(..., min_length=1)
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    auth_type: str = Field(default="none", description="none, api_key, basic, bearer, oauth2")
    sync_interval_minutes: int = Field(default=15, ge=1, le=1440)
    enabled: bool = True
    building_id: Optional[str] = None
    params: dict = Field(default_factory=dict, description="Provider-specific parameters")


class IntegrationResponse(BaseModel):
    """Integration details response."""
    id: str
    name: str
    api_standard: str
    base_url: str
    auth_type: str
    status: str
    last_sync: Optional[str]
    last_error: Optional[str]
    enabled: bool
    sync_interval_minutes: int


class IntegrationTestResult(BaseModel):
    """Result of connection test."""
    success: bool
    message: str


class SyncResultResponse(BaseModel):
    """Result of a sync operation."""
    success: bool
    timestamp: str
    records_fetched: int
    records_stored: int
    errors: list[str]
    warnings: list[str]
    duration_seconds: float


class WeatherConfigRequest(BaseModel):
    """Configuration for weather integration."""
    name: str = Field(default="Weather Data")
    provider: str = Field(default="open_meteo", description="open_meteo, openweathermap, noaa")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    address: Optional[str] = Field(None, description="Street address to geocode (alternative to lat/lon)")
    api_key: Optional[str] = Field(None, description="Required for openweathermap")
    sync_interval_minutes: int = Field(default=60, ge=15, le=1440)
    building_id: Optional[str] = Field(None, description="Building ID to get address from")


@router.get("", response_model=list[IntegrationResponse])
async def list_integrations(
    user: User = Depends(get_current_user),
):
    """List all configured integrations."""
    manager = get_integration_manager()
    connectors = manager.list_connectors()

    return [
        IntegrationResponse(
            id=c.get("id", ""),
            name=c["name"],
            api_standard=c["standard"],
            base_url="",  # Don't expose URL in list
            auth_type="",
            status=c["status"],
            last_sync=c["last_sync"],
            last_error=c["last_error"],
            enabled=c["enabled"],
            sync_interval_minutes=15,
        )
        for c in connectors
    ]


@router.post("", response_model=IntegrationResponse)
async def create_integration(
    request: IntegrationCreate,
    user: User = Depends(get_current_user),
):
    """Create a new integration connection."""
    try:
        api_standard = APIStandard(request.api_standard.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid API standard. Supported: {[s.value for s in APIStandard]}"
        )

    config = ConnectionConfig(
        name=request.name,
        api_standard=api_standard,
        base_url=request.base_url,
        api_key=request.api_key,
        username=request.username,
        password=request.password,
        auth_type=request.auth_type,
        sync_interval_minutes=request.sync_interval_minutes,
        enabled=request.enabled,
        building_id=request.building_id,
        params=request.params,
    )

    manager = get_integration_manager()

    try:
        connector_id = manager.register_connector(config)
        connector = manager.get_connector(connector_id)

        return IntegrationResponse(
            id=connector_id,
            name=config.name,
            api_standard=config.api_standard.value,
            base_url=config.base_url,
            auth_type=config.auth_type,
            status=connector.status.value,
            last_sync=None,
            last_error=None,
            enabled=config.enabled,
            sync_interval_minutes=config.sync_interval_minutes,
        )

    except Exception as e:
        logger.exception("Failed to create integration")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weather", response_model=IntegrationResponse)
async def create_weather_integration(
    request: WeatherConfigRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a weather data integration (simplified setup).

    You can provide:
    - latitude/longitude directly
    - address to geocode
    - building_id to use the building's address
    """
    provider_urls = {
        "open_meteo": "https://api.open-meteo.com/v1",
        "openweathermap": "https://api.openweathermap.org/data/2.5",
        "noaa": "https://api.weather.gov",
    }

    if request.provider not in provider_urls:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider. Supported: {list(provider_urls.keys())}"
        )

    if request.provider == "openweathermap" and not request.api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenWeatherMap requires an API key"
        )

    # Determine address/coordinates
    address = request.address
    latitude = request.latitude
    longitude = request.longitude
    building_name = None

    # If building_id provided, get address from building
    if request.building_id:
        building = crud.get_building(db, request.building_id)
        if building:
            if building.address:
                address = building.address
            building_name = building.name

    # Need either coordinates or an address
    if not address and (latitude is None or longitude is None):
        raise HTTPException(
            status_code=400,
            detail="Please provide latitude/longitude, an address, or a building_id with an address"
        )

    # Generate name if not provided
    name = request.name
    if name == "Weather Data" and building_name:
        name = f"Weather - {building_name}"

    config = ConnectionConfig(
        name=name,
        api_standard=APIStandard.WEATHER,
        base_url=provider_urls[request.provider],
        api_key=request.api_key,
        auth_type="api_key" if request.api_key else "none",
        sync_interval_minutes=request.sync_interval_minutes,
        enabled=True,
        building_id=request.building_id,
        params={
            "provider": request.provider,
            "latitude": latitude or 0.0,
            "longitude": longitude or 0.0,
            "address": address or "",
        },
    )

    manager = get_integration_manager()
    connector_id = manager.register_connector(config)
    connector = manager.get_connector(connector_id)

    return IntegrationResponse(
        id=connector_id,
        name=config.name,
        api_standard=config.api_standard.value,
        base_url=config.base_url,
        auth_type=config.auth_type,
        status=connector.status.value,
        last_sync=None,
        last_error=None,
        enabled=config.enabled,
        sync_interval_minutes=config.sync_interval_minutes,
    )


class PowerMonitorConfigRequest(BaseModel):
    """Configuration for power monitor integration."""
    name: str = Field(default="Power Monitor")
    provider: str = Field(default="emporia", description="emporia, iotawatt, shelly, home_assistant, generic")
    email: Optional[str] = Field(None, description="Account email (for Emporia)")
    password: Optional[str] = Field(None, description="Account password (for Emporia)")
    base_url: Optional[str] = Field(None, description="Device/API base URL (for generic/local devices)")
    api_key: Optional[str] = Field(None, description="API key if required")
    auth_type: str = Field(default="none", description="none, api_key, basic, bearer")
    sync_interval_minutes: int = Field(default=5, ge=1, le=1440)
    building_id: Optional[str] = Field(None, description="Building to associate with")
    device_preset: Optional[str] = Field(None, description="Use preset config: iotawatt, shelly, home_assistant")
    custom_endpoints: Optional[dict] = Field(None, description="Custom API endpoint paths")
    custom_mapping: Optional[dict] = Field(None, description="Custom data field mapping")


class LiveReadingResponse(BaseModel):
    """Live power reading response."""
    channel_id: str
    channel_name: Optional[str] = None
    watts: float
    kwh: Optional[float] = None
    voltage: Optional[float] = None
    current: Optional[float] = None
    timestamp: str


@router.post("/power-monitor", response_model=IntegrationResponse)
async def create_power_monitor_integration(
    request: PowerMonitorConfigRequest,
    user: User = Depends(get_current_user),
):
    """Create a power monitoring integration (Emporia, IoTaWatt, Shelly, etc.).

    For Emporia:
    - Provide email and password for your Emporia account

    For local devices (IoTaWatt, Shelly):
    - Provide base_url pointing to the device

    For Home Assistant:
    - Provide base_url and api_key (long-lived access token)
    """
    from ...integrations.generic_power import get_preset_config

    # Build params
    params = {
        "provider": request.provider.lower(),
        "email": request.email,
        "password": request.password,
    }

    # Apply preset config if specified
    if request.device_preset:
        preset = get_preset_config(request.device_preset)
        if preset:
            params["endpoints"] = preset.get("endpoints", {})
            params["data_mapping"] = preset.get("data_mapping", {})
            params["response_config"] = preset.get("response_config", {})

    # Apply custom overrides
    if request.custom_endpoints:
        params["endpoints"] = {**params.get("endpoints", {}), **request.custom_endpoints}
    if request.custom_mapping:
        params["data_mapping"] = {**params.get("data_mapping", {}), **request.custom_mapping}

    # Determine base URL
    base_url = request.base_url or ""
    if request.provider.lower() == "emporia":
        base_url = "https://api.emporiaenergy.com"

    config = ConnectionConfig(
        name=request.name,
        api_standard=APIStandard.POWER_MONITOR,
        base_url=base_url,
        api_key=request.api_key,
        username=request.email,
        password=request.password,
        auth_type=request.auth_type,
        sync_interval_minutes=request.sync_interval_minutes,
        enabled=True,
        building_id=request.building_id,
        params=params,
    )

    manager = get_integration_manager()

    try:
        connector_id = manager.register_connector(config)
        connector = manager.get_connector(connector_id)

        return IntegrationResponse(
            id=connector_id,
            name=config.name,
            api_standard=config.api_standard.value,
            base_url=config.base_url,
            auth_type=config.auth_type,
            status=connector.status.value,
            last_sync=None,
            last_error=None,
            enabled=config.enabled,
            sync_interval_minutes=config.sync_interval_minutes,
        )

    except Exception as e:
        logger.exception("Failed to create power monitor integration")
        raise HTTPException(status_code=500, detail=str(e))


# Supported API standards info
@router.get("/standards/info")
async def get_supported_standards():
    """Get information about supported API standards."""
    return {
        "standards": [
            {
                "id": "haystack",
                "name": "Project Haystack",
                "description": "Open source standard for semantic data models in smart buildings",
                "auth_types": ["none", "basic", "bearer", "scram"],
                "website": "https://project-haystack.org/",
            },
            {
                "id": "greenbutton",
                "name": "Green Button",
                "description": "Standard for utility energy usage data (ESPI)",
                "auth_types": ["bearer", "oauth2"],
                "website": "https://www.greenbuttondata.org/",
            },
            {
                "id": "weather",
                "name": "Weather APIs",
                "description": "External weather data (Open-Meteo, OpenWeatherMap, NOAA)",
                "auth_types": ["none", "api_key"],
                "providers": ["open_meteo", "openweathermap", "noaa"],
            },
            {
                "id": "power_monitor",
                "name": "Power Monitors",
                "description": "Real-time energy monitoring devices (Emporia, IoTaWatt, Shelly, etc.)",
                "auth_types": ["none", "api_key", "basic", "credentials"],
                "providers": ["emporia", "iotawatt", "shelly", "home_assistant", "generic"],
            },
        ]
    }


@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    user: User = Depends(get_current_user),
):
    """Get integration details."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    return IntegrationResponse(
        id=integration_id,
        name=connector.config.name,
        api_standard=connector.config.api_standard.value,
        base_url=connector.config.base_url,
        auth_type=connector.config.auth_type,
        status=connector.status.value,
        last_sync=connector.last_sync.isoformat() if connector.last_sync else None,
        last_error=connector.last_error,
        enabled=connector.config.enabled,
        sync_interval_minutes=connector.config.sync_interval_minutes,
    )


@router.delete("/{integration_id}")
async def delete_integration(
    integration_id: str,
    user: User = Depends(get_current_user),
):
    """Delete an integration."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    await connector.disconnect()
    manager.unregister_connector(integration_id)

    return {"message": "Integration deleted"}


@router.post("/{integration_id}/test", response_model=IntegrationTestResult)
async def test_integration(
    integration_id: str,
    user: User = Depends(get_current_user),
):
    """Test an integration connection."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    success, message = await connector.test_connection()

    return IntegrationTestResult(success=success, message=message)


@router.post("/{integration_id}/sync", response_model=SyncResultResponse)
async def sync_integration(
    integration_id: str,
    background_tasks: BackgroundTasks,
    since: Optional[datetime] = None,
    user: User = Depends(get_current_user),
):
    """Trigger a manual sync for an integration."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    result = await connector.sync(since)

    return SyncResultResponse(
        success=result.success,
        timestamp=result.timestamp.isoformat(),
        records_fetched=result.records_fetched,
        records_stored=result.records_stored,
        errors=result.errors,
        warnings=result.warnings,
        duration_seconds=result.duration_seconds,
    )


@router.get("/{integration_id}/points")
async def get_integration_points(
    integration_id: str,
    user: User = Depends(get_current_user),
):
    """Fetch available data points from an integration."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    try:
        points = await connector.fetch_points()
        return {"points": points, "count": len(points)}
    except Exception as e:
        logger.exception("Failed to fetch points")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{integration_id}/enable")
async def enable_integration(
    integration_id: str,
    user: User = Depends(get_current_user),
):
    """Enable an integration."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    connector.config.enabled = True
    return {"message": "Integration enabled"}


@router.post("/{integration_id}/disable")
async def disable_integration(
    integration_id: str,
    user: User = Depends(get_current_user),
):
    """Disable an integration."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    connector.config.enabled = False
    await connector.disconnect()
    return {"message": "Integration disabled"}


@router.get("/{integration_id}/live", response_model=list[LiveReadingResponse])
async def get_live_readings(
    integration_id: str,
    user: User = Depends(get_current_user),
):
    """Get live/current power readings from a power monitor integration."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    # Check if it's a power monitor connector
    if not hasattr(connector, 'get_live_reading'):
        raise HTTPException(
            status_code=400,
            detail="This integration does not support live readings"
        )

    try:
        readings = await connector.get_live_reading()

        # Get channel names if available
        channel_names = {}
        if hasattr(connector, 'channels'):
            channel_names = {ch.id: ch.name for ch in connector.channels}

        return [
            LiveReadingResponse(
                channel_id=r.channel_id,
                channel_name=channel_names.get(r.channel_id),
                watts=r.watts,
                kwh=r.kwh,
                voltage=r.voltage,
                current=r.current,
                timestamp=r.timestamp.isoformat(),
            )
            for r in readings
        ]

    except Exception as e:
        logger.exception("Failed to get live readings")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{integration_id}/devices")
async def get_power_monitor_devices(
    integration_id: str,
    user: User = Depends(get_current_user),
):
    """Get discovered devices from a power monitor integration."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    if not hasattr(connector, 'discover_devices'):
        raise HTTPException(
            status_code=400,
            detail="This integration does not support device discovery"
        )

    try:
        devices = await connector.discover_devices()
        return {"devices": devices, "count": len(devices)}

    except Exception as e:
        logger.exception("Failed to discover devices")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{integration_id}/channels")
async def get_power_monitor_channels(
    integration_id: str,
    device_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Get monitoring channels from a power monitor integration."""
    manager = get_integration_manager()
    connector = manager.get_connector(integration_id)

    if not connector:
        raise HTTPException(status_code=404, detail="Integration not found")

    if not hasattr(connector, 'discover_channels'):
        raise HTTPException(
            status_code=400,
            detail="This integration does not support channel discovery"
        )

    try:
        # Use provided device_id or the connector's default
        target_device = device_id or getattr(connector, '_device_id', None)

        if not target_device:
            # Try to get first device
            if hasattr(connector, 'discover_devices'):
                devices = await connector.discover_devices()
                if devices:
                    target_device = str(devices[0].get("deviceGid", devices[0].get("id", "")))

        if not target_device:
            return {"channels": [], "count": 0, "message": "No device found"}

        channels = await connector.discover_channels(target_device)

        return {
            "channels": [
                {
                    "id": ch.id,
                    "name": ch.name,
                    "channel_number": ch.channel_number,
                    "device_id": ch.device_id,
                    "type": ch.channel_type,
                    "multiplier": ch.multiplier,
                }
                for ch in channels
            ],
            "count": len(channels),
            "device_id": target_device,
        }

    except Exception as e:
        logger.exception("Failed to discover channels")
        raise HTTPException(status_code=500, detail=str(e))
