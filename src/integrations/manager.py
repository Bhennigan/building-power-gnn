"""Integration manager for handling multiple API connections."""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
import logging

from .base import APIConnector, APIStandard, ConnectionConfig, SyncResult, ConnectionStatus
from .haystack import HaystackConnector
from .greenbutton import GreenButtonConnector
from .weather import WeatherConnector
from .emporia import EmporiaConnector
from .generic_power import GenericPowerConnector

logger = logging.getLogger(__name__)


class IntegrationManager:
    """Manages multiple API integrations and synchronization."""

    # Map of API standards to connector classes
    CONNECTOR_CLASSES = {
        APIStandard.HAYSTACK: HaystackConnector,
        APIStandard.GREENBUTTON: GreenButtonConnector,
        APIStandard.WEATHER: WeatherConnector,
        APIStandard.POWER_MONITOR: EmporiaConnector,  # Default to Emporia
    }

    # Provider-specific power monitor connectors
    POWER_MONITOR_PROVIDERS = {
        "emporia": EmporiaConnector,
        "generic": GenericPowerConnector,
        "iotawatt": GenericPowerConnector,
        "shelly": GenericPowerConnector,
        "home_assistant": GenericPowerConnector,
    }

    def __init__(self):
        self._connectors: dict[str, APIConnector] = {}
        self._sync_tasks: dict[str, asyncio.Task] = {}
        self._running = False

    def register_connector(self, config: ConnectionConfig) -> str:
        """Register a new API connector.

        Args:
            config: Connection configuration

        Returns:
            Connector ID
        """
        if not config.id:
            import uuid
            config.id = str(uuid.uuid4())

        # Handle power monitor providers
        if config.api_standard == APIStandard.POWER_MONITOR:
            provider = config.params.get("provider", "emporia").lower()
            connector_class = self.POWER_MONITOR_PROVIDERS.get(provider)
            if not connector_class:
                raise ValueError(f"Unsupported power monitor provider: {provider}")
        else:
            connector_class = self.CONNECTOR_CLASSES.get(config.api_standard)
            if not connector_class:
                raise ValueError(f"Unsupported API standard: {config.api_standard}")

        connector = connector_class(config)
        self._connectors[config.id] = connector

        logger.info(f"Registered connector: {config.name} ({config.api_standard.value})")
        return config.id

    def unregister_connector(self, connector_id: str) -> None:
        """Remove a connector."""
        if connector_id in self._sync_tasks:
            self._sync_tasks[connector_id].cancel()
            del self._sync_tasks[connector_id]

        if connector_id in self._connectors:
            del self._connectors[connector_id]
            logger.info(f"Unregistered connector: {connector_id}")

    def get_connector(self, connector_id: str) -> Optional[APIConnector]:
        """Get a connector by ID."""
        return self._connectors.get(connector_id)

    def list_connectors(self) -> list[dict]:
        """List all registered connectors with their status."""
        return [c.get_status() for c in self._connectors.values()]

    async def test_connector(self, connector_id: str) -> tuple[bool, str]:
        """Test a specific connector."""
        connector = self._connectors.get(connector_id)
        if not connector:
            return False, "Connector not found"

        return await connector.test_connection()

    async def sync_connector(
        self,
        connector_id: str,
        since: Optional[datetime] = None
    ) -> SyncResult:
        """Manually trigger sync for a connector."""
        connector = self._connectors.get(connector_id)
        if not connector:
            return SyncResult(success=False, errors=["Connector not found"])

        return await connector.sync(since)

    async def sync_all(self, since: Optional[datetime] = None) -> dict[str, SyncResult]:
        """Sync all enabled connectors."""
        results = {}

        tasks = []
        for conn_id, connector in self._connectors.items():
            if connector.config.enabled:
                tasks.append((conn_id, connector.sync(since)))

        for conn_id, task in tasks:
            try:
                results[conn_id] = await task
            except Exception as e:
                logger.exception(f"Sync failed for {conn_id}")
                results[conn_id] = SyncResult(success=False, errors=[str(e)])

        return results

    async def start_background_sync(self) -> None:
        """Start background synchronization tasks."""
        if self._running:
            return

        self._running = True

        for conn_id, connector in self._connectors.items():
            if connector.config.enabled and connector.config.sync_interval_minutes > 0:
                task = asyncio.create_task(
                    self._sync_loop(conn_id, connector.config.sync_interval_minutes)
                )
                self._sync_tasks[conn_id] = task

        logger.info("Started background sync tasks")

    async def stop_background_sync(self) -> None:
        """Stop all background synchronization tasks."""
        self._running = False

        for task in self._sync_tasks.values():
            task.cancel()

        self._sync_tasks.clear()
        logger.info("Stopped background sync tasks")

    async def _sync_loop(self, connector_id: str, interval_minutes: int) -> None:
        """Background sync loop for a connector."""
        while self._running:
            try:
                connector = self._connectors.get(connector_id)
                if connector and connector.config.enabled:
                    logger.debug(f"Running scheduled sync for {connector_id}")
                    await connector.sync()

                await asyncio.sleep(interval_minutes * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in sync loop for {connector_id}")
                await asyncio.sleep(60)  # Wait before retry

    async def disconnect_all(self) -> None:
        """Disconnect all connectors."""
        await self.stop_background_sync()

        for connector in self._connectors.values():
            try:
                await connector.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting {connector.config.name}: {e}")


# Global manager instance
_manager: Optional[IntegrationManager] = None


def get_integration_manager() -> IntegrationManager:
    """Get the global integration manager instance."""
    global _manager
    if _manager is None:
        _manager = IntegrationManager()
    return _manager
