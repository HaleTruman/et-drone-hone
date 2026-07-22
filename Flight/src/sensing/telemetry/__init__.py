"""Telemetry transport and synchronization."""

from core.schemas import CollisionEvent, MavlinkTelemetry, RaceStatus, TrackGate

from .mavlink_client import MavlinkClient
from .sync import DataSynchronizer

MavlinkBridge = MavlinkClient

__all__ = [
    "CollisionEvent",
    "DataSynchronizer",
    "MavlinkBridge",
    "MavlinkClient",
    "MavlinkTelemetry",
    "RaceStatus",
    "TrackGate",
]
