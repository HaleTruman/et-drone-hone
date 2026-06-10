"""Telemetry transport and synchronization."""

from core.schemas import CollisionEvent, RaceStatus, TelemetrySample, TrackGate

from .mavlink_client import MavlinkClient
from .sync import DataSynchronizer

MavlinkBridge = MavlinkClient

__all__ = [
    "CollisionEvent",
    "DataSynchronizer",
    "MavlinkBridge",
    "MavlinkClient",
    "RaceStatus",
    "TelemetrySample",
    "TrackGate",
]
