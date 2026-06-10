from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RouteStateSnapshot:
    active_gate_index: int | None
    previous_active_gate_index: int | None
    gate_index_changed: bool
    route_velocity_local_ned_mps: tuple[float, float, float] | None
    velocity_source: str | None
    active_gate_position_local_ned_m: tuple[float, float, float] | None = None
    active_gate_id: int | None = None

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouteStateTracker:
    """Normalize simulator route status and velocity for target selection."""

    def __init__(self, *, velocity_source: str = "local_position"):
        if velocity_source not in {"local_position", "odometry"}:
            raise ValueError("velocity_source must be 'local_position' or 'odometry'")
        self.velocity_source = velocity_source
        self.active_gate_index: int | None = None
        self.last_snapshot: RouteStateSnapshot | None = None

    def update(self, bridge: Any, telemetry: Any | None) -> RouteStateSnapshot:
        previous = self.active_gate_index
        active = self._active_gate_index(bridge)
        changed = active is not None and previous is not None and active != previous
        if active is not None:
            self.active_gate_index = active
        velocity, source = self._route_velocity(bridge, telemetry)
        active_gate = self._active_track_gate(bridge, self.active_gate_index)
        self.last_snapshot = RouteStateSnapshot(
            active_gate_index=self.active_gate_index,
            previous_active_gate_index=previous,
            gate_index_changed=changed,
            route_velocity_local_ned_mps=velocity,
            velocity_source=source,
            active_gate_position_local_ned_m=None
            if active_gate is None
            else tuple(float(value) for value in active_gate.position_local_ned_m),
            active_gate_id=None if active_gate is None else int(active_gate.gate_id),
        )
        return self.last_snapshot

    def snapshot(self) -> dict[str, Any]:
        payload = (
            {
                "active_gate_index": self.active_gate_index,
                "previous_active_gate_index": None,
                "gate_index_changed": False,
                "route_velocity_local_ned_mps": None,
                "velocity_source": None,
                "active_gate_position_local_ned_m": None,
                "active_gate_id": None,
            }
            if self.last_snapshot is None
            else self.last_snapshot.to_log_dict()
        )
        payload["configured_velocity_source"] = self.velocity_source
        return payload

    @staticmethod
    def _active_gate_index(bridge: Any) -> int | None:
        race_status = getattr(bridge, "race_status", None)
        if race_status is None:
            return None
        value = getattr(race_status, "active_gate_index", None)
        return None if value is None else int(value)

    def _route_velocity(self, bridge: Any, telemetry: Any | None) -> tuple[tuple[float, float, float] | None, str | None]:
        if self.velocity_source == "local_position":
            latest_local_position = getattr(bridge, "latest_local_position", None)
            if isinstance(latest_local_position, dict):
                velocity = latest_local_position.get("velocity_local_ned_mps")
                if velocity is not None:
                    return tuple(float(value) for value in velocity), "local_position_ned"
        if telemetry is not None:
            velocity = getattr(telemetry, "velocity_local_ned_mps", None)
            if velocity is not None:
                return tuple(float(value) for value in velocity), "odometry"
        return None, None

    @staticmethod
    def _active_track_gate(bridge: Any, active_gate_index: int | None) -> Any | None:
        if active_gate_index is None:
            return None
        track_gates = getattr(bridge, "track_gates", None)
        if not track_gates:
            return None
        for gate in track_gates:
            if int(getattr(gate, "gate_id", -1)) == int(active_gate_index):
                return gate
        if 0 <= int(active_gate_index) < len(track_gates):
            return track_gates[int(active_gate_index)]
        return None
