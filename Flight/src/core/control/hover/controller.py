from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.coordinates import normalize_quaternion
from core.schemas import VehicleState


@dataclass(frozen=True)
class HoverAttitudeConfig:
    # Approximate normalized total-thrust trim; tune in flight for exact hover.
    hover_thrust: float = 0.274
    min_thrust: float = 0.0
    max_thrust: float = 1.0


class HoverPIDController:
    """Hover controller that emits desired-attitude and thrust targets."""

    def __init__(
        self,
        dt_s: float | None = None,
        *,
        config: HoverAttitudeConfig | None = None,
        hover_thrust: float | None = None,
        min_thrust: float | None = None,
        max_thrust: float | None = None,
    ):
        base = config or HoverAttitudeConfig()
        self.config = HoverAttitudeConfig(
            hover_thrust=float(base.hover_thrust if hover_thrust is None else hover_thrust),
            min_thrust=float(base.min_thrust if min_thrust is None else min_thrust),
            max_thrust=float(base.max_thrust if max_thrust is None else max_thrust),
        )
        self.dt_s = None if dt_s is None else float(dt_s)

    def update(
        self,
        vehicle_state: VehicleState,
        desired_attitude_quaternion: Iterable[float],
    ) -> tuple[np.ndarray, float]:
        _ = vehicle_state
        desired = normalize_quaternion(desired_attitude_quaternion)
        thrust = float(
            np.clip(
                float(self.config.hover_thrust),
                float(self.config.min_thrust),
                float(self.config.max_thrust),
            )
        )
        return desired.astype(float), thrust


class HoverController:
    """State-facing hover controller that commands attitude and normalized thrust."""

    def __init__(
        self,
        *,
        desired_attitude_quaternion: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
        pid_controller: HoverPIDController | None = None,
        config: HoverAttitudeConfig | None = None,
        dt_s: float | None = None,
    ):
        self.desired_attitude_quaternion = normalize_quaternion(desired_attitude_quaternion)
        self.pid_controller = pid_controller or HoverPIDController(config=config, dt_s=dt_s)
        self.last_payload: dict[str, object] | None = None

    def compute_control(
        self,
        state: VehicleState,
        desired_attitude_quaternion: Iterable[float] | None = None,
    ) -> dict[str, object]:
        desired = (
            self.desired_attitude_quaternion
            if desired_attitude_quaternion is None
            else normalize_quaternion(desired_attitude_quaternion)
        )
        quaternion, thrust = self.pid_controller.update(state, desired)
        payload: dict[str, object] = {
            "quaternion": quaternion.astype(float).tolist(),
            "thrust": float(thrust),
        }
        self.last_payload = payload
        return payload
