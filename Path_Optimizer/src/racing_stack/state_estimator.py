import numpy as np


class StateEstimator:
    def __init__(self, initial_state: np.ndarray | None = None):
        self._last_sim_time_ns: int | None = None
        self._last_velocity_local_ned_mps: np.ndarray | None = None
        self.reset(initial_state)

    def update_from_telemetry(self, telemetry: object) -> None:
        dt_s = (
            0.0
            if self._last_sim_time_ns is None
            else (telemetry.sim_time_ns - self._last_sim_time_ns) / 1e9
        )
        self._last_sim_time_ns = telemetry.sim_time_ns
        velocity = np.asarray(telemetry.velocity_local_ned_mps, dtype=float)
        integration_velocity = (
            velocity
            if self._last_velocity_local_ned_mps is None
            else 0.5 * (self._last_velocity_local_ned_mps + velocity)
        )
        self._last_velocity_local_ned_mps = velocity
        self.integrate_velocity(integration_velocity, dt_s)
        self._state[3:6] = velocity
        self.update_attitude(telemetry.attitude, telemetry.body_rates_rps)

    def integrate_velocity(self, velocity_local_ned_mps: np.ndarray, dt_s: float) -> None:
        velocity = np.asarray(velocity_local_ned_mps, dtype=float)
        self._state[0:3] += velocity * float(dt_s)
        self._state[3:6] = velocity

    def vision_correction(self, position_local_ned_m: np.ndarray, weight: float = 1.0) -> None:
        weight = float(np.clip(weight, 0.0, 1.0))
        observed = np.asarray(position_local_ned_m, dtype=float)
        self._state[0:3] = (1.0 - weight) * self._state[0:3] + weight * observed

    def update_attitude(self, quaternion: np.ndarray, body_rates_rps: np.ndarray) -> None:
        quat = np.asarray(quaternion, dtype=float)
        self._state[6:10] = quat / max(np.linalg.norm(quat), 1e-12)
        self._state[10:13] = np.asarray(body_rates_rps, dtype=float)

    def get_13_state(self) -> np.ndarray:
        return self._state.copy()

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        self._last_sim_time_ns = None
        self._last_velocity_local_ned_mps = None
        self._state = np.zeros(13, dtype=float)
        self._state[6] = 1.0
        if initial_state is not None:
            self._state[:] = np.asarray(initial_state, dtype=float)
