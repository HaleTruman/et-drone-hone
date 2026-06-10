from collections.abc import Iterable

import numpy as np


class QuadrotorSimulatorHarness:
    def __init__(self, model: object, initial_state: np.ndarray, dt_s: float = 0.01):
        self.model = model
        self.initial_state = np.asarray(initial_state, dtype=float).copy()
        self.dt_s = dt_s
        self.reset()

    def step(
        self,
        u: np.ndarray,
        external_force_i: np.ndarray | None = None,
        external_moment_b: np.ndarray | None = None,
    ) -> np.ndarray:
        x = self.state
        dt = self.dt_s
        derivative = self.model.state_derivative
        k1 = derivative(x, u, external_force_i, external_moment_b)
        k2 = derivative(x + dt * k1 / 2.0, u, external_force_i, external_moment_b)
        k3 = derivative(x + dt * k2 / 2.0, u, external_force_i, external_moment_b)
        k4 = derivative(x + dt * k3, u, external_force_i, external_moment_b)
        self.state = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.state[6:10] /= max(np.linalg.norm(self.state[6:10]), 1e-12)
        return self.state.copy()

    def reset(self) -> np.ndarray:
        self.state = self.initial_state.copy()
        return self.state.copy()

    def run_trajectory(self, controls: Iterable[np.ndarray]) -> np.ndarray:
        states = [self.state.copy()]
        for control in controls:
            states.append(self.step(np.asarray(control, dtype=float)))
        return np.asarray(states)
