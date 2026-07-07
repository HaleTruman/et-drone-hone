from collections.abc import Callable

import numpy as np

from sensing.gates import GateMap


class PathManager:
    def __init__(self, spline_generator: Callable[[np.ndarray], object] | None = None):
        self.spline_generator = spline_generator
        self._waypoints = np.empty((0, 3))

    def update_from_gate_map(self, gate_map: GateMap, limit: int = 3) -> np.ndarray:
        self._waypoints = np.asarray(
            [gate.position_relative_ned_m or gate.position_local_ned_m for gate in gate_map.get_next_gates(limit)],
            dtype=float,
        )
        return self.get_waypoints()

    def generate_spline(self) -> object:
        if self.spline_generator is None:
            raise NotImplementedError("Inject the chosen spline generator.")
        return self.spline_generator(self.get_waypoints())

    def get_waypoints(self) -> np.ndarray:
        return self._waypoints.copy()
