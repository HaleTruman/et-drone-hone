from typing import Any

import numpy as np


class SE3GeometricController:
    def compute_control(self, state: np.ndarray, reference: Any) -> dict[str, Any]:
        raise NotImplementedError("Implement the optional SE(3) tracking controller.")

    def position_error(self, position: np.ndarray, desired_position: np.ndarray) -> np.ndarray:
        return np.asarray(position, dtype=float) - np.asarray(desired_position, dtype=float)

    def attitude_error(self, rotation: np.ndarray, desired_rotation: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Choose and implement the SO(3) attitude-error metric.")
