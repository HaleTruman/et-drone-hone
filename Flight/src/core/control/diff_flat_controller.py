from typing import Any


class DifferentialFlatnessController:
    def compute_commands(self, reference_traj: Any) -> dict[str, Any]:
        raise NotImplementedError("Implement flat-output derivative tracking.")

    def flat_to_thrust_and_rates(self, flat_output: Any) -> tuple[float, tuple[float, float, float]]:
        raise NotImplementedError("Map position and yaw derivatives to thrust and body rates.")
