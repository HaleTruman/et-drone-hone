import json
import os

import numpy as np
import yaml

from engine.planner import PlanningEngine
from quadrotor.model import Quadrotor


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_reference_trajectory(
    *,
    course_path: str,
    state: np.ndarray,
    reference_trajectory: dict,
) -> None:
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "artifacts",
        "reference_trajectory.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    x = np.asarray(reference_trajectory["x"], dtype=float)
    payload = {
        "course_path": os.path.abspath(course_path),
        "drone_state": {
            "state_vector": state.tolist(),
            "position": state[0:3].tolist(),
            "velocity": state[3:6].tolist(),
            "quaternion": state[6:10].tolist(),
            "rates": state[10:13].tolist(),
        },
        "reference_trajectory": {
            "t": np.asarray(reference_trajectory["t"], dtype=float).tolist(),
            "pos": x[:, 0:3].tolist(),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def initial_state(config: dict) -> np.ndarray:
    state = np.zeros(13, dtype=float)
    state[0:3] = np.asarray(config["X_0"]["p_0"], dtype=np.float64)
    state[3:6] = np.asarray(config["X_0"]["v_0"], dtype=np.float64)
    state[6:10] = np.asarray(config["X_0"]["q_0"], dtype=np.float64)
    state[6:10] /= np.linalg.norm(state[6:10])
    state[10:13] = np.asarray(config["X_0"]["omega_0"], dtype=np.float64)
    return state


def extract_gate_positions(course: dict) -> np.ndarray:
    gates = course["targets"]
    key = "position_cm"
    

    positions: list[list[float]] = []
    for item in gates:
        position = item if key is None else item.get(key) or item.get("position")
        
        positions.append(
            [
                float(position["x"]) * 0.01,
                float(position["y"]) * 0.01,
                float(position["z"]) * 0.01,
            ]
        )

    return np.asarray(positions, dtype=float)

def main() -> int:
    quad_params = load_yaml("src/quadrotor/params.yaml")
    
    sim_config = load_yaml("src/config/settings.yaml")
    course_path = "course_model/targets-SimBlank-20260216_194648.json"
    course = load_json(course_path)

    quadrotor = Quadrotor(quad_params)
    planner = PlanningEngine(params=quad_params)

    state = initial_state(sim_config)
    gates = extract_gate_positions(course)

    print(gates)

    reference_trajectory = planner.generate_initial_guess(gates, x0=state)

    print(reference_trajectory)

    save_reference_trajectory(
        course_path=course_path,
        state=state,
        reference_trajectory=reference_trajectory,
    )

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
