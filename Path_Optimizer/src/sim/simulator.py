import sys
from pathlib import Path

import numpy as np
import yaml


def _ensure_src_on_path() -> Path:
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    return here


SRC_ROOT = _ensure_src_on_path()

from controller import RateController, load_params
from quadrotor.model import Quadrotor


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def initial_state(config: dict) -> np.ndarray:
    state = np.zeros(13, dtype=float)
    state[0:3] = np.asarray(config["drone"]["p_0"], dtype=float)
    state[3:6] = np.asarray(config["drone"]["v_0"], dtype=float)
    state[6:10] = np.asarray(config["drone"]["q_0"], dtype=float)
    state[6:10] /= np.linalg.norm(state[6:10])
    state[10:13] = np.asarray(config["drone"]["omega_0"], dtype=float)
    return state


def step_dynamics(quadrotor: Quadrotor, state: np.ndarray, motor_command: np.ndarray, dt: float) -> np.ndarray:
    next_state = state + dt * quadrotor.state_derivative(state, motor_command)
    next_state[6:10] /= np.linalg.norm(next_state[6:10])
    return next_state


def run_axis_test(
    quadrotor: Quadrotor,
    controller: RateController,
    sim_config: dict,
    axis_name: str,
    axis_index: int,
    commanded_rate: float,
    step_time: float,
    duration: float,
) -> dict:
    dt = float(sim_config["dt"])
    state = initial_state(sim_config)
    controller.reset()

    steps = int(duration / dt)
    time_history = np.arange(steps, dtype=float) * dt
    rate_history = np.zeros((steps, 3), dtype=float)
    motor_history = np.zeros((steps, 4), dtype=float)

    for step_idx, current_time in enumerate(time_history):
        desired_rates = np.zeros(3, dtype=float)
        if current_time >= step_time:
            desired_rates[axis_index] = commanded_rate

        motor_command = controller.update(state, desired_rates)
        state = step_dynamics(quadrotor, state, motor_command, dt)
        rate_history[step_idx] = state[10:13]
        motor_history[step_idx] = motor_command

    achieved = rate_history[:, axis_index]
    window_start = int(step_time / dt)
    return {
        "axis": axis_name,
        "commanded_rate": commanded_rate,
        "final_rate": achieved[-1],
        "peak_rate": np.max(np.abs(achieved[window_start:])),
        "mean_motor": np.mean(motor_history[-200:], axis=0),
    }


def main() -> int:
    quad_params = load_yaml(SRC_ROOT / "quadrotor" / "quad_params.yaml")
    controller_params = load_params(SRC_ROOT / "controller" / "rate" / "params.yaml")
    sim_config = load_yaml(SRC_ROOT / "sim" / "sim_config.yaml")

    sim_config["dt"] = 0.001

    quadrotor = Quadrotor(quad_params)
    controller = RateController(controller_params)

    tests = [
        ("roll", 0, 1.0),
        ("pitch", 1, 1.0),
        ("yaw", 2, 0.8),
    ]

    print("Simple attitude-rate step test at 1000 Hz")
    for axis_name, axis_index, commanded_rate in tests:
        result = run_axis_test(
            quadrotor=quadrotor,
            controller=controller,
            sim_config=sim_config,
            axis_name=axis_name,
            axis_index=axis_index,
            commanded_rate=commanded_rate,
            step_time=0.1,
            duration=1.0,
        )
        mean_motor = np.array2string(result["mean_motor"], precision=3, floatmode="fixed")
        print(
            f"{result['axis']:>5s}: cmd={result['commanded_rate']:.3f} rad/s  "
            f"final={result['final_rate']:.3f} rad/s  "
            f"peak={result['peak_rate']:.3f} rad/s  "
            f"mean_motor={mean_motor}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
