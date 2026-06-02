"""Generate MAVLink-shaped telemetry from the existing quadrotor model."""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from core.quadrotor.model import Quadrotor
from core.simulator.harness import QuadrotorSimulatorHarness
from sensing.telemetry.mavlink_bridge import TelemetrySample


@dataclass(frozen=True)
class MavlinkMessage:
    """A transport-neutral representation of one MAVLink telemetry message."""

    message_type: str
    sim_time_ns: int
    fields: dict[str, Any]


class TelemetrySimulator:
    """Simulate a quadrotor with MAVLink-shaped telemetry and attitude targets."""

    def __init__(
        self,
        telemetry_hz: float = 100.0,
        heartbeat_hz: float = 2.0,
        physics_hz: float = 100.0,
        initial_altitude_m: float = 0.0,
        noise_seed: int = 7,
    ):
        if telemetry_hz <= 0.0 or heartbeat_hz <= 0.0 or physics_hz <= 0.0:
            raise ValueError("rates must be positive")
        physics_steps = round(physics_hz / telemetry_hz)
        if physics_steps < 1 or not math.isclose(physics_steps * telemetry_hz, physics_hz):
            raise ValueError("physics_hz must be an integer multiple of telemetry_hz")

        params_path = Path(__file__).resolve().parents[1] / "quadrotor" / "params.yaml"
        self.model = Quadrotor(self._load_params(params_path))

        initial_state = np.zeros(13, dtype=float)
        initial_state[2] = -float(initial_altitude_m) if initial_altitude_m else 0.0
        initial_state[6] = 1.0
        self.telemetry_hz = float(telemetry_hz)
        self.heartbeat_hz = float(heartbeat_hz)
        self.noise_seed = int(noise_seed)
        self._physics_steps = physics_steps
        self._motor_command = np.zeros(4, dtype=float)
        self._harness = QuadrotorSimulatorHarness(self.model, initial_state, dt_s=1.0 / physics_hz)

    def set_motor_command(self, motor_command: np.ndarray) -> None:
        self._motor_command = np.clip(np.asarray(motor_command, dtype=float), 0.0, 1.0)

    def apply_attitude_target(self, target: dict[str, Any]) -> None:
        """Apply a minimal stabilized-controller interpretation of SET_ATTITUDE_TARGET."""

        desired = np.asarray(target["quaternion"], dtype=float)
        desired /= np.linalg.norm(desired)
        current = self._harness.state[6:10]
        attitude_error = self._quaternion_multiply(desired, self._quaternion_conjugate(current))[1:4]
        desired_rates = np.asarray(target.get("body_rates_rps", [0.0, 0.0, 0.0]), dtype=float)
        rate_error = desired_rates - self._harness.state[10:13]
        roll, pitch, yaw = 0.08 * attitude_error + 0.01 * rate_error
        collective = float(np.clip(target["thrust"], 0.0, 1.0))
        self.set_motor_command(
            collective
            + np.array(
                [-roll + pitch - yaw, -roll - pitch + yaw, roll - pitch - yaw, roll + pitch + yaw],
                dtype=float,
            )
        )

    def messages(self, duration_s: float) -> Iterator[MavlinkMessage]:
        """Yield HEARTBEAT, TIMESYNC, ATTITUDE, and HIGHRES_IMU messages."""

        heartbeat_interval = max(1, round(self.telemetry_hz / self.heartbeat_hz))
        for step, sample in enumerate(self.telemetry_samples(duration_s)):
            if step % heartbeat_interval == 0:
                yield MavlinkMessage(
                    "HEARTBEAT",
                    sample.sim_time_ns,
                    {
                        "type": "MAV_TYPE_QUADROTOR",
                        "autopilot": "MAV_AUTOPILOT_GENERIC",
                        "base_mode": "MAV_MODE_FLAG_SAFETY_ARMED",
                        "system_status": sample.system_status,
                    },
                )
            yield MavlinkMessage("TIMESYNC", sample.sim_time_ns, {"tc1": 0, "ts1": sample.sim_time_ns})

            roll, pitch, yaw = self._euler_from_quaternion(sample.attitude)
            rollspeed, pitchspeed, yawspeed = sample.body_rates_rps
            yield MavlinkMessage(
                "ATTITUDE",
                sample.sim_time_ns,
                {
                    "time_boot_ms": sample.sim_time_ns // 1_000_000,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "rollspeed": rollspeed,
                    "pitchspeed": pitchspeed,
                    "yawspeed": yawspeed,
                },
            )
            acceleration = sample.raw["acceleration_local_ned_mps2"]
            yield MavlinkMessage(
                "HIGHRES_IMU",
                sample.sim_time_ns,
                {
                    "time_usec": sample.sim_time_ns // 1_000,
                    "xacc": acceleration[0],
                    "yacc": acceleration[1],
                    "zacc": acceleration[2],
                    "xgyro": rollspeed,
                    "ygyro": pitchspeed,
                    "zgyro": yawspeed,
                    # Expected simulator API extension used by the racing stack.
                    "velocity_local_ned_mps": sample.velocity_local_ned_mps,
                },
            )

    def telemetry_samples(self, duration_s: float) -> Iterator[TelemetrySample]:
        """Yield bridge-ready samples while stepping the powered-off model."""

        if duration_s < 0.0:
            raise ValueError("duration must be non-negative")
        self._harness.reset()
        rng = np.random.default_rng(self.noise_seed)
        step_count = int(math.floor(duration_s * self.telemetry_hz)) + 1
        for step in range(step_count):
            if step:
                for _ in range(self._physics_steps):
                    self._harness.step(self._motor_command)
            state = self._harness.state
            acceleration = self.model.state_derivative(state, self._motor_command)[3:6]
            attitude = self._noisy_attitude(state[6:10], rng)
            velocity = state[3:6] + rng.normal(0.0, 0.03, size=3)
            body_rates = state[10:13] + rng.normal(0.0, 0.002, size=3)
            acceleration = acceleration + rng.normal(0.0, 0.05, size=3)
            yield TelemetrySample(
                sim_time_ns=round(step / self.telemetry_hz * 1_000_000_000),
                attitude=tuple(float(value) for value in attitude),
                velocity_local_ned_mps=tuple(float(value) for value in velocity),
                body_rates_rps=tuple(float(value) for value in body_rates),
                system_status="MAV_STATE_ACTIVE",
                raw={
                    "source": "quadrotor_model",
                    "acceleration_local_ned_mps2": tuple(float(value) for value in acceleration),
                },
            )

    @staticmethod
    def _noisy_attitude(attitude: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noisy = np.asarray(attitude, dtype=float) + rng.normal(0.0, 0.0005, size=4)
        return noisy / np.linalg.norm(noisy)

    @staticmethod
    def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quaternion
        return np.array([qw, -qx, -qy, -qz], dtype=float)

    @staticmethod
    def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=float,
        )

    @staticmethod
    def _euler_from_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
        qw, qx, qy, qz = quaternion
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        sin_pitch = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return roll, pitch, yaw

    @staticmethod
    def _load_params(path: Path) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator:
                params[key.strip()] = ast.literal_eval(value.partition("#")[0].strip())
        return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Print falling-drone MAVLink telemetry as JSON lines.")
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--telemetry-hz", type=float, default=100.0)
    parser.add_argument("--physics-hz", type=float, default=100.0)
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--initial-altitude-m", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=7)
    args = parser.parse_args()
    simulator = TelemetrySimulator(
        telemetry_hz=args.telemetry_hz,
        physics_hz=args.physics_hz,
        heartbeat_hz=args.heartbeat_hz,
        initial_altitude_m=args.initial_altitude_m,
        noise_seed=args.noise_seed,
    )
    for message in simulator.messages(args.duration_s):
        print(json.dumps(asdict(message), separators=(",", ":")), flush=True)
