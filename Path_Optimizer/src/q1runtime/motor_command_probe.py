from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sensing.telemetry import MavlinkBridge

from .config import Q1RuntimeConfig


@dataclass(frozen=True)
class MotorProbeConfig:
    endpoint: str = Q1RuntimeConfig.mavlink_endpoint
    command_hz: float = 30.0
    sample_hz: float = 30.0
    pulse_s: float = 0.25
    rest_s: float = 0.75
    start_delay_s: float = 0.5
    include_basic_cases: bool = True
    motor_values: tuple[float, ...] = (0.05, 0.08, 0.1, 0.12)
    motor_patterns: tuple[str, ...] = ("equal",)
    mixer_base_values: tuple[float, ...] = ()
    mixer_deltas: tuple[float, ...] = ()
    mixer_axes: tuple[str, ...] = ("roll", "pitch", "yaw")
    arm_on_start: bool = True
    disarm_on_exit: bool = True
    reset_on_start: bool = False
    reset_between_cases: bool = False
    reset_wait_s: float = 2.0
    reset_ready_timeout_s: float = 20.0
    reset_stable_s: float = 2.0
    reset_stable_max_speed_mps: float = 0.05
    post_reset_delay_s: float = 2.0
    abort_max_speed_mps: float | None = None
    abort_max_displacement_m: float | None = None
    abort_max_body_rate_rps: float | None = None
    abort_on_new_collision: bool = False
    abort_collision_threat_level: int | None = None
    abort_max_collision_impact_kg_mps: float | None = None
    precase_clearance: bool = False
    precase_clearance_motor_value: float = 0.3
    precase_clearance_s: float = 0.12
    precase_clearance_rest_s: float = 0.0
    precase_clearance_repeat: int = 1
    precase_collision_quiet_s: float = 0.0
    precase_collision_quiet_timeout_s: float = 3.0
    progress: bool = False
    output_path: Path | None = Path("logs/q1runtime/motor-command-probe.json")


class MotorCommandProbe:
    """Probe direct SET_ACTUATOR_CONTROL_TARGET equal-motor commands."""

    def __init__(self, bridge: Any, config: MotorProbeConfig | None = None):
        self.bridge = bridge
        self.config = config or MotorProbeConfig()
        self._last_safety_abort_reason: str | None = None
        self._clearance_reports: list[dict[str, Any]] = []

    def run_live(self) -> dict[str, Any]:
        self.bridge.connect()
        self.bridge.start_heartbeat()
        self.bridge.subscribe_telemetry()
        self._wait_for_telemetry()
        if self.config.reset_on_start:
            self._reset_simulator("sending simulator reset before motor probe")
        if self.config.arm_on_start:
            self.bridge.arm()
            time.sleep(0.2)
        self._run_precase_clearance("initial")
        if self.config.start_delay_s > 0.0:
            self._progress(f"waiting start_delay_s={self.config.start_delay_s:g} before motor commands")
            time.sleep(self.config.start_delay_s)

        results: list[dict[str, Any]] = []
        try:
            cases = self.cases()
            for index, case in enumerate(cases):
                if index > 0 and self.config.reset_between_cases:
                    self._reset_simulator(f"resetting simulator before case {index + 1}/{len(cases)}")
                    if self.config.arm_on_start:
                        self.bridge.arm()
                        time.sleep(0.2)
                    self._run_precase_clearance(f"case_{index + 1}")
                    if self.config.start_delay_s > 0.0:
                        time.sleep(self.config.start_delay_s)
                command = case["command"]
                start = self.snapshot()
                self._progress(f"[{index + 1}/{len(cases)}] start {case['name']} motors={command}")
                self._last_safety_abort_reason = None
                samples = self._stream_motor(command, self.config.pulse_s, safety_origin=start)
                safety_abort_reason = self._last_safety_abort_reason
                rest_samples = self._stream_motor([0.0, 0.0, 0.0, 0.0], self.config.rest_s, safety_origin=start)
                samples.extend(rest_samples)
                if safety_abort_reason is None:
                    safety_abort_reason = self._last_safety_abort_reason
                end = self.snapshot()
                result = self.summarize_case(case, start, end, samples)
                result["collision_summary"] = self._collision_summary(start.get("collision_count"))
                result["safety_abort_reason"] = safety_abort_reason
                results.append(result)
                self._progress(
                    f"[{index + 1}/{len(cases)}] done {case['name']} "
                    f"disp={result['displacement_local_ned_m']} max_speed={result['max_speed_mps']} "
                    f"collisions={result['collision_summary']['new_count']}"
                )
                if safety_abort_reason is not None:
                    self._progress(f"stopping sweep after safety abort: {safety_abort_reason}")
                    break
        finally:
            self._stream_motor([0.0, 0.0, 0.0, 0.0], 0.2)
            if self.config.disarm_on_exit:
                try:
                    self.bridge.disarm()
                except Exception:
                    pass
            self.bridge.shutdown()

        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "completed_case_count": len(results),
            "results": results,
            "clearance_reports": self._clearance_reports,
            "bridge_snapshot": self._bridge_snapshot(),
        }
        if self.config.output_path is not None:
            path = Path(self.config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    def summarize_case(
        self,
        case: dict[str, Any],
        start: dict[str, Any],
        end: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        displacement = None
        if start["position_local_ned_m"] is not None and end["position_local_ned_m"] is not None:
            displacement = (
                np.asarray(end["position_local_ned_m"], dtype=float)
                - np.asarray(start["position_local_ned_m"], dtype=float)
            ).tolist()
        speeds = [
            float(np.linalg.norm(np.asarray(sample["velocity_local_ned_mps"], dtype=float)))
            for sample in samples
            if sample["velocity_local_ned_mps"] is not None
        ]
        body_rate_delta = None
        if start["body_rates_rps"] is not None and end["body_rates_rps"] is not None:
            body_rate_delta = (
                np.asarray(end["body_rates_rps"], dtype=float)
                - np.asarray(start["body_rates_rps"], dtype=float)
            ).tolist()
        body_rates = [
            sample["body_rates_rps"]
            for sample in samples
            if sample["body_rates_rps"] is not None
        ]
        max_body_rate = None
        if body_rates:
            max_body_rate = float(np.max(np.linalg.norm(np.asarray(body_rates, dtype=float), axis=1)))
        return {
            "name": case["name"],
            "motor_value": case["value"],
            "motor_pattern": case["pattern"],
            "mixer_axis": case.get("mixer_axis"),
            "mixer_delta": case.get("mixer_delta"),
            "mixer_base": case.get("mixer_base"),
            "motor_command": case["command"],
            "start": start,
            "end": end,
            "samples": samples,
            "displacement_local_ned_m": displacement,
            "max_speed_mps": None if not speeds else max(speeds),
            "body_rate_delta_rps": body_rate_delta,
            "max_body_rate_rps": max_body_rate,
        }

    def cases(self) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        if self.config.include_basic_cases:
            for value in self.config.motor_values:
                value = float(value)
                for pattern in self.config.motor_patterns:
                    pattern = str(pattern).lower()
                    if pattern == "equal":
                        command = [value, value, value, value]
                    elif pattern == "front":
                        command = [value, value, 0.0, 0.0]
                    elif pattern == "rear":
                        command = [0.0, 0.0, value, value]
                    elif pattern == "left":
                        command = [value, 0.0, value, 0.0]
                    elif pattern == "right":
                        command = [0.0, value, 0.0, value]
                    elif pattern == "diag_a":
                        command = [value, 0.0, 0.0, value]
                    elif pattern == "diag_b":
                        command = [0.0, value, value, 0.0]
                    else:
                        raise ValueError(f"Unsupported motor pattern: {pattern}")
                    cases.append({"name": f"{pattern}_{value:g}", "pattern": pattern, "value": value, "command": command})
        for base in self.config.mixer_base_values:
            base = float(base)
            for delta in self.config.mixer_deltas:
                delta = float(delta)
                for axis in self.config.mixer_axes:
                    axis = str(axis).lower()
                    if axis not in {"roll", "pitch", "yaw"}:
                        raise ValueError(f"Unsupported mixer axis: {axis}")
                    for sign in (1.0, -1.0):
                        signed_delta = sign * delta
                        command = self._mixer_command(base, axis, signed_delta)
                        cases.append(
                            {
                                "name": f"mixer_{axis}_{'plus' if sign > 0 else 'minus'}{delta:g}_base_{base:g}",
                                "pattern": f"mixer_{axis}",
                                "value": base,
                                "mixer_axis": axis,
                                "mixer_delta": signed_delta,
                                "mixer_base": base,
                                "command": command,
                            }
                        )
        return cases

    def _mixer_command(self, base: float, axis: str, delta: float) -> list[float]:
        roll = delta if axis == "roll" else 0.0
        pitch = delta if axis == "pitch" else 0.0
        yaw = delta if axis == "yaw" else 0.0
        command = np.asarray(
            [
                base - roll + pitch - yaw,
                base - roll - pitch + yaw,
                base + roll - pitch - yaw,
                base + roll + pitch + yaw,
            ],
            dtype=float,
        )
        return [float(value) for value in np.clip(command, 0.0, 1.0)]

    def snapshot(self) -> dict[str, Any]:
        telemetry = self.bridge.get_latest_telemetry()
        actuator = getattr(self.bridge, "latest_actuator_output", None)
        return {
            "monotonic_s": time.monotonic(),
            "sim_time_ns": None if telemetry is None else int(telemetry.sim_time_ns),
            "position_local_ned_m": None
            if telemetry is None or telemetry.position_local_ned_m is None
            else [float(value) for value in telemetry.position_local_ned_m],
            "velocity_local_ned_mps": None
            if telemetry is None
            else [float(value) for value in telemetry.velocity_local_ned_mps],
            "body_rates_rps": None
            if telemetry is None
            else [float(value) for value in telemetry.body_rates_rps],
            "attitude": None if telemetry is None else [float(value) for value in telemetry.attitude],
            "euler_deg": None if telemetry is None else self._euler_deg(telemetry.attitude),
            "reset_count": None if telemetry is None else telemetry.reset_count,
            "actuator_output": None
            if not actuator
            else [float(value) for value in actuator.get("motor_commands", [])],
            "collision_count": len(getattr(self.bridge, "collisions", []) or []),
            "latest_collision": self._latest_collision(),
        }

    @staticmethod
    def _euler_deg(quaternion: Any) -> list[float]:
        w, x, y, z = [float(value) for value in quaternion]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return [math.degrees(value) for value in (roll, pitch, yaw)]

    def _latest_collision(self) -> dict[str, Any] | None:
        collisions = getattr(self.bridge, "collisions", []) or []
        if not collisions:
            return None
        collision = collisions[-1]
        return {
            "collision_id": int(getattr(collision, "collision_id")),
            "threat_level": int(getattr(collision, "threat_level")),
            "impact_kg_mps": float(getattr(collision, "impact_kg_mps")),
        }

    def _collision_summary(self, start_count: Any | None = None) -> dict[str, Any]:
        collisions = getattr(self.bridge, "collisions", []) or []
        start_index = 0 if start_count is None else max(0, int(start_count))
        new_collisions = collisions[start_index:]
        impacts = [float(getattr(collision, "impact_kg_mps")) for collision in new_collisions]
        threat_levels = [int(getattr(collision, "threat_level")) for collision in new_collisions]
        ids = sorted({int(getattr(collision, "collision_id")) for collision in new_collisions})
        latest = None
        if new_collisions:
            collision = new_collisions[-1]
            latest = {
                "collision_id": int(getattr(collision, "collision_id")),
                "threat_level": int(getattr(collision, "threat_level")),
                "impact_kg_mps": float(getattr(collision, "impact_kg_mps")),
            }
        return {
            "start_count": start_index,
            "end_count": len(collisions),
            "new_count": len(new_collisions),
            "collision_ids": ids,
            "max_threat_level": None if not threat_levels else max(threat_levels),
            "max_impact_kg_mps": None if not impacts else max(impacts),
            "latest_collision": latest,
        }

    def _stream_motor(
        self,
        command: list[float],
        duration_s: float,
        safety_origin: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        command_interval = 1.0 / max(1e-6, self.config.command_hz)
        sample_interval = 1.0 / max(1e-6, self.config.sample_hz)
        deadline = time.monotonic() + max(0.0, duration_s)
        next_command = time.monotonic()
        next_sample = next_command
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_command:
                self.bridge.send_motor_target(command)
                next_command += command_interval
            if now >= next_sample:
                sample = self.snapshot()
                samples.append(sample)
                if safety_origin is not None:
                    reason = self._safety_breach(safety_origin, sample)
                    if reason is not None:
                        self._last_safety_abort_reason = reason
                        self.bridge.send_motor_target([0.0, 0.0, 0.0, 0.0])
                        self._progress(f"safety abort: {reason}")
                        break
                next_sample += sample_interval
            time.sleep(min(0.002, command_interval / 2.0, sample_interval / 2.0))
        return samples

    def _safety_breach(self, origin: dict[str, Any], sample: dict[str, Any]) -> str | None:
        if self.config.abort_max_speed_mps is not None and sample["velocity_local_ned_mps"] is not None:
            speed = float(np.linalg.norm(np.asarray(sample["velocity_local_ned_mps"], dtype=float)))
            if speed > float(self.config.abort_max_speed_mps):
                return f"speed {speed:.3f} m/s > {self.config.abort_max_speed_mps:.3f} m/s"

        if (
            self.config.abort_max_displacement_m is not None
            and origin["position_local_ned_m"] is not None
            and sample["position_local_ned_m"] is not None
        ):
            displacement = float(
                np.linalg.norm(
                    np.asarray(sample["position_local_ned_m"], dtype=float)
                    - np.asarray(origin["position_local_ned_m"], dtype=float)
                )
            )
            if displacement > float(self.config.abort_max_displacement_m):
                return f"displacement {displacement:.3f} m > {self.config.abort_max_displacement_m:.3f} m"

        if self.config.abort_max_body_rate_rps is not None and sample["body_rates_rps"] is not None:
            body_rate = float(np.linalg.norm(np.asarray(sample["body_rates_rps"], dtype=float)))
            if body_rate > float(self.config.abort_max_body_rate_rps):
                return f"body_rate {body_rate:.3f} rps > {self.config.abort_max_body_rate_rps:.3f} rps"

        collision_summary = self._collision_summary(origin.get("collision_count"))
        if self.config.abort_on_new_collision and collision_summary["new_count"] > 0:
            return f"new collision count {collision_summary['new_count']}"
        if (
            self.config.abort_collision_threat_level is not None
            and collision_summary["max_threat_level"] is not None
            and int(collision_summary["max_threat_level"]) >= int(self.config.abort_collision_threat_level)
        ):
            return (
                f"collision threat {collision_summary['max_threat_level']} "
                f">= {self.config.abort_collision_threat_level}"
            )
        if (
            self.config.abort_max_collision_impact_kg_mps is not None
            and collision_summary["max_impact_kg_mps"] is not None
            and float(collision_summary["max_impact_kg_mps"]) > float(self.config.abort_max_collision_impact_kg_mps)
        ):
            return (
                f"collision impact {collision_summary['max_impact_kg_mps']:.3f} kg*m/s "
                f"> {self.config.abort_max_collision_impact_kg_mps:.3f} kg*m/s"
            )

        return None

    def _run_precase_clearance(self, label: str) -> None:
        if not self.config.precase_clearance:
            return
        command = [float(self.config.precase_clearance_motor_value)] * 4
        repeat = max(1, int(self.config.precase_clearance_repeat))
        start = self.snapshot()
        self._progress(
            f"precase clearance {label}: repeats={repeat} motors={command} "
            f"pulse_s={self.config.precase_clearance_s:g} rest_s={self.config.precase_clearance_rest_s:g}"
        )
        self._last_safety_abort_reason = None
        samples: list[dict[str, Any]] = []
        for index in range(repeat):
            samples.extend(self._stream_motor(command, self.config.precase_clearance_s, safety_origin=start))
            if self._last_safety_abort_reason is not None:
                break
            if self.config.precase_clearance_rest_s > 0.0:
                samples.extend(
                    self._stream_motor([0.0, 0.0, 0.0, 0.0], self.config.precase_clearance_rest_s, safety_origin=start)
                )
            if self._last_safety_abort_reason is not None:
                break
            if index < repeat - 1 and self.config.precase_collision_quiet_s > 0.0:
                self._wait_for_collision_quiet()
        if self.config.precase_collision_quiet_s > 0.0:
            self._wait_for_collision_quiet()
        end = self.snapshot()
        report = {
            "label": label,
            "motor_command": command,
            "repeat": repeat,
            "pulse_s": float(self.config.precase_clearance_s),
            "rest_s": float(self.config.precase_clearance_rest_s),
            "start": start,
            "end": end,
            "sample_count": len(samples),
            "collision_summary": self._collision_summary(start.get("collision_count")),
            "safety_abort_reason": self._last_safety_abort_reason,
        }
        if start["position_local_ned_m"] is not None and end["position_local_ned_m"] is not None:
            report["displacement_local_ned_m"] = (
                np.asarray(end["position_local_ned_m"], dtype=float)
                - np.asarray(start["position_local_ned_m"], dtype=float)
            ).tolist()
        else:
            report["displacement_local_ned_m"] = None
        self._clearance_reports.append(report)
        self._progress(
            f"precase clearance {label} done disp={report['displacement_local_ned_m']} "
            f"collisions={report['collision_summary']['new_count']}"
        )
        if self._last_safety_abort_reason is not None:
            self._progress(f"precase clearance safety abort: {self._last_safety_abort_reason}")

    def _wait_for_collision_quiet(self) -> None:
        quiet_s = max(0.0, float(self.config.precase_collision_quiet_s))
        if quiet_s <= 0.0:
            return
        deadline = time.monotonic() + max(0.01, float(self.config.precase_collision_quiet_timeout_s))
        stable_start: float | None = None
        last_count = len(getattr(self.bridge, "collisions", []) or [])
        while time.monotonic() < deadline:
            count = len(getattr(self.bridge, "collisions", []) or [])
            if count == last_count:
                if stable_start is None:
                    stable_start = time.monotonic()
                if time.monotonic() - stable_start >= quiet_s:
                    self._progress(f"collision quiet for {quiet_s:g}s count={count}")
                    return
            else:
                last_count = count
                stable_start = None
            time.sleep(0.02)
        self._progress(
            f"collision quiet wait timed out after {self.config.precase_collision_quiet_timeout_s:g}s "
            f"count={len(getattr(self.bridge, 'collisions', []) or [])}"
        )

    def _wait_for_telemetry(self) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is not None and getattr(telemetry, "position_local_ned_m", None) is not None:
                return
            time.sleep(0.02)
        raise TimeoutError("No local-NED telemetry received for motor command probe.")

    def _reset_simulator(self, message: str) -> None:
        self._progress(message)
        previous = self.bridge.get_latest_telemetry()
        previous_reset_count = None if previous is None else getattr(previous, "reset_count", None)
        try:
            self._stream_motor([0.0, 0.0, 0.0, 0.0], 0.1)
        except Exception:
            pass
        self.bridge.send_sim_reset_command()
        time.sleep(self.config.reset_wait_s)
        self._wait_for_telemetry()
        self._wait_for_reset_ready(previous_reset_count)

    def _wait_for_reset_ready(self, previous_reset_count: Any | None) -> None:
        deadline = time.monotonic() + max(0.01, float(self.config.reset_ready_timeout_s))
        stable_start: float | None = None
        while time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is None or getattr(telemetry, "position_local_ned_m", None) is None:
                stable_start = None
                time.sleep(0.05)
                continue
            reset_count = getattr(telemetry, "reset_count", None)
            if previous_reset_count is not None and reset_count == previous_reset_count:
                stable_start = None
                time.sleep(0.05)
                continue
            velocity = getattr(telemetry, "velocity_local_ned_mps", None)
            speed = 0.0 if velocity is None else float(np.linalg.norm(np.asarray(velocity, dtype=float)))
            if speed <= float(self.config.reset_stable_max_speed_mps):
                if stable_start is None:
                    stable_start = time.monotonic()
                if time.monotonic() - stable_start >= float(self.config.reset_stable_s):
                    self._progress(
                        f"reset ready reset_count={reset_count} speed={speed:.4f} "
                        f"stable_s={self.config.reset_stable_s:g}; post_delay={self.config.post_reset_delay_s:g}"
                    )
                    if self.config.post_reset_delay_s > 0.0:
                        time.sleep(self.config.post_reset_delay_s)
                    return
            else:
                stable_start = None
            time.sleep(0.05)
        raise TimeoutError("Simulator did not reach stable post-reset telemetry before timeout.")

    def _bridge_snapshot(self) -> dict[str, Any] | None:
        snapshot = getattr(self.bridge, "snapshot", None)
        return None if not callable(snapshot) else snapshot()

    def _progress(self, message: str) -> None:
        if self.config.progress:
            print(message, flush=True)


def _float_tuple_from_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one value.")
    return values


def _str_tuple_from_csv(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one value.")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe direct equal-motor actuator commands.")
    parser.add_argument("--endpoint", default=MotorProbeConfig.endpoint)
    parser.add_argument("--command-hz", type=float, default=MotorProbeConfig.command_hz)
    parser.add_argument("--sample-hz", type=float, default=MotorProbeConfig.sample_hz)
    parser.add_argument("--pulse-s", type=float, default=MotorProbeConfig.pulse_s)
    parser.add_argument("--rest-s", type=float, default=MotorProbeConfig.rest_s)
    parser.add_argument("--start-delay-s", type=float, default=MotorProbeConfig.start_delay_s)
    parser.add_argument("--no-basic-cases", action="store_true")
    parser.add_argument("--motor-values", type=_float_tuple_from_csv, default=MotorProbeConfig.motor_values)
    parser.add_argument("--motor-patterns", type=_str_tuple_from_csv, default=MotorProbeConfig.motor_patterns)
    parser.add_argument("--mixer-base-values", type=_float_tuple_from_csv, default=MotorProbeConfig.mixer_base_values)
    parser.add_argument("--mixer-deltas", type=_float_tuple_from_csv, default=MotorProbeConfig.mixer_deltas)
    parser.add_argument("--mixer-axes", type=_str_tuple_from_csv, default=MotorProbeConfig.mixer_axes)
    parser.add_argument("--reset-on-start", action="store_true")
    parser.add_argument("--reset-between-cases", action="store_true")
    parser.add_argument("--reset-wait-s", type=float, default=MotorProbeConfig.reset_wait_s)
    parser.add_argument("--reset-ready-timeout-s", type=float, default=MotorProbeConfig.reset_ready_timeout_s)
    parser.add_argument("--reset-stable-s", type=float, default=MotorProbeConfig.reset_stable_s)
    parser.add_argument("--reset-stable-max-speed-mps", type=float, default=MotorProbeConfig.reset_stable_max_speed_mps)
    parser.add_argument("--post-reset-delay-s", type=float, default=MotorProbeConfig.post_reset_delay_s)
    parser.add_argument("--abort-max-speed-mps", type=float, default=MotorProbeConfig.abort_max_speed_mps)
    parser.add_argument("--abort-max-displacement-m", type=float, default=MotorProbeConfig.abort_max_displacement_m)
    parser.add_argument("--abort-max-body-rate-rps", type=float, default=MotorProbeConfig.abort_max_body_rate_rps)
    parser.add_argument("--abort-on-new-collision", action="store_true")
    parser.add_argument("--abort-collision-threat-level", type=int, default=MotorProbeConfig.abort_collision_threat_level)
    parser.add_argument(
        "--abort-max-collision-impact-kg-mps",
        type=float,
        default=MotorProbeConfig.abort_max_collision_impact_kg_mps,
    )
    parser.add_argument("--precase-clearance", action="store_true")
    parser.add_argument("--precase-clearance-motor-value", type=float, default=MotorProbeConfig.precase_clearance_motor_value)
    parser.add_argument("--precase-clearance-s", type=float, default=MotorProbeConfig.precase_clearance_s)
    parser.add_argument("--precase-clearance-rest-s", type=float, default=MotorProbeConfig.precase_clearance_rest_s)
    parser.add_argument("--precase-clearance-repeat", type=int, default=MotorProbeConfig.precase_clearance_repeat)
    parser.add_argument("--precase-collision-quiet-s", type=float, default=MotorProbeConfig.precase_collision_quiet_s)
    parser.add_argument(
        "--precase-collision-quiet-timeout-s",
        type=float,
        default=MotorProbeConfig.precase_collision_quiet_timeout_s,
    )
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-disarm-on-exit", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=MotorProbeConfig.output_path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = MotorProbeConfig(
        endpoint=args.endpoint,
        command_hz=args.command_hz,
        sample_hz=args.sample_hz,
        pulse_s=args.pulse_s,
        rest_s=args.rest_s,
        start_delay_s=args.start_delay_s,
        include_basic_cases=not args.no_basic_cases,
        motor_values=tuple(float(value) for value in args.motor_values),
        motor_patterns=tuple(str(value) for value in args.motor_patterns),
        mixer_base_values=tuple(float(value) for value in args.mixer_base_values),
        mixer_deltas=tuple(float(value) for value in args.mixer_deltas),
        mixer_axes=tuple(str(value) for value in args.mixer_axes),
        arm_on_start=not args.no_arm,
        disarm_on_exit=not args.no_disarm_on_exit,
        reset_on_start=bool(args.reset_on_start),
        reset_between_cases=bool(args.reset_between_cases),
        reset_wait_s=args.reset_wait_s,
        reset_ready_timeout_s=args.reset_ready_timeout_s,
        reset_stable_s=args.reset_stable_s,
        reset_stable_max_speed_mps=args.reset_stable_max_speed_mps,
        post_reset_delay_s=args.post_reset_delay_s,
        abort_max_speed_mps=args.abort_max_speed_mps,
        abort_max_displacement_m=args.abort_max_displacement_m,
        abort_max_body_rate_rps=args.abort_max_body_rate_rps,
        abort_on_new_collision=bool(args.abort_on_new_collision),
        abort_collision_threat_level=args.abort_collision_threat_level,
        abort_max_collision_impact_kg_mps=args.abort_max_collision_impact_kg_mps,
        precase_clearance=bool(args.precase_clearance),
        precase_clearance_motor_value=args.precase_clearance_motor_value,
        precase_clearance_s=args.precase_clearance_s,
        precase_clearance_rest_s=args.precase_clearance_rest_s,
        precase_clearance_repeat=args.precase_clearance_repeat,
        precase_collision_quiet_s=args.precase_collision_quiet_s,
        precase_collision_quiet_timeout_s=args.precase_collision_quiet_timeout_s,
        progress=bool(args.progress),
        output_path=args.output,
    )
    payload = MotorCommandProbe(MavlinkBridge(config.endpoint), config).run_live()
    summary = {
        "created_utc": payload["created_utc"],
        "output_path": None if config.output_path is None else str(config.output_path),
        "completed_case_count": payload["completed_case_count"],
        "results": [
            {
                "motor_value": result["motor_value"],
                "motor_pattern": result["motor_pattern"],
                "mixer_axis": result["mixer_axis"],
                "mixer_delta": result["mixer_delta"],
                "mixer_base": result["mixer_base"],
                "motor_command": result["motor_command"],
                "displacement_local_ned_m": result["displacement_local_ned_m"],
                "max_speed_mps": result["max_speed_mps"],
                "body_rate_delta_rps": result["body_rate_delta_rps"],
                "max_body_rate_rps": result["max_body_rate_rps"],
                "collision_summary": result.get("collision_summary"),
                "safety_abort_reason": result.get("safety_abort_reason"),
                "start_position_local_ned_m": result["start"]["position_local_ned_m"],
                "end_position_local_ned_m": result["end"]["position_local_ned_m"],
                "start_euler_deg": result["start"]["euler_deg"],
                "end_euler_deg": result["end"]["euler_deg"],
            }
            for result in payload["results"]
        ],
        "clearance_reports": [
            {
                "label": report["label"],
                "motor_command": report["motor_command"],
                "repeat": report["repeat"],
                "pulse_s": report["pulse_s"],
                "rest_s": report["rest_s"],
                "displacement_local_ned_m": report["displacement_local_ned_m"],
                "collision_summary": report["collision_summary"],
                "safety_abort_reason": report["safety_abort_reason"],
                "start_position_local_ned_m": report["start"]["position_local_ned_m"],
                "end_position_local_ned_m": report["end"]["position_local_ned_m"],
                "start_euler_deg": report["start"]["euler_deg"],
                "end_euler_deg": report["end"]["euler_deg"],
            }
            for report in payload["clearance_reports"]
        ],
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
