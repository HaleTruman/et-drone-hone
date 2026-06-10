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
class AttitudeProbeSnapshot:
    monotonic_s: float
    sim_time_ns: int | None
    position_local_ned_m: tuple[float, float, float] | None
    velocity_local_ned_mps: tuple[float, float, float] | None
    attitude: tuple[float, float, float, float] | None
    reset_count: int | None
    actuator_output: tuple[float, ...] | None


@dataclass(frozen=True)
class ThrustCase:
    name: str
    thrust: float
    roll_offset_rad: float = 0.0
    pitch_offset_rad: float = 0.0
    yaw_offset_rad: float = 0.0


@dataclass(frozen=True)
class ThrustCaseResult:
    name: str
    command_payload: dict[str, Any]
    start: AttitudeProbeSnapshot
    end: AttitudeProbeSnapshot
    displacement_local_ned_m: tuple[float, float, float] | None
    total_displacement_m: float | None
    max_speed_mps: float | None
    max_actuator_output: float | None
    min_actuator_output: float | None
    reset_changed: bool


@dataclass(frozen=True)
class AttitudeThrustProbeConfig:
    endpoint: str = Q1RuntimeConfig.mavlink_endpoint
    command_hz: float = 30.0
    sample_hz: float = 30.0
    pulse_s: float = 1.0
    rest_s: float = 0.75
    start_delay_s: float = 0.0
    wait_for_race_start: bool = False
    race_start_margin_s: float = 0.5
    race_start_timeout_s: float = 20.0
    thrust_values: tuple[float, ...] = (0.2, 0.35, 0.5)
    roll_offsets_rad: tuple[float, ...] = (0.0,)
    pitch_offsets_rad: tuple[float, ...] = (0.0,)
    yaw_offsets_rad: tuple[float, ...] = (0.0,)
    attitude_mode: str = "measured"
    attitude_type_mask: int = 7
    body_rates_rps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_on_start: bool = True
    disarm_on_exit: bool = True
    reset_on_start: bool = False
    reset_between_cases: bool = False
    reset_wait_s: float = 2.0
    reset_ready_timeout_s: float = 20.0
    reset_stable_s: float = 2.0
    reset_stable_max_speed_mps: float = 0.05
    post_reset_delay_s: float = 2.0
    max_speed_mps: float = 8.0
    min_z_ned_m: float = -8.0
    max_z_ned_m: float = 3.0
    safety_enabled: bool = True
    progress: bool = False
    output_path: Path | None = Path("logs/q1runtime/attitude-thrust-probe.json")


class AttitudeThrustProbe:
    """Probe SET_ATTITUDE_TARGET collective thrust while holding measured attitude."""

    def __init__(self, bridge: Any, config: AttitudeThrustProbeConfig | None = None):
        self.bridge = bridge
        self.config = config or AttitudeThrustProbeConfig()

    def cases(self) -> list[ThrustCase]:
        cases: list[ThrustCase] = []
        for thrust in self.config.thrust_values:
            for roll in self.config.roll_offsets_rad:
                for pitch in self.config.pitch_offsets_rad:
                    for yaw in self.config.yaw_offsets_rad:
                        parts = [f"thrust_{self._fmt(thrust)}"]
                        if abs(float(roll)) > 1e-12:
                            parts.append(f"roll_{self._fmt(roll)}")
                        if abs(float(pitch)) > 1e-12:
                            parts.append(f"pitch_{self._fmt(pitch)}")
                        if abs(float(yaw)) > 1e-12:
                            parts.append(f"yaw_{self._fmt(yaw)}")
                        cases.append(
                            ThrustCase(
                                "attitude_" + "_".join(parts),
                                float(thrust),
                                float(roll),
                                float(pitch),
                                float(yaw),
                            )
                        )
        return cases

    def run_live(self) -> dict[str, Any]:
        self.bridge.connect()
        self.bridge.start_heartbeat()
        self.bridge.subscribe_telemetry()
        self._wait_for_telemetry()
        if self.config.reset_on_start:
            self._reset_simulator("sending simulator reset before attitude thrust probe")
        if self.config.arm_on_start:
            self.bridge.arm()
            time.sleep(0.2)
        if self.config.wait_for_race_start:
            self._wait_for_race_start()
        if self.config.start_delay_s > 0.0:
            self._progress(f"waiting start_delay_s={self.config.start_delay_s:g} before first attitude target")
            time.sleep(self.config.start_delay_s)

        results: list[ThrustCaseResult] = []
        aborted = False
        abort_reason: str | None = None
        cleanup_error: str | None = None
        try:
            preflight_ok = True
            if self.config.safety_enabled:
                try:
                    self._raise_if_unsafe(self.snapshot())
                except RuntimeError as exc:
                    aborted = True
                    abort_reason = f"preflight_{exc}"
                    preflight_ok = False
                    self._progress(f"preflight safety abort before first command: {abort_reason}")
            if preflight_ok:
                for index, case in enumerate(self.cases()):
                    if index > 0 and self.config.reset_between_cases:
                        self._reset_simulator(f"resetting simulator before case {index + 1}/{len(self.cases())}")
                        if self.config.arm_on_start:
                            self.bridge.arm()
                            time.sleep(0.2)
                    start = self.snapshot()
                    quaternion = self.command_quaternion(start, case)
                    payload = self.build_payload(case, quaternion)
                    self._progress(f"[{index + 1}/{len(self.cases())}] start {case.name} payload={payload}")
                    try:
                        samples = self._stream_payload(payload, self.config.pulse_s)
                    except RuntimeError as exc:
                        aborted = True
                        abort_reason = str(exc)
                        self._progress(f"[{index + 1}/{len(self.cases())}] safety abort {case.name}: {abort_reason}")
                        break
                    self._stream_payload(
                        self.build_payload(ThrustCase("rest_zero_thrust", 0.0), quaternion),
                        self.config.rest_s,
                        check_safety=False,
                    )
                    end = self.snapshot()
                    result = self.summarize_case(case, payload, start, end, samples)
                    results.append(result)
                    self._progress(
                        f"[{index + 1}/{len(self.cases())}] done {case.name} "
                        f"disp={result.displacement_local_ned_m} max_speed={result.max_speed_mps} "
                        f"actuator_range=({result.min_actuator_output}, {result.max_actuator_output})"
                    )
        finally:
            try:
                self._send_zero_thrust(0.2)
            except Exception as exc:  # Keep shutdown/disarm best-effort after safety trips.
                cleanup_error = repr(exc)
                self._progress(f"cleanup zero-thrust failed: {cleanup_error}")
            if self.config.disarm_on_exit:
                try:
                    self.bridge.disarm()
                except Exception as exc:
                    cleanup_error = repr(exc) if cleanup_error is None else f"{cleanup_error}; disarm={exc!r}"
            self.bridge.shutdown()

        return self._final_payload(results, aborted, abort_reason, cleanup_error)

    def _final_payload(
        self,
        results: list[ThrustCaseResult],
        aborted: bool,
        abort_reason: str | None,
        cleanup_error: str | None,
    ) -> dict[str, Any]:
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "case_count": len(self.cases()),
            "completed_case_count": len(results),
            "aborted": aborted,
            "abort_reason": abort_reason,
            "cleanup_error": cleanup_error,
            "results": [asdict(result) for result in results],
            "bridge_snapshot": self._bridge_snapshot(),
        }
        if self.config.output_path is not None:
            path = Path(self.config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    def build_payload(self, case: ThrustCase, quaternion: tuple[float, float, float, float]) -> dict[str, Any]:
        return {
            "quaternion": [float(value) for value in quaternion],
            "thrust": float(np.clip(case.thrust, 0.0, 1.0)),
            "attitude_type_mask": int(self.config.attitude_type_mask),
            "body_rates_rps": [float(value) for value in self.config.body_rates_rps],
            "source": "q1runtime_attitude_thrust_probe",
            "probe_case": case.name,
            "roll_offset_rad": float(case.roll_offset_rad),
            "pitch_offset_rad": float(case.pitch_offset_rad),
            "yaw_offset_rad": float(case.yaw_offset_rad),
        }

    def command_quaternion(self, snapshot: AttitudeProbeSnapshot, case: ThrustCase) -> tuple[float, float, float, float]:
        mode = str(self.config.attitude_mode).lower()
        if mode == "measured":
            base = snapshot.attitude or (1.0, 0.0, 0.0, 0.0)
        elif mode == "level":
            base = (1.0, 0.0, 0.0, 0.0)
        elif mode in {"level_yaw_pi", "level-yaw-pi"}:
            base = (0.0, 0.0, 0.0, 1.0)
        else:
            raise ValueError("attitude_mode must be measured, level, or level_yaw_pi")
        offset = self._quaternion_from_roll_pitch_yaw(
            case.roll_offset_rad,
            case.pitch_offset_rad,
            case.yaw_offset_rad,
        )
        target = self._quaternion_multiply(offset, base)
        norm = float(np.linalg.norm(target))
        if norm <= 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        return tuple(float(value) for value in target / norm)

    def summarize_case(
        self,
        case: ThrustCase,
        command_payload: dict[str, Any],
        start: AttitudeProbeSnapshot,
        end: AttitudeProbeSnapshot,
        samples: list[AttitudeProbeSnapshot],
    ) -> ThrustCaseResult:
        displacement = None
        total_displacement = None
        if start.position_local_ned_m is not None and end.position_local_ned_m is not None:
            delta = np.asarray(end.position_local_ned_m, dtype=float) - np.asarray(start.position_local_ned_m, dtype=float)
            displacement = tuple(float(value) for value in delta)
            total_displacement = float(np.linalg.norm(delta))
        velocities = [sample.velocity_local_ned_mps for sample in samples if sample.velocity_local_ned_mps is not None]
        max_speed = None
        if velocities:
            max_speed = float(np.max(np.linalg.norm(np.asarray(velocities, dtype=float), axis=1)))
        actuators = [sample.actuator_output for sample in samples if sample.actuator_output is not None]
        actuator_values = [float(value) for actuator in actuators for value in actuator]
        return ThrustCaseResult(
            name=case.name,
            command_payload=command_payload,
            start=start,
            end=end,
            displacement_local_ned_m=displacement,
            total_displacement_m=total_displacement,
            max_speed_mps=max_speed,
            max_actuator_output=None if not actuator_values else max(actuator_values),
            min_actuator_output=None if not actuator_values else min(actuator_values),
            reset_changed=(
                start.reset_count is not None
                and end.reset_count is not None
                and int(start.reset_count) != int(end.reset_count)
            ),
        )

    def snapshot(self) -> AttitudeProbeSnapshot:
        telemetry = self.bridge.get_latest_telemetry()
        actuator_output = self._actuator_output()
        if telemetry is None:
            return AttitudeProbeSnapshot(time.monotonic(), None, None, None, None, None, actuator_output)
        return AttitudeProbeSnapshot(
            monotonic_s=time.monotonic(),
            sim_time_ns=self._optional_int(getattr(telemetry, "sim_time_ns", None)),
            position_local_ned_m=self._optional_tuple(getattr(telemetry, "position_local_ned_m", None), 3),
            velocity_local_ned_mps=self._optional_tuple(getattr(telemetry, "velocity_local_ned_mps", None), 3),
            attitude=self._optional_tuple(getattr(telemetry, "attitude", None), 4),
            reset_count=self._optional_int(getattr(telemetry, "reset_count", None)),
            actuator_output=actuator_output,
        )

    def _stream_payload(
        self,
        payload: dict[str, Any],
        duration_s: float,
        *,
        check_safety: bool = True,
    ) -> list[AttitudeProbeSnapshot]:
        samples: list[AttitudeProbeSnapshot] = []
        command_interval = 1.0 / max(1e-6, float(self.config.command_hz))
        sample_interval = 1.0 / max(1e-6, float(self.config.sample_hz))
        deadline_s = time.monotonic() + max(0.0, float(duration_s))
        next_command_s = time.monotonic()
        next_sample_s = next_command_s
        while time.monotonic() < deadline_s:
            now_s = time.monotonic()
            if now_s >= next_command_s:
                self.bridge.send_attitude_target(payload)
                next_command_s += command_interval
            if now_s >= next_sample_s:
                sample = self.snapshot()
                samples.append(sample)
                if check_safety and self.config.safety_enabled:
                    self._raise_if_unsafe(sample)
                next_sample_s += sample_interval
            time.sleep(min(0.002, command_interval / 2.0, sample_interval / 2.0))
        return samples

    def _send_zero_thrust(self, duration_s: float) -> None:
        snapshot = self.snapshot()
        quaternion = snapshot.attitude or (1.0, 0.0, 0.0, 0.0)
        self._stream_payload(
            self.build_payload(ThrustCase("exit_zero_thrust", 0.0), quaternion),
            duration_s,
            check_safety=False,
        )

    def _wait_for_telemetry(self) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is not None and getattr(telemetry, "position_local_ned_m", None) is not None:
                return
            time.sleep(0.02)
        raise TimeoutError("No local-NED telemetry received for attitude thrust probe.")

    def _reset_simulator(self, message: str) -> None:
        self._progress(message)
        previous = self.bridge.get_latest_telemetry()
        previous_reset_count = None if previous is None else getattr(previous, "reset_count", None)
        try:
            self._send_zero_thrust(0.1)
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

    def _wait_for_race_start(self) -> None:
        deadline = time.monotonic() + max(0.01, float(self.config.race_start_timeout_s))
        margin_ms = int(max(0.0, float(self.config.race_start_margin_s)) * 1000.0)
        while time.monotonic() < deadline:
            race_status = getattr(self.bridge, "race_status", None)
            if race_status is not None:
                sim_boot_ms = int(getattr(race_status, "sim_boot_time_ms", 0))
                race_start_ms = int(getattr(race_status, "race_start_boot_time_ms", 0))
                if race_start_ms > 0 and sim_boot_ms >= race_start_ms + margin_ms:
                    self._progress(
                        f"race start gate satisfied sim_boot_ms={sim_boot_ms} "
                        f"race_start_boot_time_ms={race_start_ms} margin_ms={margin_ms}"
                    )
                    return
            time.sleep(0.1)
        raise TimeoutError("Race start gate did not open before timeout.")

    def _raise_if_unsafe(self, sample: AttitudeProbeSnapshot) -> None:
        if sample.position_local_ned_m is not None:
            z = float(sample.position_local_ned_m[2])
            if z < self.config.min_z_ned_m or z > self.config.max_z_ned_m:
                raise RuntimeError(f"z safety bound exceeded: {z}")
        if sample.velocity_local_ned_mps is not None:
            speed = float(np.linalg.norm(np.asarray(sample.velocity_local_ned_mps, dtype=float)))
            if speed > self.config.max_speed_mps or not math.isfinite(speed):
                raise RuntimeError(f"velocity safety bound exceeded: {speed}")

    def _actuator_output(self) -> tuple[float, ...] | None:
        latest = getattr(self.bridge, "latest_actuator_output", None)
        if not latest:
            return None
        commands = latest.get("motor_commands") if isinstance(latest, dict) else None
        return None if commands is None else tuple(float(value) for value in commands)

    @staticmethod
    def _quaternion_from_roll_pitch_yaw(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(float(roll) / 2.0), np.sin(float(roll) / 2.0)
        cp, sp = np.cos(float(pitch) / 2.0), np.sin(float(pitch) / 2.0)
        cy, sy = np.cos(float(yaw) / 2.0), np.sin(float(yaw) / 2.0)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )

    @staticmethod
    def _quaternion_multiply(a: np.ndarray | tuple[float, ...], b: np.ndarray | tuple[float, ...]) -> np.ndarray:
        aw, ax, ay, az = np.asarray(a, dtype=float)
        bw, bx, by, bz = np.asarray(b, dtype=float)
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=float,
        )

    def _bridge_snapshot(self) -> dict[str, Any] | None:
        snapshot = getattr(self.bridge, "snapshot", None)
        if not callable(snapshot):
            return None
        return snapshot()

    def _progress(self, message: str) -> None:
        if self.config.progress:
            print(message, flush=True)

    @staticmethod
    def _optional_tuple(values: Any, length: int) -> Any:
        if values is None:
            return None
        converted = tuple(float(value) for value in values)
        if len(converted) != length:
            return None
        return converted

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{float(value):g}".replace("-", "neg")


def _float_tuple_from_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value.")
    return values


def _float_tuple3_from_csv(raw: str) -> tuple[float, float, float]:
    values = _float_tuple_from_csv(raw)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three comma-separated numeric values.")
    return values  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe SET_ATTITUDE_TARGET thrust authority.")
    parser.add_argument("--endpoint", default=AttitudeThrustProbeConfig.endpoint)
    parser.add_argument("--command-hz", type=float, default=AttitudeThrustProbeConfig.command_hz)
    parser.add_argument("--sample-hz", type=float, default=AttitudeThrustProbeConfig.sample_hz)
    parser.add_argument("--pulse-s", type=float, default=AttitudeThrustProbeConfig.pulse_s)
    parser.add_argument("--rest-s", type=float, default=AttitudeThrustProbeConfig.rest_s)
    parser.add_argument("--start-delay-s", type=float, default=AttitudeThrustProbeConfig.start_delay_s)
    parser.add_argument("--wait-for-race-start", action="store_true")
    parser.add_argument("--race-start-margin-s", type=float, default=AttitudeThrustProbeConfig.race_start_margin_s)
    parser.add_argument("--race-start-timeout-s", type=float, default=AttitudeThrustProbeConfig.race_start_timeout_s)
    parser.add_argument("--thrust-values", type=_float_tuple_from_csv, default=AttitudeThrustProbeConfig.thrust_values)
    parser.add_argument("--roll-offsets-rad", type=_float_tuple_from_csv, default=AttitudeThrustProbeConfig.roll_offsets_rad)
    parser.add_argument("--pitch-offsets-rad", type=_float_tuple_from_csv, default=AttitudeThrustProbeConfig.pitch_offsets_rad)
    parser.add_argument("--yaw-offsets-rad", type=_float_tuple_from_csv, default=AttitudeThrustProbeConfig.yaw_offsets_rad)
    parser.add_argument("--attitude-mode", default=AttitudeThrustProbeConfig.attitude_mode, choices=("measured", "level", "level_yaw_pi"))
    parser.add_argument("--attitude-type-mask", type=int, default=AttitudeThrustProbeConfig.attitude_type_mask)
    parser.add_argument("--body-rates", type=_float_tuple3_from_csv, default=AttitudeThrustProbeConfig.body_rates_rps)
    parser.add_argument("--reset-on-start", action="store_true")
    parser.add_argument("--reset-between-cases", action="store_true")
    parser.add_argument("--reset-wait-s", type=float, default=AttitudeThrustProbeConfig.reset_wait_s)
    parser.add_argument("--reset-ready-timeout-s", type=float, default=AttitudeThrustProbeConfig.reset_ready_timeout_s)
    parser.add_argument("--reset-stable-s", type=float, default=AttitudeThrustProbeConfig.reset_stable_s)
    parser.add_argument("--reset-stable-max-speed-mps", type=float, default=AttitudeThrustProbeConfig.reset_stable_max_speed_mps)
    parser.add_argument("--post-reset-delay-s", type=float, default=AttitudeThrustProbeConfig.post_reset_delay_s)
    parser.add_argument("--max-speed-mps", type=float, default=AttitudeThrustProbeConfig.max_speed_mps)
    parser.add_argument("--min-z-ned-m", type=float, default=AttitudeThrustProbeConfig.min_z_ned_m)
    parser.add_argument("--max-z-ned-m", type=float, default=AttitudeThrustProbeConfig.max_z_ned_m)
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-disarm-on-exit", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=AttitudeThrustProbeConfig.output_path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = AttitudeThrustProbeConfig(
        endpoint=args.endpoint,
        command_hz=args.command_hz,
        sample_hz=args.sample_hz,
        pulse_s=args.pulse_s,
        rest_s=args.rest_s,
        start_delay_s=args.start_delay_s,
        wait_for_race_start=bool(args.wait_for_race_start),
        race_start_margin_s=args.race_start_margin_s,
        race_start_timeout_s=args.race_start_timeout_s,
        thrust_values=tuple(float(value) for value in args.thrust_values),
        roll_offsets_rad=tuple(float(value) for value in args.roll_offsets_rad),
        pitch_offsets_rad=tuple(float(value) for value in args.pitch_offsets_rad),
        yaw_offsets_rad=tuple(float(value) for value in args.yaw_offsets_rad),
        attitude_mode=args.attitude_mode,
        attitude_type_mask=int(args.attitude_type_mask),
        body_rates_rps=tuple(float(value) for value in args.body_rates),
        reset_on_start=bool(args.reset_on_start),
        reset_between_cases=bool(args.reset_between_cases),
        reset_wait_s=args.reset_wait_s,
        reset_ready_timeout_s=args.reset_ready_timeout_s,
        reset_stable_s=args.reset_stable_s,
        reset_stable_max_speed_mps=args.reset_stable_max_speed_mps,
        post_reset_delay_s=args.post_reset_delay_s,
        max_speed_mps=args.max_speed_mps,
        min_z_ned_m=args.min_z_ned_m,
        max_z_ned_m=args.max_z_ned_m,
        safety_enabled=not args.no_safety,
        arm_on_start=not args.no_arm,
        disarm_on_exit=not args.no_disarm_on_exit,
        progress=bool(args.progress),
        output_path=args.output,
    )
    payload = AttitudeThrustProbe(MavlinkBridge(config.endpoint), config).run_live()
    summary = {
        "created_utc": payload["created_utc"],
        "case_count": payload["case_count"],
        "completed_case_count": payload["completed_case_count"],
        "aborted": payload["aborted"],
        "abort_reason": payload["abort_reason"],
        "cleanup_error": payload["cleanup_error"],
        "output_path": None if config.output_path is None else str(config.output_path),
        "results": [
            {
                "name": result["name"],
                "thrust": result["command_payload"]["thrust"],
                "displacement_local_ned_m": result["displacement_local_ned_m"],
                "max_speed_mps": result["max_speed_mps"],
                "min_actuator_output": result["min_actuator_output"],
                "max_actuator_output": result["max_actuator_output"],
                "reset_changed": result["reset_changed"],
            }
            for result in payload["results"]
        ],
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
