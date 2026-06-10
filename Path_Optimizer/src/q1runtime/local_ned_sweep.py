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


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class TelemetrySnapshot:
    monotonic_s: float
    sim_time_ns: int | None
    position_local_ned_m: tuple[float, float, float] | None
    velocity_local_ned_mps: tuple[float, float, float] | None
    acceleration_local_ned_mps2: tuple[float, float, float] | None
    attitude: tuple[float, float, float, float] | None
    body_rates_rps: tuple[float, float, float] | None
    reset_count: int | None
    actuator_output: tuple[float, ...] | None


@dataclass(frozen=True)
class SweepCase:
    name: str
    mode: str
    axis: str | None
    sign: int | None
    magnitude: float
    velocity_local_ned_mps: tuple[float, float, float]
    position_offset_local_ned_m: tuple[float, float, float] | None = None
    position_axes: tuple[bool, bool, bool] | None = None
    yaw_rad: float | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class SweepSafetyStop:
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True)
class SweepCaseResult:
    name: str
    mode: str
    command_payload: dict[str, Any]
    start: TelemetrySnapshot
    end: TelemetrySnapshot
    samples: tuple[TelemetrySnapshot, ...]
    displacement_local_ned_m: tuple[float, float, float] | None
    total_displacement_m: float | None
    horizontal_displacement_m: float | None
    z_displacement_m: float | None
    dominant_axis: str | None
    dominant_sign: int | None
    commanded_axis: str | None
    commanded_sign: int | None
    max_speed_mps: float | None
    mean_velocity_local_ned_mps: tuple[float, float, float] | None
    response_gain_m_per_mps_s: float | None
    cross_axis_ratio: float | None
    reset_changed: bool
    actuator_observed: bool
    safety_stop: SweepSafetyStop | None


@dataclass(frozen=True)
class LocalNedSweepConfig:
    endpoint: str = Q1RuntimeConfig.mavlink_endpoint
    command_hz: float = 30.0
    sample_hz: float = 30.0
    pulse_s: float = 0.25
    rest_s: float = 0.75
    settle_s: float = 0.25
    start_delay_s: float = 0.0
    wait_for_race_start: bool = False
    race_start_margin_s: float = 0.5
    race_start_timeout_s: float = 20.0
    reset_wait_s: float = 0.75
    arm_settle_s: float = 0.2
    stop_on_exit_s: float = 0.2
    wait_for_telemetry_timeout_s: float = 10.0
    speeds_mps: tuple[float, ...] = (0.05, 0.1, 0.25)
    position_offsets_m: tuple[float, ...] = (0.5,)
    axes: tuple[str, ...] = ("x", "y")
    signs: tuple[int, ...] = (1, -1)
    case_modes: tuple[str, ...] = ("velocity",)
    include_z: bool = False
    include_position: bool = False
    include_position_velocity: bool = False
    include_yaw: bool = False
    yaw_angles_rad: tuple[float, ...] = (math.pi / 2.0, -math.pi / 2.0)
    arm_on_start: bool = True
    disarm_on_exit: bool = True
    reset_before_first_case: bool = False
    reset_between_cases: bool = True
    abort_on_safety: bool = True
    abort_on_reset_change: bool = True
    max_case_displacement_m: float = 5.0
    max_horizontal_radius_m: float = 20.0
    min_z_ned_m: float = -8.0
    max_z_ned_m: float = 3.0
    max_speed_mps: float = 8.0
    max_cases: int | None = None
    progress: bool = False
    output_path: Path | None = Path("logs/q1runtime/local-ned-sweep.json")


class LocalNedCommandSweep:
    """Independent live probe for SET_POSITION_TARGET_LOCAL_NED command response."""

    def __init__(self, bridge: Any, config: LocalNedSweepConfig | None = None):
        self.bridge = bridge
        self.config = config or LocalNedSweepConfig()

    def build_cases(self) -> list[SweepCase]:
        axes = self._validated_axes()
        modes = self._validated_modes()
        signs = self._validated_signs()
        cases: list[SweepCase] = []
        if "velocity" in modes:
            for speed in self.config.speeds_mps:
                for axis in axes:
                    for sign in signs:
                        cases.append(
                            SweepCase(
                                name=f"velocity_{self._sign_name(sign)}{axis}_{self._fmt(speed)}mps",
                                mode="velocity",
                                axis=axis,
                                sign=sign,
                                magnitude=float(speed),
                                velocity_local_ned_mps=self._axis_vector(axis, sign * float(speed)),
                            )
                        )

        if "position" in modes:
            for offset_m in self.config.position_offsets_m:
                for axis in axes:
                    for sign in signs:
                        cases.append(
                            SweepCase(
                                name=f"position_{self._sign_name(sign)}{axis}_{self._fmt(offset_m)}m",
                                mode="position",
                                axis=axis,
                                sign=sign,
                                magnitude=float(offset_m),
                                velocity_local_ned_mps=(0.0, 0.0, 0.0),
                                position_offset_local_ned_m=self._axis_vector(axis, sign * float(offset_m)),
                                position_axes=(True, True, True),
                            )
                        )

        if "position_velocity" in modes:
            for offset_m in self.config.position_offsets_m:
                for speed in self.config.speeds_mps:
                    for axis in axes:
                        for sign in signs:
                            cases.append(
                                SweepCase(
                                    name=(
                                        f"position_velocity_{self._sign_name(sign)}{axis}_"
                                        f"{self._fmt(offset_m)}m_{self._fmt(speed)}mps"
                                    ),
                                    mode="position_velocity",
                                    axis=axis,
                                    sign=sign,
                                    magnitude=float(speed),
                                    velocity_local_ned_mps=self._axis_vector(axis, sign * float(speed)),
                                    position_offset_local_ned_m=self._axis_vector(axis, sign * float(offset_m)),
                                    position_axes=(True, True, True),
                                )
                            )

        if "yaw" in modes:
            for yaw in self.config.yaw_angles_rad:
                cases.append(
                    SweepCase(
                        name=f"yaw_{self._fmt(yaw)}rad",
                        mode="yaw",
                        axis=None,
                        sign=None,
                        magnitude=float(yaw),
                        velocity_local_ned_mps=(0.0, 0.0, 0.0),
                        yaw_rad=float(yaw),
                    )
                )

        if self.config.max_cases is not None:
            return cases[: max(0, int(self.config.max_cases))]
        return cases

    def build_payload(self, case: SweepCase, start: TelemetrySnapshot) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "velocity_local_ned_mps": [float(value) for value in case.velocity_local_ned_mps],
            "yaw_rad": None if case.yaw_rad is None else float(case.yaw_rad),
            "source": "q1runtime_local_ned_sweep",
            "sweep_case": case.name,
            "sweep_mode": case.mode,
            "sweep_axis": case.axis,
            "sweep_sign": case.sign,
            "sweep_magnitude": float(case.magnitude),
        }
        if case.position_offset_local_ned_m is not None:
            if start.position_local_ned_m is None:
                raise RuntimeError(f"Case {case.name} requires position_local_ned_m telemetry.")
            position = np.asarray(start.position_local_ned_m, dtype=float) + np.asarray(
                case.position_offset_local_ned_m, dtype=float
            )
            payload["position_local_ned_m"] = [float(value) for value in position]
            payload["position_offset_local_ned_m"] = [float(value) for value in case.position_offset_local_ned_m]
            payload["position_axes"] = [bool(value) for value in (case.position_axes or (True, True, True))]
        return payload

    def summarize_case(
        self,
        case: SweepCase,
        *,
        command_payload: dict[str, Any],
        start: TelemetrySnapshot,
        end: TelemetrySnapshot,
        samples: tuple[TelemetrySnapshot, ...],
        safety_stop: SweepSafetyStop | None = None,
    ) -> SweepCaseResult:
        displacement = None
        total_displacement = None
        horizontal_displacement = None
        z_displacement = None
        dominant_axis = None
        dominant_sign = None
        cross_axis_ratio = None
        if start.position_local_ned_m is not None and end.position_local_ned_m is not None:
            delta = np.asarray(end.position_local_ned_m, dtype=float) - np.asarray(start.position_local_ned_m, dtype=float)
            displacement = tuple(float(value) for value in delta)
            total_displacement = float(np.linalg.norm(delta))
            horizontal_displacement = float(np.linalg.norm(delta[:2]))
            z_displacement = float(delta[2])
            dominant_index = int(np.argmax(np.abs(delta)))
            dominant_axis = AXIS_NAMES[dominant_index]
            dominant_sign = 1 if float(delta[dominant_index]) >= 0.0 else -1
            if case.axis in AXIS_INDEX:
                commanded_index = AXIS_INDEX[case.axis]
                commanded_abs = abs(float(delta[commanded_index]))
                other_abs = [abs(float(delta[index])) for index in range(3) if index != commanded_index]
                cross_axis_ratio = max(other_abs) / max(commanded_abs, 1e-9)

        velocities = [snapshot.velocity_local_ned_mps for snapshot in samples if snapshot.velocity_local_ned_mps is not None]
        max_speed = None
        mean_velocity = None
        if velocities:
            velocity_array = np.asarray(velocities, dtype=float)
            speeds = np.linalg.norm(velocity_array, axis=1)
            max_speed = float(np.max(speeds))
            mean_velocity = tuple(float(value) for value in np.mean(velocity_array, axis=0))

        command_speed = float(np.linalg.norm(np.asarray(case.velocity_local_ned_mps, dtype=float)))
        duration_s = float(case.duration_s if case.duration_s is not None else self.config.pulse_s)
        response_gain = None
        if total_displacement is not None and command_speed > 1e-9 and duration_s > 1e-9:
            response_gain = total_displacement / (command_speed * duration_s)

        reset_changed = (
            start.reset_count is not None
            and end.reset_count is not None
            and int(start.reset_count) != int(end.reset_count)
        )
        actuator_observed = any(snapshot.actuator_output is not None for snapshot in (start, end, *samples))

        return SweepCaseResult(
            name=case.name,
            mode=case.mode,
            command_payload=command_payload,
            start=start,
            end=end,
            samples=samples,
            displacement_local_ned_m=displacement,
            total_displacement_m=total_displacement,
            horizontal_displacement_m=horizontal_displacement,
            z_displacement_m=z_displacement,
            dominant_axis=dominant_axis,
            dominant_sign=dominant_sign,
            commanded_axis=case.axis,
            commanded_sign=case.sign,
            max_speed_mps=max_speed,
            mean_velocity_local_ned_mps=mean_velocity,
            response_gain_m_per_mps_s=response_gain,
            cross_axis_ratio=cross_axis_ratio,
            reset_changed=reset_changed,
            actuator_observed=actuator_observed,
            safety_stop=safety_stop,
        )

    def run_live(self) -> dict[str, Any]:
        self.bridge.connect()
        self.bridge.start_heartbeat()
        self.bridge.subscribe_telemetry()
        self._wait_for_telemetry()
        if self.config.arm_on_start:
            self.bridge.arm()
            time.sleep(self.config.arm_settle_s)
        if self.config.wait_for_race_start:
            self._wait_for_race_start()
        if self.config.start_delay_s > 0.0:
            self._progress(f"waiting start_delay_s={self.config.start_delay_s:g} before first command")
            time.sleep(self.config.start_delay_s)

        cases = self.build_cases()
        results: list[SweepCaseResult] = []
        aborted = False
        abort_reason: str | None = None
        try:
            for index, case in enumerate(cases):
                self._progress(f"[{index + 1}/{len(cases)}] start {case.name} payload={case.velocity_local_ned_mps}")
                if (index == 0 and self.config.reset_before_first_case) or (index > 0 and self.config.reset_between_cases):
                    self._reset_case_origin()
                self._stream_stop(self.config.settle_s)
                start = self.snapshot()
                command_payload = self.build_payload(case, start)
                samples, safety_stop = self._stream_payload(command_payload, case.duration_s or self.config.pulse_s, start)
                self._stream_stop(self.config.rest_s)
                end = self.snapshot()
                result = self.summarize_case(
                    case,
                    command_payload=command_payload,
                    start=start,
                    end=end,
                    samples=tuple(samples),
                    safety_stop=safety_stop,
                )
                results.append(result)
                self._progress(
                    f"[{index + 1}/{len(cases)}] done {case.name} "
                    f"disp={result.displacement_local_ned_m} max_speed={result.max_speed_mps} "
                    f"safety={None if result.safety_stop is None else result.safety_stop.reason}"
                )
                if safety_stop is not None and self.config.abort_on_safety:
                    aborted = True
                    abort_reason = safety_stop.reason
                    break
        finally:
            self._stream_stop(self.config.stop_on_exit_s)
            if self.config.disarm_on_exit:
                self.bridge.disarm()
            self.bridge.shutdown()

        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "case_count": len(cases),
            "completed_case_count": len(results),
            "aborted": aborted,
            "abort_reason": abort_reason,
            "cases": [asdict(case) for case in cases],
            "results": [asdict(result) for result in results],
            "bridge_snapshot": self._bridge_snapshot(),
        }
        if self.config.output_path is not None:
            path = Path(self.config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    def summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "created_utc": payload.get("created_utc"),
            "case_count": payload.get("case_count"),
            "completed_case_count": payload.get("completed_case_count"),
            "aborted": payload.get("aborted"),
            "abort_reason": payload.get("abort_reason"),
            "output_path": None if self.config.output_path is None else str(self.config.output_path),
            "results": [
                {
                    "name": result["name"],
                    "velocity_local_ned_mps": result["command_payload"].get("velocity_local_ned_mps"),
                    "position_local_ned_m": result["command_payload"].get("position_local_ned_m"),
                    "yaw_rad": result["command_payload"].get("yaw_rad"),
                    "displacement_local_ned_m": result["displacement_local_ned_m"],
                    "total_displacement_m": result["total_displacement_m"],
                    "max_speed_mps": result["max_speed_mps"],
                    "dominant_axis": result["dominant_axis"],
                    "dominant_sign": result["dominant_sign"],
                    "cross_axis_ratio": result["cross_axis_ratio"],
                    "safety_stop": result["safety_stop"],
                }
                for result in payload.get("results", [])
            ],
        }

    def snapshot(self) -> TelemetrySnapshot:
        telemetry = self.bridge.get_latest_telemetry()
        actuator_output = self._actuator_output()
        if telemetry is None:
            return TelemetrySnapshot(
                monotonic_s=time.monotonic(),
                sim_time_ns=None,
                position_local_ned_m=None,
                velocity_local_ned_mps=None,
                acceleration_local_ned_mps2=None,
                attitude=None,
                body_rates_rps=None,
                reset_count=None,
                actuator_output=actuator_output,
            )
        return TelemetrySnapshot(
            monotonic_s=time.monotonic(),
            sim_time_ns=self._optional_int(getattr(telemetry, "sim_time_ns", None)),
            position_local_ned_m=self._optional_tuple(getattr(telemetry, "position_local_ned_m", None), 3),
            velocity_local_ned_mps=self._optional_tuple(getattr(telemetry, "velocity_local_ned_mps", None), 3),
            acceleration_local_ned_mps2=self._optional_tuple(getattr(telemetry, "acceleration_local_ned_mps2", None), 3),
            attitude=self._optional_tuple(getattr(telemetry, "attitude", None), 4),
            body_rates_rps=self._optional_tuple(getattr(telemetry, "body_rates_rps", None), 3),
            reset_count=self._optional_int(getattr(telemetry, "reset_count", None)),
            actuator_output=actuator_output,
        )

    def _stream_payload(
        self,
        payload: dict[str, Any],
        duration_s: float,
        start: TelemetrySnapshot,
    ) -> tuple[list[TelemetrySnapshot], SweepSafetyStop | None]:
        samples: list[TelemetrySnapshot] = []
        safety_stop = None
        command_interval = 1.0 / max(1e-6, float(self.config.command_hz))
        sample_interval = 1.0 / max(1e-6, float(self.config.sample_hz))
        next_command_s = time.monotonic()
        next_sample_s = next_command_s
        deadline_s = next_command_s + max(0.0, float(duration_s))
        while time.monotonic() < deadline_s:
            now_s = time.monotonic()
            if now_s >= next_command_s:
                self.bridge.send_position_target(payload)
                next_command_s += command_interval
            if now_s >= next_sample_s:
                sample = self.snapshot()
                samples.append(sample)
                safety_stop = self._safety_stop(start, sample)
                if safety_stop is not None:
                    break
                next_sample_s += sample_interval
            time.sleep(min(0.002, command_interval / 2.0, sample_interval / 2.0))
        return samples, safety_stop

    def _stream_stop(self, duration_s: float) -> None:
        stop_payload = {
            "velocity_local_ned_mps": [0.0, 0.0, 0.0],
            "yaw_rad": None,
            "source": "q1runtime_local_ned_sweep_stop",
        }
        end_s = time.monotonic() + max(0.0, float(duration_s))
        interval_s = 1.0 / max(1e-6, float(self.config.command_hz))
        while time.monotonic() < end_s:
            self.bridge.send_position_target(stop_payload)
            time.sleep(interval_s)

    def _progress(self, message: str) -> None:
        if self.config.progress:
            print(message, flush=True)

    def _reset_case_origin(self) -> None:
        self._stream_stop(0.1)
        previous = self.bridge.get_latest_telemetry()
        self.bridge.send_sim_reset_command()
        time.sleep(self.config.reset_wait_s)
        self._wait_for_telemetry(previous_telemetry=previous, require_new_sample=True)
        if self.config.arm_on_start:
            self.bridge.arm()
            time.sleep(self.config.arm_settle_s)

    def _wait_for_telemetry(self, *, previous_telemetry: Any | None = None, require_new_sample: bool = False) -> None:
        deadline = time.monotonic() + max(0.01, float(self.config.wait_for_telemetry_timeout_s))
        while time.monotonic() < deadline:
            telemetry = self.bridge.get_latest_telemetry()
            if telemetry is not None and getattr(telemetry, "position_local_ned_m", None) is not None:
                if require_new_sample and telemetry is previous_telemetry:
                    time.sleep(0.02)
                    continue
                return
            time.sleep(0.02)
        raise TimeoutError("No local-NED telemetry received for local-NED command sweep.")

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
                self._progress(
                    f"waiting race start sim_boot_ms={sim_boot_ms} "
                    f"race_start_boot_time_ms={race_start_ms} margin_ms={margin_ms}"
                )
            time.sleep(0.1)
        raise TimeoutError("Race start gate did not open before timeout.")

    def _safety_stop(self, start: TelemetrySnapshot, sample: TelemetrySnapshot) -> SweepSafetyStop | None:
        if sample.position_local_ned_m is not None:
            position = np.asarray(sample.position_local_ned_m, dtype=float)
            horizontal_radius = float(np.linalg.norm(position[:2]))
            if horizontal_radius > self.config.max_horizontal_radius_m:
                return SweepSafetyStop("horizontal_radius_exceeded", {"radius_m": horizontal_radius})
            z = float(position[2])
            if z < self.config.min_z_ned_m or z > self.config.max_z_ned_m:
                return SweepSafetyStop("z_bounds_exceeded", {"z_ned_m": z})
            if start.position_local_ned_m is not None:
                displacement = float(np.linalg.norm(position - np.asarray(start.position_local_ned_m, dtype=float)))
                if displacement > self.config.max_case_displacement_m:
                    return SweepSafetyStop("case_displacement_exceeded", {"displacement_m": displacement})

        if sample.velocity_local_ned_mps is not None:
            speed = float(np.linalg.norm(np.asarray(sample.velocity_local_ned_mps, dtype=float)))
            if speed > self.config.max_speed_mps or not math.isfinite(speed):
                return SweepSafetyStop("velocity_limit_exceeded", {"speed_mps": speed})

        if (
            self.config.abort_on_reset_change
            and start.reset_count is not None
            and sample.reset_count is not None
            and int(start.reset_count) != int(sample.reset_count)
        ):
            return SweepSafetyStop(
                "reset_count_changed",
                {"start_reset_count": int(start.reset_count), "sample_reset_count": int(sample.reset_count)},
            )
        return None

    def _validated_axes(self) -> tuple[str, ...]:
        axes = tuple(str(axis).lower() for axis in self.config.axes)
        invalid = [axis for axis in axes if axis not in AXIS_INDEX]
        if invalid:
            raise ValueError(f"Unsupported axes: {invalid}. Expected any of x,y,z.")
        if "z" in axes and not self.config.include_z:
            raise ValueError("Z-axis cases require include_z=True.")
        return axes

    def _validated_modes(self) -> tuple[str, ...]:
        modes_list = [str(mode).lower() for mode in self.config.case_modes]
        if self.config.include_position and "position" not in modes_list:
            modes_list.append("position")
        if self.config.include_position_velocity and "position_velocity" not in modes_list:
            modes_list.append("position_velocity")
        if self.config.include_yaw and "yaw" not in modes_list:
            modes_list.append("yaw")
        modes = tuple(modes_list)
        valid = {"velocity", "position", "position_velocity", "yaw"}
        invalid = [mode for mode in modes if mode not in valid]
        if invalid:
            raise ValueError(f"Unsupported case modes: {invalid}. Expected any of {sorted(valid)}.")
        if not modes:
            raise ValueError("At least one case mode is required.")
        return modes

    def _validated_signs(self) -> tuple[int, ...]:
        signs = tuple(1 if int(sign) >= 0 else -1 for sign in self.config.signs)
        if not signs:
            raise ValueError("At least one sign is required.")
        return signs

    def _axis_vector(self, axis: str, value: float) -> tuple[float, float, float]:
        vector = [0.0, 0.0, 0.0]
        vector[AXIS_INDEX[axis]] = float(value)
        return tuple(vector)  # type: ignore[return-value]

    def _actuator_output(self) -> tuple[float, ...] | None:
        latest = getattr(self.bridge, "latest_actuator_output", None)
        if not latest:
            return None
        commands = latest.get("motor_commands") if isinstance(latest, dict) else None
        return None if commands is None else tuple(float(value) for value in commands)

    def _bridge_snapshot(self) -> dict[str, Any] | None:
        snapshot = getattr(self.bridge, "snapshot", None)
        if not callable(snapshot):
            return None
        return snapshot()

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

    @staticmethod
    def _sign_name(sign: int) -> str:
        return "plus" if int(sign) >= 0 else "minus"


def _float_tuple_from_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value.")
    return values


def _axis_tuple_from_csv(raw: str) -> tuple[str, ...]:
    axes = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not axes:
        raise argparse.ArgumentTypeError("Expected at least one axis.")
    invalid = [axis for axis in axes if axis not in AXIS_INDEX]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported axes: {invalid}. Expected x,y,z.")
    return axes


def _mode_tuple_from_csv(raw: str) -> tuple[str, ...]:
    modes = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not modes:
        raise argparse.ArgumentTypeError("Expected at least one mode.")
    valid = {"velocity", "position", "position_velocity", "yaw"}
    invalid = [mode for mode in modes if mode not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported modes: {invalid}. Expected velocity,position,position_velocity,yaw.")
    return modes


def _sign_tuple_from_csv(raw: str) -> tuple[int, ...]:
    signs: list[int] = []
    for part in (part.strip().lower() for part in raw.split(",") if part.strip()):
        if part in {"+", "+1", "1", "plus", "positive", "pos"}:
            signs.append(1)
        elif part in {"-", "-1", "minus", "negative", "neg"}:
            signs.append(-1)
        else:
            raise argparse.ArgumentTypeError(f"Unsupported sign {part!r}. Expected plus/minus.")
    if not signs:
        raise argparse.ArgumentTypeError("Expected at least one sign.")
    return tuple(signs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep local-NED command surfaces against the live simulator.")
    parser.add_argument("--endpoint", default=LocalNedSweepConfig.endpoint)
    parser.add_argument("--command-hz", type=float, default=LocalNedSweepConfig.command_hz)
    parser.add_argument("--sample-hz", type=float, default=LocalNedSweepConfig.sample_hz)
    parser.add_argument("--pulse-s", type=float, default=LocalNedSweepConfig.pulse_s)
    parser.add_argument("--rest-s", type=float, default=LocalNedSweepConfig.rest_s)
    parser.add_argument("--settle-s", type=float, default=LocalNedSweepConfig.settle_s)
    parser.add_argument("--start-delay-s", type=float, default=LocalNedSweepConfig.start_delay_s)
    parser.add_argument("--wait-for-race-start", action="store_true")
    parser.add_argument("--race-start-margin-s", type=float, default=LocalNedSweepConfig.race_start_margin_s)
    parser.add_argument("--race-start-timeout-s", type=float, default=LocalNedSweepConfig.race_start_timeout_s)
    parser.add_argument("--reset-wait-s", type=float, default=LocalNedSweepConfig.reset_wait_s)
    parser.add_argument("--speeds", type=_float_tuple_from_csv, default=LocalNedSweepConfig.speeds_mps)
    parser.add_argument("--position-offsets", type=_float_tuple_from_csv, default=LocalNedSweepConfig.position_offsets_m)
    parser.add_argument("--axes", type=_axis_tuple_from_csv, default=LocalNedSweepConfig.axes)
    parser.add_argument("--signs", type=_sign_tuple_from_csv, default=LocalNedSweepConfig.signs)
    parser.add_argument("--modes", type=_mode_tuple_from_csv, default=None)
    parser.add_argument("--include-z", action="store_true")
    parser.add_argument("--include-position", action="store_true")
    parser.add_argument("--include-position-velocity", action="store_true")
    parser.add_argument("--include-yaw", action="store_true")
    parser.add_argument("--yaw-angles", type=_float_tuple_from_csv, default=LocalNedSweepConfig.yaw_angles_rad)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-case-displacement-m", type=float, default=LocalNedSweepConfig.max_case_displacement_m)
    parser.add_argument("--max-horizontal-radius-m", type=float, default=LocalNedSweepConfig.max_horizontal_radius_m)
    parser.add_argument("--min-z-ned-m", type=float, default=LocalNedSweepConfig.min_z_ned_m)
    parser.add_argument("--max-z-ned-m", type=float, default=LocalNedSweepConfig.max_z_ned_m)
    parser.add_argument("--max-speed-mps", type=float, default=LocalNedSweepConfig.max_speed_mps)
    parser.add_argument("--reset-before-first-case", action="store_true")
    parser.add_argument("--no-reset-between-cases", action="store_true")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-disarm-on-exit", action="store_true")
    parser.add_argument("--no-abort-on-safety", action="store_true")
    parser.add_argument("--no-abort-on-reset-change", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path, default=LocalNedSweepConfig.output_path)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> LocalNedSweepConfig:
    include_z = bool(args.include_z or "z" in args.axes)
    if args.modes is None:
        modes = ["velocity"]
        if args.include_position:
            modes.append("position")
        if args.include_position_velocity:
            modes.append("position_velocity")
        if args.include_yaw:
            modes.append("yaw")
    else:
        modes = list(args.modes)
    return LocalNedSweepConfig(
        endpoint=args.endpoint,
        command_hz=args.command_hz,
        sample_hz=args.sample_hz,
        pulse_s=args.pulse_s,
        rest_s=args.rest_s,
        settle_s=args.settle_s,
        start_delay_s=args.start_delay_s,
        wait_for_race_start=bool(args.wait_for_race_start),
        race_start_margin_s=args.race_start_margin_s,
        race_start_timeout_s=args.race_start_timeout_s,
        reset_wait_s=args.reset_wait_s,
        speeds_mps=tuple(float(value) for value in args.speeds),
        position_offsets_m=tuple(float(value) for value in args.position_offsets),
        axes=tuple(args.axes),
        signs=tuple(int(value) for value in args.signs),
        case_modes=tuple(modes),
        include_z=include_z,
        include_position=bool(args.include_position or "position" in modes),
        include_position_velocity=bool(args.include_position_velocity or "position_velocity" in modes),
        include_yaw=bool(args.include_yaw or "yaw" in modes),
        yaw_angles_rad=tuple(float(value) for value in args.yaw_angles),
        arm_on_start=not args.no_arm,
        disarm_on_exit=not args.no_disarm_on_exit,
        reset_before_first_case=bool(args.reset_before_first_case),
        reset_between_cases=not args.no_reset_between_cases,
        abort_on_safety=not args.no_abort_on_safety,
        abort_on_reset_change=not args.no_abort_on_reset_change,
        max_case_displacement_m=args.max_case_displacement_m,
        max_horizontal_radius_m=args.max_horizontal_radius_m,
        min_z_ned_m=args.min_z_ned_m,
        max_z_ned_m=args.max_z_ned_m,
        max_speed_mps=args.max_speed_mps,
        max_cases=args.max_cases,
        progress=bool(args.progress),
        output_path=args.output,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    if args.plan_only:
        cases = LocalNedCommandSweep(None, config).build_cases()
        print(json.dumps({"config": asdict(config), "case_count": len(cases), "cases": [asdict(case) for case in cases]}, indent=2, default=str))
        return
    sweep = LocalNedCommandSweep(MavlinkBridge(config.endpoint), config)
    payload = sweep.run_live()
    output = sweep.summary(payload) if args.summary_only else payload
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
