from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sensing.telemetry import MavlinkBridge

from .command_emitter import CommandEmitter
from .command_transform import CommandFrameTransform
from .config import Q1RuntimeConfig


@dataclass(frozen=True)
class AxisPulse:
    name: str
    velocity_local_ned_mps: tuple[float, float, float]


@dataclass(frozen=True)
class PulseResult:
    name: str
    command_velocity_local_ned_mps: tuple[float, float, float]
    start_position_local_ned_m: tuple[float, float, float]
    end_position_local_ned_m: tuple[float, float, float]
    displacement_local_ned_m: tuple[float, float, float]
    dominant_axis: str
    dominant_sign: int
    gain_m_per_mps: float


@dataclass(frozen=True)
class AuthorityProbeConfig:
    endpoint: str = Q1RuntimeConfig.mavlink_endpoint
    command_hz: float = 30.0
    speed_mps: float = 0.25
    pulse_s: float = 1.0
    rest_s: float = 0.5
    include_z: bool = False
    arm_on_start: bool = True
    disarm_on_exit: bool = True
    output_path: Path | None = None


class AuthorityProbe:
    """Measures how velocity-only SET_POSITION_TARGET_LOCAL_NED affects telemetry."""

    def __init__(
        self,
        bridge: Any,
        config: AuthorityProbeConfig | None = None,
        *,
        emitter: CommandEmitter | None = None,
    ):
        self.bridge = bridge
        self.config = config or AuthorityProbeConfig()
        self.emitter = emitter or CommandEmitter(bridge, command_mode="velocity", dry_run=False)

    def pulses(self) -> list[AxisPulse]:
        speed = float(self.config.speed_mps)
        pulses = [
            AxisPulse("+x", (speed, 0.0, 0.0)),
            AxisPulse("-x", (-speed, 0.0, 0.0)),
            AxisPulse("+y", (0.0, speed, 0.0)),
            AxisPulse("-y", (0.0, -speed, 0.0)),
        ]
        if self.config.include_z:
            pulses.extend([AxisPulse("+z", (0.0, 0.0, speed)), AxisPulse("-z", (0.0, 0.0, -speed))])
        return pulses

    def summarize_pulse(
        self,
        pulse: AxisPulse,
        *,
        start_position_local_ned_m: tuple[float, float, float],
        end_position_local_ned_m: tuple[float, float, float],
    ) -> PulseResult:
        start = np.asarray(start_position_local_ned_m, dtype=float)
        end = np.asarray(end_position_local_ned_m, dtype=float)
        displacement = end - start
        dominant_index = int(np.argmax(np.abs(displacement)))
        dominant_axis = ("x", "y", "z")[dominant_index]
        dominant_value = float(displacement[dominant_index])
        command_norm = max(float(np.linalg.norm(np.asarray(pulse.velocity_local_ned_mps, dtype=float))), 1e-9)
        return PulseResult(
            name=pulse.name,
            command_velocity_local_ned_mps=pulse.velocity_local_ned_mps,
            start_position_local_ned_m=tuple(float(value) for value in start),
            end_position_local_ned_m=tuple(float(value) for value in end),
            displacement_local_ned_m=tuple(float(value) for value in displacement),
            dominant_axis=dominant_axis,
            dominant_sign=1 if dominant_value >= 0.0 else -1,
            gain_m_per_mps=float(np.linalg.norm(displacement) / command_norm),
        )

    def recommended_transform(self, results: list[PulseResult]) -> CommandFrameTransform:
        # Conservative v1: only recommend sign flips for axes that moved dominantly on the commanded axis.
        signs = {"x": 1.0, "y": 1.0, "z": 1.0}
        for result in results:
            command = np.asarray(result.command_velocity_local_ned_mps, dtype=float)
            axis_index = int(np.argmax(np.abs(command)))
            axis = ("x", "y", "z")[axis_index]
            commanded_sign = 1 if float(command[axis_index]) >= 0.0 else -1
            if result.dominant_axis == axis and result.dominant_sign != commanded_sign:
                signs[axis] = -1.0
        return CommandFrameTransform.from_matrix(
            [[signs["x"], 0.0, 0.0], [0.0, signs["y"], 0.0], [0.0, 0.0, signs["z"]]],
            name="authority_probe_recommended",
        )

    def run_live(self) -> dict[str, Any]:
        self.bridge.connect()
        self.bridge.start_heartbeat()
        self.bridge.subscribe_telemetry()
        self._wait_for_position()
        if self.config.arm_on_start:
            self.bridge.arm()
            time.sleep(0.2)
        results: list[PulseResult] = []
        try:
            for pulse in self.pulses():
                start = self._position()
                self._stream_velocity(pulse.velocity_local_ned_mps, self.config.pulse_s)
                self._stream_velocity((0.0, 0.0, 0.0), self.config.rest_s)
                end = self._position()
                results.append(self.summarize_pulse(pulse, start_position_local_ned_m=start, end_position_local_ned_m=end))
        finally:
            self._stream_velocity((0.0, 0.0, 0.0), 0.2)
            if self.config.disarm_on_exit:
                self.bridge.disarm()
            self.bridge.shutdown()
        payload = {
            "config": asdict(self.config),
            "results": [asdict(result) for result in results],
            "recommended_transform": self.recommended_transform(results).to_payload(),
        }
        if self.config.output_path is not None:
            path = Path(self.config.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    def _stream_velocity(self, velocity: tuple[float, float, float], duration_s: float) -> None:
        end = time.monotonic() + float(duration_s)
        interval = 1.0 / max(1e-6, float(self.config.command_hz))
        while time.monotonic() < end:
            self.emitter.emit_velocity(velocity, source="q1runtime_authority_probe")
            time.sleep(interval)

    def _wait_for_position(self) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self.bridge.get_latest_telemetry() is not None and self._position_or_none() is not None:
                return
            time.sleep(0.02)
        raise TimeoutError("No local-NED telemetry received for authority probe.")

    def _position(self) -> tuple[float, float, float]:
        position = self._position_or_none()
        if position is None:
            raise RuntimeError("Telemetry does not include position_local_ned_m.")
        return position

    def _position_or_none(self) -> tuple[float, float, float] | None:
        telemetry = self.bridge.get_latest_telemetry()
        position = None if telemetry is None else getattr(telemetry, "position_local_ned_m", None)
        return None if position is None else tuple(float(value) for value in position)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe simulator velocity command authority.")
    parser.add_argument("--endpoint", default=AuthorityProbeConfig.endpoint)
    parser.add_argument("--command-hz", type=float, default=AuthorityProbeConfig.command_hz)
    parser.add_argument("--speed", type=float, default=AuthorityProbeConfig.speed_mps)
    parser.add_argument("--pulse-s", type=float, default=AuthorityProbeConfig.pulse_s)
    parser.add_argument("--rest-s", type=float, default=AuthorityProbeConfig.rest_s)
    parser.add_argument("--include-z", action="store_true")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-disarm-on-exit", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("logs/q1runtime/authority-probe.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = AuthorityProbeConfig(
        endpoint=args.endpoint,
        command_hz=args.command_hz,
        speed_mps=args.speed,
        pulse_s=args.pulse_s,
        rest_s=args.rest_s,
        include_z=args.include_z,
        arm_on_start=not args.no_arm,
        disarm_on_exit=not args.no_disarm_on_exit,
        output_path=args.output,
    )
    payload = AuthorityProbe(MavlinkBridge(config.endpoint), config).run_live()
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
