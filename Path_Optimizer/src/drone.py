"""Minimal offline rig for telemetry ingestion and control-loop timing."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

from autonomy.control.command_mapper import CommandMapper
from autonomy.control.hover import HoverPIDController
from autonomy.modes.system_mode import SystemMode, SystemModeManager
from core.logging import Logger
from core.simulator import TelemetrySimulator
from sensing.estimation.state_estimator import StateEstimator
from sensing.telemetry.mavlink_bridge import MavlinkBridge


def _event_snapshot(*, sim_time_ns: int, system_mode: SystemModeManager, bridge: MavlinkBridge) -> dict:
    return {
        "sim_time_ns": sim_time_ns,
        "system_mode": system_mode.system_mode,
        "bridge": {
            "connected": bridge.connected,
            "heartbeat_started": bridge.heartbeat_started,
            "telemetry_subscribed": bridge.telemetry_subscribed,
        },
    }


def run(duration_s: float = 3.0, loop_hz: float = 10.0, idle_s: float = 1.0) -> Path:
    if idle_s < 1.0:
        raise ValueError("idle_s must be at least 1 second")
    simulator = TelemetrySimulator(telemetry_hz=loop_hz)
    bridge = MavlinkBridge(endpoint="telemetry-simulator")
    command_mapper = CommandMapper()
    estimator = StateEstimator(initial_state=simulator._harness.initial_state)
    system_mode = SystemModeManager()
    hover_controller = HoverPIDController(
        mass_kg=simulator.model.m,
        gravity_mps2=simulator.model.g,
        thrust_coefficient=simulator.model.kf,
        dt_s=1.0 / loop_hz,
    )
    period_ns = round(1_000_000_000 / loop_hz)
    logger = Logger(
        metadata={
            "scenario": "idle_fall_then_armed_xyz_home",
            "duration_s": duration_s,
            "loop_hz": loop_hz,
            "idle_s": idle_s,
            "endpoint": bridge.endpoint,
            "noise_seed": simulator.noise_seed,
            "position_frame": "local_ned_m",
            "position_origin": "simulator_home",
            "initial_true_state": simulator._harness.initial_state,
            "target_position_local_ned_m": hover_controller.target_position_local_ned_m,
            "hover_controller": {
                "position_gain": hover_controller.position_gain,
                "velocity_gain": hover_controller.velocity_gain,
                "acceleration_pid": {
                    "kp": hover_controller.kp,
                    "ki": hover_controller.ki,
                    "kd": hover_controller.kd,
                },
                "velocity_damping": hover_controller.velocity_damping,
            },
            "quadrotor_params": {
                "mass_kg": simulator.model.m,
                "gravity_mps2": simulator.model.g,
                "thrust_coefficient": simulator.model.kf,
            },
        }
    )
    log_path = Logger.timestamped_path(Path(__file__).resolve().parents[1] / "logs")

    bridge.connect()
    bridge.start_heartbeat()
    bridge.subscribe_telemetry()
    logger.log_event(
        "initialized",
        **_event_snapshot(sim_time_ns=0, system_mode=system_mode, bridge=bridge),
    )
    print(f"connected endpoint={bridge.endpoint} system_mode={system_mode.system_mode.value} loop_hz={loop_hz:g}", flush=True)

    samples = simulator.telemetry_samples(duration_s)
    first_sample = next(samples, None)
    start_ns = time.perf_counter_ns()
    try:
        for cycle, sample in enumerate(itertools.chain([first_sample], samples) if first_sample else []):
            deadline_ns = start_ns + cycle * period_ns
            remaining_ns = deadline_ns - time.perf_counter_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000)

            bridge.update_latest_telemetry(sample)
            telemetry = bridge.get_latest_telemetry()
            if telemetry is None:
                continue

            estimator.update_from_telemetry(telemetry)
            target = None
            if system_mode.system_mode == SystemMode.IDLE and telemetry.sim_time_ns >= idle_s * 1e9:
                system_mode.update_mode("arm")
                logger.log_event(
                    "system_mode_changed",
                    **_event_snapshot(sim_time_ns=telemetry.sim_time_ns, system_mode=system_mode, bridge=bridge),
                )
            if system_mode.system_mode == SystemMode.ARMED:
                quaternion, thrust = hover_controller.update(
                    telemetry.raw["acceleration_local_ned_mps2"],
                    estimator.get_13_state()[0:3],
                    telemetry.velocity_local_ned_mps,
                )
                target = command_mapper.to_attitude_target(quaternion, thrust)
                bridge.send_attitude_target(target)
                simulator.apply_attitude_target(target)
            state = estimator.get_13_state()
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            truth_acceleration = simulator.model.state_derivative(
                simulator._harness.state,
                simulator._motor_command,
            )[3:6]
            logger.log_cycle(
                cycle=cycle,
                wall_elapsed_ms=elapsed_ms,
                deadline_lateness_ms=elapsed_ms - cycle * period_ns / 1_000_000,
                sim_time_ns=telemetry.sim_time_ns,
                system_mode=system_mode.system_mode,
                bridge={
                    "connected": bridge.connected,
                    "latest_attitude_target": bridge.latest_attitude_target,
                    "latest_position_target": bridge.latest_position_target,
                },
                telemetry=telemetry,
                estimated_state={
                    "state_vector": state,
                    "position_local_ned_m": state[0:3],
                    "velocity_local_ned_mps": state[3:6],
                    "attitude_quaternion": state[6:10],
                    "body_rates_rps": state[10:13],
                },
                command={"set_attitude_target": target},
                simulator_truth={
                    "state_vector": simulator._harness.state,
                    "position_local_ned_m": simulator._harness.state[0:3],
                    "velocity_local_ned_mps": simulator._harness.state[3:6],
                    "attitude_quaternion": simulator._harness.state[6:10],
                    "body_rates_rps": simulator._harness.state[10:13],
                    "acceleration_local_ned_mps2": truth_acceleration,
                    "motor_commands": simulator._motor_command,
                },
            )
            print(
                f"cycle={cycle:03d} wall_ms={elapsed_ms:7.1f} "
                f"sim_ms={telemetry.sim_time_ns / 1_000_000:7.1f} "
                f"system_mode={system_mode.system_mode.value} "
                f"pos_ned={state[0:3].round(3).tolist()} "
                f"vel_ned={state[3:6].round(3).tolist()} "
                f"set_attitude_target={target}",
                flush=True,
            )
    finally:
        bridge.shutdown()
        logger.log_event(
            "shutdown",
            **_event_snapshot(
                sim_time_ns=telemetry.sim_time_ns if "telemetry" in locals() else 0,
                system_mode=system_mode,
                bridge=bridge,
            ),
        )
        logger.save_run(log_path)
        print(f"shutdown complete log={log_path}", flush=True)
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--idle-s", type=float, default=1.0)
    args = parser.parse_args()
    run(duration_s=args.duration_s, loop_hz=args.loop_hz, idle_s=args.idle_s)


if __name__ == "__main__":
    main()
