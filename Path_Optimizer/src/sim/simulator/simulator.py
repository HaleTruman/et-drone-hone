from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from controller.rate.controller import RateController
from drone_mpcc_planner.planner import MPCCPlanner
from drone_mpcc_planner.reference_path import GATE_INNER_M, R, yaw_quat
from quadrotor.model import Quadrotor
from sim.utils.utils import (
    initial_state,
    load_yaml,
    project_root,
    quaternion_to_euler,
    rk4_step,
    src_root,
)


@dataclass
class SimulationResult:
    t: np.ndarray
    state: np.ndarray
    motor_commands: np.ndarray
    desired_rates: np.ndarray
    desired_euler: np.ndarray
    acceleration: np.ndarray
    thrust: np.ndarray
    speed: np.ndarray
    target_position: np.ndarray
    dt: float
    reference_path: np.ndarray | None = None
    planned_path: np.ndarray | None = None
    planned_state: np.ndarray | None = None
    planned_controls: np.ndarray | None = None
    gates: list[dict[str, Any]] | None = None
    gate_wireframes: list[np.ndarray] | None = None
    phase: np.ndarray | None = None
    planner_info: dict[str, Any] | None = None
    display_quaternion: np.ndarray | None = None

    @property
    def position(self) -> np.ndarray:
        return self.state[:, 0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[:, 3:6]

    @property
    def quaternion(self) -> np.ndarray:
        return self.state[:, 6:10]

    @property
    def rates(self) -> np.ndarray:
        return self.state[:, 10:13]

    @property
    def euler(self) -> np.ndarray:
        return np.vstack([quaternion_to_euler(q) for q in self.quaternion])


class HoverController:
    """Small outer-loop hover stabilizer backed by the existing rate controller."""

    def __init__(
        self,
        *,
        rate_controller: RateController,
        quadrotor: Quadrotor,
        config: dict[str, Any],
    ):
        hover = config["scenario"]["hover"]
        gains = config["hover_controller"]

        self.rate_controller = rate_controller
        self.quadrotor = quadrotor
        self.target_position = np.asarray(hover["target_position"], dtype=float)
        self.target_yaw = float(hover.get("target_yaw", 0.0))

        self.kp_pos = np.asarray(gains["kp_pos"], dtype=float)
        self.kd_pos = np.asarray(gains["kd_pos"], dtype=float)
        self.kp_att = np.asarray(gains["kp_att"], dtype=float)
        self.max_rate = np.asarray(gains["max_rate"], dtype=float)
        self.along_track_tolerance_m = float(gains.get("along_track_tolerance_m", 0.0))
        self.lateral_tolerance_m = float(gains.get("lateral_tolerance_m", 0.0))
        self.vertical_tolerance_m = float(gains.get("vertical_tolerance_m", 0.0))
        self.velocity_tolerance_mps = float(gains.get("velocity_tolerance_mps", 0.0))
        self.forward_accel_floor_mps2 = float(gains.get("forward_accel_floor_mps2", 0.0))
        self.near_gate_distance_m = float(gains.get("near_gate_distance_m", 4.0))
        self.near_gate_tolerance_m = float(gains.get("near_gate_tolerance_m", 0.05))
        self.post_gate_decel_mps2 = float(gains.get("post_gate_decel_mps2", 8.0))
        self.hover_capture_speed_mps = float(gains.get("hover_capture_speed_mps", 0.35))

        self.motor_min = float(rate_controller.motor_min)
        self.motor_max = float(rate_controller.motor_max)

    def update(self, state: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        pos = state[0:3]
        vel = state[3:6]
        euler = quaternion_to_euler(state[6:10])

        target_position = self.target_position
        target_velocity = np.zeros(3, dtype=float)
        feedforward_accel = np.zeros(3, dtype=float)
        target_yaw = self.target_yaw

        desired_accel = self._desired_accel(
            pos=pos,
            vel=vel,
            target_position=target_position,
            target_velocity=target_velocity,
            feedforward_accel=feedforward_accel,
        )
        desired_euler = self._desired_euler(desired_accel, target_yaw)
        desired_rates = self._desired_rates(euler, desired_euler)

        motor_command = self._motor_command(state, desired_rates, desired_accel[2], euler)
        telemetry = {
            "desired_rates": desired_rates,
            "desired_euler": desired_euler,
            "desired_accel": desired_accel,
            "target_position": target_position,
            "target_velocity": target_velocity,
            "phase": "hover",
        }
        return motor_command, telemetry

    def _desired_accel(
        self,
        *,
        pos: np.ndarray,
        vel: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        feedforward_accel: np.ndarray,
    ) -> np.ndarray:
        pos_error = self.target_position - pos
        if target_position is not self.target_position:
            pos_error = target_position - pos
        vel_error = target_velocity - vel
        return self.kp_pos * pos_error + self.kd_pos * vel_error + feedforward_accel

    def _desired_euler(self, desired_accel: np.ndarray, target_yaw: float) -> np.ndarray:
        force_axis = np.array(
            [
                -desired_accel[0],
                -desired_accel[1],
                self.quadrotor.g - desired_accel[2],
            ],
            dtype=float,
        )

        body_z = force_axis / max(np.linalg.norm(force_axis), 1e-9)
        heading = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0], dtype=float)
        body_y = np.cross(body_z, heading)
        if np.linalg.norm(body_y) < 1e-6:
            body_y = np.array([-np.sin(target_yaw), np.cos(target_yaw), 0.0], dtype=float)
        body_y /= np.linalg.norm(body_y)
        body_x = np.cross(body_y, body_z)

        rotation = np.column_stack((body_x, body_y, body_z))
        return _rotation_to_euler(rotation)

    def _desired_rates(self, euler: np.ndarray, desired_euler: np.ndarray) -> np.ndarray:
        attitude_error = desired_euler - euler
        attitude_error[2] = np.arctan2(np.sin(attitude_error[2]), np.cos(attitude_error[2]))
        return np.clip(self.kp_att * attitude_error, -self.max_rate, self.max_rate)

    def _motor_command(
        self,
        state: np.ndarray,
        desired_rates: np.ndarray,
        desired_accel_z: float,
        euler: np.ndarray,
    ) -> np.ndarray:
        base_motor_command = self.rate_controller.update(state, desired_rates)
        thrust_command = self._collective_for_vertical_accel(desired_accel_z, euler)
        collective_delta = thrust_command - float(np.mean(self.rate_controller.hover_command))
        return np.clip(
            base_motor_command + collective_delta,
            self.motor_min,
            self.motor_max,
        )

    def _collective_for_vertical_accel(self, desired_accel_z: float, euler: np.ndarray) -> float:
        roll, pitch, _ = euler
        tilt_factor = max(0.25, np.cos(roll) * np.cos(pitch))
        total_thrust = self.quadrotor.m * (self.quadrotor.g - desired_accel_z) / tilt_factor
        per_motor_thrust = max(total_thrust / 4.0, 0.0)
        return float(np.sqrt(per_motor_thrust / self.quadrotor.kf))


class TrackController(HoverController):
    """Follow an MPCC-planned trajectory after an initial hover hold."""

    def __init__(
        self,
        *,
        rate_controller: RateController,
        quadrotor: Quadrotor,
        config: dict[str, Any],
        plan: dict[str, Any],
    ):
        super().__init__(
            rate_controller=rate_controller,
            quadrotor=quadrotor,
            config=config,
        )
        gains = config["track_controller"]
        self.kp_pos = np.asarray(gains["kp_pos"], dtype=float)
        self.kd_pos = np.asarray(gains["kd_pos"], dtype=float)
        self.kp_att = np.asarray(gains["kp_att"], dtype=float)
        self.max_rate = np.asarray(gains["max_rate"], dtype=float)
        self.along_track_tolerance_m = float(gains.get("along_track_tolerance_m", 0.0))
        self.lateral_tolerance_m = float(gains.get("lateral_tolerance_m", 0.0))
        self.vertical_tolerance_m = float(gains.get("vertical_tolerance_m", 0.0))
        self.velocity_tolerance_mps = float(gains.get("velocity_tolerance_mps", 0.0))
        self.forward_accel_floor_mps2 = float(gains.get("forward_accel_floor_mps2", 0.0))

        self.hover_duration = float(config["scenario"]["track"]["hover_duration"])
        self.time_scale = float(config["scenario"]["track"].get("time_scale", 1.0))
        self.track_time = np.asarray(plan["track_time"], dtype=float) * self.time_scale
        self.final_target_position = np.asarray(plan["planned_state"][0:3, -1], dtype=float)
        self.gate_plan_time = float(plan["info"].get("gate_crossing_time_s", self.track_time[-1])) * self.time_scale
        self.gate_position = np.asarray(plan["gates"][-1]["pos"], dtype=float)
        self.gate_rotation = R(np.asarray(plan["gates"][-1]["quat"], dtype=float))
        self.gate_normal = np.asarray(self.gate_rotation[:, 0], dtype=float)
        self.gate_half_width = 0.5 * float(plan["reference"]["gate_dimensions"].get("inner_m", GATE_INNER_M))
        self.hover_after_gate = False
        self.post_gate_hover_position: np.ndarray | None = None
        self.previous_gate_progress: float | None = None
        self.plan = plan

    @property
    def planned_duration(self) -> float:
        return float(self.track_time[-1])

    def update(self, state: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        if t < self.hover_duration:
            return super().update(state, t)
        if self.hover_after_gate:
            return self._post_gate_hover_update(state, t)
        if self._gate_crossed(state[0:3]):
            self.hover_after_gate = True
            self.rate_controller.reset()
            return self._post_gate_hover_update(state, t)

        tau = min(t - self.hover_duration, self.gate_plan_time)
        target_position = self._interp_plan(tau, 0, 3)
        target_velocity = self._interp_plan(tau, 3, 6) / self.time_scale
        feedforward_accel = self._feedforward_accel(tau)

        vel_norm = np.linalg.norm(target_velocity[0:2])
        target_yaw = float(np.arctan2(target_velocity[1], target_velocity[0])) if vel_norm > 0.05 else self.target_yaw

        euler = quaternion_to_euler(state[6:10])
        desired_accel = self._race_desired_accel(
            pos=state[0:3],
            vel=state[3:6],
            target_position=target_position,
            target_velocity=target_velocity,
            feedforward_accel=feedforward_accel,
        )
        desired_euler = self._desired_euler(desired_accel, target_yaw)
        desired_rates = self._desired_rates(euler, desired_euler)
        motor_command = self._motor_command(state, desired_rates, desired_accel[2], euler)

        telemetry = {
            "desired_rates": desired_rates,
            "desired_euler": desired_euler,
            "desired_accel": desired_accel,
            "target_position": target_position,
            "target_velocity": target_velocity,
            "phase": "track",
        }
        return motor_command, telemetry

    def _race_desired_accel(
        self,
        *,
        pos: np.ndarray,
        vel: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        feedforward_accel: np.ndarray,
    ) -> np.ndarray:
        pos_error = self._soft_tracking_position_error(target_position - pos, target_velocity, pos)
        vel_error = self._soft_tracking_velocity_error(target_velocity - vel, target_velocity)
        desired_accel = self.kp_pos * pos_error + self.kd_pos * vel_error + feedforward_accel
        drive_axis = self._race_drive_axis(pos, target_velocity)
        forward_accel = float(np.dot(desired_accel, drive_axis))
        if forward_accel < self.forward_accel_floor_mps2:
            desired_accel = desired_accel + (self.forward_accel_floor_mps2 - forward_accel) * drive_axis
        return desired_accel

    def _soft_tracking_position_error(
        self,
        error: np.ndarray,
        target_velocity: np.ndarray,
        pos: np.ndarray,
    ) -> np.ndarray:
        tangent = self._tracking_tangent(target_velocity)
        vertical_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        lateral_axis = np.cross(vertical_axis, tangent)
        if np.linalg.norm(lateral_axis) < 1e-6:
            lateral_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral_axis /= np.linalg.norm(lateral_axis)

        lateral_tolerance = self.lateral_tolerance_m
        vertical_tolerance = self.vertical_tolerance_m
        distance_to_gate_plane = float(np.dot(self.gate_position - np.asarray(pos, dtype=float), self.gate_normal))
        if 0.0 < distance_to_gate_plane < self.near_gate_distance_m:
            lateral_tolerance = min(lateral_tolerance, self.near_gate_tolerance_m)
            vertical_tolerance = min(vertical_tolerance, self.near_gate_tolerance_m)

        along = self._soft_deadband_scalar(float(np.dot(error, tangent)), self.along_track_tolerance_m)
        along = max(0.0, along)
        lateral = self._soft_deadband_scalar(float(np.dot(error, lateral_axis)), lateral_tolerance)
        vertical = self._soft_deadband_scalar(float(error[2]), vertical_tolerance)
        return along * tangent + lateral * lateral_axis + vertical * vertical_axis

    def _soft_tracking_velocity_error(self, error: np.ndarray, target_velocity: np.ndarray) -> np.ndarray:
        tangent = self._tracking_tangent(target_velocity)
        vertical_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        lateral_axis = np.cross(vertical_axis, tangent)
        if np.linalg.norm(lateral_axis) < 1e-6:
            lateral_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral_axis /= np.linalg.norm(lateral_axis)

        along = self._soft_deadband_scalar(float(np.dot(error, tangent)), self.velocity_tolerance_mps)
        along = max(0.0, along)
        lateral = self._soft_deadband_scalar(float(np.dot(error, lateral_axis)), self.velocity_tolerance_mps)
        vertical = self._soft_deadband_scalar(float(error[2]), self.velocity_tolerance_mps)
        return along * tangent + lateral * lateral_axis + vertical * vertical_axis

    def _tracking_tangent(self, target_velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(target_velocity)
        if speed > 0.1:
            return target_velocity / speed
        return self.gate_normal

    def _race_drive_axis(self, pos: np.ndarray, target_velocity: np.ndarray) -> np.ndarray:
        to_gate = self.gate_position - np.asarray(pos, dtype=float)
        if np.dot(to_gate, self.gate_normal) > 0.25 and np.linalg.norm(to_gate) > 1e-6:
            return to_gate / np.linalg.norm(to_gate)
        return self._tracking_tangent(target_velocity)

    def _soft_deadband_vector(self, error: np.ndarray, tolerance: float) -> np.ndarray:
        error = np.asarray(error, dtype=float)
        norm = np.linalg.norm(error)
        if tolerance <= 0.0 or norm <= tolerance:
            return np.zeros_like(error) if norm <= tolerance else error
        return error * ((norm - tolerance) / norm)

    def _soft_deadband_scalar(self, error: float, tolerance: float) -> float:
        if tolerance <= 0.0:
            return error
        magnitude = abs(error)
        if magnitude <= tolerance:
            return 0.0
        return np.sign(error) * (magnitude - tolerance)

    def _gate_crossed(self, position: np.ndarray) -> bool:
        rel = np.asarray(position, dtype=float) - self.gate_position
        progress = float(np.dot(rel, self.gate_normal))
        local = self.gate_rotation.T @ rel
        inside_aperture = abs(float(local[1])) <= self.gate_half_width and abs(float(local[2])) <= self.gate_half_width
        crossed_plane = progress >= 0.0
        if self.previous_gate_progress is not None:
            crossed_plane = self.previous_gate_progress < 0.0 <= progress
        self.previous_gate_progress = progress
        return crossed_plane and inside_aperture

    def _post_gate_hover_update(
        self,
        state: np.ndarray,
        t: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        target_position = self.post_gate_hover_position
        if target_position is None and np.linalg.norm(state[3:6]) <= self.hover_capture_speed_mps:
            target_position = state[0:3].copy()
            self.post_gate_hover_position = target_position

        pos = state[0:3]
        vel = state[3:6]
        euler = quaternion_to_euler(state[6:10])
        if target_position is None:
            speed = np.linalg.norm(vel)
            desired_accel = -self.post_gate_decel_mps2 * vel / max(speed, 1e-9)
            target_position = pos
            phase = "post_gate_brake"
        else:
            desired_accel = self._desired_accel(
                pos=pos,
                vel=vel,
                target_position=target_position,
                target_velocity=np.zeros(3, dtype=float),
                feedforward_accel=np.zeros(3, dtype=float),
            )
            phase = "post_gate_hover"
        desired_euler = self._desired_euler(desired_accel, self.target_yaw)
        desired_rates = self._desired_rates(euler, desired_euler)
        motor_command = self._motor_command(state, desired_rates, desired_accel[2], euler)
        telemetry = {
            "desired_rates": desired_rates,
            "desired_euler": desired_euler,
            "desired_accel": desired_accel,
            "target_position": target_position,
            "target_velocity": np.zeros(3, dtype=float),
            "phase": phase,
        }
        return motor_command, telemetry

    def _interp_plan(self, tau: float, start: int, stop: int) -> np.ndarray:
        values = self.plan["planned_state"][start:stop]
        return np.array(
            [np.interp(tau, self.track_time, values[i]) for i in range(stop - start)],
            dtype=float,
        )

    def _feedforward_accel(self, tau: float) -> np.ndarray:
        return self._raw_feedforward_accel(tau) / (self.time_scale * self.time_scale)

    def _raw_feedforward_accel(self, tau: float) -> np.ndarray:
        if "planned_accel" not in self.plan:
            return np.zeros(3, dtype=float)
        values = self.plan["planned_accel"]
        return np.array([np.interp(tau, self.track_time, values[:, i]) for i in range(3)], dtype=float)


class Simulator:
    def __init__(
        self,
        *,
        quadrotor: Quadrotor,
        controller: HoverController,
        config: dict[str, Any],
        plan: dict[str, Any] | None = None,
    ):
        self.quadrotor = quadrotor
        self.controller = controller
        self.dt = float(config["dt"])
        self.duration = self._duration(config, controller)
        self.max_steps = int(config.get("max_steps", round(self.duration / self.dt)))
        self.initial_state = initial_state(config)
        self.plan = plan

    def _duration(self, config: dict[str, Any], controller: HoverController) -> float:
        if isinstance(controller, TrackController):
            margin = float(config["scenario"]["track"].get("settle_duration", 1.0))
            return controller.hover_duration + controller.planned_duration + margin
        return float(config["duration"])

    def run(self) -> SimulationResult:
        steps = min(self.max_steps, int(round(self.duration / self.dt)))
        state = np.zeros((steps + 1, 13), dtype=float)
        motor_commands = np.zeros((steps + 1, 4), dtype=float)
        desired_rates = np.zeros((steps + 1, 3), dtype=float)
        desired_euler = np.zeros((steps + 1, 3), dtype=float)
        acceleration = np.zeros((steps + 1, 3), dtype=float)
        thrust = np.zeros(steps + 1, dtype=float)
        target_position = np.zeros((steps + 1, 3), dtype=float)
        phase = np.empty(steps + 1, dtype=object)
        t = np.arange(steps + 1, dtype=float) * self.dt

        state[0] = self.initial_state
        for k in range(steps):
            motor_command, telemetry = self.controller.update(state[k], t[k])
            deriv = self.quadrotor.state_derivative(state[k], motor_command)

            motor_commands[k] = motor_command
            desired_rates[k] = telemetry["desired_rates"]
            desired_euler[k] = telemetry["desired_euler"]
            acceleration[k] = deriv[3:6]
            target_position[k] = telemetry["target_position"]
            phase[k] = telemetry["phase"]
            thrust[k] = float(np.sum(self.quadrotor.kf * np.square(motor_command)))

            state[k + 1] = rk4_step(
                self.quadrotor.state_derivative,
                state[k],
                motor_command,
                self.dt,
            )

        motor_commands[-1], telemetry = self.controller.update(state[-1], t[-1])
        desired_rates[-1] = telemetry["desired_rates"]
        desired_euler[-1] = telemetry["desired_euler"]
        target_position[-1] = telemetry["target_position"]
        phase[-1] = telemetry["phase"]
        final_deriv = self.quadrotor.state_derivative(state[-1], motor_commands[-1])
        acceleration[-1] = final_deriv[3:6]
        thrust[-1] = float(np.sum(self.quadrotor.kf * np.square(motor_commands[-1])))

        return SimulationResult(
            t=t,
            state=state,
            motor_commands=motor_commands,
            desired_rates=desired_rates,
            desired_euler=desired_euler,
            acceleration=acceleration,
            thrust=thrust,
            speed=np.linalg.norm(state[:, 3:6], axis=1),
            target_position=target_position,
            dt=self.dt,
            reference_path=None if self.plan is None else self.plan["reference"]["pos"],
            planned_path=None if self.plan is None else self.plan["planned_state"][0:3].T,
            planned_state=None if self.plan is None else self.plan["planned_state"],
            planned_controls=None if self.plan is None else self.plan["planned_controls"],
            gates=None if self.plan is None else self.plan["gates"],
            gate_wireframes=None if self.plan is None else self.plan["reference"]["gate_wireframes"],
            phase=phase,
            planner_info=None if self.plan is None else self.plan["info"],
            display_quaternion=None,
        )


def load_simulation_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else src_root() / "sim" / "config" / "settings.yaml"
    return load_yaml(config_path)


def _rotation_to_euler(rotation: np.ndarray) -> np.ndarray:
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def load_quadrotor_params(path: str | Path | None = None) -> dict[str, Any]:
    params_path = Path(path) if path is not None else src_root() / "quadrotor" / "params.yaml"
    return load_yaml(params_path)


def load_rate_params(params_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    base = Path(params_path) if params_path is not None else src_root() / "controller" / "rate"
    if base.is_dir():
        return load_yaml(base / "params.yaml"), load_yaml(base / "gains.yaml")
    return load_yaml(base), load_yaml(base.with_name("gains.yaml"))


def make_hover_simulator(config_path: str | Path | None = None) -> Simulator:
    config = load_simulation_config(config_path)
    quad_params = load_quadrotor_params()
    quad_params["g"] = float(config.get("environment", {}).get("g", quad_params["g"]))

    rate_params, rate_gains = load_rate_params()
    rate_params = dict(rate_params)
    rate_params["dt"] = float(config["dt"])

    quadrotor = Quadrotor(quad_params)
    rate_controller = RateController(rate_params, rate_gains)
    controller = HoverController(
        rate_controller=rate_controller,
        quadrotor=quadrotor,
        config=config,
    )
    return Simulator(quadrotor=quadrotor, controller=controller, config=config)


def run_hover_simulation(config_path: str | Path | None = None) -> SimulationResult:
    return make_hover_simulator(config_path).run()


def demo_track_gates() -> list[dict[str, Any]]:
    return [{"pos": [12.0, 5.0, 0.0], "quat": yaw_quat(0)}]


def build_demo_track_plan(start_state: np.ndarray) -> dict[str, Any]:
    gates = demo_track_gates()
    planner = MPCCPlanner(False)
    _, _, info = planner.plan(start_state, gates)
    planned_state = np.asarray(planner.warm_start["X"], dtype=float)
    planned_controls = np.asarray(planner.warm_start["U"], dtype=float)
    track_dt = float(info.get("dt", planner.pack[3]["dt"]))
    track_time = np.arange(planned_state.shape[1], dtype=float) * track_dt
    planned_accel = np.gradient(planned_state[3:6].T, track_dt, axis=0)
    motor_slew = np.max(np.abs(np.diff(planned_controls, axis=1))) / track_dt
    info["max_motor_slew_per_s"] = float(motor_slew)
    return {
        "gates": gates,
        "reference": planner.reference,
        "planned_state": planned_state,
        "planned_controls": planned_controls,
        "planned_accel": planned_accel,
        "track_time": track_time,
        "info": info,
    }


def make_track_simulator(config_path: str | Path | None = None) -> Simulator:
    config = load_simulation_config(config_path)
    quad_params = load_quadrotor_params()
    quad_params["g"] = float(config.get("environment", {}).get("g", quad_params["g"]))

    rate_params, rate_gains = load_rate_params()
    rate_params = dict(rate_params)
    rate_params["dt"] = float(config["dt"])

    quadrotor = Quadrotor(quad_params)
    rate_controller = RateController(rate_params, rate_gains)
    start_state = initial_state(config)
    hover_target = np.asarray(config["scenario"]["hover"]["target_position"], dtype=float)
    planner_start = start_state.copy()
    planner_start[0:3] = hover_target
    planner_start[3:6] = 0.0
    planner_start[6:10] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    planner_start[10:13] = 0.0
    plan = build_demo_track_plan(planner_start)
    controller = TrackController(
        rate_controller=rate_controller,
        quadrotor=quadrotor,
        config=config,
        plan=plan,
    )
    return Simulator(quadrotor=quadrotor, controller=controller, config=config, plan=plan)


def run_track_simulation(config_path: str | Path | None = None) -> SimulationResult:
    return make_track_simulator(config_path).run()


def run_configured_simulation(config_path: str | Path | None = None) -> SimulationResult:
    config = load_simulation_config(config_path)
    if config.get("scenario", {}).get("name") == "single_gate_min_time":
        return make_track_simulator(config_path).run()
    return make_hover_simulator(config_path).run()


def main() -> int:
    result = run_configured_simulation()
    final_pos = result.position[-1]
    final_speed = result.speed[-1]
    max_error = np.max(np.linalg.norm(result.position - result.target_position, axis=1))
    scenario = "single gate minimum-time pass" if result.gates else "hover"
    print(f"Simulated {result.t[-1]:.2f}s {scenario} from {project_root()}")
    print(f"Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}] m")
    print(f"Final speed: {final_speed:.3f} m/s")
    print(f"Max tracking error: {max_error:.3f} m")
    if result.planner_info:
        print(f"Planner status: {result.planner_info['status']}")
        if "gate_crossing_time_s" in result.planner_info:
            print(f"Planned gate crossing: {result.planner_info['gate_crossing_time_s']:.3f}s")
            print(f"Planned gate speed: {result.planner_info['gate_speed_mps']:.3f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
