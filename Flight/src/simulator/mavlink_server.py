"""Thin pymavlink adapter for the local flight-only simulator."""

from typing import Any

from simulator.telemetry import TelemetrySimulator

MAVLINK_CMD_SIM_RESET = 31000


class MavlinkServer:
    def __init__(self, simulator: TelemetrySimulator, endpoint: str = "udpout:127.0.0.1:14550", connection: Any = None):
        self.simulator = simulator
        if connection is None:
            from pymavlink import mavutil

            connection = mavutil.mavlink_connection(endpoint, source_system=1, source_component=1)
        self.connection = connection

    def poll(self) -> None:
        while True:
            message = self.connection.recv_match(blocking=False)
            if message is None:
                return
            if message.get_type() != "BAD_DATA":
                self.handle_message(message)

    def handle_message(self, message: Any) -> None:
        from pymavlink import mavutil

        message_type = message.get_type()
        if message_type == "COMMAND_LONG":
            if int(message.command) == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                self.simulator.arm() if float(message.param1) == 1.0 else self.simulator.disarm()
            elif int(message.command) == MAVLINK_CMD_SIM_RESET:
                self.simulator.reset()
        elif message_type == "SET_ACTUATOR_CONTROL_TARGET":
            if self.simulator.armed:
                self.simulator.set_motor_command(message.controls[:4])
        elif message_type == "SET_ATTITUDE_TARGET":
            if self.simulator.armed:
                self.simulator.apply_attitude_target({
                    "quaternion": message.q,
                    "body_rates_rps": [message.body_roll_rate, message.body_pitch_rate, message.body_yaw_rate],
                    "thrust": message.thrust,
                })
        elif message_type == "SET_POSITION_TARGET_LOCAL_NED":
            if not self.simulator.armed:
                return
            mask = int(message.type_mask)
            position_axes = [
                not bool(mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE),
                not bool(mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE),
                not bool(mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE),
            ]
            self.simulator.apply_position_target({
                "position_local_ned_m": None if not any(position_axes) else [message.x, message.y, message.z],
                "position_axes": position_axes,
                "velocity_local_ned_mps": [message.vx, message.vy, message.vz],
                "yaw_rad": None if mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE else message.yaw,
            })
        elif message_type == "TIMESYNC" and int(message.ts1) == 0:
            self.connection.mav.timesync_send(0, int(message.tc1))

    def emit(self, sample: Any, *, heartbeat: bool = False) -> None:
        from pymavlink import mavutil

        mav = self.connection.mav
        if heartbeat:
            base_mode = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED if self.simulator.armed else 0
            mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_QUADROTOR, mavutil.mavlink.MAV_AUTOPILOT_GENERIC, base_mode, 0, mavutil.mavlink.MAV_STATE_ACTIVE)
        position = sample.position_local_ned_m
        velocity = sample.velocity_local_ned_mps
        rates = sample.body_rates_rps
        acceleration = sample.acceleration_local_ned_mps2
        reported_time_ns = self.simulator.reported_time_ns(sample.sim_time_ns)
        boot_ms = reported_time_ns // 1_000_000
        boot_us = reported_time_ns // 1_000
        roll, pitch, yaw = self.simulator._euler_from_quaternion(sample.attitude)
        mav.attitude_send(boot_ms, roll, pitch, yaw, *rates)
        mav.local_position_ned_send(boot_ms, *position, *velocity)
        mav.odometry_send(
            boot_us,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
            *position,
            sample.attitude,
            *velocity,
            *rates,
            [0.0] * 21,
            [0.0] * 21,
            sample.reset_count,
            mavutil.mavlink.MAV_ESTIMATOR_TYPE_NAIVE,
        )
        mav.highres_imu_send(boot_us, *acceleration, *rates, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        mav.actuator_output_status_send(boot_us, 15, list(self.simulator._motor_command) + [0.0] * 28)

    def emit_collision(self, collision_id: int, impact_kg_mps: float) -> None:
        self.connection.mav.collision_send(0, collision_id, 0, 2, 0.0, 0.0, impact_kg_mps)
