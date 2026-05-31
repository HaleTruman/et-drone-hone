import numpy as np

from controller.hover import HoverPIDController
from racing_stack.flight_state import FlightMode, FlightStateMachine
from telemetry_simulator import TelemetrySimulator


def test_armed_hover_holds_initial_altitude() -> None:
    simulator = TelemetrySimulator(telemetry_hz=10.0)
    flight_state = FlightStateMachine()
    controller = HoverPIDController(
        mass_kg=simulator.model.m,
        gravity_mps2=simulator.model.g,
        thrust_coefficient=simulator.model.kf,
        dt_s=0.1,
    )

    assert flight_state.mode == FlightMode.IDLE
    flight_state.update_state("arm")
    simulator.apply_attitude_target({"quaternion": [1.0, 0.0, 0.0, 0.0], "thrust": controller.hover_motor_command})

    for sample in simulator.telemetry_samples(duration_s=3.0):
        quaternion, thrust = controller.update(sample.raw["acceleration_local_ned_mps2"], np.zeros(3))
        simulator.apply_attitude_target({"quaternion": quaternion, "thrust": thrust})

    assert flight_state.mode == FlightMode.ARMED
    assert abs(simulator._harness.state[2] - simulator._harness.initial_state[2]) < 0.05


def test_attitude_target_is_mixed_into_individual_motor_commands() -> None:
    simulator = TelemetrySimulator()

    simulator.apply_attitude_target({"quaternion": [0.999, 0.04, 0.0, 0.0], "thrust": 0.5})

    assert simulator._motor_command[0] != simulator._motor_command[2]


def test_idle_fall_then_armed_attitude_target_recovery() -> None:
    simulator = TelemetrySimulator(telemetry_hz=10.0)
    controller = HoverPIDController(
        mass_kg=simulator.model.m,
        gravity_mps2=simulator.model.g,
        thrust_coefficient=simulator.model.kf,
        dt_s=0.1,
    )

    for sample in simulator.telemetry_samples(duration_s=5.0):
        if sample.sim_time_ns >= 1_000_000_000:
            quaternion, thrust = controller.update(
                sample.raw["acceleration_local_ned_mps2"],
                np.zeros(3),
                sample.velocity_local_ned_mps,
            )
            simulator.apply_attitude_target({"quaternion": quaternion, "thrust": thrust})

    assert simulator._harness.state[5] < 0.2


def test_xyz_target_changes_quaternion_and_thrust() -> None:
    simulator = TelemetrySimulator()
    controller = HoverPIDController(
        mass_kg=simulator.model.m,
        gravity_mps2=simulator.model.g,
        thrust_coefficient=simulator.model.kf,
        dt_s=0.1,
    )

    quaternion, thrust = controller.update((0.0, 0.0, 0.0), np.array([1.0, 0.0, 1.0]))

    assert quaternion.tolist() != [1.0, 0.0, 0.0, 0.0]
    assert thrust > controller.hover_motor_command
