"""Simple gate map transform checks with sample vision observations."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import VehicleState
from mapping.gates import GateMap
from mapping.perception import VisionGateObservation, VisionObservation
from core.coordinates import quaternion_from_roll_pitch_yaw_deg, euler_from_quaternion
from validation.gate_map_orientation_view import show_gate_map_orientation_view


CAMERA_TILT_DEG = 20.0
EPSILON = 1e-9



gate_map = GateMap()
vehicle_state = VehicleState(
        sim_time_ns=0,
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=quaternion_from_roll_pitch_yaw_deg(0.0, -17.8, 0.0),
        body_rates_frd_rps=(0.0, 0.0, 0.0),
        acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
    )


print("Starting gates: ", gate_map.gates)
tilt_rad = math.radians(CAMERA_TILT_DEG)
cos_tilt = math.cos(tilt_rad)
sin_tilt = math.sin(tilt_rad)

print("GATE MAP TRANSFORM VALIDATION")
print(f"camera_tilt_deg = {CAMERA_TILT_DEG}")
print(f"vehicle_state.position_local_ned_m = {vehicle_state.position_local_ned_m}")
print(f"vehicle_state.attitude_quaternion = {vehicle_state.attitude_quaternion}\n")


observation = VisionObservation(
    frame_id=42,
    sim_time_ns=123_456_789,
    gates=[
        VisionGateObservation(
            gate_id="sample_gate_0",
            position_camera_m=(0.0, 0.0, 10.0),
            position_confidence=1.0,
            orientation_camera=(0.0, 0.0, 0.0),
            orientation_confidence=1.0,
        ),
        VisionGateObservation(
            gate_id="sample_gate_1",
            position_camera_m=(2.5, 5.4, 20.0),
            position_confidence=1.0,
            orientation_camera=(0.0, 0.0, 0.0),
            orientation_confidence=1.0,
        ),
        VisionGateObservation(
            gate_id="sample_gate_2",
            position_camera_m=(-3.0, -0.2, 29.0),
            position_confidence=1.0,
            orientation_camera=(0.0, 0.0, 0.0),
            orientation_confidence=1.0,
        )
    ],
)
print("Observation: ", observation)
records = gate_map.update_from_observation(observation, vehicle_state)
record = records[0]

for record in records:
    print(record.position_local_ned_m)
    
print("UPDATED GATE RECORD:")
print(record.position_local_ned_m)
print(euler_from_quaternion(record.quaternion, units="deg"))
print(record.quaternion)
show_gate_map_orientation_view(observation, vehicle_state, camera_tilt_deg=CAMERA_TILT_DEG)
