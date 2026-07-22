"""Simple gate map transform checks with sample vision observations."""

from __future__ import annotations

import time
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import VehicleState
from mapping.gates import GateMap
from core.coordinates import quaternion_from_roll_pitch_yaw_deg, euler_from_quaternion
from sensing.vision.service import VisionPerceptionService, VisionPerceptionConfig
from validation.gate_map_orientation_view import show_gate_map_orientation_view


CAMERA_TILT_DEG = 20.0
frame_path = SRC.parent / Path(
    r"logs\runs\run-20260722T033600Z\vision_frames\frame-00000445-1784691374665223300.jpg"
)



gate_map = GateMap()
vehicle_state = VehicleState(
        sim_time_ns=0,
        position_local_ned_m=(4.811570266543818,
            -0.15288991305016666,
            -0.11961158207688252),
        velocity_local_ned_mps=(4.47117843239257,
            -0.22568238239104813,
            -0.0725597449418581),
        attitude_quaternion=(0.9967387753323732,
            -0.00671169100087157,
            -0.07798697990371752,
            -0.019616266676581257),
        body_rates_frd_rps=(0.02716052532196045,
            0.14999257028102875,
            0.047370508313179016),
        acceleration_local_ned_mps2=(0.562250093946285,
            -0.1258951301358195,
            -0.057597492363919756),
    )


print("Starting gates: ", gate_map.gates)
tilt_rad = math.radians(CAMERA_TILT_DEG)
cos_tilt = math.cos(tilt_rad)
sin_tilt = math.sin(tilt_rad)

print("GATE MAP TRANSFORM VALIDATION")
print(f"camera_tilt_deg = {CAMERA_TILT_DEG}")
print(f"vehicle_state.position_local_ned_m = {vehicle_state.position_local_ned_m}")
print(f"vehicle_state.attitude_quaternion = {vehicle_state.attitude_quaternion}\n")


frame_parts = frame_path.stem.split("-")

start = time.perf_counter()
observation = VisionPerceptionService(VisionPerceptionConfig(backend="deterministic_0721", run_landmarker=False)).process_frame(
    frame_id=int(frame_parts[1]),
    sim_time_ns=int(frame_parts[2]),
    jpeg_bytes=frame_path.read_bytes(),
)
end = time.perf_counter()

total_time_s = end-start
total_time_ms = total_time_s * 1000

print("Observation: ", observation, "-- Processing time (ms): ", total_time_ms)
records = gate_map.update_from_observation(observation, vehicle_state)

for record in records:
    print(record.position_local_ned_m)

if records:
    record = records[0]
    print("UPDATED GATE RECORD:")
    print(record.position_local_ned_m)
    print(euler_from_quaternion(record.quaternion, units="deg"))
    print(record.quaternion)
show_gate_map_orientation_view(observation, vehicle_state, camera_tilt_deg=CAMERA_TILT_DEG)
