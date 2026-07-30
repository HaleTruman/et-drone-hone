"""Simple gate map checks with sample vision observations."""

from __future__ import annotations

import time
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapping.gates import GateMap
from core.coordinates import euler_from_quaternion
from sensing.vision.service import VisionPerceptionService, VisionPerceptionConfig
from validation.gate_map_orientation_view import show_gate_map_orientation_view


frame_path = SRC.parent / Path(
    r"logs\runs\run-20260722T033600Z\vision_frames\frame-00000445-1784691374665223300.jpg"
)



gate_map = GateMap()


print("Starting gates: ", gate_map.gates)

print("GATE MAP VALIDATION")


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
records = gate_map.update(observation)

for record in records:
    print(record.position_local_ned_m)

if records:
    record = records[0]
    print("UPDATED GATE RECORD:")
    print(record.position_local_ned_m)
    print(euler_from_quaternion(record.quaternion, units="deg"))
    print(record.quaternion)
show_gate_map_orientation_view(observation)
