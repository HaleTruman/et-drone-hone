import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomy.planning.mpcc.planner import MPCCPlanner
from autonomy.planning.mpcc.reference_path import yaw_quat
from autonomy.planning.mpcc.visualizer import visualize
from core.coordinates import quaternion_from_roll_pitch_yaw_deg

current_state = np.array([0, 0, 0, 0, 0, 0, 0.9810200504409996, -0.00014745527995722662, -0.19390626066334854, 0, 0, 0, 0], float)
gates = [
    {"pos": [12.0, 0.0, 0.0], "quat":quaternion_from_roll_pitch_yaw_deg(0, 0, 0)},
    {"pos": [24.0, 0.0, 0.0], "quat":quaternion_from_roll_pitch_yaw_deg(0, 0, 0)}
]

planner = MPCCPlanner()
q_des, thrust_cmd, info = planner.plan(current_state, gates)
print("q_des:", q_des)
print("thrust_cmd:", thrust_cmd)
ref = planner.reference
print("gate_dimensions:", ref["gate_dimensions"])
print("gate_misses_m:", info["gate_misses_m"])
print("gate_crossing_times_s:", info["gate_crossing_times_s"])
print("expected_total_time_start_to_last_gate_s:", info["expected_total_time_start_to_last_gate_s"])
visualize(ref["pos"].T, gates, planner.warm_start["X"], planner.warm_start["U"], ref)
