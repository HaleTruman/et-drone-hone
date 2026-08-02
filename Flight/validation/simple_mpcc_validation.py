import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomy.planning.simple_mpcc.planner import MPCCPlanner
from autonomy.planning.simple_mpcc.reference_path import yaw_quat
from autonomy.planning.simple_mpcc.visualizer import visualize
from core.coordinates import quaternion_from_roll_pitch_yaw_deg

GATE_APPROACH_M = 1.0
GATE_EXIT_M = 0.5
GATE_ALIGN_SPACING_FRACTION = 0.35
HOVER_THRUST_CMD = 0.265

current_state = np.array([0, 0, 0, 0, 0, 0, 0.9810200504409996, -0.00014745527995722662, -0.19390626066334854, 0, 0, 0, 0], float)
gates = [
    {"pos": [12.0, 0.0, 0.0], "quat":quaternion_from_roll_pitch_yaw_deg(0, 0, 0)},
    {"pos": [30.0, 12.0, -8.0], "quat":quaternion_from_roll_pitch_yaw_deg(0.0, -10.0, -25.0)}
]

planner = MPCCPlanner(
    False,
    gate_approach_m=GATE_APPROACH_M,
    gate_exit_m=GATE_EXIT_M,
    gate_align_spacing_fraction=GATE_ALIGN_SPACING_FRACTION,
    hover_thrust_cmd=HOVER_THRUST_CMD,
)
q_des, thrust_cmd, info = planner.plan(current_state, gates)
print("q_des:", q_des)
print("thrust_cmd:", thrust_cmd)
print("model_thrust_fraction:", info["model_thrust_fraction"])
print("thrust_output_scale:", info["thrust_output_scale"])
ref = planner.reference
print("gate_dimensions:", ref["gate_dimensions"])
print("gate_misses_m:", info["gate_misses_m"])
print("gate_crossing_times_s:", info["gate_crossing_times_s"])
print("expected_total_time_start_to_last_gate_s:", info["expected_total_time_start_to_last_gate_s"])
visualize(ref["pos"].T, gates, planner.warm_start["X"], planner.warm_start["U"], ref)
