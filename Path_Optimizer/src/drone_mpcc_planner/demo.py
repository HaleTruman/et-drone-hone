import numpy as np
from planner import MPCCPlanner
from reference_path import yaw_quat
from visualizer import visualize

current_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], float)
gates = [{"pos": [12.0, 0.0, 0.0], "quat": yaw_quat(0)}]

planner = MPCCPlanner(False)
q_des, thrust_cmd, info = planner.plan(current_state, gates)
print("q_des:", q_des)
print("thrust_cmd:", thrust_cmd)
ref = planner.reference
print("gate_dimensions:", ref["gate_dimensions"])
print("gate_misses_m:", info["gate_misses_m"])
visualize(ref["pos"].T, gates, planner.warm_start["X"], planner.warm_start["U"], ref)
