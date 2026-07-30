import numpy as np
try:
    from quadrotor_dynamics import f, compute_total_thrust, kf
    from reference_path import generate_reference, R
    from mpcc_problem import setup_mpcc_problem
    from mpcc_solver import solve_mpcc
except ImportError:
    from .quadrotor_dynamics import f, compute_total_thrust, kf
    from .reference_path import generate_reference, R
    from .mpcc_problem import setup_mpcc_problem
    from .mpcc_solver import solve_mpcc

class MPCCPlanner:
    def __init__(self, visualize_result=False):
        self.pack = setup_mpcc_problem(f)
        self.warm_start = None
        self.reference = None
        self.visualize_result = visualize_result

    def plan(self, current_state, gates):
        if len(gates) < 1:
            raise ValueError("MPCC planning requires at least one gate.")

        ref = generate_reference(current_state[0:3], gates, self.pack[3]["N"] + 1)
        X, U, theta, info = solve_mpcc(self.pack, np.asarray(current_state, float), ref, self.warm_start)
        misses = [float(np.min(np.linalg.norm(X[0:3].T - np.array(g["pos"], float), axis=1))) for g in gates]
        gate_normals = [np.asarray(R(np.asarray(g["quat"], float))[:, 0], float) for g in gates]
        gate_crossing_steps = []
        for gate, gate_normal_i in zip(gates, gate_normals):
            gate_progress = (X[0:3].T - np.asarray(gate["pos"], float)) @ gate_normal_i
            crossed = np.flatnonzero(gate_progress >= 0.0)
            gate_crossing_steps.append(
                int(crossed[0]) if len(crossed) else int(np.argmin(np.abs(gate_progress)))
            )
        gate_crossing_step = gate_crossing_steps[0]
        last_gate_crossing_step = gate_crossing_steps[-1]
        self.warm_start = {"X": X, "U": U}
        self.reference = ref
        q = X[6:10, 1]
        q = q / np.linalg.norm(q)
        thrust = float(np.clip(float(compute_total_thrust(U[:, 0])) / (4 * kf), 0, 1))
        info["theta"] = theta
        info["q_des"] = q
        info["thrust_cmd"] = thrust
        info["planned_path_local_ned_m"] = X[0:3].T.tolist()
        info["reference_path_local_ned_m"] = ref["pos"].tolist()
        info["gate_count"] = len(gates)
        info["gates"] = gates
        info["gate_misses_m"] = misses
        info["gate_crossing_steps"] = gate_crossing_steps
        info["gate_crossing_times_s"] = [
            step * float(info.get("dt", self.pack[3]["dt"])) for step in gate_crossing_steps
        ]
        info["gate_crossing_step"] = gate_crossing_step
        info["gate_crossing_time_s"] = gate_crossing_step * float(info.get("dt", self.pack[3]["dt"]))
        info["last_gate_crossing_step"] = last_gate_crossing_step
        info["last_gate_crossing_time_s"] = last_gate_crossing_step * float(info.get("dt", self.pack[3]["dt"]))
        info["expected_total_time_start_to_last_gate_s"] = info["last_gate_crossing_time_s"]
        info["gate_speed_mps"] = float(np.linalg.norm(X[3:6, gate_crossing_step]))
        info["gate_forward_speed_mps"] = float(np.dot(gate_normals[0], X[3:6, gate_crossing_step]))
        info["last_gate_speed_mps"] = float(np.linalg.norm(X[3:6, last_gate_crossing_step]))
        info["last_gate_forward_speed_mps"] = float(
            np.dot(gate_normals[-1], X[3:6, last_gate_crossing_step])
        )
        info["motor_rate_limit"] = self.pack[3].get("motor_rate_limit")
        print("Gate misses:", misses)
        if self.visualize_result:
            try:
                from visualizer import visualize
            except ImportError:
                from .visualizer import visualize
            visualize(ref["pos"].T, gates, X, U, ref)
        self.warm_start["h"] = info["dt"]
        return q, thrust, info
