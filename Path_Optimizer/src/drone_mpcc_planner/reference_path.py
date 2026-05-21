import numpy as np
GATE_OUTER_M = 2.7
GATE_INNER_M = 1.5
GATE_DEPTH_M = 0.26

def R(q):
    qw, qx, qy, qz = q
    return np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)], [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)], [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]])

def square(p, rot, x, size):
    a = size / 2
    pts = np.array([[x, -a, -a], [x, a, -a], [x, a, a], [x, -a, a], [x, -a, -a]])
    return p + pts @ rot.T

def gate_wireframe(gate):
    p = np.array(gate["pos"], float)
    rot = R(np.array(gate["quat"], float))
    xs = [-GATE_DEPTH_M / 2, GATE_DEPTH_M / 2]
    frames = [square(p, rot, x, GATE_OUTER_M) for x in xs] + [square(p, rot, x, GATE_INNER_M) for x in xs]
    a = GATE_OUTER_M / 2
    b = GATE_INNER_M / 2
    for y, z in [(-a, -a), (a, -a), (a, a), (-a, a), (-b, -b), (b, -b), (b, b), (-b, b)]:
        pts = np.array([[xs[0], y, z], [xs[1], y, z]])
        frames.append(p + pts @ rot.T)
    return frames

def yaw_quat(deg):
    a = np.deg2rad(deg) / 2
    return [np.cos(a), 0, 0, np.sin(a)]

def generate_reference(start_pos, gates, num_points=50, pass_through_m=4.0):
    gate_pts = np.array([g["pos"] for g in gates], float)
    gate_dirs = np.array([R(np.array(g["quat"], float))[:, 0] for g in gates], float)
    finish = gate_pts[-1] + gate_dirs[-1] * float(pass_through_m)
    pts = np.vstack([np.array(start_pos, float), gate_pts, finish])
    gate_i = list(range(1, len(gates) + 1))
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if s[-1] == 0:
        s[-1] = 1
    theta = s / s[-1]

    th = np.linspace(0, 1, num_points)
    xyz = np.column_stack([np.interp(th, theta, pts[:, i]) for i in range(3)])
    dxyz = np.zeros_like(xyz)
    for k, value in enumerate(th):
        seg = min(np.searchsorted(theta, value, side="right") - 1, len(pts) - 2)
        seg = max(0, seg)
        dtheta = max(theta[seg + 1] - theta[seg], 1e-9)
        dxyz[k] = (pts[seg + 1] - pts[seg]) / dtheta
    n = np.linalg.norm(dxyz, axis=1)
    tan = dxyz / np.maximum(n[:, None], 1e-9)
    wires = [w for g in gates for w in gate_wireframe(g)]
    dims = {"outer_m": GATE_OUTER_M, "inner_m": GATE_INNER_M, "depth_m": GATE_DEPTH_M}
    return {"theta": th, "px": xyz[:, 0], "py": xyz[:, 1], "pz": xyz[:, 2], "pos": xyz, "tan": tan, "length": s[-1], "gates": gates, "gate_theta": theta[gate_i], "gate_wireframes": wires, "gate_dimensions": dims}
