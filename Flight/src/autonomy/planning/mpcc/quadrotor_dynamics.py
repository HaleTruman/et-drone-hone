import numpy as np
import casadi as ca
import yaml
from pathlib import Path

PARAMS_PATH = Path(__file__).resolve().parents[3] / "core" / "quadrotor" / "params.yaml"
params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))

m = params["m"]
Ixx = params["Ixx"]
Iyy = params["Iyy"]
Izz = params["Izz"]
kf = params["kf"]
km = params["km"]
Jr = params["Jr"]
l = params["l"]
h = params["h"]
cd = params["cd"]
cr = params["cr"]
d = params["d"]
g = params["g"]
c = 2**0.5 / 2
rho = ca.DM([[l[0]*c, l[0]*c, h[0]], [-l[1]*c, l[1]*c, h[1]], [-l[2]*c, -l[2]*c, h[2]], [l[3]*c, -l[3]*c, h[3]]])

def rotation_matrix(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
        ca.horzcat(2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)),
        ca.horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)),
    )

def quaternion_to_euler(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    roll = ca.atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    pitch = ca.asin(2*(qw*qy-qz*qx))
    yaw = ca.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return ca.vertcat(roll, pitch, yaw)

def thrust_allocation(u):
    u = ca.vertcat(u[0], u[1], u[2], u[3])
    T = kf * (u*u)
    O = ca.vertcat(d[0], d[1], d[2], d[3]) * (kf**0.5) * u
    return T, O

def compute_total_thrust(u):
    T, _ = thrust_allocation(u)
    return ca.sum1(T)

def f(x, u):
    p = x[0:3]
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]
    R = rotation_matrix(q)
    T, O = thrust_allocation(u)
    vb = R.T @ v
    F = ca.vertcat(0, 0, -ca.sum1(T)) - ca.vertcat(cd[0]*vb[0]*ca.fabs(vb[0]), cd[1]*vb[1]*ca.fabs(vb[1]), cd[2]*vb[2]*ca.fabs(vb[2]))
    M = ca.vertcat(0, 0, 0)
    for i in range(4):
        M += ca.cross(rho[i, :].T, ca.vertcat(0, 0, -T[i]))
    M += ca.vertcat(0, 0, -km * (d[0]*u[0]*u[0] + d[1]*u[1]*u[1] + d[2]*u[2]*u[2] + d[3]*u[3]*u[3]))
    M += -ca.cross(w, ca.vertcat(0, 0, Jr * ca.sum1(O)))
    M += -ca.vertcat(cr[0]*w[0]*ca.fabs(w[0]), cr[1]*w[1]*ca.fabs(w[1]), cr[2]*w[2]*ca.fabs(w[2]))
    Xi = ca.vertcat(
        ca.horzcat(-q[1], -q[2], -q[3]),
        ca.horzcat(q[0], -q[3], q[2]),
        ca.horzcat(q[3], q[0], -q[1]),
        ca.horzcat(-q[2], q[1], q[0]),
    )
    Iw = ca.vertcat(Ixx*w[0], Iyy*w[1], Izz*w[2])
    wdot = ca.vertcat((M[0]-ca.cross(w, Iw)[0])/Ixx, (M[1]-ca.cross(w, Iw)[1])/Iyy, (M[2]-ca.cross(w, Iw)[2])/Izz)
    return ca.vertcat(v, R @ F / m + ca.vertcat(0, 0, g), 0.5 * Xi @ w, wdot)
