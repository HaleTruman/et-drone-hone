import casadi as ca
GATE_R = 0.7

def setup_mpcc_problem(dynamics_func, N=60, dt=0.08):
    opti = ca.Opti()
    X = opti.variable(13, N + 1)
    U = opti.variable(4, N)
    h = opti.variable()
    x0 = opti.parameter(13)
    pref = opti.parameter(3, N + 1)
    tref = opti.parameter(3, N + 1)
    gate_pos = opti.parameter(3)
    gate_normal = opti.parameter(3)

    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(opti.bounded(0, U, 1))
    opti.subject_to(opti.bounded(0.02, h, 0.18))

    J = 0
    motor_rate_limit = 4.0
    for k in range(N):
        x = X[:, k]
        u = U[:, k]
        k1 = dynamics_func(x, u)
        k2 = dynamics_func(x + h*k1/2, u)
        k3 = dynamics_func(x + h*k2/2, u)
        k4 = dynamics_func(x + h*k3, u)
        xn = x + h*(k1 + 2*k2 + 2*k3 + k4)/6
        qn = xn[6:10] / ca.sqrt(ca.sumsqr(xn[6:10]) + 1e-12)
        opti.subject_to(X[:, k + 1] == ca.vertcat(xn[0:6], qn, xn[10:13]))

        e = X[0:3, k] - pref[:, k]
        lag = ca.dot(e, tref[:, k])
        con = e - lag * tref[:, k]
        J += 120*ca.sumsqr(con) + 2*lag*lag + 0.01*ca.sumsqr(U[:, k]) + 0.02*ca.sumsqr(X[10:13, k])
        if k:
            opti.subject_to(opti.bounded(-motor_rate_limit*h, U[:, k] - U[:, k - 1], motor_rate_limit*h))
            J += 2*ca.sumsqr(U[:, k] - U[:, k - 1]) + 0.02*ca.sumsqr(X[10:13, k] - X[10:13, k - 1])
        opti.subject_to(ca.sumsqr(X[6:10, k]) >= 0.95)
        opti.subject_to(ca.sumsqr(X[6:10, k]) <= 1.05)

    gate_k = int(round(0.75 * N))
    opti.subject_to(ca.sumsqr(X[0:3, gate_k] - gate_pos) <= GATE_R**2)
    opti.subject_to(ca.dot(gate_normal, X[3:6, gate_k]) >= 0.5)
    opti.subject_to(ca.sumsqr(X[6:10, N]) >= 0.95)
    opti.subject_to(ca.sumsqr(X[6:10, N]) <= 1.05)
    J += 1000*ca.sumsqr(X[0:3, gate_k] - gate_pos) + 500*ca.sumsqr(X[0:3, N] - pref[:, N]) + 240*gate_k*h - 10*ca.dot(gate_normal, X[3:6, gate_k])

    opti.minimize(J)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 250, "tol": 1e-4})
    return opti, {"X": X, "U": U, "h": h}, {"x0": x0, "pref": pref, "tref": tref, "gate_pos": gate_pos, "gate_normal": gate_normal}, {"N": N, "dt": dt, "motor_rate_limit": motor_rate_limit, "gate_k": gate_k}
