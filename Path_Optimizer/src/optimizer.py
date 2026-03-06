import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt

g = np.array([0.0, -9.81])  # 2D gravity (x, y)

# ────────────────────────────────────────────────
# Spline helpers (unchanged)
# ────────────────────────────────────────────────

def quintic_hermite_coefficients(r0, v0, a0, r1, v1, a1, dt):
    A = np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [5, 4, 3, 2, 1, 0],
        [20, 12, 6, 2, 0, 0]
    ], dtype=float)
    b = np.array([r0, v0*dt, a0*dt**2, r1, v1*dt, a1*dt**2], dtype=float)
    return np.linalg.solve(A, b)


def evaluate_segment(c, dt, num_samples=50):
    tau = np.linspace(0, 1, num_samples)
    t = tau * dt
    powers   = np.stack([tau**5, tau**4, tau**3, tau**2, tau,   np.ones_like(tau)])
    dpowers  = np.stack([5*tau**4, 4*tau**3, 3*tau**2, 2*tau, np.ones_like(tau), np.zeros_like(tau)])
    ddpowers = np.stack([20*tau**3, 12*tau**2, 6*tau, 2*np.ones_like(tau), np.zeros_like(tau), np.zeros_like(tau)])
    r = np.dot(c, powers).T
    v = np.dot(c, dpowers).T / dt
    a = np.dot(c, ddpowers).T / dt**2
    return t, r, v, a


# ────────────────────────────────────────────────
# Rotational acceleration approximation (unchanged)
# ────────────────────────────────────────────────

def compute_omega_dot(a, jerk, snap, g=g):
    thrust = a + g[None, :]
    u1 = np.linalg.norm(thrust, axis=1, keepdims=True)
    u1 = np.maximum(u1, 1e-8)
    b3 = thrust / u1
    b3_dot = (jerk - np.sum(b3 * jerk, axis=1, keepdims=True) * b3) / u1
    omega_scalar = b3[:,0] * b3_dot[:,1] - b3[:,1] * b3_dot[:,0]
    omega = np.abs(omega_scalar)[:, np.newaxis]
    if len(omega) < 2:
        return np.zeros_like(omega.ravel())
    dt = 1.0 / (len(omega) - 1)
    return np.gradient(omega.ravel(), dt)


# ────────────────────────────────────────────────
# Trajectory evaluation for stats (unchanged)
# ────────────────────────────────────────────────

def evaluate_trajectory_stats(x, current_r, current_v, visible_r, visible_n, g=g):
    N = len(visible_r)
    dts = x[:N]
    ss  = x[N:2*N]
    a_knots = x[2*N:].reshape((N+1, 2))

    all_v_norms = []
    all_thrust_norms = []
    total_time = np.sum(dts)

    r = current_r.copy()
    v = current_v.copy()

    for i in range(N):
        r0, v0 = r, v
        r1 = visible_r[i]
        v1 = ss[i] * visible_n[i]
        a0 = a_knots[i]
        a1 = a_knots[i+1]

        c = np.stack([
            quintic_hermite_coefficients(r0[j], v0[j], a0[j], r1[j], v1[j], a1[j], dts[i])
            for j in range(2)
        ])

        _, _, v_seg, a_seg = evaluate_segment(c, dts[i], num_samples=80)

        all_v_norms.append(np.linalg.norm(v_seg, axis=1))
        all_thrust_norms.append(np.linalg.norm(a_seg + g[None,:], axis=1))

        r = r1.copy()
        v = v1.copy()

    max_speed = np.max(np.concatenate(all_v_norms)) if all_v_norms else 0.0
    max_accel = np.max(np.concatenate(all_thrust_norms)) if all_thrust_norms else 0.0

    return max_speed, max_accel, total_time


# ────────────────────────────────────────────────
# Plot callback (unchanged)
# ────────────────────────────────────────────────

class PlotCallback:
    def __init__(self, current_r, current_v, visible_r, visible_n, initial_x, interval=5):
        self.current_r = current_r.copy()
        self.current_v = current_v.copy()
        self.visible_r = visible_r.copy()
        self.visible_n = visible_n.copy()
        self.initial_x = initial_x.copy()
        self.interval = interval
        self.iter = 0

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.set_title('Drone Path Optimization – 3 Gates')
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.grid(True, alpha=0.3)

        for i, rg in enumerate(visible_r):
            self.ax.plot(rg[0], rg[1], 'ro', ms=10, label=f'Gate {i+1}' if i==0 else "")
            arr_len = 1.4
            self.ax.arrow(rg[0], rg[1],
                          arr_len * visible_n[i][0], arr_len * visible_n[i][1],
                          head_width=0.5, head_length=0.7, fc='r', ec='r', alpha=0.75)

        self.ax.plot(current_r[0], current_r[1], 'ko', ms=10, label='Start')

        self.current_line, = self.ax.plot([], [], 'b-', lw=1.6, label='Optimizing')
        self.final_line = None
        self.initial_line = None

        self.ax.legend(loc='upper left', fontsize='small')
        plt.draw()

    def plot_path(self, x, color='b', lw=1.6, ls='-', label=None, zorder=5):
        N = len(self.visible_r)
        dts = x[:N]
        ss  = x[N:2*N]
        a_knots = x[2*N:].reshape((N+1, 2))

        r_points = []
        r = self.current_r.copy()
        v = self.current_v.copy()

        for i in range(N):
            r0, v0 = r, v
            r1 = self.visible_r[i]
            v1 = ss[i] * self.visible_n[i]
            a0 = a_knots[i]
            a1 = a_knots[i+1]

            c = np.stack([
                quintic_hermite_coefficients(r0[j], v0[j], a0[j], r1[j], v1[j], a1[j], dts[i])
                for j in range(2)
            ])

            _, r_seg, _, _ = evaluate_segment(c, dts[i], num_samples=40)
            r_points.append(r_seg)
            r = r1.copy()
            v = v1.copy()

        all_r = np.vstack(r_points)
        line, = self.ax.plot(all_r[:,0], all_r[:,1], color=color, lw=lw, ls=ls,
                             label=label, zorder=zorder)
        return line

    def __call__(self, x):
        self.iter += 1
        if self.iter % self.interval != 0:
            return
        self.current_line.set_data([], [])
        self.current_line = self.plot_path(x, color='b', lw=1.6, ls='-', label='Optimizing')
        self.ax.set_title(f'Drone Path Optimization – iter {self.iter}')
        plt.draw()
        plt.pause(0.02)

    def show_initial_and_final(self, res_x=None):
        if self.initial_line is None:
            self.initial_line = self.plot_path(self.initial_x, color='0.65', lw=1.4, ls='--',
                                               label='Initial guess', zorder=4)
        if res_x is not None:
            if self.final_line is not None:
                self.final_line.remove()
            self.final_line = self.plot_path(res_x, color='#00bb44', lw=4.0, ls='-',
                                             label='Final optimized', zorder=10)
        self.ax.legend(loc='upper left', fontsize='small')
        self.ax.set_title('Drone Path Optimization – Done (3 gates)')
        plt.draw()


# ────────────────────────────────────────────────
# Main planner with progress printing in callback
# ────────────────────────────────────────────────

def plan_horizon(current_r, current_v, visible_r, visible_n, a_max, v_max, omega_dot_max, M, callback=None):
    N = M

    x0 = np.concatenate([
        np.ones(N) * 1.5,
        np.array([8.5, 9.5, 11.0]),
        np.zeros(2 * (N + 1))
    ])

    lb = np.concatenate([np.full(N, 0.08),
                         np.full(N, 0.8),
                         np.full(2*(N+1), -a_max)])
    ub = np.concatenate([np.full(N, 25.0),
                         np.full(N, v_max),
                         np.full(2*(N+1), a_max)])

    bounds = Bounds(lb, ub)

    def objective(x):
        penalty = 1.5 * max(0, v_max - x[N + (N-1)]) ** 2
        return np.sum(x[:N]) + penalty

    def constraints(x):
        dts = x[:N]
        ss  = x[N:2*N]
        a_knots = x[2*N:].reshape((N+1, 2))
        violations = []

        for i in range(N):
            r0 = current_r if i == 0 else visible_r[i-1]
            v0 = current_v if i == 0 else ss[i-1] * visible_n[i-1]
            r1 = visible_r[i]
            v1 = ss[i] * visible_n[i]
            a0 = a_knots[i]
            a1 = a_knots[i+1]

            c = np.stack([
                quintic_hermite_coefficients(r0[j], v0[j], a0[j], r1[j], v1[j], a1[j], dts[i])
                for j in range(2)
            ])

            t_seg, r_seg, v_seg, a_seg = evaluate_segment(c, dts[i], num_samples=60)
            dt_sample = dts[i] / max(1, len(v_seg) - 1)

            jerk = np.gradient(a_seg, dt_sample, axis=0)
            snap = np.gradient(jerk, dt_sample, axis=0)

            violations.append(np.max(np.linalg.norm(v_seg, axis=1)) - v_max)
            violations.append(np.max(np.linalg.norm(a_seg + g[None,:], axis=1)) - a_max)

            omega_dot = compute_omega_dot(a_seg, jerk, snap)
            violations.append(np.max(np.abs(omega_dot)) - omega_dot_max)

        return np.array(violations)

    cons = [{'type': 'ineq', 'fun': lambda x: -constraints(x)}]

    # ── Progress callback with printing ──
    global iter_counter
    iter_counter = 0

    def progress_callback(xk):
        global iter_counter
        iter_counter += 1
        if iter_counter % 8 == 0:  # print every 8th callback – adjust as needed (4, 5, 10…)
            sp, ac, tt = evaluate_trajectory_stats(xk, current_r, current_v, visible_r, visible_n)
            print(f"  iter {iter_counter:4d} | total time {tt:6.3f} s | max speed {sp:6.3f} m/s | max accel {ac:6.3f} m/s²")

    res = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP',
                   options={'maxiter': 800, 'ftol': 1e-6, 'disp': False},
                   callback=progress_callback)

    if callback is not None:
        callback.show_initial_and_final(res.x if res.success else None)

    if res.success:
        dts = res.x[:N]
        ss  = res.x[N:2*N]
        a_knots = res.x[2*N:].reshape((N+1, 2))
        return dts, ss, a_knots, res.fun, res.x
    else:
        print("Optimization failed or did not converge well:", res.message)
        return None, None, None, None, None


# ────────────────────────────────────────────────
# Run with 3 gates
# ────────────────────────────────────────────────

if __name__ == "__main__":
    current_r = np.array([0.0, 0.0])
    current_v = np.array([0.0, 0.0])

    visible_r = np.array([
        [10.0,  2.0],
        [18.0,  6.5],
        [19.0,  6.5]
    ])

    visible_n = np.array([
        [1.00,  0.00],
        [0.0, 1.0],
        [0.0, -1.0]
    ])
    visible_n = visible_n / np.linalg.norm(visible_n, axis=1, keepdims=True)

    a_max = 40
    v_max = 20
    omega_dot_max = 500
    M = 3

    dummy_x = np.concatenate([
        np.ones(M) * 1.5,
        np.array([8.5, 9.5, 11.0]),
        np.zeros(2 * (M + 1))
    ])

    # Pre-optimization stats
    print("Pre-optimization (initial guess) trajectory:")
    init_max_speed, init_max_accel, init_total_time = evaluate_trajectory_stats(
        dummy_x, current_r, current_v, visible_r, visible_n
    )
    print(f"  Max speed:      {init_max_speed:.2f} m/s")
    print(f"  Max acceleration: {init_max_accel:.2f} m/s²")
    print(f"  Total time:     {init_total_time:.2f} s")
    print("-" * 60)

    callback = PlotCallback(current_r, current_v, visible_r, visible_n, dummy_x, interval=5)

    print("\nOptimization progress (printed every 8 iterations):")
    dts, ss, a_knots, total_t, final_x = plan_horizon(
        current_r, current_v, visible_r, visible_n,
        a_max, v_max, omega_dot_max, M, callback=callback
    )

    if dts is not None:
        print("\n" + "="*60)
        print("Optimized results (3 gates):")
        print("  Segment times     :", dts.round(3), "s")
        print("  Gate exit speeds  :", ss.round(2), "m/s")
        print("  Total time        :", total_t.round(3), "s")

        final_max_speed, final_max_accel, _ = evaluate_trajectory_stats(
            final_x, current_r, current_v, visible_r, visible_n
        )
        print(f"  Final max speed:   {final_max_speed:.2f} m/s")
        print(f"  Final max accel:   {final_max_accel:.2f} m/s²")

    plt.ioff()
    plt.show()