import numpy as np
import casadi as ca


class PlanningEngine:
    def __init__(self, params):
        self.params = params
        # Pre-compute constants for speed
        self.c = np.sqrt(2.0) / 2.0
        self.I_B = np.diag([params['Ixx'], params['Iyy'], params['Izz']])
        self.I_inv = np.linalg.inv(self.I_B)
        self.l_arr = np.asarray(params['l'])
        self.h_arr = np.asarray(params['h'])
        self.cd = np.asarray(params['cd'])
        self.cr = np.asarray(params['cr'])
        self.d_arr = np.asarray(params['d'])
        self.kf = params['kf']
        self.km = params['km']
        self.Jr = params['Jr']
        self.m = params['m']
        self.g = params['g']

    def generate_initial_guess(self, waypoints, x0=None, nodes_per_segment=60):
            """Return dict with keys: 't', 'x', 'u', 'lam', 'mu', 'nu'
            waypoints should be FUTURE gates only (do NOT include origin/start)."""
            w = np.asarray(waypoints, dtype=float).reshape(-1, 3)
            M = w.shape[0]  # number of future waypoints
            if M < 1:
                raise ValueError("At least one future waypoint required.")

            if x0 is None:
                x0 = np.zeros(13, dtype=float)
                x0[6] = 1.0
            x0 = np.asarray(x0, dtype=float).copy()

            # Conservative a_max estimate
            a_max = max(3.5 * self.params['kf'] / self.params['m'] - self.params['g'], 8.0)

            total_dist = np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1)) if M > 1 else np.linalg.norm(w[0] - x0[0:3])
            v_max = min(np.sqrt(a_max * total_dist * 0.6), 25.0)
            t_total = total_dist / (0.75 * v_max) + 0.5  # small margin

            n_nodes = M * nodes_per_segment + 1
            t = np.linspace(0.0, t_total, n_nodes, dtype=float)

            x = np.zeros((n_nodes, 13), dtype=float)
            u = np.zeros((n_nodes - 1, 4), dtype=float)
            lam = np.ones((n_nodes, M), dtype=float)   # start at 1
            mu = np.zeros((n_nodes - 1, M), dtype=float)
            nu = np.zeros((n_nodes - 1, M), dtype=float)

            kf, m, g = self.params['kf'], self.params['m'], self.params['g']
            u_hover = np.sqrt(m * g / (4.0 * kf))

            # Linear interpolation along waypoints (from current position)
            positions = np.vstack([x0[0:3], w])
            cum_dist = np.cumsum(np.concatenate(([0.], np.linalg.norm(np.diff(positions, axis=0), axis=1))))
            cum_dist /= cum_dist[-1] if cum_dist[-1] > 0 else 1.0

            x[0] = x0.copy()
            lam[0] = 1.0
            lam[-1] = 0.0

            for i in range(n_nodes):
                s = float(i) / (n_nodes - 1)
                seg = np.searchsorted(cum_dist, s) - 1
                seg = np.clip(seg, 0, M)
                alpha = (s - cum_dist[seg]) / (cum_dist[seg+1] - cum_dist[seg]) if seg < M else 1.0

                p = (1 - alpha) * positions[seg] + alpha * positions[seg + 1]

                # Velocity heuristic
                speed = v_max * (0.2 + 0.6 * min(s, 1.0 - s)) if M > 1 else v_max * s
                dir_vec = positions[seg+1] - positions[seg]
                dir_vec /= np.linalg.norm(dir_vec) + 1e-9
                vel = speed * dir_vec

                x[i, 0:3] = p
                x[i, 3:6] = vel
                x[i, 6:10] = [1.0, 0.0, 0.0, 0.0]
                x[i, 10:13] = 0.0

                if i < n_nodes - 1:
                    u[i] = u_hover
                    # Place small mu activation near expected crossing
                    for j in range(M):
                        gate_s = cum_dist[j+1]
                        if abs(s - gate_s) < 2.0 / (n_nodes - 1):
                            mu[i, j] = 1.0 / nodes_per_segment

            return {'t': t, 'x': x, 'u': u, 'lam': lam, 'mu': mu, 'nu': nu}


    def _rotation_matrix(self, q):
        """Body → inertial rotation matrix from unit quaternion q = [qw, qx, qy, qz]."""
        qw, qx, qy, qz = q
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ], dtype=float)


    def _quadrotor_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Return \dot{x} (13,) using the exact 13-state nonlinear rigid-body quadrotor model
        described in quadrotor_nonlinear_model.md (full thrust, gyro, drag, quaternion kinematics, etc.).
        x = [px, py, pz, vx_I, vy_I, vz_I, qw, qx, qy, qz, p, q, r]
        u = [u1, u2, u3, u4] ∈ [0,1]
        """
        # Unpack state
        vel_I = x[3:6]
        q = x[6:10]
        omega_B = x[10:13]

        # Motor thrusts and signed rotor speeds
        T = self.kf * u**2
        Omega = self.d_arr * np.sqrt(self.kf) * u
        sum_Omega = np.sum(Omega)

        # Motor positions ρ_i in body frame (X-config)
        rho = np.zeros((4, 3))
        for i in range(4):
            li = self.l_arr[i]
            hi = self.h_arr[i]
            if i == 0:      # front-right
                rho[i] = np.array([self.c * li, self.c * li, hi])
            elif i == 1:    # rear-right
                rho[i] = np.array([-self.c * li, self.c * li, hi])
            elif i == 2:    # rear-left
                rho[i] = np.array([-self.c * li, -self.c * li, hi])
            elif i == 3:    # front-left
                rho[i] = np.array([self.c * li, -self.c * li, hi])

        # Forces in body frame
        F_thrust_B = np.array([0.0, 0.0, -np.sum(T)])
        R = self._rotation_matrix(q)
        v_B = R.T @ vel_I
        F_drag_B = -self.cd * v_B * np.abs(v_B)
        F_B = F_thrust_B + F_drag_B

        # Thrust moments
        M_thrust_B = np.zeros(3)
        for i in range(4):
            f_i = np.array([0.0, 0.0, -T[i]])
            M_thrust_B += np.cross(rho[i], f_i)

        # Motor reaction torques (yaw only)
        M_motor_B = np.array([0.0, 0.0, -self.km * np.sum(self.d_arr * u**2)])

        # Gyroscopic moments from rotor inertia
        M_gyro_B = -self.Jr * sum_Omega * np.array([omega_B[1], -omega_B[0], 0.0])

        # Rotational drag
        M_drag_B = -self.cr * omega_B * np.abs(omega_B)

        M_B = M_thrust_B + M_motor_B + M_gyro_B + M_drag_B

        # State derivatives
        dx = np.zeros(13)

        # Position kinematics
        dx[0:3] = vel_I

        # Translational dynamics (Newton)
        F_I = R @ F_B
        dx[3:6] = F_I / self.m + np.array([0.0, 0.0, self.g])

        # Quaternion kinematics
        Xi = np.array([
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]]
        ])
        dx[6:10] = 0.5 * Xi @ omega_B

        # Rotational dynamics (Euler)
        omega_cross_Iomega = np.cross(omega_B, self.I_B @ omega_B)
        dx[10:13] = self.I_inv @ (M_B - omega_cross_Iomega)

        return dx

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Single RK4 integration step of the full nonlinear dynamics."""
        k1 = self._quadrotor_dynamics(x, u)
        k2 = self._quadrotor_dynamics(x + 0.5 * dt * k1, u)
        k3 = self._quadrotor_dynamics(x + 0.5 * dt * k2, u)
        k4 = self._quadrotor_dynamics(x + dt * k3, u)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Renormalize quaternion (states 6:10)
        q_norm = np.linalg.norm(x_next[6:10])
        if q_norm > 1e-12:
            x_next[6:10] /= q_norm
        return x_next


    def _flatten_for_solver(self, guess: dict, N_nodes: int, N: int, M: int):
        t_N = guess['t'][-1]
        x_dyn_flat = guess['x'][1:].ravel()   # x0 fixed
        u_flat = guess['u'].ravel()
        lam_flat = guess['lam'].ravel()
        mu_flat = guess['mu'].ravel()
        nu_flat = guess['nu'].ravel()
        return np.concatenate([np.array([t_N]), x_dyn_flat, u_flat, lam_flat, mu_flat, nu_flat])

    def _unflatten_solution(self, sol_vec: np.ndarray, N_nodes: int, N: int, M: int, t_N: float):
        offset = 1
        x = np.zeros((N_nodes, 13))
        x[0] = np.zeros(13)  # will be set by caller
        x[1:] = sol_vec[offset:offset + (N_nodes-1)*13].reshape((N_nodes-1, 13))
        offset += (N_nodes-1)*13

        u = sol_vec[offset:offset + N*4].reshape((N, 4))
        offset += N*4

        lam = sol_vec[offset:offset + N_nodes*M].reshape((N_nodes, M))
        offset += N_nodes*M

        mu = sol_vec[offset:offset + N*M].reshape((N, M))
        offset += N*M

        nu = sol_vec[offset:offset + N*M].reshape((N, M))

        t = np.linspace(0.0, t_N, N_nodes)
        return {'t': t, 'x': x, 'u': u, 'lam': lam, 'mu': mu, 'nu': nu}


    def optimize_cpc(self,
                     guess: dict,
                     gates_rel: list | np.ndarray,   # FUTURE waypoints ONLY
                     x_current: np.ndarray | None = None,
                     d_tol: float = 1.2,
                     u_min: float = 0.0,
                     u_max: float = 1.0,
                     max_iter: int = 800,
                     tol: float = 1e-6) -> dict:
        """CPC optimization - corrected formulation"""
        if x_current is None:
            x_current = guess['x'][0].copy()

        w_rel = np.asarray(gates_rel, dtype=float).reshape(-1, 3)
        M = w_rel.shape[0]   # number of future waypoints
        if M < 1:
            raise ValueError("At least one future waypoint required.")

        N_nodes = guess['x'].shape[0]
        N = N_nodes - 1

        # Transform relative gates to inertial
        R0 = self._rotation_matrix(x_current[6:10])
        w_inertial = x_current[0:3] + (R0 @ w_rel.T).T

        opti = ca.Opti()

        relax_param = opti.parameter()          # we'll set it to a small value (e.g. 1e-4)
        opti.set_value(relax_param, 1e-4)

        T   = opti.variable()                       # final time
        X   = opti.variable(13, N_nodes)            # states
        U   = opti.variable(4, N)                   # controls
        Lam = opti.variable(M, N_nodes)             # λ
        Mu  = opti.variable(M, N)                   # μ
        Nu  = opti.variable(M, N)                   # ν slack

        opti.minimize(T)

        # Initial state (hard constraint)
        opti.subject_to(X[:, 0] == x_current)

        # Dynamics (RK4)
        dt = T / N
        for k in range(N):
            x_next = self._rk4_step_casadi(opti, X[:, k], U[:, k], dt)
            opti.subject_to(X[:, k + 1] == x_next)

        # Progress evolution: λ_{k+1} = λ_k - μ_k
        for k in range(N):
            opti.subject_to(Lam[:, k + 1] == Lam[:, k] - Mu[:, k])

        # === COMPLEMENTARITY (now over ALL nodes 0 to N) ===
        for k in range(N + 1):
            for j in range(M):
                idx = min(k, N-1)
                dist2 = ca.sum1((X[0:3, k] - w_inertial[j]) ** 2)
                # Scholtes relaxation: Mu * (dist2 - Nu) <= relax_param
                opti.subject_to(Mu[j, idx] * (dist2 - Nu[j, idx]) <= relax_param)
                # Optional: add the reverse direction softly
                opti.subject_to((dist2 - Nu[j, idx]) * Mu[j, idx] >= -relax_param)

                # Boundaries
                opti.subject_to(Lam[:, 0] == 1.0)
                opti.subject_to(Lam[:, -1] == 0.0)

        # Sequencing constraint (critical): μ_k^j <= μ_k^{j+1}
        for k in range(N):
            for j in range(M-1):
                opti.subject_to(Mu[j, k] <= Mu[j+1, k] + 1e-6)

        # Inequality constraints
        opti.subject_to(ca.vec(Mu) >= 0)
        opti.subject_to(ca.vec(Nu) >= 0)
        opti.subject_to(ca.vec(Nu) <= d_tol**2)
        opti.subject_to(ca.vec(U) >= u_min)
        opti.subject_to(ca.vec(U) <= u_max)
        opti.subject_to(ca.vec(Lam) >= 0)
        opti.subject_to(ca.vec(Lam) <= 1)
        opti.subject_to(T >= 0.5)

        # Warm-start
        opti.set_initial(T, guess['t'][-1])
        opti.set_initial(X, guess['x'].T)
        opti.set_initial(U, guess['u'].T)
        opti.set_initial(Lam, guess['lam'].T)
        opti.set_initial(Mu, guess['mu'].T)
        opti.set_initial(Nu, guess['nu'].T)

        # Solver settings (increased robustness)
        opti.solver("ipopt", {"expand": True}, {
            "max_iter": max_iter,
            "tol": tol,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 100,
            "print_level": 5,
            "mu_strategy": "adaptive",
            "linear_solver": "mumps",
            "nlp_scaling_method": "gradient-based",   # helps a lot
            "warm_start_init_point": "yes",
        })

        try:
            sol = opti.solve()
            print(f"✅ CPC optimization SUCCESS! Final time = {sol.value(T):.4f} s")
        except Exception as e:
            print(f"⚠️ Solver failed or stopped early: {e}")
            sol = opti  # return best point

        optimized = {
            't': np.linspace(0.0, sol.value(T), N_nodes),
            'x': sol.value(X).T,
            'u': sol.value(U).T,
            'lam': sol.value(Lam).T,
            'mu': sol.value(Mu).T,
            'nu': sol.value(Nu).T,
        }

        if 'sol' in locals() and hasattr(sol, 'value'):
            print("Final objective (t_N):", sol.value(T))
        else:
            print("Using debug values from last iteration")
            # Inspect last feasible-ish point
            print("Last λ:", opti.debug.value(Lam).T[-1])   # should be close to [0,0,...]
        
        return optimized


    # Helper for CasADi (symbolic RK4)
    def _rk4_step_casadi(self, opti, x, u, dt):
        """Symbolic RK4 step."""
        k1 = self._quadrotor_dynamics_casadi(x, u)
        k2 = self._quadrotor_dynamics_casadi(x + (dt/2) * k1, u)
        k3 = self._quadrotor_dynamics_casadi(x + (dt/2) * k2, u)
        k4 = self._quadrotor_dynamics_casadi(x + dt * k3, u)

        x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Quaternion renormalization
        q = x_next[6:10]
        q_norm = ca.sqrt(ca.sum1(q**2) + 1e-12)
        x_next[6:10] = q / q_norm

        return x_next


    def _quadrotor_dynamics_casadi(self, x, u):
        """Symbolic 13-state quadrotor dynamics for CasADi Opti.
        Exactly matches the NumPy version + fixes vector shapes for ca.cross.
        """
        import casadi as ca

        kf = self.kf
        km = self.km
        m = self.m
        g = self.g
        Jr = self.Jr
        I_B = ca.diag(ca.vertcat(self.params['Ixx'], self.params['Iyy'], self.params['Izz']))
        I_inv = ca.inv(I_B)

        l_arr = ca.DM(self.l_arr)
        h_arr = ca.DM(self.h_arr)
        cd = ca.DM(self.cd)
        cr = ca.DM(self.cr)
        d_arr = ca.DM(self.d_arr)
        c = self.c

        # Unpack state
        vel_I = x[3:6]
        q = x[6:10]
        omega_B = x[10:13]

        # Rotation matrix Body → Inertial
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        R = ca.vertcat(
            ca.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            ca.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            ca.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
        )

        # Motor thrusts and signed speeds
        T = kf * u**2
        Omega = d_arr * ca.sqrt(kf) * u
        sum_Omega = ca.sum1(Omega)

        # Motor positions ρ_i (make them column vectors)
        rho = ca.MX.zeros(3, 4)   # 3 x 4 for easier handling
        for i in range(4):
            li = l_arr[i]
            hi = h_arr[i]
            if i == 0:      # front-right
                rho[:, i] = ca.vertcat(c*li, c*li, hi)
            elif i == 1:    # rear-right
                rho[:, i] = ca.vertcat(-c*li, c*li, hi)
            elif i == 2:    # rear-left
                rho[:, i] = ca.vertcat(-c*li, -c*li, hi)
            elif i == 3:    # front-left
                rho[:, i] = ca.vertcat(c*li, -c*li, hi)

        # Forces in body frame
        F_thrust_B = ca.vertcat(0.0, 0.0, -ca.sum1(T))
        v_B = R.T @ vel_I
        F_drag_B = -cd * v_B * ca.fabs(v_B)
        F_B = F_thrust_B + F_drag_B

        # Thrust moments
        M_thrust_B = ca.MX.zeros(3)
        for i in range(4):
            f_i = ca.vertcat(0.0, 0.0, -T[i])          # 3x1
            rho_i = rho[:, i]                          # 3x1
            M_thrust_B += ca.cross(rho_i, f_i)

        # Motor reaction torque (yaw only)
        M_motor_B = ca.vertcat(0.0, 0.0, -km * ca.sum1(d_arr * u**2))

        # Gyroscopic moment
        M_gyro_B = -Jr * sum_Omega * ca.vertcat(omega_B[1], -omega_B[0], 0.0)

        # Rotational drag
        M_drag_B = -cr * omega_B * ca.fabs(omega_B)

        M_B = M_thrust_B + M_motor_B + M_gyro_B + M_drag_B

        # State derivatives
        dx = ca.MX.zeros(13)

        # Position
        dx[0:3] = vel_I

        # Translational dynamics
        F_I = R @ F_B
        dx[3:6] = F_I / m + ca.vertcat(0.0, 0.0, g)

        # Quaternion kinematics
        Xi = ca.vertcat(
            ca.horzcat(-q[1], -q[2], -q[3]),
            ca.horzcat(q[0], -q[3], q[2]),
            ca.horzcat(q[3], q[0], -q[1]),
            ca.horzcat(-q[2], q[1], q[0])
        )
        dx[6:10] = 0.5 * (Xi @ omega_B)

        # Rotational dynamics
        omega_cross_Iomega = ca.cross(omega_B, I_B @ omega_B)
        dx[10:13] = I_inv @ (M_B - omega_cross_Iomega)

        return dx
