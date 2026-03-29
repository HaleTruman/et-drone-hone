import numpy as np

class PlanningEngine:
    def __init__(self, params):
        self.params = params
        
    def generate_initial_guess(self, waypoints, x0=None, nodes_per_segment=60):
        """Return dict{'t':(N,), 'x':(N+1,13), 'u':(N,4), 'lam':(N+1,M), 'mu':(N,M), 'nu':(N,M)} 
        using continuous high-speed point-mass approximation."""
        w = np.asarray(waypoints, dtype=float).reshape(-1, 3)
        M = len(w) - 1
        if x0 is None:
            x0 = np.zeros(13, dtype=float)
            x0[6] = 1.0
        x0 = np.asarray(x0, dtype=float).copy()

        # Compute a_max (conservative)
        a_max = max(3.5 * self.params['kf'] / self.params['m'] - self.params['g'], 8.0)

        # Simple straight-line total distance for timing
        total_dist = np.sum(np.linalg.norm(np.diff(w, axis=0), axis=1))
        v_max = np.sqrt(a_max * total_dist * 0.6)   # rough heuristic for good speed
        v_max = min(v_max, 25.0)                    # cap for realism
        t_total = total_dist / (0.75 * v_max)       # allow some acceleration margin

        n_nodes = M * nodes_per_segment + 1
        t = np.linspace(0.0, t_total, n_nodes, dtype=float)

        x = np.zeros((n_nodes, 13), dtype=float)
        u = np.zeros((n_nodes - 1, 4), dtype=float)
        lam = np.zeros((n_nodes, M), dtype=float)
        mu = np.zeros((n_nodes - 1, M), dtype=float)
        nu = np.zeros((n_nodes - 1, M), dtype=float)

        kf, m, g = self.params['kf'], self.params['m'], self.params['g']
        u_hover = np.sqrt(m * g / (4.0 * kf))

        # Linear position interpolation (straight lines between gates)
        cum_dist = np.cumsum(np.concatenate(([0.], np.linalg.norm(np.diff(w, axis=0), axis=1))))
        cum_dist /= cum_dist[-1]

        idx = 0
        x[0] = x0.copy()
        lam[0] = 1.0

        for i in range(n_nodes):
            s = float(i) / (n_nodes - 1)          # progress along whole track [0,1]
            # Find which segment we are in
            seg = np.searchsorted(cum_dist, s) - 1
            seg = np.clip(seg, 0, M-1)
            alpha = (s - cum_dist[seg]) / (cum_dist[seg+1] - cum_dist[seg])

            p = (1 - alpha) * w[seg] + alpha * w[seg + 1]

            # Velocity: tangent to the path, magnitude ramps up then stays high
            if s < 0.15:
                speed = v_max * (s / 0.15)
            elif s > 0.85:
                speed = v_max * ((1.0 - s) / 0.15)
            else:
                speed = v_max

            # Direction of current segment
            dir_vec = w[seg+1] - w[seg]
            dir_vec /= np.linalg.norm(dir_vec) + 1e-9
            vel = speed * dir_vec

            x[i, 0:3] = p
            x[i, 3:6] = vel
            x[i, 6] = 1.0
            x[i, 7:13] = 0.0

            if i < n_nodes - 1:
                u[i] = u_hover
                # Place mu switches near the actual gate crossing times
                if abs(s - cum_dist[seg+1]) < 1.5 / (n_nodes - 1):
                    mu[i, seg] = 1.0 / nodes_per_segment

        lam[-1] = 0.0
        u[-1] = u_hover

        return {'t': t, 'x': x, 'u': u, 'lam': lam, 'mu': mu, 'nu': nu}
