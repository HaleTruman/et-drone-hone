import time
import numpy as np
try:
    from quadrotor_dynamics import m, g, kf
except ImportError:
    from .quadrotor_dynamics import m, g, kf

def solve_mpcc(opti_pack, x0, reference, warm_start=None):
    opti, v, p, cfg = opti_pack
    N = cfg["N"]
    ids = np.linspace(0, len(reference["theta"]) - 1, N + 1).round().astype(int)
    pref = reference["pos"][ids].T
    tref = reference["tan"][ids].T
    opti.set_value(p["x0"], x0)
    opti.set_value(p["pref"], pref)
    opti.set_value(p["tref"], tref)

    X0 = np.tile(np.asarray(x0, float)[:, None], (1, N + 1))
    X0[0:3, :] = pref
    X0[3:6, :] = np.gradient(pref, cfg["dt"], axis=1)
    X0[6:10, :] = np.asarray(x0, float)[6:10, None]
    if warm_start:
        X0 = warm_start.get("X", X0)
        opti.set_initial(v["U"], warm_start.get("U", (m*g/(4*kf))**0.5))
        opti.set_initial(v["h"], warm_start.get("h", cfg["dt"]))
    else:
        opti.set_initial(v["U"], (m*g/(4*kf))**0.5)
        opti.set_initial(v["h"], cfg["dt"])
    opti.set_initial(v["X"], X0)
    t = time.time()
    status = "failed"
    iters = 0
    try:
        sol = opti.solve()
        X = sol.value(v["X"])
        U = sol.value(v["U"])
        h = float(sol.value(v["h"]))
        cost = sol.value(opti.f)
        stats = sol.stats()
        status = stats.get("return_status", "unknown")
        iters = stats.get("iter_count", 0)
    except Exception:
        X = opti.debug.value(v["X"])
        U = opti.debug.value(v["U"])
        h = float(opti.debug.value(v["h"]))
        cost = opti.debug.value(opti.f)
        stats = opti.stats()
        status = stats.get("return_status", "failed")
        iters = stats.get("iter_count", 0)
    print(f"MPCC solve time: {time.time() - t:.4f} s")
    print(f"Iterations: {iters}")
    print(f"Status: {status}")
    print(f"Final cost: {float(cost):.4f}")
    print(f"Optimized dt: {h:.4f} s")
    print(f"Optimized horizon: {N*h:.4f} s")
    return X, U, 1.0, {"status": status, "iterations": iters, "cost": float(cost), "dt": h, "horizon_s": N*h}
