import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
try:
    from quadrotor_dynamics import rotation_matrix, f
except ImportError:
    from .quadrotor_dynamics import rotation_matrix, f

def equal_axes(ax, pts):
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    mid = (lo + hi) / 2
    r = max((hi - lo).max() / 2, 1)
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_box_aspect((1, 1, 1))

def euler(q):
    qw, qx, qy, qz = q
    return np.array([
        np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)),
        np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1)),
        np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)),
    ])

def visualize(initial_guess, gates, opt_states, opt_controls, reference):
    X = np.asarray(opt_states)
    U = np.asarray(opt_controls)
    t = np.arange(U.shape[1])
    thrust = np.mean(U * U, axis=0)
    thrust_x = np.r_[thrust, thrust[-1]]
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(6, 3, width_ratios=[2.4, 1, 1])
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ig = np.asarray(initial_guess)
    if ig.ndim == 2:
        ax.plot(ig[0], ig[1], ig[2], "--", color="0.5")
    for w in reference["gate_wireframes"]:
        ax.plot(w[:, 0], w[:, 1], w[:, 2], "k-")
    seg = np.stack([X[0:3, :-1].T, X[0:3, 1:].T], axis=1)
    lc = Line3DCollection(seg, cmap="viridis", linewidth=4)
    lc.set_array(thrust)
    lc.set_clim(0, 1)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.45, pad=0.02, label="collective thrust")
    ax.quiver(
        X[0, :-1:3],
        X[1, :-1:3],
        X[2, :-1:3],
        X[3, :-1:3],
        X[4, :-1:3],
        X[5, :-1:3],
        length=0.15,
        color="tab:green",
    )
    for i in range(0, X.shape[1], max(1, X.shape[1] // 9)):
        R = np.array(rotation_matrix(X[6:10, i]))
        p = X[0:3, i]
        u = thrust_x[i]
        ax.quiver(*p, *R[:, 0], length=0.55, color="r")
        ax.quiver(*p, *R[:, 1], length=0.55, color="g")
        ax.quiver(*p, *R[:, 2], length=0.35, color="b")
        ax.quiver(*p, *-R[:, 2], length=0.4 + 1.2*u, color="m", alpha=0.8)
    ax.set_title("trajectory")
    ax.set_xlabel("forward (+X NED)")
    ax.set_ylabel("right (+Y NED)")
    ax.set_zlabel("down (+Z NED)")
    pts = [X[0:3].T, reference["pos"]]
    pts += [np.asarray(w) for w in reference["gate_wireframes"]]
    equal_axes(ax, np.vstack(pts))
    ax.invert_yaxis()
    ax.invert_zaxis()
    spd = np.linalg.norm(X[3:6], axis=0)
    rpy = np.rad2deg(np.array([euler(X[6:10, i]) for i in range(X.shape[1])]))
    ax0 = fig.add_subplot(gs[0:2, 1])
    ax0.plot(rpy)
    ax0.set_title("attitude deg")
    ax0.legend(["roll", "pitch", "yaw"], fontsize=8)
    ax1 = fig.add_subplot(gs[0:2, 2])
    ax1.plot(t, thrust, "k", lw=2)
    ax1.fill_between(t, 0, thrust, color="0.85")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("collective thrust")
    ax2 = fig.add_subplot(gs[2:4, 1])
    ax2.plot(spd)
    ax2.set_title("speed")
    ax3 = fig.add_subplot(gs[2:4, 2])
    ax3.plot(X[10:13].T)
    ax3.set_title("body rates")
    acc = []
    for k in range(U.shape[1]):
        acc.append(np.array(f(X[:, k], U[:, k])).astype(float).ravel()[3:6])
    ax4 = fig.add_subplot(gs[4:6, 1])
    ax4.plot(np.array(acc))
    ax4.set_title("acceleration")
    ax5 = fig.add_subplot(gs[4:6, 2])
    ax5.plot(t, U.T)
    ax5.set_ylim(-0.05, 1.05)
    ax5.set_title("motor thrust")
    fig.suptitle("MPCC result")
    plt.tight_layout()
    plt.show()
