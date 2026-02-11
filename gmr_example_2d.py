import time
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from RL.env2D import FishGoalEnv2D_DF, get_terminal_goals  # your 2D env :contentReference[oaicite:2]{index=2}
from GMR.gmr import GMRGMM  # your GMR implementation (gmr.py)


# ----------------------------
# Demo selection (self-contained 2D version)
# ----------------------------
def select_demos_2d(boid_pos_TN2, history_points_2d, n_demos=15, time_stride=5):
    """
    boid_pos_TN2: (T, N, 2)
    history_points_2d: (H, 2) array
    Select demos (boid trajectories) whose trajectory gets closest to the last history point.
    Return list of (T', 2) arrays.
    """
    T, N, _ = boid_pos_TN2.shape
    target = np.asarray(history_points_2d[-1], dtype=float).reshape(1, 1, 2)  # (1,1,2)

    # distance of each boid to target over time: (T,N)
    d2 = np.sum((boid_pos_TN2 - target) ** 2, axis=2)
    # best approach per boid
    best = np.min(d2, axis=0)  # (N,)
    idx = np.argsort(best)[: min(n_demos, N)]

    demos = []
    for j in idx:
        traj = boid_pos_TN2[::time_stride, j, :]  # (T',2)
        demos.append(traj.astype(np.float64))
    return demos


# ----------------------------
# 2D covariance ellipses (visual)
# ----------------------------
def add_cov_ellipses(ax, mu_y, Sigma_y, step=20, n_std=1.5, n_points=40, alpha=0.20):
    """
    Draw ellipses for 2D Gaussian covariances along the trajectory.
    Returns a list of Line2D objects (so you can remove them on refresh).
    """
    lines = []
    for i in range(0, mu_y.shape[0], step):
        S = Sigma_y[i]
        # ensure symmetric
        S = 0.5 * (S + S.T)

        # eig
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, 1e-12)

        ang = np.linspace(0, 2 * np.pi, n_points)
        circle = np.stack([np.cos(ang), np.sin(ang)], axis=0)  # (2,P)

        A = V @ np.diag(np.sqrt(w) * n_std)
        ell = (A @ circle).T + mu_y[i]  # (P,2)

        ln, = ax.plot(ell[:, 0], ell[:, 1], "k", lw=1.0, alpha=alpha)
        lines.append(ln)
    return lines


def remove_lines(lines):
    for ln in lines:
        try:
            ln.remove()
        except Exception:
            pass
    lines.clear()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # --- Scenario: same as env2D.py __main__ :contentReference[oaicite:3]{index=3} ---
    start = np.array([20.0, 20.0], dtype=np.float32)
    goals = np.array(
        [
            [20.0, 20.0],
            [4.0, 20.0],
            [36.0, 20.0],
            [4.0, 4.0],
            [20.0, 4.0],
            [36.0, 36.0],
            [20.0, 36.0],
        ],
        dtype=np.float32,
    )

    W = np.array(
        [
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    segs = np.array(
        [
            [0, 25, 32, 25],
            [32, 25, 32, 31],
            [32, 31, 0, 31],
            [40, 15, 8, 15],
            [8, 15, 8, 9],
            [8, 9, 40, 9],
        ],
        dtype=np.float32,
    )

    # --- Load policy (same pattern as your example :contentReference[oaicite:4]{index=4} ) ---
    theta_path = "save/best_policy.pkl"
    try:
        action = pickle.load(open(theta_path, "rb"))["best_theta"]
    except Exception:
        # fallback: reasonable default (won't be optimal)
        action = np.array([1.2, 0.9, 0.8, 1.0, 0.15, 2.0, 0.25], dtype=np.float32)

    # --- Run env once to collect boid trajectories ---
    boid_count = 600
    max_steps = 500
    dt = 0.5

    env = FishGoalEnv2D_DF(
        boid_count=boid_count,
        bound=40.0,
        max_steps=max_steps,
        dt=dt,
        start=start,
        start_spread=2.0,
        goals=goals,
        goal_W=W,
        start_goal_idx=0,
        segs=segs,
        df_origin=np.array([0.0, 0.0], dtype=np.float32),
        df_length=40.0,
        df_R=256,
        avoid_r=2.5,
        avoid_power=1.0,
        avoid_alpha=4.0,
        df_kill_thresh=0.0,
        doAnimation=False,
        returnTrajectory=True,
    )

    env.reset(seed=0)
    _, _, _, _, info = env.step(action)
    boid_pos = np.array(info["trajectory_boid_pos"])  # (T, N, 2)

    # --- GMR parameters (mirrors your example :contentReference[oaicite:5]{index=5} ) ---
    n_demos = 15
    time_stride = 6
    n_components = 6
    cov_type = "full"

    history_len = 8
    update_period = 0.08
    update_iters = 10
    move_eps = 1e-3

    # --- init cursor/history ---
    x = start.astype(float).copy()
    history = [x.copy()]
    path = [x.copy()]

    history_points = np.array([start] * history_len, dtype=float)
    pos_demos = select_demos_2d(boid_pos, history_points, n_demos=n_demos, time_stride=time_stride)

    gmr = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
    gmr.fit(pos_demos)
    mu_y, Sigma_y, gamma, loglik = gmr.regress(T=max_steps, pos_dim=2)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # walls
    for m in range(segs.shape[0]):
        x1, y1, x2, y2 = segs[m]
        ax.plot([x1, x2], [y1, y2], "k", lw=2.0)

    # terminal goals
    terminal_goals, terminal_idx = get_terminal_goals(goals, W)
    ax.scatter(terminal_goals[:, 0], terminal_goals[:, 1], s=160, marker="*", c="r", label="Terminal goals")
    ax.scatter(start[0], start[1], s=80, c="k", label="Start")

    # demo lines
    demo_lines = []
    for _ in range(n_demos):
        ln, = ax.plot([], [], "k--", lw=1.0, alpha=0.6)
        demo_lines.append(ln)

    def set_demo_lines(demos):
        for i, ln in enumerate(demo_lines):
            if i < len(demos):
                d = demos[i]
                ln.set_data(d[:, 0], d[:, 1])
                ln.set_visible(True)
            else:
                ln.set_visible(False)

    set_demo_lines(pos_demos)

    # model mean + cov
    mu_line, = ax.plot(mu_y[:, 0], mu_y[:, 1], "k", lw=2.5, label="GMR mean")
    cov_lines = add_cov_ellipses(ax, mu_y, Sigma_y, step=25, n_std=1.5, alpha=0.18)

    # cursor + path + history
    cursor_sc = ax.scatter([x[0]], [x[1]], s=90, c="b", label="Cursor")
    path_ln, = ax.plot([x[0]], [x[1]], lw=2.0, c="b", alpha=0.8)
    hist_sc = ax.scatter([x[0]], [x[1]], s=25, c="g", alpha=0.8, label="History")

    ax.legend(loc="upper right")

    # --- mouse drag control ---
    dragging = {"on": False}

    def on_press(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # left
            dragging["on"] = True

    def on_release(event):
        if event.button == 1:
            dragging["on"] = False

    def on_move(event):
        if not dragging["on"]:
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        nonlocal_x[0] = float(event.xdata)
        nonlocal_x[1] = float(event.ydata)

    nonlocal_x = x  # alias so callback can write

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    last_update_t = time.time()
    last_x_for_update = x.copy()

    plt.ion()
    while plt.fignum_exists(fig.number):
        now = time.time()

        # integrate "cursor"
        x = nonlocal_x

        # update history/path
        if np.linalg.norm(x - history[-1]) > 1e-6:
            history.append(x.copy())
            if len(history) > history_len:
                history = history[-history_len:]
            path.append(x.copy())

        cursor_sc.set_offsets(np.array([x]))
        P = np.array(path)
        path_ln.set_data(P[:, 0], P[:, 1])

        H = np.array(history)
        hist_sc.set_offsets(H)

        # throttled GMR update
        if (now - last_update_t) >= update_period and np.linalg.norm(x - last_x_for_update) > move_eps:
            last_update_t = now
            last_x_for_update = x.copy()

            pos_demos = select_demos_2d(boid_pos, np.array(history), n_demos=n_demos, time_stride=time_stride)

            gmrUpdate = copy.deepcopy(gmr)
            gmrUpdate.update(pos_demos, n_iter=update_iters)
            mu_y, Sigma_y, gamma, loglik = gmrUpdate.regress(T=max_steps, pos_dim=2)

            set_demo_lines(pos_demos)
            mu_line.set_data(mu_y[:, 0], mu_y[:, 1])

            remove_lines(cov_lines)
            cov_lines[:] = add_cov_ellipses(ax, mu_y, Sigma_y, step=25, n_std=1.5, alpha=0.18)

        plt.pause(0.01)

    plt.ioff()
    plt.show()
