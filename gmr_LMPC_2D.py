import time
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from RL.env2D import FishGoalEnv2D_DF  
from GMR.gmr import GMRGMM           
from MPC.LMPC_solver_obs import LinearMPCController
import spatialmath as sm
from controllers.spacemouse import SpaceMouse3D

# ----------------------------
# Terminal goals (same logic as your 3D file) :contentReference[oaicite:4]{index=4}
# ----------------------------
def compute_terminal_nodes(goal_W, tol=1e-12):
    G = goal_W.shape[0]
    terminal = np.zeros(G, dtype=bool)
    for i in range(G):
        row = goal_W[i]
        outgoing = np.where(row > tol)[0]
        if len(outgoing) == 0:
            terminal[i] = True
        elif len(outgoing) == 1 and outgoing[0] == i:
            terminal[i] = True
    return terminal

def segs_to_wall_mesh(segs, height=10_000.0, thickness=0.05, z0=0.0, cap=True):
    """
    segs: (N,4) with [x1,y1,x2,y2]
    Returns:
      verts: (M,3)
      faces: (K,3) int32 triangle indices
    """
    segs = np.asarray(segs, dtype=np.float32)
    verts_all = []
    faces_all = []
    v_off = 0

    for x1, y1, x2, y2 in segs:
        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)
        d = p2 - p1
        L = float(np.linalg.norm(d))
        if L < 1e-8:
            continue

        # unit normal to the segment (2D)
        n = np.array([-d[1], d[0]], dtype=np.float32) / L
        o = 0.5 * thickness * n

        # bottom rectangle corners (counter-clockwise around the segment)
        b0 = p1 + o
        b1 = p2 + o
        b2 = p2 - o
        b3 = p1 - o

        z1 = z0 + float(height)

        # 8 vertices: bottom then top
        v = np.array([
            [b0[0], b0[1], z0],
            [b1[0], b1[1], z0],
            [b2[0], b2[1], z0],
            [b3[0], b3[1], z0],
            [b0[0], b0[1], z1],
            [b1[0], b1[1], z1],
            [b2[0], b2[1], z1],
            [b3[0], b3[1], z1],
        ], dtype=np.float32)

        # triangles (two per quad face)
        # side faces:
        f = [
            [0, 1, 5], [0, 5, 4],  # +o side
            [1, 2, 6], [1, 6, 5],  # end at p2
            [2, 3, 7], [2, 7, 6],  # -o side
            [3, 0, 4], [3, 4, 7],  # end at p1
        ]

        if cap:
            # bottom cap (z0) and top cap (z1)
            f += [
                [0, 2, 1], [0, 3, 2],  # bottom
                [4, 5, 6], [4, 6, 7],  # top
            ]

        f = np.array(f, dtype=np.int32) + v_off

        verts_all.append(v)
        faces_all.append(f)
        v_off += v.shape[0]

    if not verts_all:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    verts = np.vstack(verts_all)
    faces = np.vstack(faces_all)
    return verts, faces

# ----------------------------
# Alpha gate (same as your 3D file, dimension-agnostic) :contentReference[oaicite:5]{index=5}
# ----------------------------
def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def compute_alpha(
    u_h: np.ndarray,
    v_ref: np.ndarray,
    Sigma,
    *,
    alpha_max: float = 1.0,
    c0: float = 0.7,
    k_a: float = 10.0,
    u_deadzone: float = 1e-3,
    s0: float = 10.0,
    k_s: float = 10.0,
) -> float:
    u_h = np.asarray(u_h, dtype=float).reshape(-1)
    v_ref = np.asarray(v_ref, dtype=float).reshape(-1)

    if np.linalg.norm(u_h) < u_deadzone or np.linalg.norm(v_ref) < 1e-12:
        alpha_star = 0.0
    else:
        c = float(u_h @ v_ref) / (float(np.linalg.norm(u_h) * np.linalg.norm(v_ref)) + 1e-12)
        c = max(-1.0, min(1.0, c))
        g_align = _sigmoid(k_a * (c - c0))

        if np.isscalar(Sigma):
            s = float(Sigma)
        else:
            S = np.asarray(Sigma, dtype=float)
            s = float(np.sqrt(np.trace(S)))

        _ = _sigmoid(k_s * (s0 - s))  # computed but not used in your original file; keep behavior
        alpha_star = alpha_max * g_align

    return float(max(0.0, min(alpha_max, alpha_star)))


# ----------------------------
def point_segment_closest(p, a, b):
    ab = b - a
    ap = p - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return a.copy(), float(np.linalg.norm(p - a))
    t = float((ap @ ab) / denom)
    t = max(0.0, min(1.0, t))
    c = a + t * ab
    return c, float(np.linalg.norm(p - c))

def repel_from_segments(p, segs, margin=1.0, gain=2.0):
    """
    Smooth repulsion from nearest segment if within 'margin'.
    Returns a velocity correction vector.
    """
    best_d = 1e9
    best_c = None
    for m in range(segs.shape[0]):
        a = segs[m, 0:2]
        b = segs[m, 2:4]
        c, d = point_segment_closest(p, a, b)
        if d < best_d:
            best_d = d
            best_c = c
    if best_c is None or best_d >= margin:
        return np.zeros(2)
    # push away from closest point
    dirv = p - best_c
    n = np.linalg.norm(dirv) + 1e-12
    dirv = dirv / n
    # stronger when closer
    mag = gain * (margin - best_d) / margin
    return dirv * mag


# ----------------------------
# Demo selection for 2D (same idea as before)
# ----------------------------
def select_demos_2d(boid_pos_TN2, history_points_2d, n_demos=15, time_stride=5):
    T, N, _ = boid_pos_TN2.shape
    target = np.asarray(history_points_2d[-1], dtype=float).reshape(1, 1, 2)
    d2 = np.sum((boid_pos_TN2 - target) ** 2, axis=2)  # (T,N)
    best = np.min(d2, axis=0)                          # (N,)
    idx = np.argsort(best)[: min(n_demos, N)]
    demos = []
    for j in idx:
        traj = boid_pos_TN2[::time_stride, j, :]
        demos.append(traj.astype(np.float64))
    return demos


# ----------------------------
# Main (2D)
# ----------------------------
if __name__ == "__main__":
    # --- Same scenario as env2D.py 
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
    obs_list = segs_to_wall_mesh(segs, height=10.0, thickness=0.05, z0=-5)
    obs_list = [obs_list]

    ends_goals = compute_terminal_nodes(W)

    # --- Load policy (optional)
    theta_path = "save/best_policy.pkl"
    try:
        action = pickle.load(open(theta_path, "rb"))["best_theta"]
    except Exception:
        action = np.array([1.2, 0.9, 0.8, 1.0, 0.15, 2.0, 0.25], dtype=np.float32)

    # --- Generate fish trajectories (for demos)
    boid_count = 600
    max_steps = 500
    dt_env = 0.5

    env = FishGoalEnv2D_DF(
        boid_count=boid_count,
        bound=40.0,
        max_steps=max_steps,
        dt=dt_env,
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
    boid_pos = np.array(info["trajectory_boid_pos"])  # (T,N,2)

    # --- GMR params
    n_demos = 15
    time_stride = 6
    n_components = 6
    cov_type = "full"

    history_len = 8
    update_period = 0.06
    update_iters = 10
    move_eps = 1e-3

    # --- Control params (2D)
    dt = 0.10                 # control loop dt (seconds)
    speed_limit = 2.0         # units/sec
    wall_margin = 1.2
    wall_gain = 3.0

    # --- Cursor + history
    x = start.astype(float).copy()
    history = [x.copy()]
    path = [x.copy()]
    history_points = np.array([start] * history_len, dtype=float)

    # --- Warm start GMR
    pos_demos = select_demos_2d(boid_pos, history_points, n_demos=n_demos, time_stride=time_stride)
    gmr = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
    gmr.fit(pos_demos)
    mu_y, Sigma_y, gamma, loglik = gmr.regress(T=max_steps, pos_dim=2)

    # --- Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # walls
    for m in range(segs.shape[0]):
        x1, y1, x2, y2 = segs[m]
        ax.plot([x1, x2], [y1, y2], "k", lw=2.0)

    # goals
    ax.scatter(start[0], start[1], s=90, c="k", label="Start")
    ax.scatter(goals[ends_goals, 0], goals[ends_goals, 1], s=160, marker="*", c="r", label="Terminal goals")

    # demos
    # demo_lines = [ax.plot([], [], "k--", lw=1.0, alpha=0.5)[0] for _ in range(n_demos)]
    def set_demo_lines(demos):
        for i, ln in enumerate(demo_lines):
            if i < len(demos):
                d = demos[i]
                ln.set_data(d[:, 0], d[:, 1])
                ln.set_visible(True)
            else:
                ln.set_visible(False)
    # set_demo_lines(pos_demos)

    # mean
    # mu_line = ax.plot(mu_y[:, 0], mu_y[:, 1], "k", lw=2.2, label="GMR mean")[0]

    # cursor, history, path
    cursor_sc = ax.scatter([x[0]], [x[1]], s=90, c="b", label="Cursor")
    path_ln = ax.plot([x[0]], [x[1]], lw=2.0, c="b", alpha=0.8)[0]
    hist_sc = ax.scatter([x[0]], [x[1]], s=18, c="g", alpha=0.8, label="History")

    last_update_t = time.time()
    last_t = time.time()
    last_x_for_update = x.copy()
    alpha = 0.0
    dt = 10

    spm = SpaceMouse3D(trans_scale=10.0, deadzone=0.0, lowpass=0.0, rate_hz=200)
    spm.start()

    speed_limit = 10
    lmpc_solver = LinearMPCController(horizon=10, dt=dt, gamma = 0.05,
                                    u_min=np.array([-speed_limit, -speed_limit, -speed_limit, -speed_limit, -speed_limit, -speed_limit]),
                                    u_max=np.array([ speed_limit,  speed_limit,  speed_limit,  speed_limit,  speed_limit,  speed_limit]))

    def interpolate_traj(p0, p1, n):
        p0 = np.asarray(p0)
        p1 = np.asarray(p1)

        t = np.linspace(0.0, 1.0, n)[:, None]   # shape (n,1)
        traj = p0 + t * (p1 - p0)

        return traj

    plt.ion()
    while plt.fignum_exists(fig.number):
        now = time.time()
        trans, rot, buttons = spm.read()
        trans = [-trans[1], trans[0], trans[2]] 
        v = np.array(trans, dtype=float)

        # Compute autonomy
        d = np.linalg.norm(mu_y - x[None, :], axis=1)
        tidx = int(np.argmin(d))
        if tidx < max_steps -lmpc_solver.horizon:
            v_ref = (mu_y[tidx + 1] - mu_y[tidx]) / dt
            p0 = mu_y[tidx]
            p1 = mu_y[tidx + lmpc_solver.horizon]
            p0 = np.array([p0[0], p0[1], 0.0])
            p1 = np.array([p1[0], p1[1], 0.0])
            traj = interpolate_traj(p0, p1, lmpc_solver.horizon)
        else:
            v_ref = np.zeros(3)
            traj = None
        if np.linalg.norm(v_ref) > 1e-3:
            v_ref *= np.linalg.norm(v) / np.linalg.norm(v_ref)

        x = np.array([x[0], x[1], 0.0])
        v = np.array([v[0], v[1], 0.0])
        v_ref = np.array([v_ref[0], v_ref[1], 0.0])

        sigma = Sigma_y[tidx]
        alpha = compute_alpha(v, v_ref, sigma)

        T_i = sm.SE3.Trans(x)
        T_des_human = sm.SE3.Trans(x + v * dt)
        T_des_GMR = sm.SE3.Trans(x + v_ref * dt)

        print(obs_list)
        Uopt, Xopt, poses = lmpc_solver.solve(T_i, T_des_human, T_des_GMR, 1 - alpha, obstacles=obs_list, traj=traj, margin=0.5)
        
        if Uopt is None:
            vopt = np.zeros(2)
        else:
            vopt = Uopt[0:3]
        if np.linalg.norm(vopt) > 1e-3:
            vopt *= np.linalg.norm(v) / np.linalg.norm(vopt)

        x = x + vopt * dt
        x = np.array([x[0], x[1]])

        # record history/path
        if np.linalg.norm(x - history[-1]) > 1e-6:
            history.append(x.copy())
            if len(history) > history_len:
                history = history[-history_len:]
            path.append(x.copy())

        # update plot
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

            # set_demo_lines(pos_demos)
            # mu_line.set_data(mu_y[:, 0], mu_y[:, 1])

        plt.pause(0.001)

        if np.linalg.norm(buttons) > 0.5: # Restart by pressing a button
            x = start.astype(float).copy()
            history = [x.copy()] 
            path = [x.copy()] 

    plt.ioff()
    plt.show()
