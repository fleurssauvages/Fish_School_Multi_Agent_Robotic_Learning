import time
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from RL.env import FishGoalEnv
from GMR.gmr import GMRGMM
from controllers.spacemouse import SpaceMouse3D

from GMR.utils import select_demos, refresh_wireframes

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # ----------------------------
    # 1) Load policy + generate fish trajectories (same idea as your script)
    # ----------------------------
    theta_path = "save/best_policy.pkl"
    action = pickle.load(open(theta_path, "rb"))["best_theta"]

    boid_count = 300
    max_steps = 200

    obs_centers = np.array([
        [24.0, 18.0, 16.0],
        [28.0, 24.0, 16.0],
        [20.0, 30.0, 16.0],
        [18.0, 18.0, 12.0],
        [28.0, 24.0, 30.0],
        [24.0, 30.0, 16.0],
    ], dtype=np.float32)

    goals = [
        np.array([36.0, 12.0, 16.0], dtype=np.float32),
        np.array([36.0, 24.0, 16.0], dtype=np.float32),
        np.array([36.0, 36.0, 16.0], dtype=np.float32),
    ]

    x_start = np.array([12.0, 24.0, 16.0], dtype=np.float32)

    boids_pos_list = []
    for gi, g in enumerate(goals):
        env = FishGoalEnv(
            boid_count=boid_count,
            pred_count=0,
            max_steps=max_steps,
            dt=1,
            doAnimation=False,
            returnTrajectory=True,
            obs_centers=obs_centers,
            goal=g,
            start=x_start,
        )
        env.reset(seed=gi)
        env.goal = g
        obs, reward, terminated, truncated, info = env.step(action)
        boids_pos_list.append(info["trajectory_boid_pos"])  # (T, N, 3)

    # stack along boid dimension -> (T, 3N, 3)
    boids_pos = np.concatenate(boids_pos_list, axis=1)

    # ----------------------------
    # 2) GMR parameters
    # ----------------------------
    n_demos = 15
    time_stride = 10
    n_components = 6
    cov_type = "full"

    # history settings
    history_len = 8          # how many last positions you condition on
    update_period = 0.05      # seconds between GMR updates (throttle!)
    update_iters = 10         # EM iters per update
    move_eps = 1e-3           # don't update if you didn't move

    # ----------------------------
    # 3) SpaceMouse
    # ----------------------------
    spm = SpaceMouse3D(trans_scale=10.0, deadzone=0.0, lowpass=0.0, rate_hz=200)
    spm.start()

    # use SpaceMouse-integrated cursor as "live via"
    x = x_start.astype(float).copy()
    history = [x.copy()]  # oldest->newest
    path = [x.copy()]     # full path for plotting

    # ----------------------------
    # 4) Warm-start GMR with initial history
    # ----------------------------
    history_points = np.array([x_start]*history_len)
    pos_demos = select_demos(boids_pos, history_points, time_stride=time_stride, space_stride=1, n_demos=15)

    gmr = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
    gmr.fit(pos_demos)
    mu_y, Sigma_y, gamma, loglik = gmr.regress(T=max_steps, pos_dim=3)

    # ----------------------------
    # 5) Plot setup
    # ----------------------------
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # obstacles
    rad = env.obs_radius
    for c in env.obs_centers:
        u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:18j]
        xs = c[0] + rad * np.cos(u) * np.sin(v)
        ys = c[1] + rad * np.sin(u) * np.sin(v)
        zs = c[2] + rad * np.cos(v)
        ax.plot_surface(xs, ys, zs, alpha=0.20, linewidth=0)

    ax.scatter(x_start[0], x_start[1], x_start[2], s=80, label="Start")
    for goal in goals:
        ax.scatter(goal[0], goal[1], goal[2], s=80, label="Goal")

    # demos + model
    demo_lines = [ax.plot([], [], [], "k--", lw=1.0, alpha=0.7)[0] for _ in range(n_demos)]
    mu_line = ax.plot(mu_y[:, 0], mu_y[:, 1], mu_y[:, 2], "k", lw=2.0)[0]

    wireframes = []
    refresh_wireframes(ax, wireframes, mu_y, Sigma_y, step=30, n_std=1.5, n_points=18, alpha=0.25)

    # live cursor + history path (avoid orange)
    cursor_scatter = ax.scatter(x[0], x[1], x[2], s=70)      # default color
    path_line = ax.plot([x[0]], [x[1]], [x[2]], lw=2.0)[0]
    hist_scatter = ax.scatter([x[0]], [x[1]], [x[2]], s=12, alpha=0.8)

    ax.legend()

    # keep bounds reasonable
    c0 = (x_start + goals[1]) / 2
    span = np.linalg.norm(goals[1] - x_start) / 2 + 2.0
    ax.set_xlim(c0[0] - span, c0[0] + span)
    ax.set_ylim(c0[1] - span, c0[1] + span)
    ax.set_zlim(c0[2] - span, c0[2] + span)
    ax.view_init(elev=40, azim=-180, roll=0)

    def set_demo_lines(demos):
        for i, ln in enumerate(demo_lines):
            if i < len(demos):
                d = demos[i]
                ln.set_data(d[:, 0], d[:, 1])
                ln.set_3d_properties(d[:, 2])
                ln.set_visible(True)
            else:
                ln.set_visible(False)

    set_demo_lines(pos_demos)

    # ----------------------------
    # 6) Live loop
    # ----------------------------
    last_update_t = time.time()
    last_t = time.time()
    last_x_for_update = x.copy()

    try:
        plt.ion()
        while plt.fignum_exists(fig.number):
            now = time.time()
            dt = max(1e-4, now - last_t)

            trans, rot, buttons = spm.read() 
            v = np.array(trans, dtype=float)
            x = x + v * dt

            if np.linalg.norm(x - history[-1]) > 1e-6:
                history.append(x.copy())
                if len(history) > history_len:
                    history = history[-history_len:]
                path.append(x.copy())

            cursor_scatter._offsets3d = ([x[0]], [x[1]], [x[2]])

            P = np.array(path)
            path_line.set_data(P[:, 0], P[:, 1])
            path_line.set_3d_properties(P[:, 2])

            H = np.array(history)
            hist_scatter._offsets3d = (H[:, 0], H[:, 1], H[:, 2])

            # throttled GMR update
            if (now - last_update_t) >= update_period and np.linalg.norm(x - last_x_for_update) > move_eps:
                last_update_t = now
                last_x_for_update = x.copy()
                
                pos_demos = select_demos(boids_pos, history, time_stride=time_stride, space_stride=1, n_demos=15)

                # update + regress
                gmrUpdate = copy.deepcopy(gmr)
                gmrUpdate.update(pos_demos, n_iter=update_iters)
                mu_y, Sigma_y, gamma, loglik = gmrUpdate.regress(T=max_steps, pos_dim=3)

                # redraw model
                set_demo_lines(pos_demos)
                mu_line.set_data(mu_y[:, 0], mu_y[:, 1])
                mu_line.set_3d_properties(mu_y[:, 2])
                refresh_wireframes(ax, wireframes, mu_y, Sigma_y, step=30, n_std=1.5, n_points=18, alpha=0.25)

            plt.pause(0.001)

            if np.linalg.norm(buttons) > 0.5:
                x = x_start.astype(float).copy()
                history = [x.copy()]  # oldest->newest
                path = [x.copy()]     # full path for plotting

    finally:
        try:
            spm.stop()
        except Exception:
            pass
        plt.ioff()
        plt.show()