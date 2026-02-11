import time
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from RL.env import FishGoalEnv, make_torus_mesh, make_sphere_mesh
from GMR.gmr import GMRGMM
from controllers.spacemouse import SpaceMouse3D

from GMR.utils import select_demos, refresh_wireframes

def compute_terminal_nodes(goal_W, tol=1e-12):
    """
    Compute terminal nodes (i.e. nodes with no outgoing edges or only self-loops) in a graph.

    Parameters
    ----------
    goal_W : (G, G) array
        Weight matrix of the graph
    tol : float, optional
        Tolerance for considering a weight nonzero. Defaults to 1e-12.

    Returns
    -------
    terminal : (G,) array of bool
        Terminal nodes in the graph
    """
    G = goal_W.shape[0]
    terminal = np.zeros(G, dtype=bool)

    # iterate over all nodes
    for i in range(G):
        row = goal_W[i]

        # find indices with outgoing weight
        outgoing = np.where(row > tol)[0]

        if len(outgoing) == 0:
            # no outgoing edges
            terminal[i] = True
        elif len(outgoing) == 1 and outgoing[0] == i:
            # only self-loop
            terminal[i] = True

    return terminal

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # ----------------------------
    # 1) Load policy + generate fish trajectories
    # ----------------------------
    theta_path = "save/best_policy.pkl"
    action = pickle.load(open(theta_path, "rb"))["best_theta"]

    boid_count = 600
    max_steps = 500
    dt = 1
    verts, faces = make_torus_mesh(
                        R=3.0,
                        r=1.0,
                        segR=12,
                        segr=12,
                        center=(20.0, 20.0, 20.0)
                    )
    goals = np.array([
        [10.0, 20.0, 20.0],  # 0 - initial
        [40.0, 20.0, 20.0],  # 1
        [40.0, 30.0, 20.0],  # 2
        [40.0, 10.0, 20.0],  # 3
    ], dtype=np.float32)
    goal_W = np.array([
        [0.0, 1.0, 1.0, 1.0],  # from 0 → {1,2,3}
        [0.0, 1.0, 0.0, 0.0],  # from 1 → 0
        [0.0, 0.0, 1.0, 0.0],  # from 2 → 0
        [0.0, 0.0, 0.0, 1.0],  # from 3 → 0
    ], dtype=np.float32)

    ends_goals = compute_terminal_nodes(goal_W)
    x_start = np.array([6.0, 20.0, 20.0], dtype=np.float32)
    env = FishGoalEnv(boid_count=boid_count, max_steps=max_steps, dt=dt, 
                      doAnimation = False, returnTrajectory = True, verts=verts, faces=faces, goals=goals, goal_W=goal_W, start=x_start)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    boid_pos = np.array(info["trajectory_boid_pos"])  # (T, N, 3)

    # ----------------------------
    # 2) GMR parameters
    # ----------------------------
    n_demos = 15
    time_stride = 10
    n_components = 6
    cov_type = "full"

    # history settings
    history_len = 8         
    update_period = 0.05      
    update_iters = 10         # EM iters per update
    move_eps = 1e-3           # don't update if you didn't move

    # ----------------------------
    # 3) SpaceMouse
    # ----------------------------
    spm = SpaceMouse3D(trans_scale=10.0, deadzone=0.0, lowpass=0.0, rate_hz=200)
    spm.start()
    x = x_start.astype(float).copy()
    history = [x.copy()] 
    path = [x.copy()]     # full path for plotting

    # ----------------------------
    # 4) Warm-start GMR with initial history
    # ----------------------------
    history_points = np.array([x_start]*history_len)
    pos_demos = select_demos(boid_pos, history_points, time_stride=time_stride, space_stride=1, n_demos=15)

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
    tris = [verts[faces[f]].tolist() for f in range(faces.shape[0])]

    mesh_poly = Poly3DCollection(
        tris,
        alpha=0.25,
        linewidths=0.5,
        edgecolor="k",
        facecolor=(0.6, 0.6, 0.6, 0.25),
    )
    ax.add_collection3d(mesh_poly)

    ax.scatter(x_start[0], x_start[1], x_start[2], s=80, label="Start")
    for goal in goals[ends_goals]:
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
            dt = 20

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
                
                pos_demos = select_demos(boid_pos, history, time_stride=time_stride, space_stride=1, n_demos=15)

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

            if np.linalg.norm(buttons) > 0.5: # Restart by pressing a button
                x = x_start.astype(float).copy()
                history = [x.copy()] 
                path = [x.copy()] 

    finally:
        try:
            spm.stop()
        except Exception:
            pass
        plt.ioff()
        plt.show()