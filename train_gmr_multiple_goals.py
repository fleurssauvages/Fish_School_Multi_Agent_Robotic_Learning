import time
from matplotlib.widgets import Slider
import numpy as np
from RL.env import FishGoalEnv
import pickle
import numpy as np
import copy

import matplotlib.pyplot as plt
from RL.env import FishGoalEnv

from GMR.gmr import GMRGMM

def refresh_wireframes(ax, wireframes, mu_y, Sigma_y, step=30, n_std=1.5, n_points=20, alpha=0.08):
    for wf in wireframes:
        try:
            wf.remove()
        except Exception:
            pass
    wireframes.clear()

    # redraw new (fewer + cheaper)
    for t in range(0, len(mu_y), step):
        X, Y, Z = gaussian_ellipsoid(mu_y[t], Sigma_y[t], n_std=n_std, n_points=n_points)
        wf = ax.plot_wireframe(X, Y, Z, alpha=alpha)
        wireframes.append(wf)

def gaussian_ellipsoid(mean, cov, n_std=1.5, n_points=30):
    """
    Returns X,Y,Z points of a 3D covariance ellipsoid.
    """
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Radii
    radii = n_std * np.sqrt(np.maximum(eigvals, 0))

    # Sphere
    u = np.linspace(0, 2*np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Transform
    sphere = np.stack([x, y, z], axis=-1)
    ellipsoid = sphere @ np.diag(radii) @ eigvecs.T
    ellipsoid += mean

    return ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]

def plot_gmr_uncertainty_3d(mu_y, Sigma_y, step=10, n_std=1.5, ax=None):
    wireframes = []
    for t in range(0, len(mu_y), step):
        X, Y, Z = gaussian_ellipsoid(mu_y[t], Sigma_y[t], n_std=n_std)
        wf = ax.plot_wireframe(X, Y, Z, alpha=0.1)
        wireframes.append(wf)
    return wireframes

def select_demos_near_via_anytime(pos_demos, via_point, k=10, stride=1):
    pos_demos = np.asarray(pos_demos, float)

    # distances: (N,T)
    d_t = np.linalg.norm(pos_demos - via_point, axis=2)
    d = d_t.min(axis=0)
    idx = np.argsort(d)[:k*stride:stride]
    return idx, d[idx]

def select_demos_near_via(boids_pos, via_point, n_demos=3, space_stride=5, time_stride=10):
    idx, _ = select_demos_near_via_anytime(boids_pos, via_point, k=n_demos, stride=space_stride)
    pos_demos = []
    for i in idx:
        demo = boids_pos[:, i, :]
        valid = (np.linalg.norm(demo, axis=1) > 1e-3)
        demo = demo[valid, :]
        demo = demo[::time_stride, :]
        pos_demos.append(demo)
    return pos_demos

if __name__ == "__main__":
    # ============================================================
    # Compute Fish trajectories
    # ============================================================
    """ Generate Fish trajectories"""
    load_theta = True
    if load_theta:
        theta_path = "save/best_policy.pkl"
        action = pickle.load(open(theta_path, "rb"))['best_theta']

    boid_count = 300
    max_steps = 200
    obs_centers = np.array([
        # corridor-ish cluster
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

    boids_pos_list = []
    boids_vel_list = []
    x_start = np.array([12.0, 24.0, 16.0], dtype=np.float32)

    t0 = time.time()
    for gi, g in enumerate(goals):
        env = FishGoalEnv(
            boid_count=boid_count,
            pred_count=0,
            max_steps=max_steps,
            dt=1,
            doAnimation=False,
            returnTrajectory=True,
            obs_centers=obs_centers,
            goal = g,
            start = x_start
        )
        env.reset(seed=gi)  # different seed per goal

        # If FishGoalEnv lets you overwrite the goal (most do):
        env.goal = g

        # step once with the policy (your original pattern)
        obs, reward, terminated, truncated, info = env.step(action)

        # store
        boids_pos = info['trajectory_boid_pos']  # (T, N, 3)
        boids_vel = info['trajectory_boid_vel']  # (T, N, 3)

        boids_pos_list.append(boids_pos)
        boids_vel_list.append(boids_vel)

        if x_start is None:
            x_start = env.start.copy()

    t1 = time.time()
    print(f"Fish 3x Simulation Time: {(t1 - t0)*1000.0:.2f} ms")

    # stack along boid dimension -> (T, 3N, 3)
    boids_pos = np.concatenate(boids_pos_list, axis=1)
    boids_vel = np.concatenate(boids_vel_list, axis=1)
    
    angle = 0.0
    r = 10.0
    x_via_center = np.array([24.0, 24.0, 16.0])
    x_via = x_via_center + np.array([0.0, r * np.cos(angle), r * np.sin(angle)])

    # ============================================================
    # Fit an GMR model
    # ============================================================
    """ Parameters"""
    n_demos = 15
    space_stride = 1
    time_stride = 10
    n_components = 6
    cov_type = "full"  # "diag" or "full"

    pos_demos = select_demos_near_via(boids_pos, x_via, n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
    
    """ Fit an GMR-GMM model to the demonstrations """
    gmrInit = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
    t0 = time.time()
    gmrInit.fit(pos_demos)
    t1 = time.time()
    print(f"GMR-GMM Warmup Time: {(t1 - t0)*1000.0:.2f} ms")

    """ Update with new via point """
    t0 = time.time()
    pos_demos = select_demos_near_via(boids_pos, x_via, n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
    gmrUpdated = copy.deepcopy(gmrInit)
    gmrUpdated.update(pos_demos, n_iter=15)
    t1 = time.time()
    print(f"GMR-GMM Update Time: {(t1 - t0)*1000.0:.2f} ms")

    """ Perform regression to get mean and covariance of trajectory """
    t0 = time.time()
    mu_y, Sigma_y, gamma, loglik = gmrUpdated.regress(T=max_steps, pos_dim=3)
    t1 = time.time()
    print(f"Model Regression: {(t1 - t0)*1000.0:.2f} ms")

    # ============================================================
    # Plot Results
    # ============================================================

    fig = plt.figure(figsize=(22, 14))
    ax_traj = fig.add_subplot(111, projection='3d')
    i = 0
    demo_lines = [ax_traj.plot([], [], [], 'k--', lw=1.0)[0] for _ in range(n_demos)]

    def update_demo_lines(pos_demos):
        global demo_lines
        for i, ln in enumerate(demo_lines):
            if i < len(pos_demos):
                d = pos_demos[i]
                ln.set_data(d[:,0], d[:,1])
                ln.set_3d_properties(d[:,2])
                ln.set_visible(True)
            else:
                ln.set_visible(False)

    update_demo_lines(pos_demos)
    mu_line, = ax_traj.plot(mu_y[:,0], mu_y[:,1], mu_y[:,2], 'k', lw=2)
    mu_line.set_data(mu_y[:,0], mu_y[:,1])
    mu_line.set_3d_properties(mu_y[:,2])

    wireframes = plot_gmr_uncertainty_3d(mu_y, Sigma_y, step=10, n_std=1.5, ax=ax_traj)
    refresh_wireframes(ax_traj, wireframes, mu_y, Sigma_y,
                   step=30, n_std=1.5, n_points=20, alpha=0.3)

    rad = env.obs_radius
    for c in env.obs_centers:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        x = c[0] + rad * np.cos(u) * np.sin(v)
        y = c[1] + rad * np.sin(u) * np.sin(v)
        z = c[2] + rad * np.cos(v)
        ax_traj.plot_surface(x, y, z, color='red', alpha=0.25, linewidth=0)

    ax_traj.scatter(x_start[0], x_start[1], x_start[2], c='blue', s=100, label='Start')
    for goal in goals:
        ax_traj.scatter(goal[0], goal[1], goal[2], c='green', s=100, label='Goal')
    via_scatter = ax_traj.scatter(x_via[0], x_via[1], x_via[2], c='orange', s=100, label='Via Point')

    c = (x_start + goals[1]) / 2
    span = np.linalg.norm(goals[1] - x_start) / 2 + 1.0
    ax_traj.set_xlim(c[0] - span, c[0] + span)
    ax_traj.set_ylim(c[1] - span, c[1] + span)
    ax_traj.set_zlim(c[2] - span, c[2] + span)

    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_zlabel("z")
    plt.legend()
    
    def _on_via_xyz_change(val):
        global demos_plot, mu, wireframes, via_scatter

        # read sliders
        x_via = np.array([x_slider.val, y_slider.val, z_slider.val], dtype=float)

        # move the via point marker
        via_scatter._offsets3d = (
            np.array([x_via[0]]),
            np.array([x_via[1]]),
            np.array([x_via[2]])
        )

        # reselect demos near new via point
        pos_demos = select_demos_near_via(
            boids_pos, x_via,
            n_demos=n_demos,
            space_stride=space_stride,
            time_stride=time_stride
        )

        # update a fresh copy of the initial model
        gmrUpdated = copy.deepcopy(gmrInit)
        gmrUpdated.update(pos_demos, n_iter=15)

        # update demo lines + GMR mean + uncertainty
        update_demo_lines(pos_demos)
        mu_y, Sigma_y, gamma, loglik = gmrUpdated.regress(T=max_steps, pos_dim=3)

        mu_line.set_data(mu_y[:, 0], mu_y[:, 1])
        mu_line.set_3d_properties(mu_y[:, 2])

        refresh_wireframes(
            ax_traj, wireframes, mu_y, Sigma_y,
            step=30, n_std=1.5, n_points=20, alpha=0.3
        )

        fig.canvas.draw_idle()


    # Choose sensible slider bounds from your data (recommended)
    # Here: bounds from trajectory extents so the via point stays in the scene.
    mins = np.array([12, 12, 12])
    maxs = np.array([36, 36, 24])

    # initial value: your current x_via_center (or any point you want)
    x_via0 = np.array(x_via_center, dtype=float)

    # Layout: 3 sliders stacked
    x_slider = Slider(
        ax=fig.add_axes([0.25, 0.02, 0.15, 0.03]),
        label='Via X',
        valmin=float(mins[0]), valmax=float(maxs[0]),
        valinit=float(x_via0[0]),
    )
    y_slider = Slider(
        ax=fig.add_axes([0.45, 0.02, 0.15, 0.03]),
        label='Via Y',
        valmin=float(mins[1]), valmax=float(maxs[1]),
        valinit=float(x_via0[1]),
    )
    z_slider = Slider(
        ax=fig.add_axes([0.65, 0.02, 0.15, 0.03]),
        label='Via Z',
        valmin=float(mins[2]), valmax=float(maxs[2]),
        valinit=float(x_via0[2]),
    )

    # Attach the same callback to all three
    x_slider.on_changed(_on_via_xyz_change)
    y_slider.on_changed(_on_via_xyz_change)
    z_slider.on_changed(_on_via_xyz_change)

    plt.show()