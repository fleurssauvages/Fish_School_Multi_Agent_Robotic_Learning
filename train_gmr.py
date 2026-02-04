import time
from matplotlib.widgets import Slider
import numpy as np
from RL.env import FishGoalEnv
import pickle
import numpy as np

import matplotlib.pyplot as plt
from RL.env import FishGoalEnv

from GMR.gmr import GMRGMM
from GMR.utils import plot_gmr_uncertainty_3d, select_demos, refresh_wireframes

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
    obs_centers =  np.array([[20.0, 20.0, 20.0],
                            [28.0, 16.0, 22.0],
                            [28.0, 24.0, 18.0]], dtype=np.float32)
    t0 = time.time()
    env = FishGoalEnv(boid_count=boid_count, pred_count=0, max_steps=max_steps, dt=1, doAnimation = False, returnTrajectory = True, obs_centers=obs_centers)
    env.reset(seed=0)
    t1 = time.time()
    print(f"Fish Warmup Time: {(t1 - t0)*1000.0:.2f} ms")

    t0 = time.time()
    obs, reward, terminated, truncated, info = env.step(action)
    env.reset(seed=0)
    t1 = time.time()
    print(f"Fish Simulation Time: {(t1 - t0)*1000.0:.2f} ms")

    """Select some demos near via point"""
    # Extract trajectory from boids
    t0 = time.time()
    boids_pos = info['trajectory_boid_pos']  # (T, N, 3)
    boids_vel = info['trajectory_boid_vel']  # (T, N, 3)
    x_start = env.start
    x_goal = env.goal
    
    angle = 0.0
    r = 5.0
    x_via_center = np.array([20.0, 20.0, 20.0])
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

    pos_demos = select_demos(boids_pos, [x_via], n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
    
    """ Fit an GMR-GMM model to the demonstrations """
    gmr = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
    t0 = time.time()
    gmr.fit(pos_demos)
    t1 = time.time()
    print(f"GMR-GMM Warmup Time: {(t1 - t0)*1000.0:.2f} ms")

    """ Update with new via point """
    t0 = time.time()
    pos_demos = select_demos(boids_pos, [x_via], n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
    gmr.update(pos_demos, n_iter=15)
    t1 = time.time()
    print(f"GMR-GMM Update Time: {(t1 - t0)*1000.0:.2f} ms")

    """ Perform regression to get mean and covariance of trajectory """
    t0 = time.time()
    mu_y, Sigma_y, gamma, loglik = gmr.regress(T=max_steps, pos_dim=3)
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

    r = env.obs_radius
    for c in env.obs_centers:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        x = c[0] + r * np.cos(u) * np.sin(v)
        y = c[1] + r * np.sin(u) * np.sin(v)
        z = c[2] + r * np.cos(v)
        ax_traj.plot_surface(x, y, z, color='red', alpha=0.25, linewidth=0)

    ax_traj.scatter(x_start[0], x_start[1], x_start[2], c='blue', s=100, label='Start')
    ax_traj.scatter(x_goal[0], x_goal[1], x_goal[2], c='green', s=100, label='Goal')
    via_scatter = ax_traj.scatter(x_via[0], x_via[1], x_via[2], c='orange', s=100, label='Via Point')

    c = (x_start + x_goal) / 2
    span = np.linalg.norm(x_goal - x_start) / 2 + 1.0
    ax_traj.set_xlim(c[0] - span, c[0] + span)
    ax_traj.set_ylim(c[1] - span, c[1] + span)
    ax_traj.set_zlim(c[2] - span, c[2] + span)

    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_zlabel("z")
    plt.legend()

    def _on_slider_change(val):
        global demos_plot, mu, wireframes, via_scatter
        angle = angle_slider.val
        percent = percent_slider.val
        x_via_center = env.start + percent * (env.goal - env.start)
        x_via = x_via_center + r * np.array([0.0, np.cos(angle), np.sin(angle)])
        via_scatter._offsets3d = (
            np.array([x_via[0]]),
            np.array([x_via[1]]),
            np.array([x_via[2]])
        )

        t0 = time.time()
        pos_demos = select_demos(boids_pos, [x_via], n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
        gmr.update(pos_demos, n_iter=15)
        t1 = time.time()

        update_demo_lines(pos_demos)
        mu_y, Sigma_y, gamma, loglik = gmr.regress(T=max_steps, pos_dim=3)
        mu_line.set_data(mu_y[:,0], mu_y[:,1])
        mu_line.set_3d_properties(mu_y[:,2])

        refresh_wireframes(ax_traj, wireframes, mu_y, Sigma_y,
                   step=30, n_std=1.5, n_points=20, alpha=0.3)

        fig.canvas.draw_idle()

    angle_slider = Slider(
        ax=fig.add_axes([0.25, 0.02, 0.15, 0.03]),
        label='Via Point Angle',
        valmin=0.0,
        valmax=2.0 * np.pi,
        valinit=0.0,
    )
    angle_slider.on_changed(_on_slider_change)

    def _on_slider_percent_change(val):
        global demos_plot, mu, wireframes, via_scatter
        angle = angle_slider.val
        percent = percent_slider.val
        x_via_center = env.start + percent * (env.goal - env.start)
        x_via = x_via_center + r * np.array([0.0, np.cos(angle), np.sin(angle)])
        via_scatter._offsets3d = (
            np.array([x_via[0]]),
            np.array([x_via[1]]),
            np.array([x_via[2]])
        )

        t0 = time.time()
        pos_demos = select_demos(boids_pos, [x_via], n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
        gmr.update(pos_demos, n_iter=15)
        t1 = time.time()

        update_demo_lines(pos_demos)
        mu_y, Sigma_y, gamma, loglik = gmr.regress(T=max_steps, pos_dim=3)
        mu_line.set_data(mu_y[:,0], mu_y[:,1])
        mu_line.set_3d_properties(mu_y[:,2])

        refresh_wireframes(ax_traj, wireframes, mu_y, Sigma_y,
                   step=30, n_std=1.5, n_points=20, alpha=0.3)

        fig.canvas.draw_idle()

    percent_slider = Slider(
        ax=fig.add_axes([0.65, 0.02, 0.15, 0.03]),
        label='Via Point Percent',
        valmin=0.0,
        valmax=1.0,
        valinit=0.5,
    )
    percent_slider.on_changed(_on_slider_percent_change)

    plt.show()