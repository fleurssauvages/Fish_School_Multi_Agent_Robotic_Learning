import time
import numpy as np
from RL.env import FishGoalEnv
import pickle
import numpy as np

import matplotlib.pyplot as plt
from RL.env import FishGoalEnv

from HMM.hmm import HMMGMM

def gaussian_ellipsoid(mean, cov, n_std=2.0, n_points=30):
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

def plot_gmr_uncertainty_3d(mu_y, Sigma_y, step=10, n_std=2.0, ax=None):
    ax.plot(mu_y[:,0], mu_y[:,1], mu_y[:,2], 'k', lw=2)

    for t in range(0, len(mu_y), step):
        X, Y, Z = gaussian_ellipsoid(mu_y[t], Sigma_y[t], n_std=n_std)
        ax.plot_wireframe(X, Y, Z, alpha=0.1)

def select_demos_near_via_anytime(pos_demos, via_point, k=10, stride=1):
    pos_demos = np.asarray(pos_demos, float)

    # distances: (N,T)
    d_t = np.linalg.norm(pos_demos - via_point, axis=2)
    d = d_t.min(axis=0)
    idx = np.argsort(d)[:k*stride:stride]
    return idx, d[idx]


def demo_weights_from_via_dist(d, sigma):
    w = np.exp(-0.5 * (d / sigma)**2)
    return w / (w.sum() + 1e-12)

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
    
    x_vias = np.array([[20.0, 20.0, 25.0],
                       [20.0, 20.0, 15.0],
                       [20.0, 25.0, 20.0],
                       [20.0, 15.0, 20.0]])

    # ============================================================
    # Fit an HMM-GMM model
    # ============================================================
    """ Parameters"""
    n_demos = 5
    space_stride = 5
    time_stride = 10
    n_states = 6
    n_mix = 1
    cov_type = "diag"  # "diag" or "full"

    x_via = x_vias[0]
    pos_demos = select_demos_near_via(boids_pos, x_via, n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
    
    """ Fit an HMM-GMM model to the demonstrations """
    hmmgmm = HMMGMM(n_states=n_states, n_mix=n_mix, seed=0, cov_type= cov_type)
    t0 = time.time()
    hmmgmm.fit(pos_demos)
    t1 = time.time()
    print(f"HMM-GMM Training Time: {(t1 - t0)*1000.0:.2f} ms")

    """ Update with new via point """
    x_via = x_vias[2]
    t0 = time.time()
    pos_demos = select_demos_near_via(boids_pos, x_via, n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
    hmmgmm.update(pos_demos, n_iter=3)
    t1 = time.time()
    print(f"HMM-GMM Update Time: {(t1 - t0)*1000.0:.2f} ms")

    """ Perform regression to get mean and covariance of trajectory """
    t0 = time.time()
    mu_y, Sigma_y, gamma, loglik = hmmgmm.regress(T=max_steps, pos_dim=3)
    t1 = time.time()
    print(f"Model Regression: {(t1 - t0)*1000.0:.2f} ms")

    # ============================================================
    # Plot Results
    # ============================================================

    fig = plt.figure(figsize=(22, 14))
    ax_traj = fig.add_subplot(111, projection='3d')
    i = 0
    for demo in pos_demos:
        xs, ys, zs = demo[:, 0], demo[:, 1], demo[:, 2]
        ax_traj.plot(xs, ys, zs, 'k--', linewidth=1.0)
    plot_gmr_uncertainty_3d(mu_y, Sigma_y, step=10, n_std=1.0, ax=ax_traj)
    r = env.obs_radius
    for c in env.obs_centers:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        x = c[0] + r * np.cos(u) * np.sin(v)
        y = c[1] + r * np.sin(u) * np.sin(v)
        z = c[2] + r * np.cos(v)
        ax_traj.plot_surface(x, y, z, color='red', alpha=0.25, linewidth=0)

    ax_traj.scatter(x_start[0], x_start[1], x_start[2], c='blue', s=100, label='Start')
    ax_traj.scatter(x_goal[0], x_goal[1], x_goal[2], c='green', s=100, label='Goal')
    ax_traj.scatter(x_via[0], x_via[1], x_via[2], c='orange', s=100, label='Via Point')

    c = (x_start + x_goal) / 2
    span = np.linalg.norm(x_goal - x_start) / 2 + 1.0
    ax_traj.set_xlim(c[0] - span, c[0] + span)
    ax_traj.set_ylim(c[1] - span, c[1] + span)
    ax_traj.set_zlim(c[2] - span, c[2] + span)

    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_zlabel("z")
    plt.legend()

    plt.show()