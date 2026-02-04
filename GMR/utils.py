import numpy as np

# ---------------------------------------
# History-conditioned demo selection
# ---------------------------------------
def reps_closest_to_point(pos_all, target):
    """
    pos_all: (T,N,3)
    target: (3,)
    returns reps: (N,3) representative point per demo
    """
    d = np.linalg.norm(pos_all - target[None, None, :], axis=2)  # (T,N)
    t_star = d.argmin(axis=0)                                   # (N,)
    reps = pos_all[t_star, np.arange(pos_all.shape[1]), :]       # (N,3)
    return reps

def nms_by_rep_points(score, reps, k, min_rep_dist=0.5):
    """
    score: (N,) lower is better
    reps: (N,3) representative points
    Greedy pick by best score, suppress reps within min_rep_dist
    """
    order = np.argsort(score)
    selected = []
    suppressed = np.zeros(len(score), dtype=bool)

    for j in order:
        if suppressed[j]:
            continue
        selected.append(j)
        if len(selected) >= k:
            break
        d = np.linalg.norm(reps - reps[j], axis=1)
        suppressed |= (d < min_rep_dist)
        suppressed[j] = False  # keep selected

    return np.array(selected, dtype=int)

def select_demos_by_history(pos_all, history_pts, k=10, stride=1,
                           w_min=0.05, w_max=1.0):
    pos_all = np.asarray(pos_all, float)
    history_pts = np.asarray(history_pts, float)
    T, N, _ = pos_all.shape
    L = history_pts.shape[0]

    # distances (T, N, L)
    d = np.linalg.norm(pos_all[:, :, None, :] - history_pts[None, None, :, :], axis=3)

    # per demo: best match over time for each history point -> (N, L)
    d_min_over_time = d.min(axis=0)

    # recency weights (oldest small, newest big)
    w = np.linspace(w_min, w_max, L)
    w = w / (w.sum() + 1e-12)

    # weighted average distance across ALL history points
    score = (d_min_over_time * w[None, :]).sum(axis=1)  # (N,)

    latest = history_pts[-1]
    reps = reps_closest_to_point(pos_all, latest)  # (N,3)
    idx = nms_by_rep_points(score, reps, k=k, min_rep_dist=0.1)
    return idx, score[idx]

def select_demos(boids_pos, history_pts, n_demos=3, space_stride=5, time_stride=10):
    idx, _ = select_demos_by_history(boids_pos, history_pts, k=n_demos, stride=space_stride)
    pos_demos = []
    for i in idx:
        demo = boids_pos[:, i, :]
        valid = (np.linalg.norm(demo, axis=1) > 1e-2)
        demo = demo[valid, :]
        demo = demo[::time_stride, :]
        pos_demos.append(demo)
    return pos_demos

# ----------------------------
# Plot helpers
# ----------------------------
def gaussian_ellipsoid(mean, cov, n_std=1.5, n_points=30):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    radii = n_std * np.sqrt(np.maximum(eigvals, 0))

    u = np.linspace(0, 2*np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=-1)
    ellipsoid = sphere @ np.diag(radii) @ eigvecs.T
    ellipsoid += mean
    return ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]


def refresh_wireframes(ax, wireframes, mu_y, Sigma_y, step=30, n_std=1.5, n_points=18, alpha=0.25):
    for wf in wireframes:
        try:
            wf.remove()
        except Exception:
            pass
    wireframes.clear()

    for t in range(0, len(mu_y), step):
        X, Y, Z = gaussian_ellipsoid(mu_y[t], Sigma_y[t], n_std=n_std, n_points=n_points)
        wf = ax.plot_wireframe(X, Y, Z, alpha=alpha, linewidth=0.5)
        wireframes.append(wf)

def plot_gmr_uncertainty_3d(mu_y, Sigma_y, step=10, n_std=1.5, ax=None):
    wireframes = []
    for t in range(0, len(mu_y), step):
        X, Y, Z = gaussian_ellipsoid(mu_y[t], Sigma_y[t], n_std=n_std)
        wf = ax.plot_wireframe(X, Y, Z, alpha=0.1)
        wireframes.append(wf)
    return wireframes