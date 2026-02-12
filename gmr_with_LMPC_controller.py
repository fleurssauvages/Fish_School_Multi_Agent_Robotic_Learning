import time
import copy
import pickle
import numpy as np
import random

from RL.env import FishGoalEnv, make_torus_mesh, make_sphere_mesh, merge_meshes

from controllers.spacemouse import SpaceMouse3D
from controllers.utils import compute_terminal_nodes
from controllers.utils import make_ref_trajs

from vispy import app, scene
from vispy.scene import visuals
from renderer.utils import add_transparent_sphere, add_wireframe, add_floor_grid, add_backwall_grid, add_transparent_mesh_with_wireframe, set_mesh_color, apply_collision_revert, build_trimesh_obstacles

from GMR.gmr import GMRGMM
from controllers.utils import compute_alpha, interpolate_traj
from GMR.utils import select_demos
from MPC.LMPC_solver_obs import LinearMPCController
import spatialmath as sm


def bayesian_potential_field(
    x,
    goals,
    v_h,
    belief=None,
    beta=8.0,            # rationality / sharpness of likelihood
    sigma_u=0.3,         # action noise for Gaussian likelihood (optional)
    use_gaussian=False,  # if True, use Gaussian around optimal action; else Boltzmann on cost
    v_max=0.3,
    kp=0.1,
    tau=0.0,             # optional: min MAP belief to assist
    eps=1e-9,
):
    """
    Jain-style: recursive Bayesian inference over discrete goals, then assist toward MAP goal.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7233691/

    Returns
    -------
    v_ref : (d,) assistance direction (velocity-like; treat as force if you want)
    belief : (N,) updated belief over goals
    i_map : int index of MAP goal
    """

    x = np.asarray(x, float)
    goals = np.asarray(goals, float)
    v_h = np.asarray(v_h, float)

    N, d = goals.shape

    # init belief if needed
    if belief is None:
        belief = np.ones(N, dtype=float) / N
    else:
        belief = np.asarray(belief, float)
        belief = belief / (belief.sum() + eps)

    # define "optimal" action for each goal: move toward goal (unit direction)
    delta = goals - x                              # (N,d)
    dist = np.linalg.norm(delta, axis=1) + eps     # (N,)
    u_star = delta / dist[:, None]                 # (N,d) unit vector toward each goal

    # likelihood P(v_h | goal)
    vh_norm = np.linalg.norm(v_h)
    if vh_norm < 1e-6:
        # no new evidence; keep belief, no assist
        i_map = int(np.argmax(belief))
        return np.zeros(d), belief, i_map

    vhat = v_h / (vh_norm + eps)

    if use_gaussian:
        # Gaussian likelihood around u_star (in direction space)
        # L_i ∝ exp(-||vhat - u_star||^2 / (2 sigma_u^2))
        diff2 = np.sum((u_star - vhat)**2, axis=1)
        logL = -0.5 * diff2 / (sigma_u**2 + eps)
    else:
        # Boltzmann on a simple "cost": cost_i = 1 - cos(u_star, vhat)
        # L_i ∝ exp(-beta * cost_i) = exp(beta * cos)
        cos = u_star @ vhat
        logL = beta * cos

    # Bayes update: b_t(g) ∝ L(g) * b_{t-1}(g)
    log_post = np.log(belief + eps) + logL
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= (post.sum() + eps)

    belief = post
    i_map = int(np.argmax(belief))

    # optional: only assist if confident enough
    if belief[i_map] < tau:
        return np.zeros(d), belief, i_map

    # proportional attractor toward MAP goal (cap magnitude)
    e = goals[i_map] - x
    v_ref = kp * e
    n = np.linalg.norm(v_ref)
    if n > v_max:
        v_ref = v_ref * (v_max / (n + eps))

    return v_ref, belief, i_map

def traj_potential_field(
    x,
    ref_trajs,
    v_h,
    belief=None,
    beta=8.0,              # same meaning as in bayesian_potential_field
    v_max=0.3,
    kp=0.1,
    tau=0.0,
    # trajectory specifics
    lookahead=10,           # 0 = closest point; >0 biases forward along the polyline
    eps=1e-9,
):
    """
    Jain-style Bayes inference, but hypotheses are trajectories instead of goals.

    Exact analogue of bayesian_potential_field:
      - hypotheses: k = 0..K-1 trajectories
      - u_star(k): unit direction toward a reference point on traj k
      - likelihood: exp(beta * cos(u_star, vhat))
      - control: v_ref = kp * (p_ref - x), capped by v_max

    Returns:
      v_ref : (d,)
      belief: (K,)
      i_map : int
    """
    x = np.asarray(x, float)
    v_h = np.asarray(v_h, float)
    K = len(ref_trajs)
    d = x.shape[0]

    # init belief
    if belief is None:
        belief = np.ones(K, dtype=float) / K
    else:
        belief = np.asarray(belief, float)
        belief = belief / (belief.sum() + eps)

    # if no human motion, no new evidence (same behavior as your goal function)
    vh_norm = np.linalg.norm(v_h)
    if vh_norm < 1e-6:
        i_map = int(np.argmax(belief))
        return np.zeros(d), belief, i_map

    vhat = v_h / (vh_norm + eps)

    # For each trajectory hypothesis, pick a reference point p_ref(k)
    p_refs = np.zeros((K, d))
    for k, traj in enumerate(ref_trajs):
        traj = np.asarray(traj, float)  # (T,d)
        diff = traj - x[None, :]
        j = int(np.argmin(np.sum(diff * diff, axis=1)))

        # optional: move forward along the trajectory to create "progress"
        j2 = min(j + int(lookahead), traj.shape[0] - 1)
        p_refs[k] = traj[j2]

    # "optimal action" directions toward each trajectory reference point (unit vectors)
    delta = p_refs - x[None, :]                          # (K,d)
    dist = np.linalg.norm(delta, axis=1) + eps           # (K,)
    u_star = delta / dist[:, None]                       # (K,d)

    # likelihood P(v_h | traj_k) using same Boltzmann-on-cosine
    cos = u_star @ vhat                                  # (K,)
    logL = beta * cos

    # Bayes update: b_t(k) ∝ L(k) * b_{t-1}(k)
    log_post = np.log(belief + eps) + logL
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= (post.sum() + eps)
    belief = post

    i_map = int(np.argmax(belief))

    if belief[i_map] < tau:
        return np.zeros(d), belief, i_map

    # proportional attractor toward selected trajectory point (cap magnitude)
    e = p_refs[i_map] - x
    v_ref = kp * e
    n = np.linalg.norm(v_ref)
    if n > v_max:
        v_ref = v_ref * (v_max / (n + eps))

    return v_ref, belief, i_map


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    doGMR = True
    doLMPC = False
    doPotential = False
    doGraphPotential = False
    doSwitchTarget = True

    # ----------------------------
    # 1) Load policy + generate fish trajectories
    # ----------------------------
    theta_path = "save/best_policy.pkl"
    action = pickle.load(open(theta_path, "rb"))["best_theta"]

    boid_count = 1200
    max_steps = 500
    dt = 1
    x_start = np.array([6.0, 20.0, 20.0], dtype=np.float32)

    d_goals = 34
    d_obs = 24
    angle = 20
    obs_centers = np.array([
        [x_start[0] + d_obs, x_start[1], x_start[2]],
        [x_start[0] + d_obs * np.cos(np.deg2rad(angle)), x_start[1] + d_obs * np.sin(np.deg2rad(angle)), x_start[2]],
        [x_start[0] + d_obs * np.cos(np.deg2rad(-angle)), x_start[1] + d_obs * np.sin(np.deg2rad(-angle)), x_start[2]],
        [15.0, 20.0, 20.0]
    ])
    t1 = make_torus_mesh(R=3.0, r=1.0, segR=12, segr=8, center=obs_centers[0])
    t2 = make_torus_mesh(R=3.0, r=1.0, segR=12, segr=8, center=obs_centers[1], yaw=np.deg2rad(angle))
    t3 = make_torus_mesh(R=3.0, r=1.0, segR=12, segr=8, center=obs_centers[2], yaw=np.deg2rad(-angle))
    s1 = make_sphere_mesh(R=3.0, seg_theta=8, seg_phi=8, center=obs_centers[3])
    obs_list = [t1, t2, t3, s1]

    verts, faces = merge_meshes(obs_list)

    goals = np.array([
        [x_start[0], x_start[1], x_start[2]],  # 0 - initial
        [x_start[0] + d_goals, x_start[1], x_start[2]],
        [x_start[0] + d_goals * np.cos(np.deg2rad(angle)), x_start[1] + d_goals * np.sin(np.deg2rad(angle)), x_start[2]],
        [x_start[0] + d_goals * np.cos(np.deg2rad(-angle)), x_start[1] + d_goals * np.sin(np.deg2rad(-angle)), x_start[2]],
    ], dtype=np.float32)
    goals_radius = 3.0
    goal_W = np.array([
        [0.0, 1.0, 1.0, 1.0],  # from 0 → {1,2,3}
        [0.0, 1.0, 0.0, 0.0],  # from 1 → 0
        [0.0, 0.0, 1.0, 0.0],  # from 2 → 0
        [0.0, 0.0, 0.0, 1.0],  # from 3 → 0
    ], dtype=np.float32)

    ends_goals = compute_terminal_nodes(goal_W)
    env = FishGoalEnv(boid_count=boid_count, max_steps=max_steps, dt=dt, 
                      doAnimation = False, returnTrajectory = True, verts=verts, faces=faces, goals=goals, goal_W=goal_W, start=x_start)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    boid_pos = np.array(info["trajectory_boid_pos"])  # (T, N, 3)

    # history settings
    history_len = 8         

    # ----------------------------
    # 3) SpaceMouse
    # ----------------------------
    spm = SpaceMouse3D(trans_scale=10.0, deadzone=0.0, lowpass=0.0, rate_hz=200)
    spm.start()
    x = x_start.astype(float).copy()
    history = [x.copy()] 
    path = [x.copy()]     # full path for plotting

    # ----------------------------
    # 4) Warm-start Controller
    # ----------------------------
    dt = 10

    if doGMR:
        update_period = 0.05      
        update_iters = 3         
        move_eps = 1e-3           

        pos_demos = select_demos(boid_pos, history, time_stride=1, n_demos=5)

        gmr = GMRGMM(n_components=6, seed=0, cov_type="full")
        gmr.fit(pos_demos)
        mu_y, Sigma_y, _, _ = gmr.regress(T=max_steps, pos_dim=3)
        last_update = time.time()
        last_x = x_start.copy() 

    if doLMPC:
        u_min = np.array([-10]*6)
        u_max = np.array([10]*6)
        lmpc_solver = LinearMPCController(horizon=10, dt=dt, gamma=10, u_min=u_min, u_max=u_max)

    if doGraphPotential:
        trajectories_points = [
            [x_start, [15.0, 14.0, 20.0], goals[3]],
            [x_start, [15.0, 26.0, 20.0], goals[2]],
            [x_start, [15.0, 26.0, 20.0], [25.0, 20.0, 20.0], goals[1]],
            [x_start, [15.0, 14.0, 20.0], [25.0, 20.0, 20.0], goals[1]],
            [x_start, [15.0, 20.0, 26.0], [25.0, 20.0, 20.0], goals[1]],
            [x_start, [15.0, 20.0, 14.0], [25.0, 20.0, 20.0], goals[1]],
        ]
        ref_trajs = make_ref_trajs(trajectories_points, step=0.5)

    # ----------------------------
    # 5) Plot setup
    # ----------------------------
    canvas = scene.SceneCanvas(keys="interactive", fullscreen=True, show=True, bgcolor="white")
    view = canvas.central_widget.add_view()

    # Turntable camera similar to matplotlib view_init(elev=40, azim=-180)
    cam = scene.cameras.TurntableCamera(fov=45, elevation=25, azimuth=-90, roll=0)
    all_pts = np.vstack([obs_centers, np.asarray(goals), x_start[None, :]])
    center = all_pts.mean(axis=0)
    span = np.max(np.linalg.norm(all_pts - center, axis=1))

    cam.center = tuple(center)
    cam.distance = float(span * 2.0)      # pull back enough to see everything
    cam.fov = 45

    view.camera = cam

    # Enable blending so alpha works (transparent spheres)
    canvas.context.set_state(blend=True, depth_test=True, blend_func=('src_alpha', 'one_minus_src_alpha'))

    axis = visuals.XYZAxis(parent=view.scene)

    if doGraphPotential:
        ref_lines = []
        for traj in ref_trajs:
            line = visuals.Line(
                np.array(traj, dtype=np.float32),
                color=(0.0, 0.0, 0.0, 0.25),  # light gray
                width=2.0,
                method="gl",
                parent=view.scene,
            )
            ref_lines.append(line)


    # --------------------------
    # Start + goals as markers
    # --------------------------
    markers = visuals.Markers(parent=view.scene)
    colors = np.array([(1.0, 0.5, 0.0, 0.25)]*len(goals), dtype=np.float32)
    goal_spheres = []
    for c, col in zip(goals[ends_goals, :], colors):
        goal_spheres.append(add_transparent_sphere(view, c, goals_radius, col))
        add_wireframe(view, c, goals_radius, line_color=(0, 0, 0, 0.05), width=1.0)

    user_size = 0.3
    user_marker = add_transparent_sphere(view, x_start, user_size, (0.0, 0.5, 0.8, 0.5))
    user_transform = scene.transforms.STTransform(translate=tuple(x))
    user_marker.transform = user_transform

    hist_markers = visuals.Markers(parent=view.scene)
    path_line = visuals.Line(parent=view.scene, method="gl", width=4.0)
    
    hist_markers.set_data(np.array(history, dtype=np.float32),
                        face_color=(0.0, 0.0, 0.6, 0.6), size=6, edge_width=0)
    path_line.set_data(np.array(path, dtype=np.float32),
                    color=(0.0, 0.0, 0.6, 0.2))

    # --------------------------
    # Obstacles
    # --------------------------
    obs_colors = [
        (0.70, 0.55, 0.40, 0.20),
        (0.7, 0.7, 0.7, 0.20),
        (0.55, 0.15, 0.55, 0.15),
        (0.7, 0.7, 0.7, 0.20),
    ]

    for obs, cols, center in zip(obs_list, obs_colors, obs_centers):
        verts, faces = obs[0], obs[1]
        mesh_vis, wire_vis = add_transparent_mesh_with_wireframe(
            view,
            verts,
            faces,
            mesh_rgba=(0.6, 0.6, 0.6, 0.25),
            wire_rgba=(0, 0, 0, 0.35),
            wire_width=1.0,
            shading="flat",
        )

    # --------------------------
    # Grid for better visibility
    # --------------------------
    mins = np.array([0, 0, 0], dtype=np.float32)
    maxs = np.array([40, 40, 40], dtype=np.float32)
    add_floor_grid(view, z=mins[2], xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]), step=2.0)
    add_backwall_grid(view, x=maxs[0], zlim=(mins[2], maxs[2]), ylim=(mins[1], maxs[1]), step=2.0)

    corners = np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ], dtype=np.float32)

    edges = [(0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)]

    for a,b in edges:
        visuals.Line(np.vstack([corners[a], corners[b]]),
                    color=(0,0,0,0.15), width=2.0, parent=view.scene)
    
    goals = goals[ends_goals]
    desired_goal = random.randint(0, len(goals)-1)
    set_mesh_color(goal_spheres[desired_goal], (0.8, 0.0, 0.0, 0.3))

    obs_tm = build_trimesh_obstacles(obs_list)
    last_update = time.time()
    last_x = x.copy()
    belief = None
    targetHasChanged = False
    v_ref_prev = 0.0
    v_ref = 0.0

    tidx_prev = 0

    # ----------------------------
    # 6) Live loop
    # ----------------------------
    def on_frame(event):
        global x, history, path, user_marker, desired_goal, dt, user_size, belief, targetHasChanged, tidx, tidx_prev, mu_new, Sigma_new
        trans, rot, buttons = spm.read()
        v = np.array(trans, dtype=float)

        x_prev = x.copy()

        # your motion update
        if doGMR:
            global mu_y, Sigma_y, last_update, last_x
            W = 30  # ~0.5s at dt=10 indexing; tune
            lo = tidx_prev
            hi = min(max_steps - 1, tidx_prev + W)

            d = np.linalg.norm(mu_y[lo:hi+1] - x[None, :], axis=1)
            tidx = lo + int(np.argmin(d))

            # enforce non-decreasing index (optional)
            tidx = max(tidx, tidx_prev)
            tidx_prev = tidx
            if tidx < max_steps - 1:
                v_ref = (mu_y[tidx + 1] - x) / dt
                v_ref = v_ref * np.linalg.norm(v) / np.linalg.norm(v_ref)
            else:
                v_ref = np.zeros(3)
            sigma = Sigma_y[tidx]
            alpha = compute_alpha(v, v_ref, sigma)

            if doLMPC:
                T_i = sm.SE3.Trans(x)
                T_des_human = sm.SE3.Trans(x + v * dt)
                T_des_GMR = sm.SE3.Trans(x + v_ref * dt)
                t_max = min(max_steps - 1, tidx + lmpc_solver.horizon)
                traj = interpolate_traj(x, mu_y[t_max], lmpc_solver.horizon)

                Uopt, _, _ = lmpc_solver.solve(T_i, T_des_human, T_des_GMR, 1 - alpha, obstacles=obs_list, traj=traj, margin=user_size*2.0)
                if Uopt is None:
                    v_opt = np.zeros(3)
                else:
                    v_opt = Uopt[0:3]
            else:                
                v_opt = (1-alpha) * v + alpha * v_ref
            x_new = x + v_opt * dt

            if (time.time() - last_update) >= update_period and np.linalg.norm(x - last_x) > move_eps:
                last_x = x.copy()        
                pos_demos = select_demos(boid_pos, history, time_stride=1, n_demos=5)

                # update + regress
                gmrUpdate = copy.deepcopy(gmr)
                gmrUpdate.update(pos_demos, n_iter=update_iters)
                mu_new, Sigma_new, _, _ = gmrUpdate.regress(T=max_steps, pos_dim=3)
                blend = 0.2  # 0.1–0.3
                mu_y = (1 - blend) * mu_y + blend * mu_new
                Sigma_y = (1 - blend) * Sigma_y + blend * Sigma_new
                last_update = time.time()
        elif doPotential:
            v_ref, belief, i_map = bayesian_potential_field(x, goals, v, belief = belief)
            alpha = compute_alpha(v, v_ref, np.eye(3))
            x_new = x + (1-alpha) * v + alpha * v_ref
        elif doGraphPotential:
            v_ref, belief, i_map  = traj_potential_field(x, ref_trajs, v, belief = belief)
            alpha = compute_alpha(v, v_ref, np.eye(3))
            x_new = x + (1-alpha) * v + alpha * v_ref
        else:
            x_new = x + v * dt

        # bounds clip first (optional)
        x_new = np.clip(x_new, mins, maxs)

        # collision revert
        x_new, collided = apply_collision_revert(
            x_prev=x_prev,
            x_new=x_new,
            user_radius=user_size,
            obstacle_meshes=obs_tm
        )
        x = x_new

        if collided:
            set_mesh_color(user_marker, (1.0, 0.0, 0.0, 0.5))
        else:
            set_mesh_color(user_marker, (0.0, 0.5, 0.8, 0.5))

        if np.linalg.norm(x - history[-1]) > 1e-6:
            history.append(x.copy())
            if len(history) > history_len:
                history = history[-history_len:]
            path.append(x.copy())


        # --- restart on button press ---
        if np.linalg.norm(buttons) > 0.5 and np.linalg.norm(x_start - x) > 0.1:
            x = x_start.astype(float).copy()
            history = [x.copy()]
            path = [x.copy()]
            for goal in goal_spheres:
                set_mesh_color(goal, (1.0, 0.5, 0.0, 0.25))
            desired_goal = random.randint(0, len(goals)-1)
            set_mesh_color(goal_spheres[desired_goal], (0.8, 0.0, 0.0, 0.3))
            targetHasChanged = False

        x = np.clip(x, mins, maxs)

        user_transform.translate = tuple(x)
        hist_markers.set_data(np.array(history, dtype=np.float32),
                            face_color=(0.0, 0.5, 0.8, 0.3), size=6, edge_width=0)
        path_line.set_data(np.array(path, dtype=np.float32),
                        color=(0.0, 0.5, 0.8, 0.3))
        
        if np.linalg.norm(x - goals[desired_goal]) < goals_radius:
            set_mesh_color(goal_spheres[desired_goal], (0.0, 0.8, 0.0, 0.3))

        if not targetHasChanged and doSwitchTarget and np.linalg.norm(x - x_start) > d_goals / 2.5:
            set_mesh_color(goal_spheres[desired_goal], (1.0, 0.5, 0.0, 0.25))
            desired_goal = random.randint(0, len(goals)-1)
            set_mesh_color(goal_spheres[desired_goal], (0.8, 0.0, 0.0, 0.3))
            targetHasChanged = True

        return

    dt_visu = 1.0 / 60
    timer = app.Timer(interval=dt_visu, connect=on_frame, start=True)
    try:
        app.run()
    finally:
        try:
            spm.stop()
        except Exception:
            pass