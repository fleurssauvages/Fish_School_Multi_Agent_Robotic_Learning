import time
from matplotlib.widgets import Slider
import numpy as np
from RL.env import FishGoalEnv
import pickle
import numpy as np

import warnings
import sys

def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    # ANSI yellow
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    return f"{YELLOW}{message}{RESET}\n"

warnings.formatwarning = custom_warning_format

import matplotlib.pyplot as plt
from RL.env import FishGoalEnv

from GMR.gmr import GMRGMM
from controllers.spacemouse import SpaceMouse3D

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p

from MPC.LMPC_solver_obs import LinearMPCController
from MPC.QP_solver import MultiDroneCBFQP
import spatialmath as sm

def build_gmr_targets(mu, pos, v_human, dt, horizon,
                        tidx=None, lookahead_min=2, lookahead_max=25,
                        k_speed=8.0):
    """
    mu: (T,3) GMR mean in drone/world coordinates
    pos: (3,) current drone position
    v_human: (3,) human commanded velocity (world frame)
    dt, horizon: controller params

    Returns:
    x_des_GMR: (3,) terminal autonomous desired position
    traj_des_GMR: (horizon,3) time-indexed reference over horizon
    tidx: nearest index used
    """
    T = mu.shape[0]

    # 1) choose nearest point if not provided
    if tidx is None:
        d = np.linalg.norm(mu - pos[None, :], axis=1)
        tidx = int(np.argmin(d))

    # 2) estimate local path tangent at tidx
    i0 = max(tidx - 1, 0)
    i1 = min(tidx + 1, T - 1)
    tangent = mu[i1] - mu[i0]
    n = np.linalg.norm(tangent)
    if n < 1e-8:
        tangent = np.array([1.0, 0.0, 0.0])
        n = 1.0
    tangent = tangent / n

    # 3) user intent along the path (signed)
    v = np.asarray(v_human, float).reshape(3,)
    v_par = float(np.dot(v, tangent))  # >0 means "along the path"

    # 4) convert that to a lookahead (in indices)
    #    - baseline lookahead_min
    #    - add more if user pushes along the path
    #    - clamp to [lookahead_min, lookahead_max]
    la = lookahead_min + int(np.clip(k_speed * v_par, 0.0, lookahead_max - lookahead_min))
    la = int(np.clip(la, lookahead_min, lookahead_max))

    # terminal target index
    idx_term = int(np.clip(tidx + la, 0, T - 1))
    x_des_GMR = mu[idx_term].copy()

    # 5) horizon trajectory: next horizon points (with end padding)
    idxs = np.arange(tidx, tidx + horizon)
    idxs = np.clip(idxs, 0, T - 1)
    traj_des_GMR = mu[idxs].copy()  # (horizon,3)

    return x_des_GMR, traj_des_GMR, tidx

def c_intent(v_human, v_ref=11.5):
    speed = np.linalg.norm(v_human)
    return np.clip(speed / v_ref, 0.0, 1.0)

def c_path(pos, mu, tidx, d0=0.5):
    d = np.linalg.norm(pos - mu[tidx])
    return np.exp(-d / d0)   # 1 near path, →0 far

def c_obstacle(pos, obstacles, d_safe=0.3):
    if not obstacles:
        return 1.0

    d_min = min(np.linalg.norm(pos - o["center"]) - o["radius"] for o in obstacles)
    return np.clip(d_min / d_safe, 0.0, 1.0)

def c_gmr(Sigma, u0=0.5):
    u = np.trace(Sigma)
    return np.exp(-u / u0)

def c_curvature(mu, tidx, k0=1.0):
    if tidx <= 1 or tidx >= len(mu) - 2:
        return 1.0

    t1 = mu[tidx] - mu[tidx - 1]
    t2 = mu[tidx + 1] - mu[tidx]
    t1 /= np.linalg.norm(t1) + 1e-9
    t2 /= np.linalg.norm(t2) + 1e-9

    kappa = np.linalg.norm(t2 - t1)   # discrete curvature
    return np.exp(-kappa / k0)

def c_state(pos, mu, tidx, Sigma, obstacles):
    return np.min(
        [c_path(pos, mu, tidx),
        c_gmr(Sigma),
        c_obstacle(pos, obstacles),
        c_curvature(mu, tidx)]
    )

def compute_w(v_human, pos, mu, tidx, Sigma_t, obstacles,
                w_prev=0.0, alpha=0.8):
    """
    Returns w_human in [0,1]:
    1 = fully trust human
    0 = fully automate
    """
    # intent gate: needs to be scaled to your actual v_human range
    intent = c_intent(v_human, v_ref=0.3)      # tune v_ref

    # state gate (optional but fine)
    state = c_state(pos, mu, tidx, Sigma_t, obstacles)

    w_raw = intent * state                     # human trust
    w = alpha * w_prev + (1 - alpha) * w_raw
    return float(np.clip(w, 0.0, 1.0))

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

def conf_from_cov_logdet(Sigma, logdet_ref, beta=2.0):
    """Scalar in [0,1], high when Sigma is tight."""
    unc = float(np.log(np.linalg.det(Sigma) + 1e-12))
    conf = 1.0 / (1.0 + np.exp(beta * (unc - logdet_ref)))
    return float(np.clip(conf, 0.0, 1.0))

def conf_from_gamma(gamma_t):
    """Scalar in [0,1], high when one component dominates."""
    gamma_t = np.asarray(gamma_t, float).reshape(-1)
    if gamma_t.size == 0:
        return 1.0
    return float(np.clip(np.max(gamma_t), 0.0, 1.0))

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
                             [18.0, 15.0, 20.0],
                             [14.0, 20.0, 25.0],
                            [28.0, 15.0, 20.0],
                            [28.0, 25.0, 20.0],
                            [32.0, 20.0, 25.0],
                            [30.0, 20.0, 15.0]], dtype=np.float32)
    
    t0 = time.time()
    env = FishGoalEnv(boid_count=boid_count, pred_count=0, max_steps=max_steps, dt=1, doAnimation = False, returnTrajectory = True, obs_centers=obs_centers, obs_radius = 2.5)
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
    
    angle = 0.0
    r = 5.0
    x_via_center = np.array([20.0, 20.0, 20.0])
    x_via = x_via_center + np.array([0.0, r * np.cos(angle), r * np.sin(angle)])

    # ============================================================
    # Fit a GMR model
    # ============================================================
    """ Parameters"""
    n_demos = 5
    space_stride = 5
    time_stride = 2
    n_components = 6
    cov_type = "full"  # "diag" or "full"

    pos_demos = select_demos_near_via(boids_pos, x_via, n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
    
    """ Fit an GMR-GMM model to the demonstrations """
    gmr = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
    gmr.fit(pos_demos)

    # ============================================================
    # Drone Env
    # ============================================================

    CTRL_HZ = 200
    dt_ctrl = 1.0 / CTRL_HZ
    sim_duration = 6.0 # seconds
    traj_duration = 1.5 # seconds

    # ============================================================
    # Space Mouse
    # ============================================================
    scale = 0.1
    deadzone = 0.0
    spm = SpaceMouse3D(trans_scale=4000.0, deadzone=deadzone, lowpass=0.0, rate_hz=CTRL_HZ)
    spm.start()

    start = np.zeros((1, 3))
    start[:, :] = env.start
    
    envDrones = VelocityAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=start * scale,
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB,
        gui=True,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    action = np.zeros((1, 4))
    def resetDroneEnv():
        envDrones.reset()
        # Disable all GUI elements
        p.connect(p.DIRECT, options="--fullscreen")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        def create_sphere_obstacle(center, radius, add_collision = True, color = [1, 0, 0, 0.6]):
            collision = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
            )

            visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=color,
            )
            if add_collision:
                body = p.createMultiBody(
                    baseMass=0.0,  # static obstacle
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=center.tolist(),
                )
            else:
                body = p.createMultiBody(
                    baseVisualShapeIndex=visual,
                    basePosition=center.tolist(),
                )
            return body
        
        obstacle_ids = []
        obstacle_list = []
        for obs in env.obs_centers:
            obstacle_ids.append(create_sphere_obstacle(obs * scale, env.obs_radius * scale))
            obstacle_list.append({"center": obs * scale, "radius": env.obs_radius * scale})
        create_sphere_obstacle(env.goal * scale, 0.3, add_collision = False, color = [0, 1, 0, 0.6])
        return obstacle_list, obstacle_ids
    
    obstacle_list, obstacle_ids = resetDroneEnv()

    def update_camera_from_velocity(drone_pos):
        yaw = np.degrees(np.arctan2(-1.0, 0.0))

        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=yaw,
            cameraPitch=-25,
            cameraTargetPosition=drone_pos + np.array([0, 0, 0.1]),
        )
    

    # ============================================================
    # LMPC SetUp
    # ============================================================
    speed_limit = 5.0 # max LMPC speed
    horizon = 15 # Horizon steps, each of dt seconds
    gamma = 0.05 # weight for control effort in LMPC
    obstacle_margin = 0.15 # margin to consider around obstacles in LMPC, i.e the size of the drone plus some margin

    lmpc_solver = LinearMPCController(horizon=horizon, dt=dt_ctrl, gamma = gamma,
                                    u_min=np.array([-speed_limit, -speed_limit, -speed_limit, -speed_limit, -speed_limit, -speed_limit]),
                                    u_max=np.array([ speed_limit,  speed_limit,  speed_limit,  speed_limit,  speed_limit,  speed_limit]))
    Uopt = np.zeros((6 * lmpc_solver.horizon,))
    v_nom = np.zeros((1, 3))
    cbf_qp_solver = MultiDroneCBFQP(num_drones=1, dt=dt_ctrl)

    sim_duration = 30.0

    w_prev = 1.0

    # ============================================================
    # Wind Perturbation
    # ============================================================

    WIND_FORCE_N = 0.02          # tune
    WIND_DIR = np.array([0.0, 1.0, 0.0])  # +x wind
    WIND_DIR = WIND_DIR / (np.linalg.norm(WIND_DIR) + 1e-9)

    # ============================================================
    # Main Loop
    # ============================================================

    for step in range(int(sim_duration * CTRL_HZ)):
        # Get Drone State
        state_i = envDrones._getDroneStateVector(0)
        pos = state_i[0:3]
        vel = state_i[10:13]
        omega = state_i[13:16]
        xi0 = np.hstack((vel, omega)) # initial state for LMPC: current linear and angular velocities
        xi0 = np.clip(xi0, -speed_limit/2, speed_limit/2) # clip to avoid too large initial velocities, happens during inter-collisions for example

        # Apply perturbation
        F_wind = WIND_FORCE_N * WIND_DIR
        p.applyExternalForce(
            objectUniqueId=envDrones.DRONE_IDS[0],
            linkIndex=-1,
            forceObj=F_wind.tolist(),
            posObj=pos,
            flags=p.WORLD_FRAME)

        # Update GMR
        pos_demos = select_demos_near_via(boids_pos, pos/scale, n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
        gmr.update(pos_demos, n_iter=15)
        mu, sigma, gamma, loglik = gmr.regress(T=max_steps, pos_dim=3)

        # Scale to drone world
        mu = mu * scale
        sigma = sigma * (scale**2)

        # Compute desired position of the user
        v_human, omega_human, buttons = spm.read()
        x_des_human = pos + v_human * dt_ctrl * horizon
        T_des_human = sm.SE3.Trans(x_des_human)

        # Compute current tidx
        d = np.linalg.norm(mu - pos[None, :], axis=1)
        tidx = int(np.argmin(d))

        # CHOOSE THE DESIRED POSITION ON GMR (FEED FORWARD)
        x_des_GMR, traj_des_GMR, tidx = build_gmr_targets(
            mu=mu,                # already scaled to drone world
            pos=pos,
            v_human=v_human,       # world-frame velocity command
            dt=dt_ctrl,
            horizon=lmpc_solver.horizon,
            tidx=tidx,             # you already computed nearest index; reuse it
            lookahead_min=2,
            lookahead_max=25,
            k_speed=8.0            # tune: bigger = more “go further” for same input
        )
        T_des_GMR = sm.SE3.Trans(x_des_GMR)

        # Confidence
        w = compute_w(v_human, pos, mu, tidx, sigma[tidx], obstacles=obstacle_list, w_prev=w_prev)
        w_prev = w

        if np.linalg.norm(v_human) < 1e-5:
            w = 1.0

        # LMPC
        T_i = sm.SE3.Trans(pos)
        Uopt, Xopt, poses = lmpc_solver.solve(T_i, T_des_human, T_des_GMR, w, xi0=xi0, obstacles=obstacle_list, traj=traj_des_GMR, margin=obstacle_margin)

        if Uopt is None:
            Uopt = np.zeros(3)

        v_nom = Uopt[0:3]    
        speed = np.linalg.norm(v_nom)
        direction = v_nom / speed if speed > 1e-6 else np.zeros(3)
        action[0, 0:3] = direction
        action[0, 3] = np.clip(speed, -5.0, 5.0)

        if np.linalg.norm(buttons) > 0.5:
            resetDroneEnv()

        obs, reward, terminated, truncated, info = envDrones.step(action)

        update_camera_from_velocity(pos)

        for b_id in obstacle_ids:
            contact_points = p.getContactPoints(bodyA=envDrones.DRONE_IDS[0], bodyB=b_id)
            if len(contact_points) > 0:
                warnings.warn("Collision")

        elapsed = time.time() - t0
        if elapsed < dt_ctrl:
            time.sleep(dt_ctrl - elapsed)
        else:
            time.sleep(dt_ctrl)
            

    
        
