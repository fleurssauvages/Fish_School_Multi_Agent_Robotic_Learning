import time
from matplotlib.widgets import Slider
import numpy as np
from RL.env import FishGoalEnv
import pickle
import numpy as np

import matplotlib.pyplot as plt
from RL.env import FishGoalEnv

from GMR.gmr import GMRGMM
from controllers.spacemouse import SpaceMouse3D

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p

from MPC.QP_solver import DroneCBFQP

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
                            [28.0, 16.0, 22.0],
                            [28.0, 24.0, 18.0],
                            [32.0, 20.0, 25.0],
                            [30.0, 20.0, 15.0]], dtype=np.float32)
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
    n_demos = 5
    space_stride = 5
    time_stride = 2
    n_components = 6
    cov_type = "full"  # "diag" or "full"

    pos_demos = select_demos_near_via(boids_pos, x_via, n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
    
    """ Fit an GMR-GMM model to the demonstrations """
    gmr = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
    t0 = time.time()
    gmr.fit(pos_demos)
    t1 = time.time()
    print(f"GMR-GMM Warmup Time: {(t1 - t0)*1000.0:.2f} ms")

    """ Update with new via point """
    t0 = time.time()
    pos_demos = select_demos_near_via(boids_pos, x_via, n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
    gmr.update(pos_demos, n_iter=15)
    t1 = time.time()
    print(f"GMR-GMM Update Time: {(t1 - t0)*1000.0:.2f} ms")

    """ Perform regression to get mean and covariance of trajectory """
    t0 = time.time()
    mu_y, Sigma_y, gamma, loglik = gmr.regress(T=max_steps, pos_dim=3)
    t1 = time.time()
    print(f"Model Regression: {(t1 - t0)*1000.0:.2f} ms")

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
    spm = SpaceMouse3D(trans_scale=2000.0, deadzone=deadzone, lowpass=0.0, rate_hz=CTRL_HZ)
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
        return obstacle_list
    
    obstacle_list = resetDroneEnv()

    def update_camera_from_velocity(drone_pos):
        yaw = np.degrees(np.arctan2(-1.0, 0.0))

        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=yaw,
            cameraPitch=-25,
            cameraTargetPosition=drone_pos + np.array([0, 0, 0.1]),
        )
    

    sim_duration = 30.0

    qp = DroneCBFQP(1, dt_ctrl)
    L = 1
    k_h = 1.0 
    kfb = 0.0      # how much path correction you want

    k_h = 0.05 
    kfb = 0.95    # how much path correction you want

    for step in range(int(sim_duration * CTRL_HZ)):
        t0 = time.time()

        # Get Drone State
        state_i = envDrones._getDroneStateVector(0)
        pos = state_i[0:3]

        # Update GMR
        pos_demos = select_demos_near_via(boids_pos, pos/scale, n_demos=n_demos, space_stride=space_stride, time_stride=time_stride)
        gmr.update(pos_demos, n_iter=15)
        mu, sigma, gamma, loglik = gmr.regress(T=max_steps, pos_dim=3)

        # scale to drone world
        mu = mu * scale
        sigma = sigma * (scale**2)

        # choose nearest point on THIS mu
        d = np.linalg.norm(mu - pos[None, :], axis=1)
        tidx = int(np.argmin(d))

        # reference at tidx
        x_ref = mu[tidx]
        Sigma_ref = sigma[tidx]

        # --- build a scalar confidence for this timestep ---
        # 1) covariance-based confidence needs a reference scale computed from this regression
        logdets = np.array([np.log(np.linalg.det(sigma[t]) + 1e-12) for t in range(len(sigma))], dtype=float)
        logdet_ref = float(np.percentile(logdets, 10))

        conf_cov = conf_from_cov_logdet(Sigma_ref, logdet_ref, beta=2.0)

        # 2) ambiguity confidence from responsibilities (if gamma is per-time)
        # If gamma returned by your regress is shape (T,K), use gamma[tidx]
        # If gamma is something else, fall back to conf_gamma=1.0
        conf_gamma = 1.0
        try:
            if gamma is not None:
                gamma_arr = np.asarray(gamma)
                if gamma_arr.ndim == 2 and gamma_arr.shape[0] == len(mu):
                    conf_gamma = conf_from_gamma(gamma_arr[tidx])
        except Exception:
            conf_gamma = 1.0

        conf = float(np.clip(conf_cov * conf_gamma, 0.0, 1.0))

        # --- corrective term (pull to GMR) scaled by conf ---
        v_fb = conf * (x_ref - pos)

        # --- Human input ---
        v_human, v_rot, buttons = spm.read()
        v_human = np.asarray(v_human, float)  # don't hardcode 1000; tune HUMAN_GAIN

        # --- blend ---
        v_nom = k_h * v_human + kfb * v_fb
        v_nom *= np.linalg.norm(v_human) / np.linalg.norm(v_nom + 1e-12)
        
        print(v_nom, v_human, v_fb)
        # Solve QP
        t0 = time.time()
        v_opt, slack = qp.solve(
            v_nom=v_nom.reshape(1,3),
            positions=pos.reshape(1,3),
            obstacles= [],
            v_max=100.0,
            d_obs_margin=0.1,
            alpha_obs=10.0,
        )
        v_cmd = v_opt.reshape(3)
        t1 = time.time()
        # print(f"Time to solve: {(t1 - t0)*1000.0:.2f} ms")
        
        speed = np.linalg.norm(v_cmd)
        direction = v_cmd / speed if speed > 1e-6 else np.zeros(3)
        action[0, 0:3] = direction
        action[0, 3] = np.clip(speed, -5.0, 5.0)

        if np.linalg.norm(buttons) > 0.5:
            resetDroneEnv()

        obs, reward, terminated, truncated, info = envDrones.step(action)

        update_camera_from_velocity(pos)

        elapsed = time.time() - t0
        if elapsed < dt_ctrl:
            time.sleep(dt_ctrl - elapsed)
        else:
            time.sleep(dt_ctrl)
            

    
        
