import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import time

import numpy as np
from numba import njit, prange

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt

import pickle

# -----------------------------
# Helpers
# -----------------------------
@njit(cache=True, fastmath=True)
def _norm3(v0, v1, v2):
    return math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)


@njit(cache=True, fastmath=True)
def _clamp_len(v0, v1, v2, max_len):
    n = _norm3(v0, v1, v2)
    if n <= 1e-12:
        return 0.0, 0.0, 0.0
    if n > max_len:
        s = max_len / n
        return v0 * s, v1 * s, v2 * s
    return v0, v1, v2


@njit(cache=True, fastmath=True)
def _lcg_rand01(seed_arr):
    m = 4294967296.0
    a = 1664525.0
    c = 1.0
    seed = seed_arr[0]
    seed = (a * seed + c) % m
    seed_arr[0] = seed
    return seed / m


@njit(cache=True, fastmath=True)
def _cubic_interpolate(v0, v1, v2, v3, x):
    # Paul Breeuwsma coefficients
    x2 = x * x
    a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3
    a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3
    a2 = -0.5 * v0 + 0.5 * v2
    a3 = v1
    return a0 * x * x2 + a1 * x2 + a2 * x + a3


@njit(cache=True, fastmath=True)
def _noise(time, cum_wavlen, rv0, rv1, rv2, rv3, seed_arr):
    wavelen = 0.3
    if time >= cum_wavlen:
        # Wavelen segment
        cum_wavlen = cum_wavlen + wavelen
        rv0, rv1, rv2 = rv1, rv2, rv3
        rv3 = _lcg_rand01(seed_arr)

    frac = (time % wavelen) / wavelen
    value = _cubic_interpolate(rv0, rv1, rv2, rv3, frac)
    return (value * 2.0 - 1.0), cum_wavlen, rv0, rv1, rv2, rv3


# -----------------------------
# Core rules
# -----------------------------
@njit(cache=True, fastmath=True)
def _bounds_steer(px, py, pz, bound_size):
    min_b = 0.0
    max_b = bound_size
    sx = 0.0
    sy = 0.0
    sz = 0.0

    if px < min_b:
        sx = min_b - px
    elif px > max_b:
        sx = max_b - px

    if py < min_b:
        sy = (min_b - py) * 2.0
    elif py > max_b:
        sy = max_b - py

    if pz < min_b:
        sz = min_b - pz
    elif pz > max_b:
        sz = max_b - pz

    sy = sy * 2.0 
    return sx, sy, sz

@njit(cache=True, fastmath=True)
def _reynolds(i, pos, vel, count, sep_r, ali_r, coh_r):
    px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]

    sep0 = sep1 = sep2 = 0.0
    ali0 = ali1 = ali2 = 0.0
    coh0 = coh1 = coh2 = 0.0
    max_d2 = max(sep_r * sep_r, ali_r * ali_r, coh_r * coh_r)

    for j in range(count):
        if j == i:
            continue
        dx = px - pos[j, 0]
        dy = py - pos[j, 1]
        dz = pz - pos[j, 2]
        d2 = dx * dx + dy * dy + dz * dz
        if d2 <= 1e-24 or d2 > max_d2:
            continue

        d = math.sqrt(d2)
        if d2 < sep_r * sep_r:
            mag = 1.0 - d / sep_r
            sep0 += (dx / d) * mag
            sep1 += (dy / d) * mag
            sep2 += (dz / d) * mag

        if d2 < ali_r * ali_r:
            mag = 1.0 - d / ali_r
            vx, vy, vz = vel[j, 0], vel[j, 1], vel[j, 2]
            vn = _norm3(vx, vy, vz)
            if vn > 1e-12:
                ali0 += (vx / vn) * mag
                ali1 += (vy / vn) * mag
                ali2 += (vz / vn) * mag

        if d2 < coh_r * coh_r:
            mag = 1.0 - d / coh_r
            coh0 += (-dx / d) * mag
            coh1 += (-dy / d) * mag
            coh2 += (-dz / d) * mag

    sep0, sep1, sep2 = _clamp_len(sep0, sep1, sep2, 1.0)
    ali0, ali1, ali2 = _clamp_len(ali0, ali1, ali2, 1.0)
    coh0, coh1, coh2 = _clamp_len(coh0, coh1, coh2, 1.0)
    return sep0, sep1, sep2, ali0, ali1, ali2, coh0, coh1, coh2

@njit(cache=True, fastmath=True)
def _obstacle_avoid(px, py, pz, cx, cy, cz, sphere_r, avoid_r):
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    d2 = dx*dx + dy*dy + dz*dz

    influence = sphere_r + avoid_r
    if influence <= 0.0:
        return 0.0, 0.0, 0.0
    infl2 = influence * influence

    # reject if outside influence or almost at center
    if d2 >= infl2 or d2 <= 1e-18:
        return 0.0, 0.0, 0.0

    d = math.sqrt(d2)

    # normalized proximity in [0,1]
    x = 1.0 - d / influence

    # exponential shaping (steeper near obstacle)
    alpha = 4.0
    mag = (math.exp(alpha * x) - 1.0)

    invd = 1.0 / d
    return (dx * invd) * mag, (dy * invd) * mag, (dz * invd) * mag

@njit(cache=True, fastmath=True)
def _predator_avoid(i, pos, predators_pos, pred_count, avoid_r):
    predator_radius = 5.0
    px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]
    sx = sy = sz = 0.0

    if pred_count <= 0 or avoid_r <= 0.0:
        return 0.0, 0.0, 0.0

    r2 = avoid_r * avoid_r

    for j in range(pred_count):
        dx = px - predators_pos[j, 0]
        dy = py - predators_pos[j, 1]
        dz = pz - predators_pos[j, 2]
        d2 = dx*dx + dy*dy + dz*dz
        if d2 < r2 and d2 > 1e-18:
            d = math.sqrt(d2)
            mag = 1.0 - d / (avoid_r + predator_radius)
            invd = 1.0 / d
            sx += (dx * invd) * mag
            sy += (dy * invd) * mag
            sz += (dz * invd) * mag

    sx, sy, sz = _clamp_len(sx, sy, sz, 1.0)
    return sx, sy, sz


@njit(cache=True, fastmath=True)
def _velocity_attack_step(pred_i, predators_pos, predators_vel, pred_own_time,
                          pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
                          boid_pos, boid_count, dt):
    rest_time = 5 * dt
    attack_time = 100 * dt

    t = pred_own_time[pred_i]

    if pred_rest[pred_i] == 1:
        if (t - pred_rest_start[pred_i]) > rest_time:
            pred_rest[pred_i] = 0
            pred_attack_start[pred_i] = t

            px, py, pz = predators_pos[pred_i, 0], predators_pos[pred_i, 1], predators_pos[pred_i, 2]
            closest = 0
            closest_dist = 1e18
            for k in range(boid_count):
                dx = px - boid_pos[k, 0]
                dy = py - boid_pos[k, 1]
                dz = pz - boid_pos[k, 2]
                d = _norm3(dx, dy, dz)
                if d < closest_dist:
                    closest_dist = d
                    closest = k
            pred_prey_idx[pred_i] = closest

        return 0.0, 0.0, 0.0

    prey = pred_prey_idx[pred_i]
    if prey < 0 or prey >= boid_count:
        pred_rest[pred_i] = 1
        pred_rest_start[pred_i] = t
        pred_prey_idx[pred_i] = -1
        return 0.0, 0.0, 0.0

    dx = boid_pos[prey, 0] - predators_pos[pred_i, 0]
    dy = boid_pos[prey, 1] - predators_pos[pred_i, 1]
    dz = boid_pos[prey, 2] - predators_pos[pred_i, 2]
    dist = _norm3(dx, dy, dz)

    speed_up_time = t - pred_attack_start[pred_i]
    if speed_up_time > attack_time or dist < 1.0:
        pred_rest[pred_i] = 1
        pred_rest_start[pred_i] = t
        pred_prey_idx[pred_i] = -1
        return 0.0, 0.0, 0.0

    if dist > 1e-12:
        ux, uy, uz = dx / dist, dy / dist, dz / dist
    else:
        ux, uy, uz = 0.0, 0.0, 0.0

    vx, vy, vz = predators_vel[pred_i, 0], predators_vel[pred_i, 1], predators_vel[pred_i, 2]
    ux -= vx
    uy -= vy
    uz -= vz

    tau = attack_time * 0.3
    s = speed_up_time / (speed_up_time + tau)
    ux *= s
    uy *= s
    uz *= s

    max_speed = 0.01
    dt_ref = 0.01
    max_speed_dt = max_speed * (dt / dt_ref)
    ux, uy, uz = _clamp_len(ux, uy, uz, max_speed_dt)
    return ux, uy, uz


@njit(cache=True, fastmath=True, parallel=True)
def step_sim(
    boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
    pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
    pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
    seed_arr, dt,
    *,
    bound_size,
    boid_count,
    pred_count,
    rule_scalar,
    rule_scalar_p,
    max_speed,
    max_speed_p,
    sep_r,
    ali_r,
    coh_r,
    pred_avoid_r,
    obs_avoid_r,
    sep_s,
    ali_s,
    coh_s,
    bnd_s,
    rand_s,
    pred_avoid_s,
    obs_avoid_s,
    attack_s,
    rand_wavelen_scalar,
    obs_radius,
    goal_gain,
    goal_x,
    goal_y,
    goal_z,
    obs_centers
):
    """One simulation step.
    """

    if (max_speed == 0.0 and max_speed_p == 0.0) or boid_count <= 0:
        return

    # --- Boids ---
    for i in prange(boid_count):
        boid_time[i] += dt

        ax = ay = az = 0.0

        sep0, sep1, sep2, ali0, ali1, ali2, coh0, coh1, coh2 = _reynolds(
            i, boid_pos, boid_vel, boid_count, sep_r, ali_r, coh_r
        )

        ax += sep0 * sep_s
        ay += sep1 * sep_s
        az += sep2 * sep_s

        ax += ali0 * ali_s
        ay += ali1 * ali_s
        az += ali2 * ali_s

        ax += coh0 * coh_s
        ay += coh1 * coh_s
        az += coh2 * coh_s

        # bounds
        sx, sy, sz = _bounds_steer(boid_pos[i, 0], boid_pos[i, 1], boid_pos[i, 2], bound_size)
        ax += sx * bnd_s
        ay += sy * bnd_s
        az += sz * bnd_s

        # random motion (smooth noise)
        if rand_s != 0.0:
            t = boid_time[i] * rand_wavelen_scalar * math.sqrt(dt)

            rv0, rv1, rv2, rv3 = boid_noise_vals[i, 0, 0], boid_noise_vals[i, 0, 1], boid_noise_vals[i, 0, 2], boid_noise_vals[i, 0, 3]
            nx, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.0, boid_noise_cum[i, 0], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 0] = cwl
            boid_noise_vals[i, 0, 0], boid_noise_vals[i, 0, 1], boid_noise_vals[i, 0, 2], boid_noise_vals[i, 0, 3] = rv0, rv1, rv2, rv3

            rv0, rv1, rv2, rv3 = boid_noise_vals[i, 1, 0], boid_noise_vals[i, 1, 1], boid_noise_vals[i, 1, 2], boid_noise_vals[i, 1, 3]
            ny, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.1, boid_noise_cum[i, 1], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 1] = cwl
            boid_noise_vals[i, 1, 0], boid_noise_vals[i, 1, 1], boid_noise_vals[i, 1, 2], boid_noise_vals[i, 1, 3] = rv0, rv1, rv2, rv3

            rv0, rv1, rv2, rv3 = boid_noise_vals[i, 2, 0], boid_noise_vals[i, 2, 1], boid_noise_vals[i, 2, 2], boid_noise_vals[i, 2, 3]
            nz, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.2, boid_noise_cum[i, 2], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 2] = cwl
            boid_noise_vals[i, 2, 0], boid_noise_vals[i, 2, 1], boid_noise_vals[i, 2, 2], boid_noise_vals[i, 2, 3] = rv0, rv1, rv2, rv3

            ax += nx * rand_s
            ay += (ny * 0.2) * rand_s
            az += nz * rand_s

        # predator avoid
        if pred_avoid_s != 0.0 and pred_count > 0:
            sx, sy, sz = _predator_avoid(i, boid_pos, pred_pos, pred_count, pred_avoid_r)
            ax += sx * pred_avoid_s
            ay += sy * pred_avoid_s
            az += sz * pred_avoid_s

        # obstacle avoid
        if obs_avoid_s != 0.0 and obs_centers.shape[0] > 0:
            for o in range(obs_centers.shape[0]):
                cx = obs_centers[o, 0]
                cy = obs_centers[o, 1]
                cz = obs_centers[o, 2]
                sx, sy, sz = _obstacle_avoid(
                    boid_pos[i, 0], boid_pos[i, 1], boid_pos[i, 2],
                    cx, cy, cz, obs_radius, obs_avoid_r
                )
                ax += sx * obs_avoid_s
                ay += sy * obs_avoid_s
                az += sz * obs_avoid_s

        # goal attraction
        if goal_gain != 0.0:
            dx = goal_x - boid_pos[i, 0]
            dy = goal_y - boid_pos[i, 1]
            dz = goal_z - boid_pos[i, 2]
            d = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12

            pwr = 3.0
            mag = goal_gain * (d ** pwr)
            mag = min(mag, goal_gain * 8.0)

            ax += (dx / d) * mag
            ay += (dy / d) * mag
            az += (dz / d) * mag

        # integrate
        ax *= rule_scalar
        ay *= rule_scalar
        az *= rule_scalar

        boid_vel[i, 0] += ax * dt
        boid_vel[i, 1] += ay * dt
        boid_vel[i, 2] += az * dt
        boid_vel[i, 0], boid_vel[i, 1], boid_vel[i, 2] = _clamp_len(
            boid_vel[i, 0], boid_vel[i, 1], boid_vel[i, 2], max_speed
        )

        boid_pos[i, 0] += boid_vel[i, 0] * dt
        boid_pos[i, 1] += boid_vel[i, 1] * dt
        boid_pos[i, 2] += boid_vel[i, 2] * dt

    # --- Predators ---
    for i in range(pred_count):
        pred_time[i] += dt

        ax = ay = az = 0.0

        # Predators have fixed Reynolds rules
        sep0, sep1, sep2, _, _, _, _, _, _ = _reynolds(
            i, pred_pos, pred_vel, pred_count, 1.0, 1.0, 1.0
        )
        ax += sep0 * 1.0
        ay += sep1 * 1.0
        az += sep2 * 1.0

        sx, sy, sz = _bounds_steer(pred_pos[i, 0], pred_pos[i, 1], pred_pos[i, 2], bound_size)
        ax += sx * (bnd_s / 1.5)
        ay += sy * (bnd_s / 1.5)
        az += sz * (bnd_s / 1.5)

        # Predators random motion
        if rand_s != 0.0:
            t = pred_time[i] * rand_wavelen_scalar * math.sqrt(dt)
            rv0, rv1, rv2, rv3 = pred_noise_vals[i, 0, 0], pred_noise_vals[i, 0, 1], pred_noise_vals[i, 0, 2], pred_noise_vals[i, 0, 3]
            nx, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.0, pred_noise_cum[i, 0], rv0, rv1, rv2, rv3, seed_arr)
            pred_noise_cum[i, 0] = cwl
            pred_noise_vals[i, 0, 0], pred_noise_vals[i, 0, 1], pred_noise_vals[i, 0, 2], pred_noise_vals[i, 0, 3] = rv0, rv1, rv2, rv3

            rv0, rv1, rv2, rv3 = pred_noise_vals[i, 1, 0], pred_noise_vals[i, 1, 1], pred_noise_vals[i, 1, 2], pred_noise_vals[i, 1, 3]
            ny, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.1, pred_noise_cum[i, 1], rv0, rv1, rv2, rv3, seed_arr)
            pred_noise_cum[i, 1] = cwl
            pred_noise_vals[i, 1, 0], pred_noise_vals[i, 1, 1], pred_noise_vals[i, 1, 2], pred_noise_vals[i, 1, 3] = rv0, rv1, rv2, rv3

            rv0, rv1, rv2, rv3 = pred_noise_vals[i, 2, 0], pred_noise_vals[i, 2, 1], pred_noise_vals[i, 2, 2], pred_noise_vals[i, 2, 3]
            nz, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.2, pred_noise_cum[i, 2], rv0, rv1, rv2, rv3, seed_arr)
            pred_noise_cum[i, 2] = cwl
            pred_noise_vals[i, 2, 0], pred_noise_vals[i, 2, 1], pred_noise_vals[i, 2, 2], pred_noise_vals[i, 2, 3] = rv0, rv1, rv2, rv3

            ax += nx * (1.0 / 2.0)
            ay += ny * (1.0 / 2.0)
            az += nz * (1.0 / 2.0)

        if obs_centers.shape[0] > 0 and obs_avoid_s != 0.0:
            for o in range(obs_centers.shape[0]):
                cx = obs_centers[o, 0]
                cy = obs_centers[o, 1]
                cz = obs_centers[o, 2]
                sx, sy, sz = _obstacle_avoid(
                    pred_pos[i, 0], pred_pos[i, 1], pred_pos[i, 2],
                    cx, cy, cz, obs_radius, obs_avoid_r
                )
                ax += sx * 1.0
                ay += sy * 1.0
                az += sz * 1.0

        # The goal is an obstacle for predators
        sx, sy, sz = _obstacle_avoid(
            pred_pos[i, 0], pred_pos[i, 1], pred_pos[i, 2],
            goal_x, goal_y, goal_z, 1.0, obs_avoid_r
        )
        ax += sx * 1.0
        ay += sy * 1.0
        az += sz * 1.0

        # Attack behavior
        vx, vy, vz = _velocity_attack_step(
            i, pred_pos, pred_vel, pred_time,
            pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
            boid_pos, boid_count, dt
        )
        vx *= (attack_s * 0.5)
        vy *= (attack_s * 0.5)
        vz *= (attack_s * 0.5)
        pred_vel[i, 0] += vx * dt
        pred_vel[i, 1] += vy * dt
        pred_vel[i, 2] += vz * dt

        ax *= rule_scalar_p
        ay *= rule_scalar_p
        az *= rule_scalar_p

        pred_vel[i, 0] += ax * dt
        pred_vel[i, 1] += ay * dt
        pred_vel[i, 2] += az * dt
        pred_vel[i, 0], pred_vel[i, 1], pred_vel[i, 2] = _clamp_len(
            pred_vel[i, 0], pred_vel[i, 1], pred_vel[i, 2], max_speed_p
        )

        pred_pos[i, 0] += pred_vel[i, 0] * dt
        pred_pos[i, 1] += pred_vel[i, 1] * dt
        pred_pos[i, 2] += pred_vel[i, 2] * dt

def _init_agents(total_boids, total_preds, start_xyz, start_spread, seed=0.1):
    sx, sy, sz = float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])

    boid_pos = np.empty((total_boids, 3), dtype=np.float32)
    boid_vel = np.zeros((total_boids, 3), dtype=np.float32)
    boid_time = np.zeros((total_boids,), dtype=np.float32)

    pred_pos = np.zeros((total_preds, 3), dtype=np.float32)
    pred_vel = np.zeros((total_preds, 3), dtype=np.float32)
    pred_time = np.zeros((total_preds,), dtype=np.float32)

    rng = np.random.default_rng(int(seed * 1e6) % (2**32 - 1))
    offsets = rng.normal(0.0, 1.0, size=(total_boids, 3))
    norms = np.linalg.norm(offsets, axis=1) + 1e-12
    offsets = offsets / norms[:, None]
    radii = rng.random(total_boids) ** (1.0/3.0)
    offsets = offsets * (radii[:, None] * float(start_spread))
    boid_pos[:, 0] = sx + offsets[:, 0]
    boid_pos[:, 1] = sy + offsets[:, 1]
    boid_pos[:, 2] = sz + offsets[:, 2]

    boid_noise_cum = np.zeros((total_boids, 3), dtype=np.float32)
    boid_noise_vals = np.empty((total_boids, 3, 4), dtype=np.float32)
    pred_noise_cum = np.zeros((total_preds, 3), dtype=np.float32)
    pred_noise_vals = np.empty((total_preds, 3, 4), dtype=np.float32)

    seed_arr = np.array([math.floor(seed * 4294967296.0)], dtype=np.float32)

    for i in prange(total_boids):
        for a in range(3):
            boid_noise_vals[i, a, 0] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 1] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 2] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 3] = _lcg_rand01(seed_arr)

    for i in prange(total_preds):
        for a in range(3):
            pred_noise_vals[i, a, 0] = _lcg_rand01(seed_arr)
            pred_noise_vals[i, a, 1] = _lcg_rand01(seed_arr)
            pred_noise_vals[i, a, 2] = _lcg_rand01(seed_arr)
            pred_noise_vals[i, a, 3] = _lcg_rand01(seed_arr)

    pred_rest = np.ones((total_preds,), dtype=np.int64)
    pred_rest_start = np.zeros((total_preds,), dtype=np.float32)
    pred_attack_start = np.zeros((total_preds,), dtype=np.float32)
    pred_prey_idx = -np.ones((total_preds,), dtype=np.int64)

    seed_arr[0] = math.floor(seed * 4294967296.0)

    return (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
            pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
            pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
            seed_arr)

@njit(cache=True, fastmath=True)
def update_events_numba(
    boid_pos, boid_vel,
    pred_pos, pred_count,
    alive, reached, eaten, t_reach,
    goal, goal_radius, eat_radius,
    step_idx, dt
):
    """
    Returns:
      n_active_after  (int): number of boids still alive after updates
      n_new_goal      (int): number that reached goal this step
      n_new_eaten     (int): number eaten this step
    """

    gx = goal[0]
    gy = goal[1]
    gz = goal[2]

    gr2 = goal_radius * goal_radius
    er2 = eat_radius * eat_radius

    t_now = (step_idx + 1) * dt

    n_active_after = 0
    n_new_goal = 0
    n_new_eaten = 0

    # Loop all boids once; skip dead ones
    for i in range(boid_pos.shape[0]):
        if not alive[i]:
            continue
        if reached[i] or eaten[i]:
            continue

        # --- goal check ---
        dx = boid_pos[i, 0] - gx
        dy = boid_pos[i, 1] - gy
        dz = boid_pos[i, 2] - gz
        d2g = dx*dx + dy*dy + dz*dz

        if d2g <= gr2:
            reached[i] = True
            alive[i] = False
            t_reach[i] = t_now
            n_new_goal += 1
            continue

        # --- Predator capture check ---
        if pred_count > 0:
            captured = False
            for j in range(pred_count):
                dxp = boid_pos[i, 0] - pred_pos[j, 0]
                dyp = boid_pos[i, 1] - pred_pos[j, 1]
                dzp = boid_pos[i, 2] - pred_pos[j, 2]
                d2p = dxp*dxp + dyp*dyp + dzp*dzp
                if d2p <= er2:
                    captured = True
                    break

            if captured:
                eaten[i] = True
                alive[i] = False
                boid_vel[i, 0] = 0.0
                boid_vel[i, 1] = 0.0
                boid_vel[i, 2] = 0.0
                n_new_eaten += 1
                continue

        # Still alive and not reached/eaten
        n_active_after += 1

    return n_active_after, n_new_goal, n_new_eaten

@njit(cache=True, fastmath=True)
def mean_time_to_goal(t_reach, reached):
    s = 0.0
    c = 0
    for i in range(t_reach.shape[0]):
        if reached[i]:
            v = t_reach[i]
            if not math.isnan(v):
                s += v
                c += 1
    if c == 0:
        return np.nan
    return s / c

@njit(cache=True, fastmath=True)
def count_reached_eaten(reached, eaten):
    n_goal = 0
    n_eaten = 0
    for i in range(reached.shape[0]):
        if reached[i]:
            n_goal += 1
        if eaten[i]:
            n_eaten += 1
    return n_goal, n_eaten


@dataclass
class EpisodeMetrics:
    frac_goal: float
    avg_time_to_goal: float
    frac_eaten: float
    diversity_entropy: float

@njit(cache=True, fastmath=True)
def heading_entropy(vel, alive_mask, n_az=12, n_el=6):
    # count bins
    n_bins = n_az * n_el
    counts = np.zeros(n_bins, dtype=np.int32)

    two_pi = 2.0 * math.pi
    inv_two_pi = 1.0 / two_pi
    inv_pi = 1.0 / math.pi

    total = 0

    for i in range(vel.shape[0]):
        if not alive_mask[i]:
            continue

        vx = vel[i, 0]
        vy = vel[i, 1]
        vz = vel[i, 2]
        sp2 = vx*vx + vy*vy + vz*vz
        if sp2 <= 1e-24:
            continue

        sp = math.sqrt(sp2)
        vx /= sp
        vy /= sp
        vz /= sp

        az = math.atan2(vy, vx)
        if vz > 1.0:
            vz = 1.0
        elif vz < -1.0:
            vz = -1.0
        el = math.asin(vz)

        az_bin = int(math.floor((az + math.pi) * inv_two_pi * n_az))
        el_bin = int(math.floor((el + (math.pi / 2.0)) * inv_pi * n_el))

        if az_bin < 0:
            az_bin = 0
        elif az_bin >= n_az:
            az_bin = n_az - 1

        if el_bin < 0:
            el_bin = 0
        elif el_bin >= n_el:
            el_bin = n_el - 1

        idx = el_bin * n_az + az_bin
        counts[idx] += 1
        total += 1

    if total == 0:
        return 0.0

    H = 0.0
    inv_total = 1.0 / total
    for b in range(n_bins):
        c = counts[b]
        if c > 0:
            p = c * inv_total
            H -= p * math.log(p)

    Hmax = math.log(n_bins)
    return H / (Hmax + 1e-12)


def _sample_point_in_box(rng: np.random.Generator, low: float, high: float) -> np.ndarray:
    return rng.uniform(low, high, size=(3,)).astype(np.float32)

def _non_overlapping_centers(rng: np.random.Generator,
                             k: int,
                             bound: float,
                             margin: float,
                             min_sep: float,
                             avoid_points: Tuple[np.ndarray, ...]) -> np.ndarray:
    centers = np.zeros((k, 3), dtype=np.float32)
    placed = 0
    max_tries = 20000
    tries = 0
    while placed < k and tries < max_tries:
        tries += 1
        c = _sample_point_in_box(rng, margin, bound - margin)
        ok = True
        for ap in avoid_points:
            if np.linalg.norm(c - ap) < min_sep:
                ok = False
                break
        if not ok:
            continue
        for j in range(placed):
            if np.linalg.norm(c - centers[j]) < min_sep:
                ok = False
                break
        if not ok:
            continue
        centers[placed] = c
        placed += 1

    if placed < k:
        for i in range(placed, k):
            centers[i] = _sample_point_in_box(rng, margin, bound - margin)
    return centers

class FishGoalEnv(gym.Env):
    """Parameter-optimization RL environment.

    One `step(action)` runs a full rollout using the action as behavior scalars.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        boid_count: int = 200,
        pred_count: int = 8,
        bound: float = 40.0,
        max_steps: int = 2000,
        dt: float = 0.01,
        start_spread: float = 3.0,
        avoid_radius: float = 6.0,
        eat_radius: float = 2.0,
        goal_radius: float = 2.0,
        seed: int = 0,
        # reward weights
        w_goal: float = 1.0,
        w_eaten: float = 1.0,
        w_time: float = 0.2,
        w_div: float = 0.1,
        # fixed sim params (you can override)
        rule_scalar: float = 1.0,
        rule_scalar_p: float = 1.0,
        max_speed: float = 0.18,
        max_speed_p: float = 0.28,
        sep_r: float = 1.6,
        ali_r: float = 4.0,
        coh_r: float = 5.5,
        pred_avoid_r: float = 6.0,
        rand_wavelen_scalar: float = 0.01,
        attack_s: float = 1.0,
        doAnimation: bool = False,
        returnTrajectory: bool = False,
        start =  np.array([6.0, 20.0, 20.0], dtype=np.float32),
        goal =  np.array([34.0, 20.0, 20.0], dtype=np.float32),
        obs_centers =  np.array([[20.0, 20.0, 20.0],
                                          [12.0, 28.0, 22.0],
                                          [28.0, 14.0, 26.0]], dtype=np.float32),
        obs_radius: float = 3.5,
    ):
        super().__init__()

        self.start = start
        self.goal = goal
        self.obs_centers = obs_centers
        self.obs_radius = obs_radius

        self.boid_count = int(boid_count)
        self.pred_count = int(pred_count)

        self.bound = float(bound)
        self.max_steps = int(max_steps)
        self.dt = float(dt)

        self.start_spread = float(start_spread)
        self.avoid_radius = float(avoid_radius)
        self.eat_radius = float(eat_radius)
        self.goal_radius = float(goal_radius)

        self.rule_scalar = float(rule_scalar)
        self.rule_scalar_p = float(rule_scalar_p)
        self.max_speed = float(max_speed)
        self.max_speed_p = float(max_speed_p)

        self.sep_r = float(sep_r)
        self.ali_r = float(ali_r)
        self.coh_r = float(coh_r)
        self.pred_avoid_r = float(pred_avoid_r)
        self.rand_wavelen_scalar = float(rand_wavelen_scalar)
        self.attack_s = float(attack_s)

        self._rng = np.random.default_rng(int(seed))

        self.w_goal = float(w_goal)
        self.w_eaten = float(w_eaten)
        self.w_time = float(w_time)
        self.w_div = float(w_div)

        # Action: 9 scalars (sep, ali, coh, bnd, rand, pred_avoid, obs_avoid, goal_gain, obs_gain)
        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(9,), dtype=np.float32)

        self._last_obs: Optional[np.ndarray] = None
        self._episode_seed: Optional[int] = None

        self._alive = np.empty((self.boid_count,), dtype=np.bool_)
        self._reached = np.empty((self.boid_count,), dtype=np.bool_)
        self._eaten = np.empty((self.boid_count,), dtype=np.bool_)
        self._t_reach = np.empty((self.boid_count,), dtype=np.float32)

        self.doAnimation = doAnimation
        self.returnTrajectory = returnTrajectory
        if self.returnTrajectory:
            self.trajectory_boid_pos = np.empty((self.max_steps, self.boid_count, 3), dtype=np.float32)
            self.trajectory_boid_vel = np.empty((self.max_steps, self.boid_count, 3), dtype=np.float32)
        self._warmup()

    def _warmup(self):
        (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
         pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
         pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
         seed_arr) = _init_agents(8, 2, self.start, 1.0, seed=0.123)

        step_sim(
            boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
            pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
            pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
            seed_arr, self.dt,
            bound_size=self.bound,
            boid_count=8,
            pred_count=2,
            rule_scalar=self.rule_scalar,
            rule_scalar_p=self.rule_scalar_p,
            max_speed=self.max_speed,
            max_speed_p=self.max_speed_p,
            sep_r=self.sep_r,
            ali_r=self.ali_r,
            coh_r=self.coh_r,
            pred_avoid_r=self.pred_avoid_r,
            obs_avoid_r=self.avoid_radius,
            sep_s=1.0,
            ali_s=1.0,
            coh_s=1.0,
            bnd_s=1.0,
            rand_s=0.1,
            pred_avoid_s=1.0,
            obs_avoid_s=1.0,
            attack_s=self.attack_s,
            rand_wavelen_scalar=self.rand_wavelen_scalar,
            obs_radius=self.obs_radius,
            goal_gain=0.0,
            goal_x=self.goal[0],
            goal_y=self.goal[1],
            goal_z=self.goal[2],
            obs_centers=self.obs_centers,
        )

        if self.doAnimation:
            self.fig = plt.figure(figsize=(11, 8))
            self.ax = self.fig.add_subplot(111, projection="3d")
            plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.98)

            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")
            self.ax.set_xlim3d(0, self.bound)
            self.ax.set_ylim3d(0, self.bound)
            self.ax.set_zlim3d(0, self.bound)

            self.boid_scatter = self.ax.scatter(boid_pos[:, 0],
                              boid_pos[:, 1],
                              boid_pos[:, 2],
                              s=6, depthshade=False)

            self.pred_scatter = self.ax.scatter(pred_pos[:, 0],
                                    pred_pos[:, 1],
                                    pred_pos[:, 2],
                                    s=30, marker="^", depthshade=False, color="red")

            self.goal_scatter = self.ax.scatter([self.goal[0]], [self.goal[1]], [self.goal[2]],
                                    s=80, marker="*", depthshade=False)
            
            for (cx, cy, cz) in self.obs_centers:
                r = self.obs_radius
                u = np.linspace(0, 2 * np.pi, 12)
                v = np.linspace(0, np.pi, 12)
                x = cx + r * np.outer(np.cos(u), np.sin(v))
                y = cy + r * np.outer(np.sin(u), np.sin(v))
                z = cz + r * np.outer(np.ones_like(u), np.cos(v))
                for k in range(0, x.shape[0], 4):
                    self.ax.plot(x[k, :], y[k, :], z[k, :], linewidth=0.8)[0]
                for k in range(0, x.shape[1], 3):
                    self.ax.plot(x[:, k], y[:, k], z[:, k], linewidth=0.8)[0]
            self.ax.view_init(elev=20, azim=25)
            plt.ion()
            plt.show()

    def _sample_start(self) -> np.ndarray:
        margin = max(1.0, self.obs_radius + self.avoid_radius + 1.0)
        return _sample_point_in_box(self._rng, margin, self.bound - margin)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        episode_seed = int(self._rng.integers(0, 2**31 - 1))
        self._episode_seed = episode_seed

    def step(self, action):
        if self._episode_seed is None:
            raise RuntimeError("Call reset() before step().")

        metrics, info = self._rollout_episode(np.asarray(action, dtype=np.float32).reshape(-1))

        time_pen = 0.0
        if not math.isnan(metrics.avg_time_to_goal):
            time_pen = metrics.avg_time_to_goal / (self.max_steps * self.dt + 1e-12)

        reward = (
            self.w_goal * metrics.frac_goal
            - self.w_eaten * metrics.frac_eaten
            - self.w_time * time_pen
            + self.w_div * metrics.diversity_entropy
        )

        info.update({
            "frac_goal": metrics.frac_goal,
            "avg_time_to_goal": metrics.avg_time_to_goal,
            "frac_eaten": metrics.frac_eaten,
            "diversity_entropy": metrics.diversity_entropy,
            "reward": float(reward),
        })

        obs = self._last_obs if self._last_obs is not None else np.zeros((6,), dtype=np.float32)

        self._episode_seed = None

        return obs, float(reward), True, False, info

    def _rollout_episode(self, action: np.ndarray) -> Tuple[EpisodeMetrics, Dict]:
        if action.shape[0] != 8:
            raise ValueError(f"action must have shape (8,), got {action.shape}")

        seed = int(self._episode_seed)
        rng = np.random.default_rng(seed)

        start = self.start
        goal = self.goal

        (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
         pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
         pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
         seed_arr) = _init_agents(
            total_boids=self.boid_count,
            total_preds=self.pred_count,
            start_xyz=start,
            start_spread=self.start_spread,
            seed=float((seed % 1000000) / 1000000.0 + 0.123),
        )

        if self.pred_count > 0:
            margin_p = max(1.0, 1.5 * self.eat_radius)
            pred_centers = _non_overlapping_centers(
                rng, self.pred_count, self.bound, self.bound / 10,
                min_sep=2.0 * self.eat_radius,
                avoid_points=(start, goal),
            )
            pred_pos[:] = pred_centers

        sep_s = float(action[0])
        ali_s = float(action[1])
        coh_s = float(action[2])
        bnd_s = float(action[3])
        rand_s = float(action[4])
        pred_avoid_s = float(action[5])
        obs_avoid_s = float(action[6])
        goal_gain = float(action[7])

        self._alive.fill(True)
        self._reached.fill(False)
        self._eaten.fill(False)
        self._t_reach.fill(np.nan)

        n_goal = 0
        n_eaten = 0

        for step in range(self.max_steps):
            step_sim(
                boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
                pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
                pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
                seed_arr, self.dt,
                bound_size=self.bound,
                boid_count=self.boid_count,
                pred_count=self.pred_count,
                rule_scalar=self.rule_scalar,
                rule_scalar_p=self.rule_scalar_p,
                max_speed=self.max_speed,
                max_speed_p=self.max_speed_p,
                sep_r=self.sep_r,
                ali_r=self.ali_r,
                coh_r=self.coh_r,
                pred_avoid_r=self.pred_avoid_r,
                obs_avoid_r=self.avoid_radius,
                sep_s=sep_s,
                ali_s=ali_s,
                coh_s=coh_s,
                bnd_s=bnd_s,
                rand_s=rand_s,
                pred_avoid_s=pred_avoid_s,
                obs_avoid_s=obs_avoid_s,
                attack_s=self.attack_s,
                rand_wavelen_scalar=self.rand_wavelen_scalar,
                obs_radius=self.obs_radius,
                goal_gain=goal_gain,
                goal_x=goal[0],
                goal_y=goal[1],
                goal_z=goal[2],
                obs_centers=self.obs_centers,
            )
            
            n_active, n_new_goal, n_new_eaten = update_events_numba(
                boid_pos, boid_vel,
                pred_pos, self.pred_count,
                self._alive, self._reached, self._eaten, self._t_reach,
                goal, self.goal_radius, self.eat_radius,
                step, self.dt
            )
            n_goal += n_new_goal
            n_eaten += n_new_eaten
            if n_active == 0:
                break

            if self.doAnimation and plt.fignum_exists(self.fig.number):
                self.boid_scatter._offsets3d = (boid_pos[:, 0], boid_pos[:, 1], boid_pos[:, 2])
                self.pred_scatter._offsets3d = (pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2])
                plt.draw()
                plt.pause(0.001)

            if self.returnTrajectory:
                self.trajectory_boid_pos[step, :, :] = boid_pos
                self.trajectory_boid_vel[step, :, :] = boid_vel

        frac_goal = n_goal / float(self.boid_count)
        frac_eaten = n_eaten / float(self.boid_count)

        avg_time_to_goal = float(mean_time_to_goal(self._t_reach, self._reached))  # nan if none
        alive_mask = ~self._eaten
        diversity = float(heading_entropy(boid_vel, alive_mask))

        metrics = EpisodeMetrics(
            frac_goal=frac_goal,
            avg_time_to_goal=avg_time_to_goal,
            frac_eaten=frac_eaten,
            diversity_entropy=diversity,
        )

        info = {
            "goal": self.goal,
            "start": self.start,
            "obstacles_used": int(self.obs_centers.shape[0]),
            "predators": int(self.pred_count),
            "reached_count": n_goal,
            "eaten_count": n_eaten,
            "steps_executed": step + 1 if self.max_steps > 0 else 0,
        }

        if self.returnTrajectory:
            info["trajectory_boid_pos"] = self.trajectory_boid_pos
            info["trajectory_boid_vel"] = self.trajectory_boid_vel
        else:
            info["trajectory_boid_pos"] = None
        return metrics, info

if __name__ == "__main__":
    env = FishGoalEnv(boid_count=600, pred_count=4, max_steps=500, dt=1, doAnimation = True, returnTrajectory = False)
    env.reset(seed=0)
    """
    Parameters order:
    0: separation scalar
    1: alignment scalar
    2: cohesion scalar
    3: boundary scalar
    4: randomness scalar
    5: predator avoidance scalar
    6: obstacle avoidance scalar
    7: goal attraction gain
    """
    action = np.array([1.0, #0: separation scalar
                       1.0, #1: alignment scalar
                       1.0, #2: cohesion scalar
                       1.0, #3: boundary scalar
                       1.0, #4: randomness scalar
                       10.0, #5: predator avoidance scalar
                       1.0, #6: obstacle avoidance scalar
                       0.3, #7: goal attraction gain
                       ], dtype=np.float32)
    
    load_theta = True
    if load_theta:
        theta_path = "save/best_policy.pkl"
        action = pickle.load(open(theta_path, "rb"))['best_theta']
        print("Loaded theta:", action)

    t = []
    for _ in range(10):
        t0 = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        env.reset(seed=0)
        t1 = time.time()
        t.append(t1 - t0)
    print(f"Median step time: {np.median(t)*1000.0:.2f} ms")