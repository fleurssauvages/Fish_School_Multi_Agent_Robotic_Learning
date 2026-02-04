### Fish School / Boids-Inspired Multi-Agent Simulation and Learning for GMR Modelisation

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![RL](https://img.shields.io/badge/reinforcement--learning-probabilistic-green)](#)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

This repository implements a **multi-agent collective motion framework** inspired by **fish schools and boid models**, and extended with **probabilistic reinforcement learning** and **latent policy parameters** (θ) for drones simulation.

The codebase supports:
- Emergent collective motion without explicit goals (free roaming)
- Goal-directed collective navigation
- Replayable environments to animate or re-execute a learned parameter vector θ
- Reinforcement learning over low-dimensional behavior parameters
- Live Prediction using GMM-GMR

---

## 🐠 Biological & Algorithmic Inspiration

This work is inspired by classical and modern studies of collective schooling, in particular:

- **Boids model** (Reynolds, 1987)
- Erik Martin Vetemaa Bachelor Thesis, https://github.com/vetemaa/fish-simulation and https://thesis.cs.ut.ee/a62fdbba-061c-475d-851e-8c49d61b09df

---

## 📂 Project Structure

env/
- env.py — Core multi-agent environment
- env.py can be run as main file to visualize a given action vector parameter.

scripts/
- free_roam.py — Emergent motion without a goal
- goal_roam.py — Goal-directed collective navigation with tunable actions using sliders
- train_rl.py — Reinforcement learning over θ to learn the "best" behaviour given a reward
- train_gmr.py - GMR to predict and extract a behaviour
- train_gmr_multiple_goals.py - Extended to multiple goals

---

## 🎥 Demos

<div align="center">

### 🔹Free-roam: Fishes don't have any goal besides avoid obstacles and predators
<img src="images/free_roam.gif" width="600" alt="traj">

### 🔹 Goal-oriented roamn: Fishes try to reach a given goal (gains can be manually tuned)
<img src="images/target_roam.gif" width="800" alt="RL">

### 🔹 Goal-oriented roam after RL-optimization
<img src="images/RL_roam.gif" width="800" alt="RL">

### 🔹 GMR Fitting
<img src="images/GMR.png" width="800" alt="GMR">

</div>
---


## 🧪 Main Scripts

### Free Roam — Emergent Collective Motion

Purely emergent behavior with no explicit goal, driven only by local interactions, similar to Erik Martin Vetemaa implementation, with modified behaviour.

### Goal Roam — Collective Navigation

Adds a global goal while preserving decentralized interactions.

### Environment & Replay
RL/env.py

Deterministic replay and visualization of learned θ.

### Reinforcement Learning
train_rl.py

Policy search over θ using probabilistic reinforcement learning using Power RL.

### GMR Fitting
train_gmr.py

Fit a GMR regression on fishes trajectories.

---

## 📜 License

MIT License — see LICENSE file.

---

## ⭐ Acknowledgments

- Craig Reynolds — Boids (1987)