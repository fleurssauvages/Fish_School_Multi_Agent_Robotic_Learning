import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp


class MultiDroneCBFQP:
    """
    Multi-drone CBF-QP safety filter.

    Decision variables:
        v = [v0x,v0y,v0z, v1x,v1y,v1z, ...]  in R^(3N)
    Optionally augmented with slacks s >= 0 for each inequality constraint.

    Objective:
        minimize sum_i ||v_i - v_nom_i||^2  + rho_slack * ||s||^2 (if enabled)

    Constraints:
        - box constraints on velocity components: -v_max <= v <= v_max
        - obstacle CBF: for each drone i and obstacle o (sphere):
            h = ||p_i - c||^2 - (r + margin)^2
            2(p_i-c)^T v_i + alpha*h >= 0
        - inter-drone CBF: for each pair (i,j):
            h = ||p_i - p_j||^2 - d_safe^2
            2(p_i-p_j)^T (v_i - v_j) + alpha*h >= 0
    """

    def __init__(self, num_drones: int, dt: float):
        self.N = int(num_drones)
        self.dt = float(dt)
        self.nv = 3 * self.N  # stacked translational velocities
        self._last_solution = None

    @staticmethod
    def _as_obstacles_list(obstacles):
        """Normalize obstacle input to list of dicts with keys center,radius."""
        if obstacles is None:
            return []
        out = []
        for obs in obstacles:
            c = np.asarray(obs["center"], dtype=float).reshape(3)
            r = float(obs["radius"])
            out.append({"center": c, "radius": r})
        return out

    def solve(
        self,
        v_nom,
        positions,
        obstacles,
        v_max,
        d_obs_margin=0.10,
        d_safe=0.30,
        alpha_obs=2.0,
        alpha_pair=2.0,
        use_slack=True,
        rho_slack=1e4,
        solver="osqp",
    ):
        """
        Parameters (match your desired call):
            v_nom:      (N,3) nominal velocities (m/s), world frame
            positions:  (N,3) current positions (m), world frame
            obstacles:  list of {'center': (3,), 'radius': float} in world frame
            v_max:      scalar speed bound per component (m/s) (box constraints)
            d_obs_margin: inflate obstacle radius by this margin (m)
            d_safe:     min inter-drone separation distance (m)
            alpha_obs:  CBF gain for obstacle constraints
            alpha_pair: CBF gain for inter-drone constraints
            use_slack:  add slacks to guarantee feasibility
            rho_slack:  slack penalty (bigger => fewer violations)
        Returns:
            v_opt: (N,3)
            slack: (m,) if use_slack else None
        """
        # ---- sanitize inputs ----
        v_nom = np.asarray(v_nom, dtype=float).reshape(self.N, 3)
        pos = np.asarray(positions, dtype=float).reshape(self.N, 3)
        obstacles = self._as_obstacles_list(obstacles)

        v_max = float(v_max)
        d_obs_margin = float(d_obs_margin)
        d_safe = float(d_safe)
        alpha_obs = float(alpha_obs)
        alpha_pair = float(alpha_pair)
        rho_slack = float(rho_slack)

        # ---- build base QP: min ||v - v_nom||^2 ----
        # 0.5 v^T H v + f^T v  with H = 2I, f = -2 v_nom
        H = 2.0 * np.eye(self.nv)
        f = -2.0 * v_nom.reshape(-1)

        # ---- inequality constraints G v <= h ----
        G_rows = []
        h_rows = []

        if alpha_obs > 0.0:
            # (A) Obstacle CBF constraints (one per drone per obstacle)
            # 2(p-c)^T v_i + alpha*h >= 0
            # => -2(p-c)^T v_i <= alpha*h
            for i in range(self.N):
                p_i = pos[i]
                for obs in obstacles:
                    c = obs["center"]
                    r = obs["radius"] + d_obs_margin
                    d = p_i - c
                    h_val = float(d @ d - r * r)

                    a = -2.0 * d  # row so that a^T v_i <= alpha*h
                    b = alpha_obs * h_val

                    row = np.zeros(self.nv)
                    row[3 * i : 3 * i + 3] = a
                    G_rows.append(row)
                    h_rows.append(b)

        if alpha_pair > 0.0:
            # (B) Inter-drone CBF constraints (one per pair)
            # 2(p_i-p_j)^T (v_i - v_j) + alpha*h >= 0
            # => (-2d^T) v_i + (2d^T) v_j <= alpha*h
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    d = pos[i] - pos[j]
                    h_val = float(d @ d - d_safe * d_safe)

                    row = np.zeros(self.nv)
                    row[3 * i : 3 * i + 3] = -2.0 * d
                    row[3 * j : 3 * j + 3] = +2.0 * d
                    G_rows.append(row)
                    h_rows.append(alpha_pair * h_val)

        if len(G_rows) == 0:
            # No constraints: just clip components
            v_opt = np.clip(v_nom, -v_max, v_max)
            return v_opt, None

        G = np.vstack(G_rows)
        h = np.asarray(h_rows, dtype=float)

        # ---- box bounds on v ----
        lb = -v_max * np.ones(self.nv)
        ub = +v_max * np.ones(self.nv)

        # ---- solve ----
        if not use_slack:
            v = solve_qp(
                sp.csc_matrix(H), f,
                G=sp.csc_matrix(G), h=h,
                lb=lb, ub=ub,
                solver=solver,
                initvals=self._last_solution,
            )
            if v is None:
                # fallback: stop
                return np.zeros((self.N, 3)), None
            self._last_solution = v
            return v.reshape(self.N, 3), None

        # ---- slack augmentation: z = [v; s],  s >= 0 ----
        m = G.shape[0]
        nZ = self.nv + m

        H_aug = np.zeros((nZ, nZ))
        H_aug[: self.nv, : self.nv] = H
        H_aug[self.nv :, self.nv :] = 2.0 * rho_slack * np.eye(m)

        f_aug = np.zeros(nZ)
        f_aug[: self.nv] = f

        # [G  -I] [v; s] <= h
        G_aug = np.zeros((m, nZ))
        G_aug[:, : self.nv] = G
        G_aug[:, self.nv :] = -np.eye(m)

        lb_aug = np.hstack([lb, np.zeros(m)])
        ub_aug = np.hstack([ub, np.inf * np.ones(m)])

        z = solve_qp(
            sp.csc_matrix(H_aug), f_aug,
            G=sp.csc_matrix(G_aug), h=h,
            lb=lb_aug, ub=ub_aug,
            solver=solver,
            initvals=None,
        )
        if z is None:
            return np.zeros((self.N, 3)), None

        v_opt = z[: self.nv].reshape(self.N, 3)
        slack = z[self.nv :]
        return v_opt, slack