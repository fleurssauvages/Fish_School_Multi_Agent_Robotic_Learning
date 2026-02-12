import numpy as np
from scipy.spatial.transform import Rotation as R
from qpsolvers import solve_qp
import scipy.sparse as sp
from scipy.linalg import block_diag
from scipy.interpolate import CubicSpline
from numba import njit
import math

""" ------------------------
MPC problem assembly based from 
Alberto, Nicolas Torres, et al. "Linear Model Predictive Control in SE (3) for online trajectory planning in dynamic workspaces." (2022).
https://hal.science/hal-03790059/document
------------------------"""

#------------------------
# Helpers for constraints
#------------------------
def build_spline(p0, p1, T=1.0):
    """
    p0, p1: (3,) start and goal
    Returns spline function s(t) with t in [0, T]
    """
    t = np.array([0.0, T])
    p = np.vstack([p0, p1])
    return CubicSpline(t, p, axis=0)

def sample_spline(spline, horizon, dt):
    ts = np.linspace(0, horizon*dt, horizon)
    return np.array([spline(t) for t in ts])

@njit(cache=True, fastmath=True)
def closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = ab[0]*ap[0] + ab[1]*ap[1] + ab[2]*ap[2]
    d2 = ac[0]*ap[0] + ac[1]*ap[1] + ac[2]*ap[2]
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = ab[0]*bp[0] + ab[1]*bp[1] + ab[2]*bp[2]
    d4 = ac[0]*bp[0] + ac[1]*bp[1] + ac[2]*bp[2]
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1*d4 - d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3 + 1e-12)
        return a + v * ab

    cp = p - c
    d5 = ab[0]*cp[0] + ab[1]*cp[1] + ab[2]*cp[2]
    d6 = ac[0]*cp[0] + ac[1]*cp[1] + ac[2]*cp[2]
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5*d2 - d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6 + 1e-12)
        return a + w * ac

    va = d3*d6 - d5*d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-12)
        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc + 1e-12)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w

@njit(cache=True, fastmath=True)
def closest_point_on_mesh(p, verts, faces):
    best_d2 = 1e30
    best_q = np.zeros(3, dtype=np.float32)

    for fi in range(faces.shape[0]):
        ia = faces[fi, 0]
        ib = faces[fi, 1]
        ic = faces[fi, 2]

        a = verts[ia]
        b = verts[ib]
        c = verts[ic]

        q = closest_point_on_triangle(p, a, b, c)
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        dz = p[2] - q[2]
        d2 = dx*dx + dy*dy + dz*dz

        if d2 < best_d2:
            best_d2 = d2
            best_q[0] = q[0]
            best_q[1] = q[1]
            best_q[2] = q[2]

    # direction normal = from surface to point
    vx = p[0] - best_q[0]
    vy = p[1] - best_q[1]
    vz = p[2] - best_q[2]
    nrm = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-12

    n = np.zeros(3, dtype=np.float32)
    n[0] = vx / nrm
    n[1] = vy / nrm
    n[2] = vz / nrm

    return best_q, n, math.sqrt(best_d2)

def corridor_planes_from_spline(p_ref, obstacles, margin=0.05, max_dist=None):
    """
    p_ref: (horizon,3)
    margin: safety margin
    max_dist: optional, skip obstacle if reference point is farther than this (speeds up)
    Returns A_x, b_x so that A_x X <= b_x, with X stacked (6*horizon,)
    """
    horizon = p_ref.shape[0]
    A_x = []
    b_x = []

    for k in range(horizon):
        p_k = p_ref[k]

        for obs in obstacles:
            V = np.asarray(obs[0], dtype=float)
            F = np.asarray(obs[1], dtype=np.int32)

            p_surf, n, d = closest_point_on_mesh(p_k, V, F)
            if (max_dist is not None) and (d > max_dist):
                continue
            if np.linalg.norm(n) < 1e-6:
                continue

            # inequality: n^T x_k >= n^T p_surf + margin
            # -> -n^T x_k <= -(n^T p_surf + margin)
            row = np.zeros(6 * horizon)
            row[6*k : 6*k+3] = -n
            rhs = -(float(np.dot(n, p_surf)) + float(margin))

            A_x.append(row)
            b_x.append(rhs)

    if len(A_x) == 0:
        return None, None
    return np.vstack(A_x), np.array(b_x)

class LinearMPCController:
    def __init__(self, horizon=10, dt=0.05, gamma=1e-3, u_min=None, u_max=None):
        self.n = 6
        self.horizon = horizon
        self.dt = dt
        self.gamma = gamma
        self.u_min = u_min
        self.u_max = u_max
        self.H = np.zeros((self.n*self.horizon, self.n*self.horizon)) # Hessian
        self.g = np.zeros((self.n*self.horizon, self.n*self.horizon)) # Gradient
        self.A = np.zeros((0, self.n*self.horizon))  # Inequality onstraints
        self.b = np.zeros(0)  # Inequality constraint bounds
        self.eqA = np.zeros((0, self.n*self.horizon))  # Equality constraints
        self.eqb = np.zeros(0) # Equality constraint bounds
        self.lb = -np.ones(self.n) * np.inf  # Lower bounds
        self.ub = np.ones(self.n) * np.inf   # Upper bounds
        self.dt = dt # Time step of the controller loop / simulation
        self.solution = None
        pass
    
    def solve(self, ini_pose, des_pose_human, des_pose_auto, w, xi0=None, obstacles=None, traj=None, margin=0.05, culling_dist=1):
        """
        Solve finite-horizon linear MPC:
        minimize ||X - X_des||^2 + gamma ||U||^2
        s.t. X = A_big x0 + B_big U
        u_min <= U <= u_max (component-wise)
        x0: current state in tangent space (6,)
        xi_des: target in tangent space that is repeated along horizon (6,)
        A,B: system (6x6), here A = I6, B = dlog * dt
        h: horizon length (# time-steps)
        dt: time step (for info only)
        gamma: weight on input norm
        u_min/u_max: (6,) or arrays of length 6*h
        Returns optimal U (stacked) and the predicted X sequence.
        """

        ini_pose = np.array(ini_pose)
        des_pose_human = np.array(des_pose_human)
        des_pose_auto  = np.array(des_pose_auto)

        # current pose inverse
        Xc_inv = np.linalg.inv(ini_pose)

        # tangent at current
        xi_current = se3_log(np.eye(4))

        # two desired targets in SAME tangent space
        xi_h = se3_log(Xc_inv @ des_pose_human)
        xi_a = se3_log(Xc_inv @ des_pose_auto)

        # blend (scalar weight)
        w = float(np.clip(w, 0.0, 1.0))
        xi_des = w * xi_h + (1.0 - w) * xi_a

        dlog = compute_dlog_approx(xi_current)

        # big matrices (unchanged)
        A_big = np.eye(self.n * self.horizon)
        singleB = compute_dlog_approx(xi_current) * self.dt
        B_big = np.kron(np.tril(np.ones((self.horizon, self.horizon))), singleB)

        Xd = np.tile(xi_des, self.horizon)
        Xprev = np.tile(xi_current, self.horizon)
        # Cost weights
        Q = np.eye(6)       # weight for pose error
        Q_big = block_diag(*[Q for _ in range(self.horizon-1)])  # (12h x 12h)
        Q_big = block_diag(Q_big, np.eye(6) * 150)  # Final state also considered, strong weight on final pose and velocity

        # Cost: (A_big x0 + B_big U - Xd)^T (A_big x0 + B_big U - Xd) + gamma * U^T U
        # Variable is stacked X_pred(= A_big @ np.tile(x0, h) + B_big @ U) and U
        # Formulation according to QP, see QP for more details
        self.H = 2 * (B_big.T @ Q_big @ B_big + self.gamma * np.eye(self.n*self.horizon))
        self.g = 2 * (B_big.T @ Q_big @ (A_big @ Xprev - Xd))
        
        # Constraints on U
        if self.u_min is not None:
            self.lb = np.tile(self.u_min, self.horizon)
        if self.u_max is not None:
            self.ub = np.tile(self.u_max, self.horizon)
        
        # --- reset each solve (safer) ---
        self.A = np.zeros((0, self.n*self.horizon))
        self.b = np.zeros(0)

        if xi0 is not None and self.u_min is not None and self.u_max is not None:
            # U0 <= xi0 + du0 where du0 the deviation allowed from current velocity, here du0 = u_max
            # -U0 <= -(xi0 - du0)
            G_add = np.zeros((12, self.n * self.horizon))
            G_add[0:6, 0:6] = np.eye(6)
            G_add[6:12, 0:6] = -np.eye(6)
            h_add = np.hstack([xi0 + self.ub[:6], -(xi0 - self.ub[:6])])

            # Append to existing inequalities (self.A, self.b)
            self.A = G_add
            self.b = h_add

        # build spline in world
        if obstacles is not None and traj is not None and np.linalg.norm(ini_pose[:3,3] - des_pose_auto[:3,3]) > 0.01:
            p_world = traj[:self.horizon].copy()
            # --- convert reference + obstacles to local/tangent-ish coordinates ---
            R0 = ini_pose[:3,:3]
            p0 = ini_pose[:3,3]
            p_ref = (R0.T @ (p_world - p0).T).T

            obstacles_local = []
            for obs in obstacles:
                V = np.asarray(obs[0], dtype=float)
                F = np.asarray(obs[1], dtype=np.int32)

                centroid = V.mean(axis=0)
                bound_r = np.max(np.linalg.norm(V - centroid, axis=1))
                if np.linalg.norm(centroid - p0) - bound_r < culling_dist:
                    V_local = (R0.T @ (V - p0).T).T
                    obstacles_local.append([V_local, F])

            A_x, b_x = corridor_planes_from_spline(p_ref, obstacles_local, margin=margin)

            if A_x is not None:
                G_corr = A_x @ B_big
                h_corr = b_x - A_x @ (A_big @ Xprev)
                self.A = np.vstack([self.A, G_corr])
                self.b = np.hstack([self.b, h_corr])
                
        # Solve QP
        Uopt = solve_qp(sp.csc_matrix(self.H), self.g, G=sp.csc_matrix(self.A), h=self.b, \
            A=sp.csc_matrix(self.eqA), b=self.eqb, lb=self.lb, ub=self.ub, solver="osqp", initvals=self.solution, verbose=False)
        self.solution = Uopt
        
        if Uopt is None:
            return None, None, None
        
        # Reconstruct predicted X sequence
        Xopt = (A_big @ Xprev + B_big @ Uopt).reshape(self.horizon, self.n)
        
        # Convert predicted x sequence back to poses
        T = ini_pose
        poses = [T]
        for i in range(self.horizon):
            u_i = Uopt[self.n*i:self.n*(i+1)]
            T = T @ se3_exp(dlog @ u_i * self.dt)
            poses.append(T)
        return Uopt, Xopt, poses
    
# ------------------------
# Example
# ------------------------

def example():
    # Current pose: identity slightly translated/rotated
    cur_pos = np.array([0.4, 0.0, 0.4])
    cur_quat = np.array([0.0, 0.0, 0.0, 1.0])  # x,y,z,w
    T_ini = pose_to_matrix(cur_pos, cur_quat)
    p, q = matrix_to_pose(T_ini)
    print(f" Initial: pos = {p}, quat = {q}")

    # Target pose: rotated 45deg around z, translated
    tgt_pos = np.array([0.4, 0.0, 1.5])
    tgt_quat = R.from_euler('z', 45, degrees=True).as_quat()
    T_des = pose_to_matrix(tgt_pos, tgt_quat)
    p, q = matrix_to_pose(T_des)
    print(f" Desired: pos = {p}, quat = {q}")

    # run one MPC step
    lmpc_solver = LinearMPCController(horizon=100, dt=0.05, gamma = 0.001,
                                    u_min=np.array([-0.5, -0.5, -0.5, -1.0, -1.0, -1.0]),
                                    u_max=np.array([ 0.5,  0.5,  0.5,  1.0,  1.0,  1.0]))
    Uopt, Xopt, poses = lmpc_solver.solve(T_ini, T_des)
    
    print("\nPredicted poses:")
    for i, T in enumerate(poses):
        p, q = matrix_to_pose(T)
        print(f" step {i}: pos = {p}, quat = {q}")

# ------------------------
# Utils
# ------------------------

def skew(v):
    """Return 3x3 skew-symmetric matrix for vector v (3,)."""
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]], dtype=float)

def so3_left_jacobian(phi):
    """
    Left Jacobian J(φ) of SO(3) (3x3).
    phi: vector (3,) rotation vector (axis * angle)
    """
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * skew(phi) + (1.0/12.0) * (skew(phi) @ skew(phi))
    axis_hat = skew(phi / theta)
    J = (np.eye(3)
         + (1 - np.cos(theta)) / (theta**2) * axis_hat
         + (theta - np.sin(theta)) / (theta**3) * (axis_hat @ axis_hat))
    return J

def so3_left_jacobian_inv(phi):
    """Inverse of the left Jacobian J^{-1}(phi)."""
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        # series expansion
        return np.eye(3) - 0.5 * skew(phi) + (1.0/12.0) * (skew(phi) @ skew(phi))
    axis = phi / theta
    A = 0.5 * skew(axis)
    cot_term = (1 / theta - 0.5 / np.tan(theta / 2))
    return np.eye(3) - 0.5 * skew(phi) + cot_term * (skew(phi) @ skew(phi)) / (theta**2)

def se3_exp(xi):
    """
    Exponential map from se(3) (6-vector) to SE(3) homogeneous matrix (4x4).
    xi = [v (3,), omega (3,)] where omega is rotation vector (axis*angle).
    """
    v = xi[:3]
    omega = xi[3:]
    theta = np.linalg.norm(omega)
    Rm = R.from_rotvec(omega).as_matrix()
    if theta < 1e-8:
        J = np.eye(3) + 0.5 * skew(omega) + (1.0/6.0) * (skew(omega) @ skew(omega))
    else:
        J = so3_left_jacobian(omega)  # V matrix
    p = J @ v
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = p
    return T

def se3_log(T):
    """
    Log map from SE(3) (4x4 matrix) to se(3) 6-vector [v, omega].
    Uses: omega = log(R) (rotvec), v = J^{-1}(omega) * p
    """
    Rm = T[:3, :3]
    p = T[:3, 3]
    rot = R.from_matrix(Rm)
    omega = rot.as_rotvec()
    J_inv = so3_left_jacobian_inv(omega)
    v = J_inv @ p
    xi = np.concatenate([v, omega])
    return xi

def pose_to_matrix(position, quaternion):
    """Return 4x4 homogeneous matrix from position (3,) and quaternion (x,y,z,w)."""
    Rm = R.from_quat(quaternion).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = position
    return T

def matrix_to_pose(T):
    """Return (pos, quat) from 4x4 matrix. quat as (x,y,z,w)."""
    pos = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat

def compute_dlog_approx(xi):
    """
    Approximate dlog at xi (6,) by block-diagonal of J_inv(omega)
    where xi = [v, omega].
    This yields a 6x6 matrix approximating mapping such that x_{k+1} = x_k + dlog * u * dt
    """
    omega = xi[3:]
    J_inv = so3_left_jacobian_inv(omega)
    # Block diag: translational & rotational
    dlog = np.zeros((6,6))
    dlog[:3, :3] = J_inv
    dlog[3:, 3:] = J_inv
    return dlog

if __name__ == "__main__":
    example()
