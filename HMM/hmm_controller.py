import numpy as np
from dataclasses import dataclass
from scipy.special import logsumexp

# ============================================================
# Shared controller using an HMM-GMR skill model
# - infers the model command u_model
# - computes blending factor lambda from confidence
# - outputs u = (1-lambda)*u_human + lambda*u_model
# ============================================================

def _safe_log(x, eps=1e-12):
    return np.log(x + eps)

def _chol_logpdf(x, mu, cov, jitter=1e-12):
    """
    Stable log N(x | mu, cov) via Cholesky. Returns scalar.
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    cov = np.asarray(cov).copy()
    d = x.size
    cov.flat[::d+1] += jitter
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov.flat[::d+1] += 1e-6
        L = np.linalg.cholesky(cov)
    y = np.linalg.solve(L, x - mu)
    quad = float(y @ y)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

def _gmm_logpdf(x, weights, means, covars):
    """
    log sum_m w_m N(x | mu_m, cov_m)
    weights: (M,)
    means: (M,D)
    covars: (M,D,D)
    """
    M = weights.shape[0]
    logs = np.empty(M)
    for m in range(M):
        logs[m] = _safe_log(weights[m]) + _chol_logpdf(x, means[m], covars[m])
    return float(logsumexp(logs))

def _entropy(p):
    p = np.asarray(p, float)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class SharedControlDebug:
    alpha: np.ndarray              # (K,) filtered state belief at time t
    alpha_pred: np.ndarray         # (K,) predicted belief for t+1 (after transition)
    loglik_t: float                # log p(x_t | history) up to normalization constant
    entropy: float                 # entropy(alpha_pred)
    cov_trace: float               # trace(Sigma_pred)
    confidence: float              # combined confidence in [0,1]
    lam: float                     # lambda used
    u_model: np.ndarray            # (pos_dim,) command from model
    u_human: np.ndarray            # (pos_dim,) human command
    u_shared: np.ndarray           # (pos_dim,) blended command


class HMMGMRSharedController:
    """
    Shared controller built on top of an hmmlearn.GMMHMM trained on [t, pos] (time-augmented).
    It performs:
      - filtering (online) to estimate p(z_t | x_{1:t})
      - one-step prediction of p(z_{t+1} | x_{1:t})
      - GMR-style state-averaged next-position mean/cov
      - converts that to a command u_model (by default a velocity toward predicted next position)
      - sets lambda from confidence (entropy + covariance + innovation)
    """

    def __init__(
        self,
        hmmgmm,                 # your HMMGMM wrapper or directly a fitted hmmlearn.hmm.GMMHMM
        pos_dim=3,
        dt=1.0,
        use_time=True,
        time_mode="normalized",  # "normalized" expects t in [0,1]; "steps" uses step index / scaling
        time_scale=1.0,          # if time_mode="steps", t_feature = step_idx * time_scale
        # lambda shaping:
        lambda_min=0.0,
        lambda_max=1.0,
        # confidence fusion weights:
        w_entropy=1.0,
        w_cov=1.0,
        w_innov=1.0,
        # confidence -> lambda gain:
        conf_k=6.0,              # steeper -> more switch-like
        conf_center=0.5,         # where sigmoid is 0.5
        # model command gains:
        k_p=1.0,                 # u_model = k_p * (mu_next - y)/dt  (velocity-like)
        u_clip=None,             # float or array-like max abs per dimension
    ):
        self.hmm = hmmgmm.model if hasattr(hmmgmm, "model") else hmmgmm
        self.pos_dim = int(pos_dim)
        self.D = (1 if use_time else 0) + self.pos_dim
        self.dt = float(dt)
        self.use_time = bool(use_time)
        self.time_mode = time_mode
        self.time_scale = float(time_scale)

        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)

        self.w_entropy = float(w_entropy)
        self.w_cov = float(w_cov)
        self.w_innov = float(w_innov)

        self.conf_k = float(conf_k)
        self.conf_center = float(conf_center)

        self.k_p = float(k_p)
        self.u_clip = u_clip

        # state: filtered belief over states
        self._log_alpha = None
        self._step = 0

        # cache state-level GMR moments for position-only (mixture-marginalized per state)
        self._state_mu, self._state_S = self._compute_state_moments_pos()

    def reset(self):
        self._log_alpha = None
        self._step = 0

    # ---------- core pieces ----------

    def _compute_state_moments_pos(self):
        """
        For each HMM state k, marginalize its emission GMM over mixtures to get:
          state_mu[k] in R^{pos_dim}
          state_S[k]  in R^{pos_dim x pos_dim}
        Using only the position dimensions (excluding time dim).
        """
        K = self.hmm.n_components
        M = self.hmm.n_mix

        state_mu = np.zeros((K, self.pos_dim))
        state_S = np.zeros((K, self.pos_dim, self.pos_dim))

        # hmmlearn stores: weights_[k,m], means_[k,m,D], covars_[k,m,D,D]
        # Our convention: if use_time, dim0 is time, dims 1: are position.
        # If use_time=False, dims 0: are position directly.
        pos_slice = slice(1, 1 + self.pos_dim) if self.use_time else slice(0, self.pos_dim)

        for k in range(K):
            w = self.hmm.weights_[k]  # (M,)
            mu_km = self.hmm.means_[k, :, pos_slice]                # (M, pos_dim)
            cov_km = self.hmm.covars_[k, :, pos_slice, pos_slice]   # (M, pos_dim, pos_dim)

            mu = np.sum(w[:, None] * mu_km, axis=0)
            second = np.zeros((self.pos_dim, self.pos_dim))
            for m in range(M):
                second += w[m] * (cov_km[m] + np.outer(mu_km[m], mu_km[m]))
            S = second - np.outer(mu, mu)

            state_mu[k] = mu
            state_S[k] = S

        return state_mu, state_S

    def _time_feature(self, t=None):
        """
        Return scalar time feature consistent with training.
        """
        if not self.use_time:
            return None
        if t is not None:
            return float(t)
        if self.time_mode == "normalized":
            # If you're calling online without known horizon, pass t explicitly.
            # Fallback: map step to [0,1] with a soft saturation.
            return float(np.tanh(self._step * 0.01))
        elif self.time_mode == "steps":
            return float(self._step * self.time_scale)
        else:
            raise ValueError(f"Unknown time_mode: {self.time_mode}")

    def _log_emission_per_state(self, y, t=None):
        """
        Compute log p(x | state=k) for all states k (vector length K).
        Uses the state's emission GMM (mixture over m).
        """
        y = np.asarray(y, float).reshape(self.pos_dim)
        K = self.hmm.n_components
        M = self.hmm.n_mix

        if self.use_time:
            tt = self._time_feature(t)
            x = np.zeros(1 + self.pos_dim)
            x[0] = tt
            x[1:] = y
            means = self.hmm.means_          # (K,M,D)
            covars = self.hmm.covars_        # (K,M,D,D)
        else:
            x = y
            means = self.hmm.means_
            covars = self.hmm.covars_

        out = np.empty(K)
        for k in range(K):
            out[k] = _gmm_logpdf(x, self.hmm.weights_[k], means[k], covars[k])
        return out

    def _filter_update(self, logB):
        """
        One-step filtering update for log alpha:
          alpha_t ∝ B_t ⊙ (A^T alpha_{t-1})
        in log-space.
        """
        A = self.hmm.transmat_
        K = A.shape[0]
        logA_T = _safe_log(A.T)  # (K,K)

        if self._log_alpha is None:
            # initialize from startprob
            log_alpha_pred = _safe_log(self.hmm.startprob_)  # (K,)
        else:
            # prediction step: logsumexp over previous states
            # log_alpha_pred[k] = logsumexp_j (logA_T[k,j] + log_alpha_prev[j])
            log_alpha_pred = logsumexp(logA_T + self._log_alpha[None, :], axis=1)

        # update with emission
        log_alpha = log_alpha_pred + logB
        log_alpha -= logsumexp(log_alpha)  # normalize
        self._log_alpha = log_alpha
        return np.exp(log_alpha)

    def _predict_next_belief(self, alpha):
        """
        alpha_next = A^T alpha
        """
        return self.hmm.transmat_.T @ alpha

    def _predict_next_pos_distribution(self, alpha_next):
        """
        Mixture over states: y_{t+1} ~ sum_k alpha_next[k] N(state_mu[k], state_S[k])
        Returns mean and covariance.
        """
        mu = alpha_next @ self._state_mu  # (pos_dim,)
        S = np.zeros((self.pos_dim, self.pos_dim))
        for k in range(alpha_next.size):
            dm = (self._state_mu[k] - mu).reshape(self.pos_dim, 1)
            S += alpha_next[k] * (self._state_S[k] + dm @ dm.T)
        return mu, S

    def _model_command(self, y, mu_next):
        """
        Default: "velocity-like" command toward predicted next mean.
        """
        y = np.asarray(y, float).reshape(self.pos_dim)
        mu_next = np.asarray(mu_next, float).reshape(self.pos_dim)
        u = self.k_p * (mu_next - y) / max(self.dt, 1e-12)
        if self.u_clip is not None:
            c = np.asarray(self.u_clip, float)
            if c.size == 1:
                u = np.clip(u, -float(c), float(c))
            else:
                u = np.clip(u, -c.reshape(self.pos_dim), c.reshape(self.pos_dim))
        return u

    def _confidence(self, y, mu_next, Sigma_next, alpha_next, logB):
        """
        Build a confidence scalar in [0,1] from:
          - low entropy of alpha_next (phase certainty)
          - small trace(Sigma_next) (trajectory certainty)
          - small innovation (y close to predicted mean)
          - emission strength (optional, via logB)
        """
        # 1) state certainty via normalized entropy
        H = _entropy(alpha_next)
        Hmax = np.log(alpha_next.size + 1e-12)
        conf_state = 1.0 - (H / (Hmax + 1e-12))  # [0,1]

        # 2) model uncertainty via trace
        tr = float(np.trace(Sigma_next))
        conf_cov = 1.0 / (1.0 + tr)  # in (0,1], scale-free-ish

        # 3) innovation (Mahalanobis distance)
        y = np.asarray(y, float).reshape(self.pos_dim)
        mu_next = np.asarray(mu_next, float).reshape(self.pos_dim)
        S = Sigma_next + 1e-9 * np.eye(self.pos_dim)
        try:
            L = np.linalg.cholesky(S)
            z = np.linalg.solve(L, (y - mu_next))
            maha2 = float(z @ z)
        except np.linalg.LinAlgError:
            maha2 = float(np.sum((y - mu_next) ** 2) / (tr + 1e-9))
        conf_innov = np.exp(-0.5 * maha2)  # (0,1]

        # 4) optional emission confidence: how peaked the evidence is
        # Here we measure peakiness of normalized exp(logB)
        B = np.exp(logB - logsumexp(logB))
        conf_emit = float(np.max(B))  # [1/K, 1]

        # Combine (weighted product-like blend)
        # Normalize conf_emit to [0,1]
        conf_emit = (conf_emit - 1.0 / B.size) / (1.0 - 1.0 / B.size + 1e-12)
        conf_emit = float(np.clip(conf_emit, 0.0, 1.0))

        # Weighted sum (simple + robust)
        ws = self.w_entropy + self.w_cov + self.w_innov + 0.5  # include emission as fixed 0.5 weight
        conf = (
            self.w_entropy * conf_state +
            self.w_cov * conf_cov +
            self.w_innov * conf_innov +
            0.5 * conf_emit
        ) / max(ws, 1e-12)

        return float(np.clip(conf, 0.0, 1.0)), H, tr

    def _lambda_from_conf(self, conf):
        """
        Map confidence in [0,1] to lambda in [lambda_min, lambda_max] with a sigmoid.
        """
        s = _sigmoid(self.conf_k * (conf - self.conf_center))
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * s
        return float(np.clip(lam, self.lambda_min, self.lambda_max))

    # ---------- public step ----------

    def step(self, y, u_human, t=None, return_debug=False):
        """
        y:        current position (pos_dim,)
        u_human:  human command (pos_dim,) e.g. desired velocity or delta
        t:        optional time feature (consistent with training), e.g. normalized [0,1]
        """
        y = np.asarray(y, float).reshape(self.pos_dim)
        u_human = np.asarray(u_human, float).reshape(self.pos_dim)

        # emission
        logB = self._log_emission_per_state(y, t=t)  # (K,)

        # filter
        alpha = self._filter_update(logB)            # (K,)

        # predict next state belief
        alpha_pred = self._predict_next_belief(alpha)

        # predict next position distribution (state-mixture)
        mu_next, Sigma_next = self._predict_next_pos_distribution(alpha_pred)

        # model command (velocity-like)
        u_model = self._model_command(y, mu_next)

        # confidence -> lambda
        conf, H, tr = self._confidence(y, mu_next, Sigma_next, alpha_pred, logB)
        lam = self._lambda_from_conf(conf)

        # shared command
        u_shared = (1.0 - lam) * u_human + lam * u_model

        # step counter
        self._step += 1

        if not return_debug:
            return u_shared

        dbg = SharedControlDebug(
            alpha=alpha,
            alpha_pred=alpha_pred,
            loglik_t=float(logsumexp(logB)),   # not absolute sequence likelihood; useful as a scalar evidence proxy
            entropy=H,
            cov_trace=tr,
            confidence=conf,
            lam=lam,
            u_model=u_model,
            u_human=u_human,
            u_shared=u_shared,
        )
        return u_shared, dbg


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # Suppose you already trained:
    #   hmmgmm = fit_hmm_gmm_time_augmented(pos_demos, ...)
    # and you have a loop with current position y and user command u_human:

    # controller = HMMGMRSharedController(hmmgmm, pos_dim=3, dt=1/60, use_time=True)

    # For demo purposes only:
    pass
