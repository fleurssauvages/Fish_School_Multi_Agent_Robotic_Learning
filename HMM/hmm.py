import numpy as np
from dataclasses import dataclass
from hmmlearn.hmm import GMMHMM
from scipy.special import logsumexp

# ============================================================
# Dataclass wrapper
# ============================================================

@dataclass
class HMMGMM:
    model: object  # hmmlearn.hmm.GMMHMM
    D: int         # full obs dim = 1 + pos_dim

    @property
    def n_states(self):
        return self.model.n_components

    @property
    def n_mix(self):
        return self.model.n_mix


# ============================================================
# Utilities
# ============================================================

def normalize_demos_list(pos_demos):
    pos_demos = np.asarray(pos_demos, float) if not isinstance(pos_demos, list) else pos_demos
    if isinstance(pos_demos, list):
        return [d if d.shape[0] >= d.shape[1] else d.T for d in pos_demos]

    if pos_demos.ndim != 3:
        raise ValueError("pos_demos must be list[(T,D)] or array (N,T,D)/(N,D,T)")

    if pos_demos.shape[1] < pos_demos.shape[2]:  # (N,T,D)
        return [pos_demos[i] for i in range(pos_demos.shape[0])]
    else:                                        # (N,D,T)
        return [pos_demos[i].T for i in range(pos_demos.shape[0])]


# ============================================================
# Training
# ============================================================

def fit_hmm_gmm_time_augmented(
    pos_demos,
    n_states=8,
    n_mix=2,
    seed=0,
    min_covar=1e-6,
    n_iter=200,
):
    demos = normalize_demos_list(pos_demos)

    seqs, lengths = [], []
    for Y in demos:
        T, Dp = Y.shape
        t = np.linspace(0.0, 1.0, T)[:, None]
        seqs.append(np.hstack([t, Y]))
        lengths.append(T)

    X_all = np.vstack(seqs)

    hmm = GMMHMM(
        n_components=n_states,
        n_mix=n_mix,
        covariance_type="full",
        random_state=seed,
        n_iter=n_iter,
        min_covar=min_covar,
        verbose=False,
    )
    hmm.fit(X_all, lengths)
    return HMMGMM(hmm, X_all.shape[1])


# ============================================================
# FAST emission likelihoods
# ============================================================

def compute_logB_time_only(t_grid, logw, mu_t, var_t):
    """
    Vectorized time-only emission:
    logB[t,k] = log p(t | state k)
    """
    T = t_grid.shape[0]
    K, M = logw.shape

    const = logw - 0.5 * np.log(2.0 * np.pi * var_t)
    invvar = 1.0 / var_t

    logB = np.empty((T, K))
    for t in range(T):
        tt = t_grid[t]
        diff2 = (tt - mu_t) ** 2
        v = const - 0.5 * diff2 * invvar
        logB[t] = logsumexp(v, axis=1)
    return logB


def state_log_emission_full(hmmgmm, x):
    """
    Full emission likelihood (time + pos), used ONLY at via points.
    """
    hmm = hmmgmm.model
    K, M = hmm.n_components, hmm.n_mix

    out = np.zeros(K)
    for k in range(K):
        tmp = np.zeros(M)
        for m in range(M):
            mu = hmm.means_[k, m]
            cov = hmm.covars_[k, m]
            diff = x - mu
            try:
                L = np.linalg.cholesky(cov)
                y = np.linalg.solve(L, diff)
                quad = y @ y
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                tmp[m] = (
                    np.log(hmm.weights_[k, m] + 1e-12)
                    - 0.5 * (len(x) * np.log(2*np.pi) + logdet + quad)
                )
            except np.linalg.LinAlgError:
                tmp[m] = -np.inf
        out[k] = logsumexp(tmp)
    return out


# ============================================================
# Forward–Backward (logB already computed!)
# ============================================================

def forward_backward_from_logB(hmm, logB):
    T, K = logB.shape
    log_start = np.log(hmm.startprob_ + 1e-12)
    log_trans = np.log(hmm.transmat_ + 1e-12)

    logalpha = np.empty((T, K))
    logalpha[0] = log_start + logB[0]
    for t in range(1, T):
        logalpha[t] = logB[t] + logsumexp(
            logalpha[t-1][:, None] + log_trans, axis=0
        )

    logbeta = np.zeros((T, K))
    for t in range(T-2, -1, -1):
        logbeta[t] = logsumexp(
            log_trans + (logB[t+1] + logbeta[t+1])[None, :], axis=1
        )

    loggamma = logalpha + logbeta
    loggamma -= logsumexp(loggamma, axis=1, keepdims=True)
    gamma = np.exp(loggamma)
    loglik = logsumexp(logalpha[-1])
    return gamma, loglik


# ============================================================
# Regression
# ============================================================

def hmm_gmm_regress(
    hmmgmm,
    T,
    pos_dim=3,
    via_points=None,
):
    hmm = hmmgmm.model
    K, M = hmm.n_components, hmm.n_mix

    if via_points is None:
        via_points = {}

    # --- fast time-only emission
    t_grid = np.linspace(0.0, 1.0, T)
    mu_t = hmm.means_[:, :, 0]
    var_t = hmm.covars_[:, :, 0, 0] + 1e-12
    logw = np.log(hmm.weights_ + 1e-12)

    logB = compute_logB_time_only(t_grid, logw, mu_t, var_t)

    # --- patch via points (slow, but few)
    for idx, p in via_points.items():
        x = np.zeros(1 + pos_dim)
        x[0] = t_grid[idx]
        x[1:] = p
        logB[idx] = state_log_emission_full(hmmgmm, x)

    gamma, loglik = forward_backward_from_logB(hmm, logB)

    # --- state means / covs
    state_mu = np.zeros((K, pos_dim))
    state_S = np.zeros((K, pos_dim, pos_dim))
    for k in range(K):
        w = hmm.weights_[k]
        mu_km = hmm.means_[k, :, 1:]
        cov_km = hmm.covars_[k, :, 1:, 1:]
        mu = np.sum(w[:, None] * mu_km, axis=0)
        S = np.zeros((pos_dim, pos_dim))
        for m in range(M):
            S += w[m] * (cov_km[m] + np.outer(mu_km[m], mu_km[m]))
        S -= np.outer(mu, mu)
        state_mu[k] = mu
        state_S[k] = S

    mu_y = gamma @ state_mu
    Sigma_y = np.zeros((T, pos_dim, pos_dim))
    for t in range(T):
        for k in range(K):
            d = (state_mu[k] - mu_y[t]).reshape(pos_dim, 1)
            Sigma_y[t] += gamma[t, k] * (state_S[k] + d @ d.T)

    return mu_y, Sigma_y, gamma, loglik


# ============================================================
# Sampling
# ============================================================

def sample_trajectory(hmmgmm, T, pos_dim=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    X, _ = hmmgmm.model.sample(T, random_state=rng)
    return X[:, 1:1+pos_dim]
