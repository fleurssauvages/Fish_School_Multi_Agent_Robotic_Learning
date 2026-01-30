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
    
    def __init__(self, n_states=8, n_mix=2, seed=0, min_covar=1e-6, n_iter=200):
        self.model = GMMHMM(
            n_components=n_states,
            n_mix=n_mix,
            covariance_type="full",
            random_state=seed,
            n_iter=n_iter,
            min_covar=min_covar,
            verbose=False,
            init_params="mcw",   # init means/covars/weights, but not start/trans
            params="stmcw", 
        )
        self.model.startprob_ = np.ones(n_states) / n_states
        self.model.transmat_  = np.ones((n_states, n_states)) / n_states

    # ============================================================
    # Training
    # ============================================================

    def fit(self,pos_demos):
        demos = normalize_demos_list(pos_demos)

        seqs, lengths = [], []
        for Y in demos:
            T, Dp = Y.shape
            t = np.linspace(0.0, 1.0, T)[:, None]
            seqs.append(np.hstack([t, Y]))
            lengths.append(T)

        X_all = np.vstack(seqs)

        self.model.fit(X_all, lengths)

    # ============================================================
    # Regression
    # ============================================================

    def regress(self, T, pos_dim=3):
        # --- fast time-only emission
        t_grid = np.linspace(0.0, 1.0, T)
        mu_t = self.model.means_[:, :, 0]
        var_t = self.model.covars_[:, :, 0, 0] + 1e-12
        logw = np.log(self.model.weights_ + 1e-12)

        logB = compute_logB_time_only(t_grid, logw, mu_t, var_t)
        gamma, loglik = forward_backward_from_logB(self.model, logB)

        # --- state means / covs
        K, M = self.model.n_components, self.model.n_mix
        state_mu = np.zeros((K, pos_dim))
        state_S = np.zeros((K, pos_dim, pos_dim))
        for k in range(K):
            w = self.model.weights_[k]
            mu_km = self.model.means_[k, :, 1:]
            cov_km = self.model.covars_[k, :, 1:, 1:]
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
# Utilities for demos
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
