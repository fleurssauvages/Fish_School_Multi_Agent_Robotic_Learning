import numpy as np
from dataclasses import dataclass
from hmmlearn.hmm import GMMHMM
from scipy.special import logsumexp
from numba import njit

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
    
    def __init__(self, n_states=8, n_mix=2, seed=0, min_covar=1e-6, n_iter=200, cov_type="diag"):
        self.model = GMMHMM(
            n_components=n_states,
            n_mix=n_mix,
            covariance_type=cov_type,
            random_state=seed,
            n_iter=n_iter,
            min_covar=min_covar,
            verbose=False,
            init_params="mcw",
            params="stmcw", 
        )
        self.model.startprob_ = np.ones(n_states) / n_states
        self.model.transmat_  = np.ones((n_states, n_states)) / n_states

    # ============================================================
    # Training
    # ============================================================

    def _pack_demos_with_time(self, demos):
        demos = normalize_demos_list(demos)

        lengths = np.fromiter((Y.shape[0] for Y in demos), dtype=np.int64)
        N = int(lengths.sum())
        Dp = demos[0].shape[1]          # position dim
        X_all = np.empty((N, 1 + Dp), dtype=np.float64)

        start = 0
        for Y, T in zip(demos, lengths):
            X_all[start:start+T, 0] = np.linspace(0.0, 1.0, T)
            X_all[start:start+T, 1:] = Y
            start += T

        return X_all, lengths.tolist()

    def fit(self,pos_demos):
        X_all, lengths = self._pack_demos_with_time(pos_demos)
        self.model.fit(X_all, lengths)

    def update(self, pos_demos, n_iter=10):
        X_all, lengths = self._pack_demos_with_time(pos_demos)
        self.model.init_params = ""
        self.model.n_iter = n_iter
        self.model.fit(X_all, lengths)

    # ============================================================
    # Regression
    # ============================================================

    def regress(self, T, pos_dim=3):
        # --- fast time-only emission
        t_grid = np.linspace(0.0, 1.0, T)
        mu_t = self.model.means_[:, :, 0]
        covtype = self.model.covariance_type
        D = self.model.means_.shape[-1]  # total dim = 1 + pos_dim

        if covtype == "full":
            var_t = self.model.covars_[:, :, 0, 0] + 1e-12
            # Cov(y,t) needed for conditional: shape (K,M,pos_dim)
            cov_yy = self.model.covars_[:, :, 1:, 1:]
        elif covtype == "diag":
            # Only diagonal variances exist; cross-covariances are zero
            var_t = self.model.covars_[:, :, 0] + 1e-12
            # diag variances for y: shape (K,M,pos_dim)
            cov_yy = np.zeros((self.n_states, self.n_mix, D-1, D-1))
            diag_y = self.model.covars_[:, :, 1:]  # (K,M,pos_dim)
            # fill diagonal matrices
            diag_y = self.model.covars_[:, :, 1:] + 1e-12        # (K,M,D)
            I = np.eye(D-1)[None, None, :, :]                    # (1,1,D,D)
            cov_yy = diag_y[:, :, :, None] * I     
        else:
            raise ValueError(f"Unsupported covariance_type: {covtype}")

        logw = np.log(self.model.weights_ + 1e-12)

        logB = compute_logB(t_grid, logw, mu_t, var_t)
        gamma, loglik = forward_backward_from_logB_numba(
            self.model.startprob_,
            self.model.transmat_,
            logB
        )

        # --- state means / covs
        w = self.model.weights_              # (K,M)
        mu_km = self.model.means_[:, :, 1:]  # (K,M,Dy)

        # state mean: (K,Dy)
        state_mu = np.einsum("km,kmd->kd", w, mu_km)

        # second moment: sum_m w_m (Cov_m + mu_m mu_m^T)
        ExxT = np.einsum("km,kmdn->kdn", w, cov_yy) + np.einsum("km,kmd,kmn->kdn", w, mu_km, mu_km)

        # covariance: E[xx^T] - E[x]E[x]^T
        state_S = ExxT - np.einsum("kd,kn->kdn", state_mu, state_mu)  # (K,Dy,Dy)

        mu_y = gamma @ state_mu
        Sigma_within = np.einsum("tk,kij->tij", gamma, state_S)

        diff = state_mu[None, :, :] - mu_y[:, None, :]          # (T,K,D)
        Sigma_between = np.einsum("tk,tk i,tk j->tij", gamma, diff, diff)

        Sigma_y = Sigma_within + Sigma_between

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

def compute_logB(t_grid, logw, mu_t, var_t):
    """
    Vectorized time-only emission:
    logB[t,k] = log p(t | state k)
    """
    t = t_grid[:, None, None]          # (T,1,1)
    mu = mu_t[None, :, :]              # (1,K,M)
    var = var_t[None, :, :]            # (1,K,M)

    const = logw[None, :, :] - 0.5 * np.log(2.0 * np.pi * var)   # (1,K,M)
    invvar = 1.0 / var
    diff2 = (t - mu) ** 2

    v = const - 0.5 * diff2 * invvar   # (T,K,M)
    return logsumexp(v, axis=2)        # (T,K)

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

@njit(cache=True, fastmath=True)
def _logsumexp_1d(a):
    """Stable logsumexp for 1D array."""
    amax = a[0]
    for i in range(1, a.shape[0]):
        if a[i] > amax:
            amax = a[i]
    s = 0.0
    for i in range(a.shape[0]):
        s += np.exp(a[i] - amax)
    return amax + np.log(s)

@njit(cache=True, fastmath=True)
def forward_backward_from_logB_numba(startprob, transmat, logB):
    """
    Numba-accelerated forward-backward where:
      startprob: (K,)
      transmat:  (K,K)
      logB:      (T,K)  already computed
    returns:
      gamma: (T,K)
      loglik: float
    """
    T, K = logB.shape

    # log-space start & transition
    log_start = np.empty(K)
    for k in range(K):
        log_start[k] = np.log(startprob[k] + 1e-12)

    log_trans = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            log_trans[i, j] = np.log(transmat[i, j] + 1e-12)

    # forward
    logalpha = np.empty((T, K))
    for k in range(K):
        logalpha[0, k] = log_start[k] + logB[0, k]

    tmp = np.empty(K)
    for t in range(1, T):
        for j in range(K):
            # tmp[i] = logalpha[t-1,i] + log_trans[i,j]
            for i in range(K):
                tmp[i] = logalpha[t - 1, i] + log_trans[i, j]
            logalpha[t, j] = logB[t, j] + _logsumexp_1d(tmp)

    # backward
    logbeta = np.zeros((T, K))
    tmp2 = np.empty(K)
    for t in range(T - 2, -1, -1):
        for i in range(K):
            # tmp2[j] = log_trans[i,j] + logB[t+1,j] + logbeta[t+1,j]
            for j in range(K):
                tmp2[j] = log_trans[i, j] + logB[t + 1, j] + logbeta[t + 1, j]
            logbeta[t, i] = _logsumexp_1d(tmp2)

    # gamma
    gamma = np.empty((T, K))
    for t in range(T):
        # loggamma = logalpha + logbeta, then normalize row with logsumexp
        for k in range(K):
            tmp[k] = logalpha[t, k] + logbeta[t, k]
        norm = _logsumexp_1d(tmp)
        for k in range(K):
            gamma[t, k] = np.exp((logalpha[t, k] + logbeta[t, k]) - norm)

    # loglik = logsumexp(logalpha[T-1])
    for k in range(K):
        tmp[k] = logalpha[T - 1, k]
    loglik = _logsumexp_1d(tmp)

    return gamma, loglik