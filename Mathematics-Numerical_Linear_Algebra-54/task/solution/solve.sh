#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def noise_aware_coupling(A: "np.ndarray", sigma: "np.ndarray",
                                 alpha: float) -> tuple:
    A = np.asarray(A, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if sigma.shape != (m,):
        raise ValueError("sigma must have shape (m,) matching A's rows")
    if np.any(sigma <= 0):
        raise ValueError("all entries of sigma must be strictly positive")
    if not (alpha > 0):
        raise ValueError("alpha must be strictly positive")
    row_norms = np.linalg.norm(A, axis=1)
    if np.any(row_norms == 0):
        raise ValueError("A must not contain a zero row")
    frob2 = float(np.sum(A ** 2))
    # Corollary 2.3(ii): p_i proportional to sigma_i * ||a_i||; the weights
    # then follow from the coupling (3): p_i w_i / ||a_i||^2 = alpha / ||A||_F^2.
    p = sigma * row_norms
    p = p / np.sum(p)
    w = alpha * row_norms ** 2 / (p * frob2)
    return p, w

import numpy as np

def aabk_spectral_bound(A: "np.ndarray", w: "np.ndarray", alpha: float,
                                tau: int) -> float:
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if w.shape != (m,):
        raise ValueError("w must have shape (m,) matching A's rows")
    if not (alpha > 0):
        raise ValueError("alpha must be strictly positive")
    if not (isinstance(tau, (int, np.integer)) and tau >= 1):
        raise ValueError("tau must be a positive integer")
    frob2 = float(np.sum(A ** 2))
    # eq. (6): T = W/(2 tau) + alpha/(2 ||A||_F^2) (1 - 1/tau) A A^T
    T = np.diag(w) / (2.0 * tau) + alpha / (2.0 * frob2) * (1.0 - 1.0 / tau) * (A @ A.T)
    return float(np.linalg.eigvalsh(T).max())

import numpy as np

def _soft_shrink(v: "np.ndarray", lam: float) -> "np.ndarray":
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def sparse_bregman_distance(x_star: "np.ndarray", y: "np.ndarray",
                                    lam: float) -> float:
    x_star = np.asarray(x_star, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_star.ndim != 1 or x_star.shape != y.shape:
        raise ValueError("x_star and y must be 1D arrays of the same shape")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    # Section 3: f*(x*) = (1/2)||S_lam(x*)||^2 and
    # D_f^{x*}(x, y) = f*(x*) - <x*, y> + f(y).
    f_conj = 0.5 * float(np.sum(_soft_shrink(x_star, lam) ** 2))
    f_y = lam * float(np.sum(np.abs(y))) + 0.5 * float(np.sum(y ** 2))
    return float(f_conj - float(x_star @ y) + f_y)

import numpy as np

def aabk_beta_init(A: "np.ndarray", sigma: "np.ndarray", p: "np.ndarray",
                           w: "np.ndarray", tau: int, breg0: float) -> float:
    A = np.asarray(A, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if sigma.shape != (m,) or p.shape != (m,) or w.shape != (m,):
        raise ValueError("sigma, p and w must have shape (m,) matching A's rows")
    if np.any(sigma <= 0):
        raise ValueError("all entries of sigma must be strictly positive")
    row_norms2 = np.sum(A ** 2, axis=1)
    if np.any(row_norms2 == 0):
        raise ValueError("A must not contain a zero row")
    if not (isinstance(tau, (int, np.integer)) and tau >= 1):
        raise ValueError("tau must be a positive integer")
    if breg0 < 0:
        raise ValueError("breg0 must be nonnegative")
    # Theorem 2.1: beta_0 = tau * D_f^{x*_0}(x_0, x_hat) / Tr(P W^2 Sigma D^{-2}),
    # with D = Diag(||a_i||) and Sigma = Diag(sigma_i^2).
    c_noise = float(np.sum(p * w ** 2 * sigma ** 2 / row_norms2))
    return float(tau * breg0 / c_noise)

import itertools

import numpy as np

def _sigma_tilde_min(A: "np.ndarray") -> float:
    # min over nonempty column subsets J (A_J != 0) of the smallest positive
    # singular value of A_J (definition used by the cited error bound).
    m, n = A.shape
    best = np.inf
    for r in range(1, n + 1):
        for J in itertools.combinations(range(n), r):
            sub = A[:, J]
            if not np.any(sub):
                continue
            s = np.linalg.svd(sub, compute_uv=False)
            s = s[s > 1e-12 * s.max()]
            best = min(best, float(s.min()))
    return best


def error_bound_gamma(A: "np.ndarray", x_hat: "np.ndarray", lam: float) -> float:
    A = np.asarray(A, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    if A.ndim != 2 or not np.any(A):
        raise ValueError("A must be a nonzero 2D array")
    m, n = A.shape
    if x_hat.shape != (n,):
        raise ValueError("x_hat must have shape (n,) matching A's columns")
    if not np.any(x_hat):
        raise ValueError("x_hat must have at least one nonzero entry")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    # Schöpfer–Lorenz (Lemma 3.1): D_f^{x*}(x, x_hat) <= gamma_SL ||Ax - b||^2 with
    #   gamma_SL = (|x_hat|_min + 2 lam) / (sigma~_min(A)^2 |x_hat|_min).
    # The paper's Assumption 3.2 is the reverse inequality, so theta = 1/gamma_SL,
    # and the paper defines gamma := theta / ||A||_F^2.
    x_min = float(np.min(np.abs(x_hat[x_hat != 0])))
    theta = _sigma_tilde_min(A) ** 2 * x_min / (x_min + 2.0 * lam)
    return float(theta / np.sum(A ** 2))

import numpy as np

def aabk_averaged_direction(A: "np.ndarray", x: "np.ndarray", w: "np.ndarray",
                                    batch: "np.ndarray",
                                    b_noisy: "np.ndarray") -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    batch = np.asarray(batch)
    b_noisy = np.asarray(b_noisy, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if x.shape != (n,):
        raise ValueError("x must have shape (n,) matching A's columns")
    if w.shape != (m,):
        raise ValueError("w must have shape (m,) matching A's rows")
    if batch.ndim != 1 or batch.size == 0:
        raise ValueError("batch must be a non-empty 1D array")
    if b_noisy.shape != batch.shape:
        raise ValueError("b_noisy must have the same shape as batch")
    if np.any(batch < 0) or np.any(batch >= m):
        raise ValueError("batch contains an index out of bounds for A")
    tau = batch.size
    rows = A[batch]                                    # (tau, n)
    row_norms2 = np.sum(rows ** 2, axis=1)             # (tau,)
    # eq. (4): d_k = (1/tau) sum_j w_{i_j} (<a_{i_j}, x> - b~_j) / ||a_{i_j}||^2 a_{i_j}
    coeff = w[batch] * (rows @ x - b_noisy) / row_norms2
    return (coeff @ rows) / tau

import numpy as np

def aabk_adaptive_step(beta_k: float, alpha: float, gamma: float,
                               sigma_max_T: float) -> tuple:
    if beta_k < 0:
        raise ValueError("beta_k must be nonnegative")
    if not (alpha > 0):
        raise ValueError("alpha must be strictly positive")
    if not (gamma > 0):
        raise ValueError("gamma must be strictly positive")
    if not (sigma_max_T > 0):
        raise ValueError("sigma_max_T must be strictly positive")
    # eq. (7): eta_k = alpha gamma beta_k / (1 + 2 alpha gamma sigma_max(T) beta_k),
    #          beta_{k+1} = beta_k (1 - alpha gamma eta_k / 2).
    eta_k = alpha * gamma * beta_k / (1.0 + 2.0 * alpha * gamma * sigma_max_T * beta_k)
    beta_next = beta_k * (1.0 - alpha * gamma * eta_k / 2.0)
    return float(eta_k), float(beta_next)

import numpy as np

def bregman_dual_update(x_star: "np.ndarray", d: "np.ndarray", eta: float,
                                lam: float) -> tuple:
    x_star = np.asarray(x_star, dtype=float)
    d = np.asarray(d, dtype=float)
    if x_star.ndim != 1 or x_star.shape != d.shape:
        raise ValueError("x_star and d must be 1D arrays of the same shape")
    if eta < 0:
        raise ValueError("eta must be nonnegative")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    # eq. (5): x*_{k+1} = x*_k - eta_k d_k, x_{k+1} = grad f*(x*_{k+1}) = S_lam(x*_{k+1}).
    x_star_next = x_star - eta * d
    x_next = np.sign(x_star_next) * np.maximum(np.abs(x_star_next) - lam, 0.0)
    return x_star_next, x_next

import numpy as np

def aabk_pipeline(A: "np.ndarray", x_hat: "np.ndarray", sigma: "np.ndarray",
                          lam: float, alpha: float, tau: int, K: int,
                          seed: int) -> float:
    A = np.asarray(A, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if x_hat.shape != (n,):
        raise ValueError("x_hat must have shape (n,) matching A's columns")
    if sigma.shape != (m,):
        raise ValueError("sigma must have shape (m,) matching A's rows")
    if np.any(sigma <= 0):
        raise ValueError("all entries of sigma must be strictly positive")
    if np.any(np.sum(A ** 2, axis=1) == 0):
        raise ValueError("A must not contain a zero row")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    if not np.any(x_hat):
        raise ValueError("x_hat must have at least one nonzero entry")
    if not (alpha > 0):
        raise ValueError("alpha must be strictly positive")
    if not (isinstance(tau, (int, np.integer)) and tau >= 1):
        raise ValueError("tau must be a positive integer")
    if not (isinstance(K, (int, np.integer)) and K >= 1):
        raise ValueError("K must be a positive integer")

    b = A @ x_hat
    p, w = noise_aware_coupling(A, sigma, alpha)
    sigma_max_T = aabk_spectral_bound(A, w, alpha, tau)
    gamma = error_bound_gamma(A, x_hat, lam)

    x_star = np.zeros(n)
    x = np.sign(x_star) * np.maximum(np.abs(x_star) - lam, 0.0)
    breg0 = sparse_bregman_distance(x_star, x_hat, lam)
    beta = aabk_beta_init(A, sigma, p, w, tau, breg0)

    rng = np.random.default_rng(seed)
    for k in range(K):
        batch = rng.choice(m, size=tau, replace=True, p=p)
        eps = rng.normal(0.0, sigma[batch])
        b_noisy = b[batch] + eps
        d = aabk_averaged_direction(A, x, w, batch, b_noisy)
        eta, beta = aabk_adaptive_step(beta, alpha, gamma, sigma_max_T)
        x_star, x = bregman_dual_update(x_star, d, eta, lam)

    diff = x - x_hat
    return float(np.sum(diff ** 2))
SCICODE_GOLD_EOF
