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
def construct_scaled_logdet(n: int, mu: float) -> float:
    

    if n < 1:
        raise ValueError("n must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")
    idx = np.arange(1, n + 1, dtype=float)
    lam = (idx ** (-2)) / mu
    return float(np.sum(np.log1p(lam)))

import numpy as np
def nystrom_frobenius_norm(A: np.ndarray, Omega: np.ndarray) -> float:
    import numpy as np
    from numpy.linalg import norm, solve, pinv, eigh

    A = np.asarray(A, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Omega.ndim != 2 or Omega.shape[0] != A.shape[0]:
        raise ValueError("Omega must have shape (n, s) matching A")
    if Omega.shape[1] < 1:
        raise ValueError("Omega must have at least one column")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Omega)):
        raise ValueError("A and Omega must be finite")

    Y = A @ Omega
    s = Omega.shape[1]
    nu = np.finfo(float).eps * norm(Y, 2)
    C = Omega.T @ Y + nu * np.eye(s)
    try:
        CinvYT = solve(C, Y.T)
    except np.linalg.LinAlgError:
        CinvYT = pinv(C) @ Y.T
    Ahat = Y @ CinvYT
    Ahat = 0.5 * (Ahat + Ahat.T)
    evals, evecs = eigh(Ahat)
    evals = np.maximum(evals, 0.0)
    Ahat = (evecs * evals) @ evecs.T
    return float(norm(Ahat, "fro"))

import numpy as np
def leave_one_out_errF2(A: np.ndarray, Omega: np.ndarray) -> float:
    import numpy as np
    from numpy.linalg import norm, solve, pinv, eigh

    A = np.asarray(A, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Omega.ndim != 2 or Omega.shape[0] != A.shape[0]:
        raise ValueError("Omega must have shape (n, s) matching A")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Omega)):
        raise ValueError("A and Omega must be finite")
    s = Omega.shape[1]
    if s < 2:
        raise ValueError("leave-one-out requires at least 2 sketch columns")
    errs = []
    for i in range(s):
        mask = np.ones(s, dtype=bool)
        mask[i] = False
        Om_i = Omega[:, mask]
        Y = A @ Om_i
        si = Om_i.shape[1]
        nu = np.finfo(float).eps * norm(Y, 2)
        C = Om_i.T @ Y + nu * np.eye(si)
        try:
            CinvYT = solve(C, Y.T)
        except np.linalg.LinAlgError:
            CinvYT = pinv(C) @ Y.T
        Ahat_i = 0.5 * ((Y @ CinvYT) + (Y @ CinvYT).T)
        evals, evecs = eigh(Ahat_i)
        evals = np.maximum(evals, 0.0)
        Ahat_i = (evecs * evals) @ evecs.T
        errs.append(norm((A - Ahat_i) @ Omega[:, i]) ** 2)
    return float(np.mean(errs))

import numpy as np
def detective_one_sample_flag(
    errF2_fine: float,
    errF2_coarse: float,
    ell: int,
    m: int,
    beta: float,
) -> float:
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0, 1)")
    if ell < 1 or m < 1:
        raise ValueError("ell and m must be positive")
    denom = (1.0 - beta) * beta * ell + m
    lhs = (m / denom) * errF2_coarse
    return 1.0 if lhs >= errF2_fine else 0.0

import numpy as np

def preconditioner_logdet(A: np.ndarray, Omega: np.ndarray) -> float:
    import numpy as np
    from numpy.linalg import norm, solve, pinv, eigh

    A = np.asarray(A, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Omega.ndim != 2 or Omega.shape[0] != A.shape[0]:
        raise ValueError("Omega must have shape (n, s) matching A")
    if Omega.shape[1] < 1:
        raise ValueError("Omega must have at least one column")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Omega)):
        raise ValueError("A and Omega must be finite")

    Y = A @ Omega
    s = Omega.shape[1]
    nu = np.finfo(float).eps * norm(Y, 2)
    C = Omega.T @ Y + nu * np.eye(s)
    try:
        CinvYT = solve(C, Y.T)
    except np.linalg.LinAlgError:
        CinvYT = pinv(C) @ Y.T
    Ahat = 0.5 * ((Y @ CinvYT) + (Y @ CinvYT).T)
    evals = np.maximum(eigh(Ahat)[0], 0.0)
    return float(np.sum(np.log1p(evals)))

import numpy as np

def slq_preconditioned_quadratic(
    A: np.ndarray, Omega: np.ndarray, w: np.ndarray, m: int
) -> float:
    import numpy as np
    from numpy.linalg import norm, solve, pinv, eigh

    A = np.asarray(A, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if m < 1:
        raise ValueError("m must be positive")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if Omega.ndim != 2 or Omega.shape[0] != A.shape[0]:
        raise ValueError("Omega must have shape (n, s) matching A")
    if Omega.shape[1] < 1:
        raise ValueError("Omega must have at least one column")
    if w.shape[0] != A.shape[0]:
        raise ValueError("w must have length n")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(Omega)):
        raise ValueError("A and Omega must be finite")
    if not np.all(np.isfinite(w)):
        raise ValueError("w must be finite")

    Y = A @ Omega
    s = Omega.shape[1]
    nu = np.finfo(float).eps * norm(Y, 2)
    C = Omega.T @ Y + nu * np.eye(s)
    try:
        CinvYT = solve(C, Y.T)
    except np.linalg.LinAlgError:
        CinvYT = pinv(C) @ Y.T
    Ahat = 0.5 * ((Y @ CinvYT) + (Y @ CinvYT).T)
    evals, evecs = eigh(Ahat)
    evals = np.maximum(evals, 0.0)
    inv_sqrt = 1.0 / np.sqrt(evals + 1.0)

    beta0 = norm(w)
    if beta0 == 0.0:
        return 0.0
    Q = [w / beta0]
    alphas = []
    betas = []
    beta = 0.0
    for j in range(m):
        v = evecs @ (inv_sqrt * (evecs.T @ Q[j]))
        v = (A + np.eye(A.shape[0])) @ v
        v = evecs @ (inv_sqrt * (evecs.T @ v))
        if j > 0:
            v = v - beta * Q[j - 1]
        alpha = float(np.dot(Q[j], v))
        v = v - alpha * Q[j]
        for qi in Q:
            v = v - np.dot(qi, v) * qi
        beta_next = norm(v)
        alphas.append(alpha)
        betas.append(beta_next)
        if beta_next < 1e-14:
            break
        Q.append(v / beta_next)
        beta = beta_next
    k = len(alphas)
    T = np.diag(alphas)
    for i in range(k - 1):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]
    tevals, tevecs = eigh(T)
    tevals = np.maximum(tevals, 1e-300)
    logT_11 = float(tevecs[0] @ (np.log(tevals) * tevecs[0]))
    return float((beta0 ** 2) * logT_11)

import numpy as np
def nuclear_residual_certificate(lam: np.ndarray, r: int, p: int) -> float:
    

    lam = np.asarray(lam, dtype=float).reshape(-1)
    if p < 2:
        raise ValueError("p must be >= 2")
    if r < p:
        raise ValueError("r must be at least p")
    if lam.size < r:
        raise ValueError("lam must have length at least r")
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    k = r - p
    tail = float(np.sum(np.log1p(np.maximum(lam[k:], 0.0))))
    return float((1.0 + k / (p - 1)) * tail)

import numpy as np

def detective_logdet_certificate(
    n: int = 32,
    mu: float = 1e-2,
    ell: int = 16,
    m: int = 5,
    beta: float = 0.75,
    seed: int = 0,
    p: int = 2,
    cert_scale: float = 1e-3,
) -> float:
    import numpy as np

    if n < 1 or mu <= 0:
        raise ValueError("invalid n or mu")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0, 1)")
    if ell < 1 or m < 1:
        raise ValueError("ell and m must be positive")

    logdet_exact = construct_scaled_logdet(n, mu)

    idx = np.arange(1, n + 1, dtype=float)
    lam = (idx ** (-2)) / mu
    A = np.diag(lam)
    fro_exact = float(np.sqrt(np.sum(lam ** 2)))

    beta_ell = int(np.floor(beta * ell))
    beta2_ell = int(np.floor((beta ** 2) * ell))
    if beta_ell < 2 or beta2_ell < 2:
        raise ValueError("ranks too small for leave-one-out")

    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal((n, beta_ell))

    # The coarse diagnostic reuses the leading beta2_ell columns of the same
    # sketch; drawing a fresh block would advance the generator stream.
    err_fine = leave_one_out_errF2(A, Omega)
    err_coarse = leave_one_out_errF2(A, Omega[:, :beta2_ell])
    flag = detective_one_sample_flag(err_fine, err_coarse, ell, m, beta)

    if flag >= 0.5:
        Psi = rng.standard_normal((n, ell - beta_ell))
        Omega_use = np.hstack([Omega, Psi])
        r = ell
        fro_hat = nystrom_frobenius_norm(A, Omega_use)
        t1 = preconditioner_logdet(A, Omega_use)
        w = rng.standard_normal(n)
        t2 = slq_preconditioned_quadratic(A, Omega_use, w, m)
    else:
        Omega_use = Omega
        r = beta_ell
        fro_hat = nystrom_frobenius_norm(A, Omega_use)
        t1 = preconditioner_logdet(A, Omega_use)
        N = int(np.floor((ell + m - beta_ell) / m))
        if N < 1:
            raise ValueError("no SLQ probes available on alpha-rank branch")
        acc = 0.0
        for _i in range(N):
            w = rng.standard_normal(n)
            acc += slq_preconditioned_quadratic(A, Omega_use, w, m)
        t2 = acc / N

    # Loewner-order consequences of 0 <= Ahat <= A (see module docstring).
    tol_f = 1e-8 * max(1.0, abs(fro_exact))
    tol_l = 1e-8 * max(1.0, abs(logdet_exact))
    if not np.isfinite(fro_hat) or fro_hat < -tol_f or fro_hat > fro_exact + tol_f:
        raise ValueError(
            "Nystrom Frobenius norm violates 0 <= ||Ahat||_F <= ||A||_F"
        )
    if not np.isfinite(t1) or t1 < -tol_l or t1 > logdet_exact + tol_l:
        raise ValueError(
            "preconditioner log-determinant exceeds exact tr log(A+I)"
        )

    E_hat = t1 + t2
    R = nuclear_residual_certificate(lam, r, p)
    return float(E_hat + cert_scale * R)
SCICODE_GOLD_EOF
