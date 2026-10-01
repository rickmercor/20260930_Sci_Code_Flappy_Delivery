#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math
import numpy
import scipy.special

import numpy as np


def feeding_branching(A: "np.ndarray") -> tuple:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2D array with shape (N,N), N>=1")
    if np.any((A != 0.0) & (A != 1.0)):
        raise ValueError("A entries must be 0 or 1")
    N = A.shape[0]
    rho = (A.sum(axis=1) > 0).astype(float)
    sigma = (A.sum(axis=0) > 0).astype(float)
    row = A.sum(axis=1, keepdims=True)
    col = A.sum(axis=0)
    chi = np.divide(A, row, out=np.zeros((N, N)), where=row != 0)
    beta = np.divide(A, col, out=np.zeros((N, N)), where=col != 0)
    return rho, sigma, chi, beta

import numpy as np


def _trophic_levels(A: "np.ndarray") -> "np.ndarray":
    N = A.shape[0]
    t = np.ones(N, dtype=float)
    for _ in range(10000):
        new = np.ones(N, dtype=float)
        for i in range(N):
            prey = np.flatnonzero(A[i] == 1.0)
            if prey.size:
                new[i] += float(t[prey].mean())
        if np.max(np.abs(new - t)) < 1e-15:
            return new
        t = new
    raise ValueError("Levine trophic levels did not converge")


def row_timescales(A: "np.ndarray", R: float = 42.0) -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2D array with shape (N,N), N>=1")
    if np.any((A != 0.0) & (A != 1.0)):
        raise ValueError("A entries must be 0 or 1")
    if not (isinstance(R, (int, float)) and np.isfinite(R) and float(R) > 0.0):
        raise ValueError("R must be finite and > 0")
    t = _trophic_levels(A)
    return np.power(float(R), -0.25 * (t - 1.0)).astype(float)

import numpy as np


def community_matrix(
    A: "np.ndarray",
    phi: "np.ndarray",
    gamma: "np.ndarray",
    psi: "np.ndarray",
    mu: "np.ndarray",
    lam: "np.ndarray",
    alpha: "np.ndarray",
    rho: "np.ndarray",
    sigma: "np.ndarray",
    chi: "np.ndarray",
    beta: "np.ndarray",
) -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    phi = np.asarray(phi, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    psi = np.asarray(psi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    lam = np.asarray(lam, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    rho = np.asarray(rho, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    chi = np.asarray(chi, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2D array with shape (N,N), N>=1")
    if np.any((A != 0.0) & (A != 1.0)):
        raise ValueError("A entries must be 0 or 1")
    N = A.shape[0]
    for name, v in [("phi", phi), ("gamma", gamma), ("psi", psi), ("mu", mu),
                    ("alpha", alpha), ("rho", rho), ("sigma", sigma)]:
        if v.shape != (N,) or not np.all(np.isfinite(v)):
            raise ValueError(name + " must be finite with shape (N,)")
    for name, m in [("lam", lam), ("chi", chi), ("beta", beta)]:
        if m.shape != (N, N) or not np.all(np.isfinite(m)):
            raise ValueError(name + " must be finite with shape (N, N)")
    J = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                val = (1.0 - rho[i]) * phi[i]
                val += rho[i] * (gamma[i] * chi[i, i] * lam[i, i] + psi[i])
                val -= (1.0 - sigma[i]) * mu[i]
                val -= sigma[i] * np.sum(beta[:, i] * lam[:, i] * ((gamma - 1.0) * chi[:, i] + 1.0))
            else:
                val = 0.0
                if chi[i, j] != 0.0:
                    val += rho[i] * gamma[i] * chi[i, j] * lam[i, j]
                if beta[j, i] != 0.0:
                    val -= sigma[i] * beta[j, i] * psi[j]
                mask = (beta[:, i] != 0.0) & (chi[:, j] != 0.0)
                val -= sigma[i] * np.sum(beta[:, i] * lam[:, j] * (gamma - 1.0) * chi[:, j] * mask)
            J[i, j] = alpha[i] * val
    return J

import numpy as np


def reactivity(M: "np.ndarray") -> float:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] < 1:
        raise ValueError("M must be a square 2D array with shape (N,N), N>=1")
    if not np.all(np.isfinite(M)):
        raise ValueError("M must be finite")
    if np.any(np.abs(M) > 1e150):
        raise ValueError("M entries must satisfy abs(M) <= 1e150")
    S = 0.5 * M + 0.5 * M.T
    if not np.all(np.isfinite(S)):
        raise ValueError("symmetric part is not finite")
    return float(np.linalg.eigvalsh(S)[-1])

import numpy as np


def apparent_competition_motifs(A: "np.ndarray") -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2D array with shape (N,N), N>=1")
    if np.any((A != 0.0) & (A != 1.0)):
        raise ValueError("A entries must be 0 or 1")
    rows = []
    for p in range(A.shape[0]):
        prey = np.flatnonzero(A[p] == 1.0)
        for ii in range(prey.size):
            for jj in range(ii + 1, prey.size):
                a = int(prey[ii])
                b = int(prey[jj])
                if A[a, b] == 0.0 and A[b, a] == 0.0:
                    if a > b:
                        a, b = b, a
                    rows.append([p, a, b])
    if not rows:
        return np.zeros((0, 3), dtype=int)
    rows.sort()
    return np.asarray(rows, dtype=int)

import numpy as np


def max_motif_reactivity(S: "np.ndarray", motifs: "np.ndarray") -> float:
    S = np.asarray(S, dtype=float)
    motifs = np.asarray(motifs, dtype=int)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] < 1:
        raise ValueError("S must be a square 2D array with shape (N,N), N>=1")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must be finite")
    if motifs.ndim != 2 or motifs.shape[0] < 1 or motifs.shape[1] < 1:
        raise ValueError("motifs must be a nonempty 2D index array")
    N = S.shape[0]
    best = None
    for row in motifs:
        if row.size != len(set(int(x) for x in row)):
            raise ValueError("motif row must contain distinct indices")
        if np.any(row < 0) or np.any(row >= N):
            raise ValueError("motif index out of range")
        idx = np.asarray(row, dtype=int)
        val = float(np.linalg.eigvalsh(S[np.ix_(idx, idx)])[-1])
        best = val if best is None else max(best, val)
    return float(best)

import numpy as np


def reactivity_mode_mass(
    A: "np.ndarray",
    phi: "np.ndarray",
    gamma: "np.ndarray",
    psi: "np.ndarray",
    mu: "np.ndarray",
    lam: "np.ndarray",
    R: float = 42.0,
) -> float:
    rho, sigma, chi, beta = feeding_branching(A)
    alpha = row_timescales(A, R)
    J = community_matrix(A, phi, gamma, psi, mu, lam, alpha, rho, sigma, chi, beta)
    _ = reactivity(J)
    motifs = apparent_competition_motifs(A)
    S = 0.5 * J + 0.5 * J.T
    if not np.all(np.isfinite(S)):
        raise ValueError("symmetric part is not finite")
    r_motif = max_motif_reactivity(S, motifs)
    best_rows = []
    for row in motifs:
        idx = np.asarray(row, dtype=int)
        val = float(np.linalg.eigvalsh(S[np.ix_(idx, idx)])[-1])
        if np.isclose(val, r_motif, rtol=0.0, atol=1e-12):
            best_rows.append([int(x) for x in row])
    best_rows.sort()
    idx = np.array(best_rows[0], dtype=int)
    v = np.linalg.eigh(S)[1][:, -1]
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        raise ValueError("leading reactivity mode has norm 0")
    v = v / nrm
    return float(np.sum(v[idx] ** 2))
SCICODE_GOLD_EOF
