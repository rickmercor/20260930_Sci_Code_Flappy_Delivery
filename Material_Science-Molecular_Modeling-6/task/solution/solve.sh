#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Step 1: greedy multicoloring of the orbital-interaction graph."""

import numpy as np


def greedy_multicoloring(D, dcut):
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.all(np.isfinite(D)):
        raise ValueError("nonfinite distance matrix")
    if not np.isfinite(dcut) or dcut <= 0:
        raise ValueError("invalid cutoff")
    n = D.shape[0]
    colors = np.full(n, -1, dtype=np.int64)
    for j in range(n):
        forbidden = set()
        for i in range(j):
            if D[i, j] < dcut:
                forbidden.add(int(colors[i]))
        c = 0
        while c in forbidden:
            c += 1
        colors[j] = c
    return colors

"""Step 2: chromatic superposition state matrix with the declared signs."""

import numpy as np


def css_matrix(colors):
    colors = np.asarray(colors)
    if colors.ndim != 1 or colors.size == 0:
        raise ValueError("colors must be a nonempty vector")
    if colors.min() < 0:
        raise ValueError("negative color index")
    n = colors.shape[0]
    nc = int(colors.max()) + 1
    R = np.zeros((n, nc))
    for j in range(n):
        s = 1.0 if ((3 * (j + 1)) % 7) >= 3 else -1.0
        R[j, int(colors[j])] = s
    return R

"""Step 3: orthonormal block-Krylov basis with the declared conventions."""

import numpy as np


def block_krylov_basis(A, B, v):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
        raise ValueError("invalid Krylov dimension")
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if B.ndim == 1:
        B = B[:, None]
    if A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]:
        raise ValueError("shape mismatch")
    cols = []
    blk = B.copy()
    for _ in range(int(v)):
        for c in range(blk.shape[1]):
            cols.append(blk[:, c].copy())
        blk = A @ blk
    Q = []
    for w in cols:
        for q in Q:
            w = w - (q @ w) * q
        for q in Q:
            w = w - (q @ w) * q
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-10:
            continue
        w = w / nrm
        for comp in w:
            if abs(comp) > 1e-8:
                if comp < 0.0:
                    w = -w
                break
        Q.append(w)
    if not Q:
        raise ValueError("no Krylov column survived")
    return np.column_stack(Q)

"""Step 4: inverse square root of the projected overlap."""

import numpy as np


def projected_inv_sqrt(S, Q):
    S = np.asarray(S, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if S.shape[0] != S.shape[1] or Q.shape[0] != S.shape[0]:
        raise ValueError("shape mismatch")
    SK = Q.T @ S @ Q
    SK = (SK + SK.T) / 2.0
    lam, U = np.linalg.eigh(SK)
    if float(lam.min()) < 1e-10:
        raise ValueError("projected overlap not positive definite")
    return U @ np.diag(lam ** -0.5) @ U.T

"""Step 5: extraction of the sparse operator from the CSS block."""

import numpy as np


def css_extract(Mtilde, R, colors, D, dcut):
    Mtilde = np.asarray(Mtilde, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    colors = np.asarray(colors)
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[1] != n or Mtilde.shape[0] != n or R.shape != Mtilde.shape or colors.shape[0] != n:
        raise ValueError("inconsistent shapes")
    out = np.zeros((n, n))
    for nn in range(n):
        c = int(colors[nn])
        s = R[nn, c]
        for mm in range(n):
            if D[mm, nn] < dcut:
                out[mm, nn] = Mtilde[mm, c] * s
    return (out + out.T) / 2.0

"""Step 6: CSS reconstruction of the overlap inverse square root."""

import numpy as np


def css_inv_sqrt(S, D, dcut_s, v_s):
    colors_s = greedy_multicoloring(D, dcut_s)
    O = css_matrix(colors_s)
    Q = block_krylov_basis(S, O, v_s)
    SKinv = projected_inv_sqrt(S, Q)
    Y = Q @ (SKinv @ (Q.T @ O))
    return css_extract(Y, O, colors_s, D, dcut_s)

"""Step 7: CSS density matrix via the transformed Hamiltonian."""

import numpy as np


def css_density(H, S, D, dcut_rho, dcut_s, v, v_s, beta, mu):
    H = np.asarray(H, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    Shat = css_inv_sqrt(S, D, dcut_s, v_s)
    Hp = Shat @ H @ Shat
    Hp = (Hp + Hp.T) / 2.0
    colors_r = greedy_multicoloring(D, dcut_rho)
    R = css_matrix(colors_r)
    Rt = Shat @ R
    Q = block_krylov_basis(Hp, Rt, v)
    HK = Q.T @ Hp @ Q
    HK = (HK + HK.T) / 2.0
    eps, U = np.linalg.eigh(HK)
    f = 1.0 / (1.0 + np.exp(beta * (eps - mu)))
    rhoK = U @ np.diag(f) @ U.T
    rho_t = Shat @ (Q @ (rhoK @ (Q.T @ Rt)))
    return css_extract(rho_t, R, colors_r, D, dcut_rho)

"""Step 8 (final orchestrator): CSS audit over the two declared datasets."""

import numpy as np

_N = 20
_DCUT_RHO = 3.0
_DCUT_S = 5.0
_BETA = 6.0


def _build(variant):
    a = 3 if variant == 1 else 5
    H = np.zeros((_N, _N))
    S = np.zeros((_N, _N))
    for i in range(_N):
        for j in range(_N):
            if abs(i - j) <= 3:
                H[i, j] = (((i + 1) * (j + 1) + a) % 13) / 13.0 - 0.5
            if abs(i - j) <= 2 and i != j:
                S[i, j] = (((i + 2) * (j + 2) + 2 * a) % 7) / 35.0 - 0.1
        S[i, i] = 1.4 + (((i + 1) * a) % 5) / 25.0
    H = (H + H.T) / 2.0
    S = (S + S.T) / 2.0
    mu = 0.15 if variant == 1 else -0.1
    return H, S, mu


def _distances():
    idx = np.arange(_N, dtype=float)
    return np.abs(idx[:, None] - idx[None, :])


def _probe():
    return np.array([((3 * (i + 1)) % 7 - 3) / 4.0 for i in range(_N)])


def _exact(H, S, beta, mu):
    lam, V = np.linalg.eigh(S)
    Sinvh = V @ np.diag(lam ** -0.5) @ V.T
    Hp = Sinvh @ H @ Sinvh
    Hp = (Hp + Hp.T) / 2.0
    eps, W = np.linalg.eigh(Hp)
    C = Sinvh @ W
    f = 1.0 / (1.0 + np.exp(beta * (eps - mu)))
    return C @ np.diag(f) @ C.T, Sinvh


def css_audit(v, v_s):
    for x in (v, v_s):
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1:
            raise ValueError("invalid Krylov dimension")
    D = _distances()
    u = _probe()
    rows = []
    for variant in (1, 2):
        H, S, mu = _build(variant)
        rho = css_density(H, S, D, _DCUT_RHO, _DCUT_S, v, v_s, _BETA, mu)
        P_exact, Sinvh_exact = _exact(H, S, _BETA, mu)
        Shat = css_inv_sqrt(S, D, _DCUT_S, v_s)
        eband = float(np.trace(rho @ H))
        ne = float(np.trace(rho @ S))
        err_p = float(np.linalg.norm(rho - P_exact))
        err_s = float(np.linalg.norm(Shat - Sinvh_exact))
        nc_r = float(greedy_multicoloring(D, _DCUT_RHO).max() + 1)
        nc_s = float(greedy_multicoloring(D, _DCUT_S).max() + 1)
        sp = float(u @ rho @ u)
        rows.append([eband, ne, err_p, err_s, nc_r, nc_s, sp])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
