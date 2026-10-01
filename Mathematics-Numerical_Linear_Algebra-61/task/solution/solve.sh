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

def construct_consistent_system(m, n, s, seed):
    if not isinstance(m, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("m and n must be integers")
    m, n = int(m), int(n)
    if n < 2 or m <= n:
        raise ValueError("require m > n >= 2")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("s must be positive and finite")
    rng = np.random.default_rng(int(seed))
    U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
    V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    A = U @ np.diag(s) @ V.T
    x_star = rng.standard_normal(n)
    b = A @ x_star
    return np.concatenate(([float(m), float(n)], A.ravel(), b))

import numpy as np

def _unpack_system(pack):
    pack = np.asarray(pack, dtype=float).reshape(-1)
    if pack.size < 2:
        raise ValueError("pack is too short")
    m, n = int(round(float(pack[0]))), int(round(float(pack[1])))
    if m < 1 or n < 1:
        raise ValueError("packed shapes must be positive")
    need = 2 + m * n + m
    if pack.size != need:
        raise ValueError("pack length does not match header")
    A = pack[2 : 2 + m * n].reshape(m, n)
    b = pack[2 + m * n :]
    return A, b

def extract_row_blocks(pack, indices):
    A, b = _unpack_system(pack)
    m, n = A.shape
    idx = np.asarray(indices, dtype=int).reshape(-1)
    mp = idx.size
    if mp < 1 or mp >= m:
        raise ValueError("require 1 <= mp < m")
    if np.unique(idx).size != mp:
        raise ValueError("indices must be distinct")
    if np.any(idx < 0) or np.any(idx >= m):
        raise ValueError("indices out of range")
    mask = np.ones(m, dtype=bool)
    mask[idx] = False
    Ir = np.where(mask)[0]
    A_p, b_p = A[idx], b[idx]
    A_r, b_r = A[Ir], b[Ir]
    mr = A_r.shape[0]
    return np.concatenate(
        ([float(mp), float(n), float(mr)], A_p.ravel(), b_p, A_r.ravel(), b_r)
    )

import numpy as np

def _unpack_constraint_init_pack(block_pack):
    p = np.asarray(block_pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("block pack is too short")
    mp, n, mr = [int(round(float(x))) for x in p[:3]]
    if min(mp, n, mr) < 1:
        raise ValueError("packed shapes must be positive")
    need = 3 + mp * n + mp + mr * n + mr
    if p.size != need:
        raise ValueError("block pack length does not match header")
    i = 3
    A_p = p[i : i + mp * n].reshape(mp, n)
    i += mp * n
    b_p = p[i : i + mp]
    i += mp
    A_r = p[i : i + mr * n].reshape(mr, n)
    i += mr * n
    b_r = p[i:]
    return A_p, b_p, A_r, b_r

def constraint_initial_iterate(block_pack):
    A_p, b_p, _A_r, _b_r = _unpack_constraint_init_pack(block_pack)
    A_p = np.asarray(A_p, dtype=float)
    b_p = np.asarray(b_p, dtype=float).reshape(-1)
    if A_p.ndim != 2 or A_p.shape[0] < 1 or A_p.shape[1] < 1:
        raise ValueError("constraint block A_p must be a nonempty 2d array")
    if b_p.shape != (A_p.shape[0],):
        raise ValueError("b_p must have shape (mp,)")
    if not np.all(np.isfinite(A_p)) or not np.all(np.isfinite(b_p)):
        raise ValueError("constraint block must be finite")
    return np.linalg.pinv(A_p) @ b_p

import numpy as np

def draw_gaussian_sketches(m_r, q, n_steps, seed):
    for name, val in (("m_r", m_r), ("q", q), ("n_steps", n_steps)):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    m_r, q, n_steps = int(m_r), int(q), int(n_steps)
    if min(m_r, q, n_steps) < 1:
        raise ValueError("require m_r, q, n_steps >= 1")
    rng = np.random.default_rng(int(seed))
    pieces = [np.array([float(n_steps), float(m_r), float(q)])]
    for _ in range(n_steps):
        pieces.append(rng.standard_normal((m_r, q)).ravel())
    return np.concatenate(pieces)

import numpy as np

def _unpack_blocks(block_pack):
    p = np.asarray(block_pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("block pack is too short")
    mp, n, mr = [int(round(float(v))) for v in p[:3]]
    if min(mp, n, mr) < 1:
        raise ValueError("packed shapes must be positive")
    need = 3 + mp * n + mp + mr * n + mr
    if p.size != need:
        raise ValueError("block pack length does not match header")
    i = 3
    A_p = p[i : i + mp * n].reshape(mp, n)
    i += mp * n
    b_p = p[i : i + mp]
    i += mp
    A_r = p[i : i + mr * n].reshape(mr, n)
    i += mr * n
    b_r = p[i:]
    return A_p, b_p, A_r, b_r

def projected_sketched_direction(block_pack, x, S):
    A_p, _b_p, A_r, b_r = _unpack_blocks(block_pack)
    n = A_p.shape[1]
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape != (n,):
        raise ValueError("x must have shape (n,)")
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != A_r.shape[0] or S.shape[1] < 1:
        raise ValueError("S must have shape (m_r, q) with q >= 1")
    r = A_r @ x - b_r
    Sr = S.T @ r
    g = -A_r.T @ (S @ Sr)
    d = g - np.linalg.pinv(A_p) @ (A_p @ g)
    return d

import numpy as np

def window_orthogonalize(d, window):
    d = np.asarray(d, dtype=float).reshape(-1)
    n = d.size
    if n < 1 or not np.all(np.isfinite(d)):
        raise ValueError("d must be a nonempty finite vector")
    W = np.asarray(window, dtype=float)
    if W.size == 0:
        W = np.zeros((n, 0))
    if W.ndim != 2 or W.shape[0] != n:
        raise ValueError("window must have shape (n, w)")
    p = d.copy()
    for j in range(W.shape[1]):
        pj = W[:, j]
        nrm2 = float(np.dot(pj, pj))
        if nrm2 <= 0.0:
            raise ValueError("window directions must be nonzero")
        p = p - (np.dot(d, pj) / nrm2) * pj
    return p

import numpy as np

def line_search_step(A_r, b_r, x, S, p):
    A_r = np.asarray(A_r, dtype=float)
    if A_r.ndim != 2 or min(A_r.shape) < 1:
        raise ValueError("A_r must be a nonempty 2d array")
    m_r, n = A_r.shape
    b_r = np.asarray(b_r, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    if b_r.shape != (m_r,) or x.shape != (n,) or p.shape != (n,):
        raise ValueError("incompatible shapes")
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != m_r or S.shape[1] < 1:
        raise ValueError("S must have shape (m_r, q) with q >= 1")
    nrm2 = float(np.dot(p, p))
    if nrm2 <= 0.0:
        raise ValueError("search direction must be nonzero")
    r = A_r @ x - b_r
    Sr = S.T @ r
    delta = float(np.dot(Sr, Sr) / nrm2)
    return x + delta * p

import numpy as np

def _unpack_blocks(block_pack):
    p = np.asarray(block_pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("block pack is too short")
    mp, n, mr = [int(round(float(v))) for v in p[:3]]
    if min(mp, n, mr) < 1:
        raise ValueError("packed shapes must be positive")
    need = 3 + mp * n + mp + mr * n + mr
    if p.size != need:
        raise ValueError("block pack length does not match header")
    i = 3
    A_p = p[i : i + mp * n].reshape(mp, n)
    i += mp * n
    b_p = p[i : i + mp]
    i += mp
    A_r = p[i : i + mr * n].reshape(mr, n)
    i += mr * n
    b_r = p[i:]
    return A_p, b_p, A_r, b_r


def _unpack_sketches(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("sketch pack is too short")
    n_steps, m_r, q = [int(round(float(v))) for v in p[:3]]
    if min(n_steps, m_r, q) < 1:
        raise ValueError("packed sketch shapes must be positive")
    need = 3 + n_steps * m_r * q
    if p.size != need:
        raise ValueError("sketch pack length does not match header")
    out = []
    i = 3
    for _ in range(n_steps):
        out.append(p[i : i + m_r * q].reshape(m_r, q))
        i += m_r * q
    return out


def run_constrained_sketch_entry(
    m, n, s, data_seed, sketch_seed, indices, q, n_steps, ell, coord
):
    for name, val in (("q", q), ("n_steps", n_steps), ("ell", ell), ("coord", coord)):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    q, n_steps, ell, coord = int(q), int(n_steps), int(ell), int(coord)
    if min(q, n_steps, ell) < 1:
        raise ValueError("require q, n_steps, ell >= 1")
    n = int(n)
    if coord < 0 or coord >= n:
        raise ValueError("coord out of range")

    sys_pack = construct_consistent_system(m, n, s, data_seed)
    block_pack = extract_row_blocks(sys_pack, indices)
    A_p, _b_p, A_r, b_r = _unpack_blocks(block_pack)
    m_r = A_r.shape[0]
    x = constraint_initial_iterate(block_pack)
    sk_pack = draw_gaussian_sketches(m_r, q, n_steps, sketch_seed)
    sketches = _unpack_sketches(sk_pack)
    hist = []
    for k in range(n_steps):
        S = sketches[k]
        d = projected_sketched_direction(block_pack, x, S)
        j0 = max(k - ell + 1, 0)
        if k == 0 or j0 >= k:
            window = np.zeros((n, 0))
        else:
            window = np.column_stack(hist[j0:k])
        p = window_orthogonalize(d, window)
        x = line_search_step(A_r, b_r, x, S, p)
        hist.append(p)
    return float(x[coord])
SCICODE_GOLD_EOF
