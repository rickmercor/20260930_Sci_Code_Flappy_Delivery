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

def assemble_sketched_instance(n, d, s, data_seed, sketch_seed):
    if not isinstance(n, (int, np.integer)) or not isinstance(d, (int, np.integer)):
        raise ValueError("n and d must be integers")
    n, d = int(n), int(d)
    if n < 2 or d < 1:
        raise ValueError("require n >= 2 and d >= 1")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("s must be positive and finite")
    rng = np.random.default_rng(int(data_seed))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    A = (Q * s) @ Q.T
    A = 0.5 * (A + A.T)
    b = rng.standard_normal(n)
    Omega = np.random.default_rng(int(sketch_seed)).standard_normal((d, n))
    return np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))

import numpy as np

def _unpack_instance(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("instance pack is too short")
    n, d = [int(round(float(v))) for v in p[:2]]
    if min(n, d) < 1:
        raise ValueError("packed shapes must be positive")
    need = 2 + n * n + n + d * n
    if p.size != need:
        raise ValueError("instance pack length does not match header")
    A = p[2 : 2 + n * n].reshape(n, n)
    b = p[2 + n * n : 2 + n * n + n]
    Omega = p[2 + n * n + n :].reshape(d, n)
    return A, b, Omega

def _pack_start_state(n, m, n_hess, U, H, leftover, h_next):
    return np.concatenate(
        (
            [float(n), float(m), float(n_hess)],
            np.asarray(U, dtype=float).ravel(),
            np.asarray(H, dtype=float).ravel(),
            np.asarray(leftover, dtype=float).ravel(),
            [float(h_next)],
        )
    )

def start_krylov_pack(inst_pack, m):
    _A, b, Omega = _unpack_instance(inst_pack)
    n = b.shape[0]
    d = Omega.shape[0]
    if not isinstance(m, (int, np.integer)):
        raise ValueError("m must be an integer")
    m = int(m)
    if m < 1 or m >= n:
        raise ValueError("require 1 <= m < n")
    if d < m:
        raise ValueError("require d >= m")
    beta = float(np.linalg.norm(Omega @ b))
    if not np.isfinite(beta) or beta == 0.0:
        raise ValueError("Omega b must be nonzero")
    U = np.zeros((n, m))
    U[:, 0] = b / beta
    H = np.zeros((m, m))
    leftover = np.zeros(n)
    return _pack_start_state(n, m, 0, U, H, leftover, 0.0)

import numpy as np

def sketched_action_pack(Omega, U, w):
    Omega = np.asarray(Omega, dtype=float)
    U = np.asarray(U, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if Omega.ndim != 2:
        raise ValueError("Omega must be 2-dimensional")
    if U.ndim != 2:
        raise ValueError("U must be 2-dimensional")
    n = U.shape[0]
    k = U.shape[1]
    if k < 1:
        raise ValueError("U must have at least one column")
    if Omega.shape[1] != n:
        raise ValueError("Omega must have shape (d, n)")
    d = Omega.shape[0]
    if d < k:
        raise ValueError("require d >= k")
    if w.shape != (n,):
        raise ValueError("w must have length n")
    if not np.all(np.isfinite(Omega)) or not np.all(np.isfinite(U)) or not np.all(np.isfinite(w)):
        raise ValueError("inputs must be finite")
    hk, *_ = np.linalg.lstsq(Omega @ U, Omega @ w, rcond=None)
    leftover = w - U @ hk
    scale = float(np.linalg.norm(Omega @ leftover))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("Arnoldi breakdown")
    return np.concatenate(([float(k)], hk.ravel(), leftover, [scale]))

import numpy as np

def _unpack_append_state(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("Krylov pack is too short")
    n, m, n_hess = [int(round(float(v))) for v in p[:3]]
    if n < 2 or m < 1 or n_hess < 0 or n_hess > m:
        raise ValueError("Krylov pack header is invalid")
    need = 3 + n * m + m * m + n + 1
    if p.size != need:
        raise ValueError("Krylov pack length does not match header")
    U = p[3 : 3 + n * m].reshape(n, m)
    H = p[3 + n * m : 3 + n * m + m * m].reshape(m, m)
    leftover = p[3 + n * m + m * m : 3 + n * m + m * m + n]
    h_next = float(p[-1])
    return n, m, n_hess, U, H, leftover, h_next

def _unpack_action(pack, n):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("action pack is too short")
    k = int(round(float(p[0])))
    if k < 1:
        raise ValueError("action width must be positive")
    need = 1 + k + n + 1
    if p.size != need:
        raise ValueError("action pack length does not match header")
    h = p[1 : 1 + k]
    leftover = p[1 + k : 1 + k + n]
    scale = float(p[-1])
    return k, h, leftover, scale

def _pack_append_state(n, m, n_hess, U, H, leftover, h_next):
    return np.concatenate(
        (
            [float(n), float(m), float(n_hess)],
            np.asarray(U, dtype=float).ravel(),
            np.asarray(H, dtype=float).ravel(),
            np.asarray(leftover, dtype=float).ravel(),
            [float(h_next)],
        )
    )

def append_krylov_column(state, action):
    n, m, n_hess, U, H, leftover, h_next = _unpack_append_state(state)
    if n_hess >= m:
        raise ValueError("Krylov pack is already complete")
    k, h, r, scale = _unpack_action(action, n)
    if k != n_hess + 1:
        raise ValueError("action width must equal n_hess + 1")
    if h.shape != (k,):
        raise ValueError("coefficient length must equal k")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("continuation scale must be positive")
    U = np.array(U, dtype=float, copy=True)
    H = np.array(H, dtype=float, copy=True)
    H[:k, n_hess] = h
    if n_hess + 1 < m:
        H[k, n_hess] = scale
        U[:, n_hess + 1] = r / scale
        leftover = np.zeros(n)
        h_next = 0.0
    else:
        leftover = r / scale
        h_next = scale
    return _pack_append_state(n, m, n_hess + 1, U, H, leftover, h_next)

import numpy as np

def _unpack_leftover_state(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("Krylov pack is too short")
    n, m, n_hess = [int(round(float(v))) for v in p[:3]]
    if n < 2 or m < 1 or n_hess < 0 or n_hess > m:
        raise ValueError("Krylov pack header is invalid")
    need = 3 + n * m + m * m + n + 1
    if p.size != need:
        raise ValueError("Krylov pack length does not match header")
    U = p[3 : 3 + n * m].reshape(n, m)
    leftover = p[3 + n * m + m * m : 3 + n * m + m * m + n]
    return n, m, n_hess, U, leftover

def leftover_euclidean_coeffs(state):
    _n, m, n_hess, U, leftover = _unpack_leftover_state(state)
    if n_hess != m:
        raise ValueError("Krylov pack is incomplete")
    G = U.T @ U
    try:
        return np.linalg.solve(G, U.T @ leftover)
    except np.linalg.LinAlgError as exc:
        raise ValueError("U^T U must be SPD") from exc

import numpy as np

def _unpack_restore_state(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("Krylov pack is too short")
    n, m, n_hess = [int(round(float(v))) for v in p[:3]]
    if n < 2 or m < 1 or n_hess < 0 or n_hess > m:
        raise ValueError("Krylov pack header is invalid")
    need = 3 + n * m + m * m + n + 1
    if p.size != need:
        raise ValueError("Krylov pack length does not match header")
    H = p[3 + n * m : 3 + n * m + m * m].reshape(m, m)
    h_next = float(p[-1])
    return n, m, n_hess, H, h_next

def restore_last_column(state, hhat):
    _n, m, n_hess, H, h_next = _unpack_restore_state(state)
    if n_hess != m:
        raise ValueError("Krylov pack is incomplete")
    hhat = np.asarray(hhat, dtype=float).reshape(-1)
    if hhat.shape != (m,):
        raise ValueError("hhat must have length m")
    if not np.all(np.isfinite(hhat)):
        raise ValueError("hhat must be finite")
    Hbar = np.array(H, dtype=float, copy=True)
    Hbar[:, -1] = Hbar[:, -1] + h_next * hhat
    return Hbar

import numpy as np

def dominant_real_eigenvalue(Hbar):
    Hbar = np.asarray(Hbar, dtype=float)
    if Hbar.ndim != 2 or Hbar.shape[0] != Hbar.shape[1]:
        raise ValueError("Hbar must be square")
    if Hbar.shape[0] < 1:
        raise ValueError("Hbar must be nonempty")
    vals = np.linalg.eigvals(Hbar)
    if np.any(np.abs(vals.imag) > 1e-8):
        raise ValueError("corrected Hessenberg must have real eigenvalues")
    return float(np.max(vals.real))

import numpy as np


def _unpack_compress_state(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("Krylov pack is too short")
    n, m, n_hess = [int(round(float(v))) for v in p[:3]]
    if n < 2 or m < 1 or n_hess < 0 or n_hess > m:
        raise ValueError("Krylov pack header is invalid")
    need = 3 + n * m + m * m + n + 1
    if p.size != need:
        raise ValueError("Krylov pack length does not match header")
    U = p[3 : 3 + n * m].reshape(n, m)
    H = p[3 + n * m : 3 + n * m + m * m].reshape(m, m)
    leftover = p[3 + n * m + m * m : 3 + n * m + m * m + n]
    h_next = float(p[-1])
    return n, m, n_hess, U, H, leftover, h_next


def _ordered_schur_vectors(M, l):
    vals, vecs = np.linalg.eig(M)
    if np.any(np.abs(vals.imag) > 1e-8):
        raise ValueError("corrected matrix must have real eigenvalues")
    vals = vals.real
    order = np.argsort(-vals, kind="stable")
    if abs(vals[order[l - 1]] - vals[order[l]]) <= 1e-12:
        raise ValueError("wanted and unwanted Ritz values must be separated")
    V, _ = np.linalg.qr(vecs[:, order[:l]].real)
    for j in range(l):
        i = int(np.argmax(np.abs(V[:, j])))
        if V[i, j] < 0.0:
            V[:, j] = -V[:, j]
    return V


def compress_ritz_pack(state, hhat, Hbar, l):
    n, m, n_hess, U, H, leftover, h_next = _unpack_compress_state(state)
    if n_hess != m:
        raise ValueError("Krylov pack is incomplete")
    hhat = np.asarray(hhat, dtype=float).reshape(-1)
    if hhat.shape != (m,) or not np.all(np.isfinite(hhat)):
        raise ValueError("hhat must be a finite vector of length m")
    Hbar = np.asarray(Hbar, dtype=float)
    if Hbar.shape != (m, m) or not np.all(np.isfinite(Hbar)):
        raise ValueError("Hbar must be a finite m-by-m matrix")
    expected = np.array(H, dtype=float, copy=True)
    expected[:, -1] = expected[:, -1] + h_next * hhat
    if not np.allclose(Hbar, expected, rtol=1e-9, atol=1e-9):
        raise ValueError("Hbar must be the last-column restoration of H")
    if not isinstance(l, (int, np.integer)):
        raise ValueError("l must be an integer")
    l = int(l)
    if l < 1 or l >= m:
        raise ValueError("require 1 <= l < m")
    V = _ordered_schur_vectors(Hbar, l)
    S = V.T @ Hbar @ V
    U_l = U @ V
    uhat = leftover - U @ hhat
    c = V.T @ (h_next * np.eye(m)[m - 1])
    return np.concatenate(
        ([float(n), float(l)], U_l.ravel(), S.ravel(), uhat.ravel(), c.ravel())
    )

import numpy as np


def _unpack_expand_instance(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("instance pack is too short")
    n, d = [int(round(float(v))) for v in p[:2]]
    if min(n, d) < 1:
        raise ValueError("packed shapes must be positive")
    need = 2 + n * n + n + d * n
    if p.size != need:
        raise ValueError("instance pack length does not match header")
    if not np.all(np.isfinite(p)):
        raise ValueError("instance pack must be finite")
    A = p[2 : 2 + n * n].reshape(n, n)
    Omega = p[2 + n * n + n :].reshape(d, n)
    return n, d, A, Omega


def _unpack_decomp(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("decomposition pack is too short")
    n, l = [int(round(float(v))) for v in p[:2]]
    if n < 2 or l < 1 or l >= n:
        raise ValueError("decomposition pack header is invalid")
    need = 2 + n * l + l * l + n + l
    if p.size != need:
        raise ValueError("decomposition pack length does not match header")
    if not np.all(np.isfinite(p)):
        raise ValueError("decomposition pack must be finite")
    U_l = p[2 : 2 + n * l].reshape(n, l)
    H_l = p[2 + n * l : 2 + n * l + l * l].reshape(l, l)
    uhat = p[2 + n * l + l * l : 2 + n * l + l * l + n]
    c = p[2 + n * l + l * l + n :]
    return n, l, U_l, H_l, uhat, c


def expand_krylov_cycle(inst_pack, decomp, m):
    n, d, A, Omega = _unpack_expand_instance(inst_pack)
    n2, l, U_l, H_l, uhat, c = _unpack_decomp(decomp)
    if n2 != n:
        raise ValueError("instance and decomposition disagree on n")
    if not isinstance(m, (int, np.integer)):
        raise ValueError("m must be an integer")
    m = int(m)
    if m < l + 1 or m >= n:
        raise ValueError("require l + 1 <= m < n")
    if d < m:
        raise ValueError("require d >= m")
    U = np.zeros((n, m))
    H = np.zeros((m, m))
    U[:, :l] = U_l
    U[:, l] = uhat
    H[:l, :l] = H_l
    H[l, :l] = c
    leftover = np.zeros(n)
    h_next = 0.0
    for k in range(l, m):
        w = A @ U[:, k]
        hk, *_ = np.linalg.lstsq(Omega @ U[:, : k + 1], Omega @ w, rcond=None)
        r = w - U[:, : k + 1] @ hk
        scale = float(np.linalg.norm(Omega @ r))
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("Arnoldi breakdown")
        H[: k + 1, k] = hk
        if k + 1 < m:
            H[k + 1, k] = scale
            U[:, k + 1] = r / scale
        else:
            leftover = r / scale
            h_next = scale
    return np.concatenate(
        (
            [float(n), float(m), float(m)],
            U.ravel(),
            H.ravel(),
            leftover.ravel(),
            [float(h_next)],
        )
    )

import numpy as np


def _prior_oracle(public_name):
    """Return a sibling * function, never a public stub. """
    oracle_name = "" + public_name
    fn = globals().get(oracle_name)
    if not callable(fn):
        raise ValueError(f"missing oracle for {public_name}")
    return fn


def _unpack_restart_instance(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("instance pack is too short")
    n, d = [int(round(float(v))) for v in p[:2]]
    if min(n, d) < 1:
        raise ValueError("packed shapes must be positive")
    need = 2 + n * n + n + d * n
    if p.size != need:
        raise ValueError("instance pack length does not match header")
    A = p[2 : 2 + n * n].reshape(n, n)
    b = p[2 + n * n : 2 + n * n + n]
    Omega = p[2 + n * n + n :].reshape(d, n)
    return A, b, Omega


def _unpack_restart_state(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("Krylov pack is too short")
    n, m, n_hess = [int(round(float(v))) for v in p[:3]]
    if n < 2 or m < 1 or n_hess < 0 or n_hess > m:
        raise ValueError("Krylov pack header is invalid")
    need = 3 + n * m + m * m + n + 1
    if p.size != need:
        raise ValueError("Krylov pack length does not match header")
    U = p[3 : 3 + n * m].reshape(n, m)
    return n, m, n_hess, U


def run_restarted_correction_norm(n, m, d, l, s, data_seed, sketch_seed):
    for name, val in (("n", n), ("m", m), ("d", d), ("l", l)):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    n, m, d, l = int(n), int(m), int(d), int(l)
    if n < 3:
        raise ValueError("require n >= 3")
    if m < 2 or m >= n:
        raise ValueError("require 2 <= m < n")
    if d < m:
        raise ValueError("require d >= m")
    if l < 1 or l >= m:
        raise ValueError("require 1 <= l < m")

    assemble_sketched_instance = _prior_oracle("assemble_sketched_instance")
    start_krylov_pack = _prior_oracle("start_krylov_pack")
    sketched_action_pack = _prior_oracle("sketched_action_pack")
    append_krylov_column = _prior_oracle("append_krylov_column")
    leftover_euclidean_coeffs = _prior_oracle("leftover_euclidean_coeffs")
    restore_last_column = _prior_oracle("restore_last_column")
    dominant_real_eigenvalue = _prior_oracle("dominant_real_eigenvalue")
    compress_ritz_pack = _prior_oracle("compress_ritz_pack")
    expand_krylov_cycle = _prior_oracle("expand_krylov_cycle")

    s_arr = np.asarray(s, dtype=float).reshape(-1)
    inst = assemble_sketched_instance(n, d, s_arr, data_seed, sketch_seed)
    A, _b, Omega = _unpack_restart_instance(inst)
    s_max = float(np.max(s_arr))
    tol = 1e-8 * max(1.0, s_max)

    # First cycle: m sketched Arnoldi steps, then the similarity correction.
    state = start_krylov_pack(inst, m)
    for _ in range(m):
        _n, _m, n_hess, U = _unpack_restart_state(state)
        w = A @ U[:, n_hess]
        action = sketched_action_pack(Omega, U[:, : n_hess + 1], w)
        state = append_krylov_column(state, action)
    hhat = leftover_euclidean_coeffs(state)
    Hbar = restore_last_column(state, hhat)
    if dominant_real_eigenvalue(Hbar) > s_max + tol:
        raise ValueError("corrected Ritz value outside the spectrum of A")

    # Restart: compress to order l, re-expand to order m, correct again.
    decomp = compress_ritz_pack(state, hhat, Hbar, l)
    state_new = expand_krylov_cycle(inst, decomp, m)
    hhat_new = leftover_euclidean_coeffs(state_new)
    Hbar_new = restore_last_column(state_new, hhat_new)
    if dominant_real_eigenvalue(Hbar_new) > s_max + tol:
        raise ValueError("corrected Ritz value outside the spectrum of A")
    return float(np.linalg.norm(hhat_new))
SCICODE_GOLD_EOF
