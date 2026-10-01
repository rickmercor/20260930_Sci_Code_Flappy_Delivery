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
from typing import Callable


def difference_potential_matrix(V: "Callable[[np.ndarray], np.ndarray]",
                                        x_grid: "np.ndarray",
                                        y_grid: "np.ndarray") -> "np.ndarray":
    if not callable(V):
        raise ValueError("V must be a callable accepting a numpy array")
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_grid and y_grid must be one-dimensional")
    if x.size < 1 or y.size < 1:
        raise ValueError("x_grid and y_grid must be non-empty")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x_grid and y_grid must contain only finite values")
    Xp = x[:, None] + 0.5 * y[None, :]
    Xm = x[:, None] - 0.5 * y[None, :]
    out = np.asarray(V(Xp), dtype=float) - np.asarray(V(Xm), dtype=float)
    if out.shape != (x.size, y.size):
        raise ValueError("V must broadcast elementwise over its argument")
    return out.astype(float)

import numpy as np


def separation_factors(D_V: "np.ndarray", rtol: float = 1e-12) -> "np.ndarray":
    A = np.asarray(D_V, dtype=float)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        raise ValueError("D_V must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(A)):
        raise ValueError("D_V must contain only finite values")
    if not isinstance(rtol, (int, float)) or not (0.0 < float(rtol) < 1.0):
        raise ValueError("rtol must satisfy 0 < rtol < 1")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    R = 0 if s.size == 0 or s[0] <= 0.0 else int(np.count_nonzero(s > float(rtol) * s[0]))
    root = np.sqrt(s[:R])
    DX = U[:, :R] * root
    DY = Vt[:R, :].T * root
    for j in range(R):
        col = DX[:, j]
        big = np.flatnonzero(np.abs(col) > 1e-12 * np.abs(col).max())
        if big.size and col[big[0]] < 0.0:
            DX[:, j] = -DX[:, j]
            DY[:, j] = -DY[:, j]
    return np.vstack([DX, DY]).astype(float)

import numpy as np


def spectral_derivative_matrix(n_points: int, period: float) -> "np.ndarray":
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("n_points must be an integer")
    n = int(n_points)
    if n < 2:
        raise ValueError("n_points must be at least 2")
    if not isinstance(period, (int, float, np.floating, np.integer)):
        raise ValueError("period must be a real number")
    p = float(period)
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError("period must be finite and positive")
    w = 2.0 * np.pi * np.fft.fftfreq(n, d=p / n)
    eye = np.eye(n, dtype=float)
    D = np.real(np.fft.ifft(1j * w[:, None] * np.fft.fft(eye, axis=0), axis=0))
    return D.astype(float)

import numpy as np


def initial_low_rank_state(f_grid: "np.ndarray", rank: int) -> "np.ndarray":
    A = np.asarray(f_grid, dtype=float)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        raise ValueError("f_grid must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(A)):
        raise ValueError("f_grid must contain only finite values")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer")
    r = int(rank)
    if r < 1 or r > min(A.shape):
        raise ValueError("rank must satisfy 1 <= rank <= min(f_grid.shape)")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if s[0] <= 0.0 or s[r - 1] <= 1e-14 * s[0]:
        raise ValueError("f_grid does not carry the requested rank; S would be singular")
    Ur = U[:, :r].copy()
    Vr = Vt[:r, :].T.copy()
    for j in range(r):
        col = Ur[:, j]
        peak = np.abs(col).max()
        anchor = int(np.flatnonzero(np.abs(col) >= (1.0 - 1e-9) * peak)[0])
        if col[anchor] < 0.0:
            Ur[:, j] = -Ur[:, j]
            Vr[:, j] = -Vr[:, j]
    return np.vstack([Ur, np.diag(s[:r]), Vr]).astype(float)

import numpy as np


def wigner_rhs(state: "np.ndarray",
                       d_x: "np.ndarray",
                       x_grid: "np.ndarray",
                       k_grid: "np.ndarray",
                       factors: "np.ndarray") -> "np.ndarray":
    Y = np.asarray(state, dtype=float)
    Dx = np.asarray(d_x, dtype=float)
    xg = np.asarray(x_grid, dtype=float)
    kg = np.asarray(k_grid, dtype=float)
    Fc = np.asarray(factors, dtype=float)
    if Y.ndim != 2:
        raise ValueError("state must be a two-dimensional array")
    if xg.ndim != 1 or kg.ndim != 1 or xg.size < 1 or kg.size < 1:
        raise ValueError("x_grid and k_grid must be non-empty one-dimensional arrays")
    nx, nk, r = xg.size, kg.size, Y.shape[1]
    if r < 1 or Y.shape[0] != nx + r + nk:
        raise ValueError("state rows must equal len(x_grid) + rank + len(k_grid)")
    if Dx.shape != (nx, nx):
        raise ValueError("the differentiation operator must be square and match the spatial grid")
    if Fc.ndim != 2 or Fc.shape[0] != nx + nk:
        raise ValueError("factors must be two-dimensional with len(x_grid) + len(k_grid) rows")
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(Dx)) and np.all(np.isfinite(Fc))):
        raise ValueError("inputs must contain only finite values")
    U = Y[:nx, :]
    S = Y[nx:nx + r, :]
    V = Y[nx + r:, :]
    rhs = -(Dx @ U) @ S @ (kg[:, None] * V).T
    if Fc.shape[1] > 0:
        # Y-truncation: move the right factors to the dual variable, multiply by each
        # dual factor, return to the wave vector, and apply 1/i.
        DX = Fc[:nx, :]
        DY = np.fft.ifftshift(Fc[nx:, :], axes=0).copy()
        if nk % 2 == 0:
            DY[nk // 2, :] = 0.0
        coef = np.fft.ifft(V, axis=0)
        for s in range(Fc.shape[1]):
            Ks = np.real(-1j * np.fft.fft(DY[:, s][:, None] * coef, axis=0))
            rhs = rhs + (DX[:, s][:, None] * U) @ S @ Ks.T
    return rhs.astype(float)

import numpy as np


def projector_splitting_step(state: "np.ndarray",
                                     increment: "np.ndarray") -> "np.ndarray":
    Y = np.asarray(state, dtype=float)
    dA = np.asarray(increment, dtype=float)
    if Y.ndim != 2 or dA.ndim != 2:
        raise ValueError("state and increment must be two-dimensional arrays")
    r = Y.shape[1]
    nx, nk = dA.shape
    if r < 1 or Y.shape[0] != nx + r + nk:
        raise ValueError("state rows must equal increment rows + rank + increment columns")
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(dA))):
        raise ValueError("inputs must contain only finite values")
    U = Y[:nx, :]
    S = Y[nx:nx + r, :]
    V = Y[nx + r:, :]

    def _h_factor(term_a, term_b):
        M = term_a + term_b
        if M.shape[0] < M.shape[1]:
            raise ValueError("factorised matrix is not of full column rank")
        scale = np.linalg.norm(term_a) + np.linalg.norm(term_b)
        sv = np.linalg.svd(M, compute_uv=False)
        if scale <= 0.0 or sv[0] <= 1e-13 * scale:
            raise ValueError("factorised matrix has cancelled to rounding error")
        if sv[-1] <= 1e-15 * sv[0]:
            raise ValueError("factorised matrix is not of full column rank")
        Q, R = np.linalg.qr(M)
        d = np.sign(np.diag(R))
        d[d == 0.0] = 1.0
        return Q * d, (R.T * d).T

    U1, Shat = _h_factor(U @ S, dA @ V)
    Stil = Shat - U1.T @ dA @ V
    V1, Rt = _h_factor(V @ Stil.T, dA.T @ U1)
    return np.vstack([U1, Rt.T, V1]).astype(float)

import numpy as np


def two_stage_tableau(b1: float) -> "np.ndarray":
    if isinstance(b1, bool) or not isinstance(b1, (int, float, np.integer, np.floating)):
        raise ValueError("b1 must be a real number")
    v = float(b1)
    if not np.isfinite(v):
        raise ValueError("b1 must be finite")
    b2 = 1.0 - v
    if b2 == 0.0:
        raise ValueError("b1 = 1 leaves no second-stage weight; no admissible tableau")
    a = 0.5 / b2
    if not (0.0 < a <= 1.0):
        raise ValueError("no admissible tableau: the stage abscissa must satisfy 0 < a <= 1")
    return np.array([a, v, b2], dtype=float)

import numpy as np


def robust_psrk_step(state: "np.ndarray",
                             d_x: "np.ndarray",
                             x_grid: "np.ndarray",
                             k_grid: "np.ndarray",
                             factors: "np.ndarray",
                             h: float,
                             b1: float) -> "np.ndarray":
    Y = np.asarray(state, dtype=float)
    Dx = np.asarray(d_x, dtype=float)
    xg = np.asarray(x_grid, dtype=float)
    kg = np.asarray(k_grid, dtype=float)
    if Y.ndim != 2:
        raise ValueError("state must be a two-dimensional array")
    if xg.ndim != 1 or kg.ndim != 1 or xg.size < 1 or kg.size < 1:
        raise ValueError("grids must be non-empty one-dimensional arrays")
    nx, nk, r = xg.size, kg.size, Y.shape[1]
    if r < 1 or Y.shape[0] != nx + r + nk:
        raise ValueError("state rows must equal len(x_grid) + rank + len(k_grid)")
    if Dx.shape != (nx, nx):
        raise ValueError("the differentiation operator must be square and match the spatial grid")
    if isinstance(h, bool) or not isinstance(h, (int, float, np.integer, np.floating)):
        raise ValueError("h must be a real number")
    hh = float(h)
    if not np.isfinite(hh) or hh <= 0.0:
        raise ValueError("h must be finite and positive")
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(Dx))):
        raise ValueError("inputs must contain only finite values")
    tab = two_stage_tableau(b1)
    a, w1, w2 = float(tab[0]), float(tab[1]), float(tab[2])
    F1 = wigner_rhs(Y, Dx, xg, kg, factors)
    stage = projector_splitting_step(Y, (a * hh) * F1)
    F2 = wigner_rhs(stage, Dx, xg, kg, factors)
    return projector_splitting_step(Y, hh * (w1 * F1 + w2 * F2)).astype(float)

import numpy as np


def wigner_dlra_target(n_points: int,
                               half_width: float,
                               theta: float,
                               rank: int,
                               t_final: float,
                               n_steps: int,
                               b1: float,
                               x_eval: float,
                               k_eval: float) -> float:
    for nm, val in (("n_points", n_points), ("rank", rank), ("n_steps", n_steps)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or int(val) < 1:
            raise ValueError(nm + " must be a positive integer")
    n, r, nt = int(n_points), int(rank), int(n_steps)
    for nm, val in (("half_width", half_width), ("theta", theta),
                    ("t_final", t_final), ("x_eval", x_eval), ("k_eval", k_eval)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)) \
           or not np.isfinite(float(val)):
            raise ValueError(nm + " must be a finite real number")
    L, th, T = float(half_width), float(theta), float(t_final)
    if L <= 0.0 or T <= 0.0:
        raise ValueError("half_width and t_final must be positive")

    period = 2.0 * L
    g = -L + (period / n) * np.arange(n)

    V = lambda t: 0.5 * t ** 2
    y = (2.0 * np.pi / period) * np.arange(-(n // 2), n - n // 2)
    D_V = difference_potential_matrix(V, g, y)
    fac = separation_factors(D_V, 1e-12)
    if fac.shape[1] != 1:
        raise ValueError("the difference potential is not of separation rank one")
    if not np.allclose(fac[:n, 0][:, None] * fac[n:, 0][None, :], D_V,
                       rtol=0.0, atol=1e-9 * max(np.abs(D_V).max(), 1.0)):
        raise ValueError("separated factors do not reproduce the difference potential")

    D = spectral_derivative_matrix(n, period)

    u = np.cos(th) * g[:, None] - np.sin(th) * g[None, :]
    v = np.sin(th) * g[:, None] + np.cos(th) * g[None, :]
    f0 = np.exp(-(u - 1.0) ** 2 / 2.0 - 2.0 * v ** 2) / np.pi
    state = initial_low_rank_state(f0, r)

    h = T / nt
    tab = two_stage_tableau(b1)
    a, w1, w2 = float(tab[0]), float(tab[1]), float(tab[2])
    F1 = wigner_rhs(state, D, g, g, fac)
    stage = projector_splitting_step(state, (a * h) * F1)
    F2 = wigner_rhs(stage, D, g, g, fac)
    state = projector_splitting_step(state, h * (w1 * F1 + w2 * F2))
    for _ in range(nt - 1):
        state = robust_psrk_step(state, D, g, g, fac, h, b1)

    ix = int(np.argmin(np.abs(g - float(x_eval))))
    ik = int(np.argmin(np.abs(g - float(k_eval))))
    tol = 1e-9 * period
    if abs(g[ix] - float(x_eval)) > tol or abs(g[ik] - float(k_eval)) > tol:
        raise ValueError("evaluation point does not lie on the grid")

    U = state[:n, :]
    S = state[n:n + r, :]
    Vf = state[n + r:, :]
    return float((U @ S @ Vf.T)[ix, ik])
SCICODE_GOLD_EOF
