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


def compute_model_constants(gamma: float, psi: float, rho: float, r: float) -> np.ndarray:
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("r", r)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    gamma, psi, rho, r = float(gamma), float(psi), float(rho), float(r)
    if not gamma > 1.0:
        raise ValueError("gamma must be strictly greater than 1")
    if not (0.0 < psi < 1.0):
        raise ValueError("psi must lie strictly between 0 and 1")
    if not (rho > r > 0.0):
        raise ValueError("the ordering rho > r > 0 must hold")

    theta = (1.0 - 1.0 / psi) / (1.0 - gamma)
    lhs_coefficient = rho / theta
    consumption_exponent = (1.0 - gamma * psi) / (1.0 - gamma)
    aggregator_exponent = 1.0 - theta
    b = rho * ((r + psi * (rho - r)) / rho) ** (1.0 / (1.0 - psi))
    return np.array([theta, lhs_coefficient, consumption_exponent,
                     aggregator_exponent, b], dtype=float)

import numpy as np


def compute_discrete_barriers(gamma: float, r: float, y1: float, y2: float, b: float,
                                      xlow: float, xbar: float, nint: int) -> np.ndarray:
    for name, val in (("gamma", gamma), ("r", r), ("y1", y1), ("y2", y2),
                      ("b", b), ("xlow", xlow), ("xbar", xbar)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    if isinstance(nint, (bool, np.bool_)) or not isinstance(nint, (int, np.integer)):
        raise ValueError("nint must be an integer")
    nint = int(nint)
    if nint < 1:
        raise ValueError("nint must be a positive integer")
    gamma, r, y1, y2, b = float(gamma), float(r), float(y1), float(y2), float(b)
    xlow, xbar = float(xlow), float(xbar)
    if not gamma > 1.0:
        raise ValueError("gamma must be strictly greater than 1")
    if not r > 0.0:
        raise ValueError("r must be strictly positive")
    if not b > 0.0:
        raise ValueError("b must be strictly positive")
    if not (y2 > y1 > 0.0):
        raise ValueError("the ordering y2 > y1 > 0 must hold")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")

    dx = (xbar - xlow) / nint
    x = xlow + dx * np.arange(nint + 1)
    low_flow = r * x + y1
    if not np.all(low_flow > 0.0):
        raise ValueError("r*x + y1 must be strictly positive at every grid node")
    wide = x + y2 / r
    if not np.all(wide > 0.0):
        raise ValueError("x + y2/r must be strictly positive at every grid node")

    p = 1.0 - gamma
    lower = low_flow ** p / p
    upper = (b * wide) ** p / p
    if not np.all(lower <= upper + 1e-14 * np.abs(upper)):
        raise ValueError("the subsolution must lie at or below the supersolution")

    block_low = np.repeat(lower[:, None], 2, axis=1)
    block_up = np.repeat(upper[:, None], 2, axis=1)
    return np.vstack([block_low, block_up]).astype(float)

import numpy as np


def compute_candidate_consumptions(V: np.ndarray, gamma: float, psi: float, rho: float,
                                           r: float, y1: float, y2: float, xlow: float,
                                           xbar: float, cap: float) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[1] != 2 or V.shape[0] < 2:
        raise ValueError("V must be a 2-D array with two columns and at least two rows")
    if not np.all(np.isfinite(V)):
        raise ValueError("V must be finite")
    if not np.all(V < 0.0):
        raise ValueError("V must be strictly negative")
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("r", r),
                      ("y1", y1), ("y2", y2), ("xlow", xlow), ("xbar", xbar), ("cap", cap)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    gamma, psi, rho, r = float(gamma), float(psi), float(rho), float(r)
    y1, y2, xlow, xbar, cap = float(y1), float(y2), float(xlow), float(xbar), float(cap)
    if not gamma > 1.0:
        raise ValueError("gamma must be strictly greater than 1")
    if not (0.0 < psi < 1.0):
        raise ValueError("psi must lie strictly between 0 and 1")
    if not rho > 0.0:
        raise ValueError("rho must be strictly positive")
    if not r > 0.0:
        raise ValueError("r must be strictly positive")
    if not (y2 > y1 > 0.0):
        raise ValueError("the ordering y2 > y1 > 0 must hold")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")

    n = V.shape[0]
    nint = n - 1
    dx = (xbar - xlow) / nint
    x = xlow + dx * np.arange(n)
    cbar = r * x[:, None] + np.array([y1, y2])[None, :]
    if not np.all(cbar > 0.0):
        raise ValueError("the zero-saving consumption must be strictly positive at every node")
    if not cap > float(cbar.max()):
        raise ValueError("cap must exceed the largest zero-saving consumption on the grid")

    d = (V[1:, :] - V[:nint, :]) / dx
    unconstrained = np.where(d > 0.0,
                             rho ** psi * np.maximum(d, 1e-300) ** (-psi), np.inf)
    weight = ((1.0 - gamma) * V) ** ((1.0 - gamma * psi) / (1.0 - gamma))

    cF = cbar.copy()
    cF[:nint, :] = np.minimum(unconstrained * weight[:nint, :], cbar[:nint, :])
    cB = cbar.copy()
    cB[1:, :] = np.maximum(np.minimum(cap, unconstrained * weight[1:, :]), cbar[1:, :])
    return np.vstack([cF, cB]).astype(float)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def evaluate_fixed_policy(policies: np.ndarray, v_init: np.ndarray, gamma: float,
                                  psi: float, rho: float, r: float, y1: float, y2: float,
                                  lam1: float, lam2: float, xlow: float, xbar: float,
                                  tol: float, max_iter: int) -> np.ndarray:
    policies = np.asarray(policies, dtype=float)
    v_init = np.asarray(v_init, dtype=float)
    if policies.ndim != 2 or policies.shape[1] != 2 or policies.shape[0] < 4 or policies.shape[0] % 2:
        raise ValueError("policies must be a 2-D array with two columns and an even row count of at least 4")
    if not np.all(np.isfinite(policies)):
        raise ValueError("policies must be finite")
    n = policies.shape[0] // 2
    if v_init.ndim != 2 or v_init.shape != (n, 2):
        raise ValueError("v_init must have shape (n, 2) with n half the policy row count")
    if not np.all(np.isfinite(v_init)):
        raise ValueError("v_init must be finite")
    if not np.all(v_init < 0.0):
        raise ValueError("v_init must be strictly negative")
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("r", r), ("y1", y1),
                      ("y2", y2), ("lam1", lam1), ("lam2", lam2), ("xlow", xlow),
                      ("xbar", xbar), ("tol", tol)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError("max_iter must be an integer")
    max_iter = int(max_iter)
    gamma, psi, rho, r = float(gamma), float(psi), float(rho), float(r)
    y1, y2, lam1, lam2 = float(y1), float(y2), float(lam1), float(lam2)
    xlow, xbar, tol = float(xlow), float(xbar), float(tol)
    if not gamma > 1.0:
        raise ValueError("gamma must be strictly greater than 1")
    if not (0.0 < psi < 1.0):
        raise ValueError("psi must lie strictly between 0 and 1")
    if not rho > 0.0:
        raise ValueError("rho must be strictly positive")
    if not r > 0.0:
        raise ValueError("r must be strictly positive")
    if not (lam1 > 0.0 and lam2 > 0.0):
        raise ValueError("both switching rates must be strictly positive")
    if not (y2 > y1 > 0.0):
        raise ValueError("the ordering y2 > y1 > 0 must hold")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")
    if not tol > 0.0:
        raise ValueError("tol must be strictly positive")
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    nint = n - 1
    dx = (xbar - xlow) / nint
    x = xlow + dx * np.arange(n)
    lam = np.array([lam1, lam2])
    cbar = r * x[:, None] + np.array([y1, y2])[None, :]
    if not np.all(cbar > 0.0):
        raise ValueError("the zero-saving consumption must be strictly positive at every node")
    cF, cB = policies[:n, :], policies[n:, :]
    sF, sB = cbar - cF, cbar - cB
    if np.any(sF < -1e-12) or np.any(sB > 1e-12):
        raise ValueError("forward saving must be non-negative and backward saving non-positive")
    sF = np.maximum(sF, 0.0); sF[nint, :] = 0.0
    sB = np.minimum(sB, 0.0); sB[0, :] = 0.0

    theta = (1.0 - 1.0 / psi) / (1.0 - gamma)
    powv = 1.0 - theta
    nu = 1.0 - 1.0 / psi
    phi = (rho / nu) * (cF ** nu + cB ** nu - cbar ** nu)

    blocks = []
    for j in (0, 1):
        main = (sF[:, j] - sB[:, j]) / dx + lam[j] + rho / theta
        blocks.append(sp.diags([sB[1:, j] / dx, main, -sF[:nint, j] / dx],
                               [-1, 0, 1], format="csr"))
    eye = sp.eye(n, format="csr")
    A = sp.bmat([[blocks[0], -lam[0] * eye], [-lam[1] * eye, blocks[1]]], format="csc")

    vf = v_init.T.reshape(-1).copy()
    pf = phi.T.reshape(-1)
    converged = False
    for _ in range(max_iter):
        w = (1.0 - gamma) * vf
        res = A @ vf - pf * w ** powv
        rn = float(np.max(np.abs(res)))
        if rn < tol:
            converged = True
            break
        jac = (A - sp.diags(pf * powv * (1.0 - gamma) * w ** (powv - 1.0))).tocsc()
        step = spl.spsolve(jac, res)
        t = 1.0
        for _ in range(60):
            trial = vf - t * step
            if np.max(trial) < 0.0:
                rt = np.max(np.abs(A @ trial - pf * ((1.0 - gamma) * trial) ** powv))
                if rt < (1.0 - 1e-4 * t) * rn:
                    break
            t *= 0.5
        vf = vf - t * step
    if not converged:
        w = (1.0 - gamma) * vf
        if float(np.max(np.abs(A @ vf - pf * w ** powv))) >= tol:
            raise ValueError("the residual tolerance was not reached within max_iter")
    return vf.reshape(2, n).T.copy()

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def _h_candidates(V, cbar, dx, gamma, psi, rho, cap):
    n = V.shape[0]
    nint = n - 1
    d = (V[1:, :] - V[:nint, :]) / dx
    unconstrained = np.where(d > 0.0, rho ** psi * np.maximum(d, 1e-300) ** (-psi), np.inf)
    weight = ((1.0 - gamma) * V) ** ((1.0 - gamma * psi) / (1.0 - gamma))
    cF = cbar.copy()
    cF[:nint, :] = np.minimum(unconstrained * weight[:nint, :], cbar[:nint, :])
    cB = cbar.copy()
    cB[1:, :] = np.maximum(np.minimum(cap, unconstrained * weight[1:, :]), cbar[1:, :])
    return cF, cB


def _h_operator(sF, sB, lam, dx, rho, theta, n):
    nint = n - 1
    blocks = []
    for j in (0, 1):
        main = (sF[:, j] - sB[:, j]) / dx + lam[j] + rho / theta
        blocks.append(sp.diags([sB[1:, j] / dx, main, -sF[:nint, j] / dx],
                               [-1, 0, 1], format="csr"))
    eye = sp.eye(n, format="csr")
    return sp.bmat([[blocks[0], -lam[0] * eye], [-lam[1] * eye, blocks[1]]], format="csc")


def _h_evaluate(A, phi, v, gamma, powv, tol, max_iter, n):
    vf = v.T.reshape(-1).copy()
    pf = phi.T.reshape(-1)
    for _ in range(max_iter):
        w = (1.0 - gamma) * vf
        res = A @ vf - pf * w ** powv
        rn = float(np.max(np.abs(res)))
        if rn < tol:
            return vf.reshape(2, n).T.copy()
        jac = (A - sp.diags(pf * powv * (1.0 - gamma) * w ** (powv - 1.0))).tocsc()
        step = spl.spsolve(jac, res)
        t = 1.0
        for _ in range(60):
            trial = vf - t * step
            if np.max(trial) < 0.0:
                rt = np.max(np.abs(A @ trial - pf * ((1.0 - gamma) * trial) ** powv))
                if rt < (1.0 - 1e-4 * t) * rn:
                    break
            t *= 0.5
        vf = vf - t * step
    w = (1.0 - gamma) * vf
    if float(np.max(np.abs(A @ vf - pf * w ** powv))) >= tol:
        raise ValueError("an inner solve failed to reach its residual tolerance")
    return vf.reshape(2, n).T.copy()


def solve_value_function(gamma: float, psi: float, rho: float, r: float, y1: float,
                                 y2: float, lam1: float, lam2: float, xlow: float,
                                 xbar: float, v_init: np.ndarray, cap: float,
                                 tol_policy: float, tol_residual: float, max_outer: int,
                                 max_inner: int) -> np.ndarray:
    v_init = np.asarray(v_init, dtype=float)
    if v_init.ndim != 2 or v_init.shape[1] != 2 or v_init.shape[0] < 2:
        raise ValueError("v_init must be a 2-D array with two columns and at least two rows")
    if not np.all(np.isfinite(v_init)):
        raise ValueError("v_init must be finite")
    if not np.all(v_init < 0.0):
        raise ValueError("v_init must be strictly negative")
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("r", r), ("y1", y1),
                      ("y2", y2), ("lam1", lam1), ("lam2", lam2), ("xlow", xlow),
                      ("xbar", xbar), ("cap", cap), ("tol_policy", tol_policy),
                      ("tol_residual", tol_residual)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    for name, val in (("max_outer", max_outer), ("max_inner", max_inner)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if int(val) < 1:
            raise ValueError(name + " must be a positive integer")
    max_outer, max_inner = int(max_outer), int(max_inner)
    gamma, psi, rho, r = float(gamma), float(psi), float(rho), float(r)
    y1, y2, lam1, lam2 = float(y1), float(y2), float(lam1), float(lam2)
    xlow, xbar, cap = float(xlow), float(xbar), float(cap)
    tol_policy, tol_residual = float(tol_policy), float(tol_residual)
    if not gamma > 1.0:
        raise ValueError("gamma must be strictly greater than 1")
    if not (0.0 < psi < 1.0):
        raise ValueError("psi must lie strictly between 0 and 1")
    if not rho > 0.0:
        raise ValueError("rho must be strictly positive")
    if not r > 0.0:
        raise ValueError("r must be strictly positive")
    if not (lam1 > 0.0 and lam2 > 0.0):
        raise ValueError("both switching rates must be strictly positive")
    if not (y2 > y1 > 0.0):
        raise ValueError("the ordering y2 > y1 > 0 must hold")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")
    if not (tol_policy > 0.0 and tol_residual > 0.0):
        raise ValueError("both tolerances must be strictly positive")

    n = v_init.shape[0]
    nint = n - 1
    dx = (xbar - xlow) / nint
    x = xlow + dx * np.arange(n)
    lam = np.array([lam1, lam2])
    cbar = r * x[:, None] + np.array([y1, y2])[None, :]
    if not np.all(cbar > 0.0):
        raise ValueError("the zero-saving consumption must be strictly positive at every node")
    if not cap > float(cbar.max()):
        raise ValueError("cap must exceed the largest zero-saving consumption on the grid")

    theta = (1.0 - 1.0 / psi) / (1.0 - gamma)
    powv = 1.0 - theta
    nu = 1.0 - 1.0 / psi
    cF, cB = cbar.copy(), cbar.copy()
    V = v_init.copy()
    converged = False
    for _ in range(max_outer):
        sF = np.maximum(cbar - cF, 0.0); sF[nint, :] = 0.0
        sB = np.minimum(cbar - cB, 0.0); sB[0, :] = 0.0
        A = _h_operator(sF, sB, lam, dx, rho, theta, n)
        phi = (rho / nu) * (cF ** nu + cB ** nu - cbar ** nu)
        V = _h_evaluate(A, phi, V, gamma, powv, tol_residual, max_inner, n)
        cFn, cBn = _h_candidates(V, cbar, dx, gamma, psi, rho, cap)
        change = sum(float(np.max(np.abs(cFn[:, j] - cF[:, j])))
                     + float(np.max(np.abs(cBn[:, j] - cB[:, j]))) for j in (0, 1))
        cF, cB = cFn, cBn
        if change < tol_policy:
            converged = True
            break
    if not converged:
        raise ValueError("the outer policy iteration failed to converge within max_outer")

    sF = np.maximum(cbar - cF, 0.0); sF[nint, :] = 0.0
    sB = np.minimum(cbar - cB, 0.0); sB[0, :] = 0.0
    return np.vstack([V, sF + sB]).astype(float)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def stationary_distribution(s: np.ndarray, lam1: float, lam2: float, h: float,
                                    xlow: float, xbar: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if s.ndim != 2 or s.shape[1] != 2 or s.shape[0] < 2:
        raise ValueError("s must be a 2-D array with two columns and at least two rows")
    if not np.all(np.isfinite(s)):
        raise ValueError("s must be finite")
    for name, val in (("lam1", lam1), ("lam2", lam2), ("h", h), ("xlow", xlow), ("xbar", xbar)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    lam1, lam2, h = float(lam1), float(lam2), float(h)
    xlow, xbar = float(xlow), float(xbar)
    if not (lam1 > 0.0 and lam2 > 0.0):
        raise ValueError("both switching rates must be strictly positive")
    if not h > 0.0:
        raise ValueError("h must be strictly positive")
    if not (lam1 * h < 1.0 and lam2 * h < 1.0):
        raise ValueError("each switching rate times h must be strictly less than one")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")

    n = s.shape[0]
    dx = (xbar - xlow) / (n - 1)
    x = xlow + dx * np.arange(n)
    lam = np.array([lam1, lam2])

    mats = []
    for j in (0, 1):
        xt = np.clip(x + h * s[:, j], x[0], x[-1])
        t = (xt - x[0]) / dx
        k = np.clip(np.floor(t).astype(int), 0, n - 2)
        w = t - k
        rows = np.concatenate([np.arange(n), np.arange(n)])
        cols = np.concatenate([k, k + 1])
        vals = np.concatenate([1.0 - w, w])
        mats.append(sp.csr_matrix((vals, (rows, cols)), shape=(n, n)))

    eye = sp.eye(n)
    M = sp.bmat([[(1.0 - lam[0] * h) * mats[0].T - eye, lam[1] * h * eye],
                 [lam[0] * h * eye, (1.0 - lam[1] * h) * mats[1].T - eye]]).tocsc()
    rhs = -M[1:, 0].toarray().ravel()
    g = np.empty(2 * n)
    g[0] = 1.0
    g[1:] = spl.spsolve(M[1:, 1:].tocsc(), rhs)
    if not np.all(np.isfinite(g)):
        raise ValueError("the invariant law could not be computed")
    total = float(g.sum() * dx)
    if not total > 0.0:
        raise ValueError("the invariant law places no positive weight on the first node")
    g = g / total
    G = g.reshape(2, n).T.copy()
    if np.min(G) < -1e-10 * max(1.0, float(np.max(np.abs(G)))):
        raise ValueError("the computed law has a negative entry")
    return np.maximum(G, 0.0).astype(float)

import numpy as np


def compute_aggregates(G: np.ndarray, y1: float, y2: float, xlow: float,
                               xbar: float) -> np.ndarray:
    G = np.asarray(G, dtype=float)
    if G.ndim != 2 or G.shape[1] != 2 or G.shape[0] < 2:
        raise ValueError("G must be a 2-D array with two columns and at least two rows")
    if not np.all(np.isfinite(G)):
        raise ValueError("G must be finite")
    if np.any(G < 0.0):
        raise ValueError("G must be non-negative")
    for name, val in (("y1", y1), ("y2", y2), ("xlow", xlow), ("xbar", xbar)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    y1, y2, xlow, xbar = float(y1), float(y2), float(xlow), float(xbar)
    if not (y2 > y1 > 0.0):
        raise ValueError("the ordering y2 > y1 > 0 must hold")
    if not xbar > xlow:
        raise ValueError("xbar must exceed xlow")

    n = G.shape[0]
    dx = (xbar - xlow) / (n - 1)
    total = float(G.sum() * dx)
    if abs(total - 1.0) > 1e-8:
        raise ValueError("the distribution must have total mass one")

    x = xlow + dx * np.arange(n)
    K = float(np.sum(x[:, None] * G) * dx)
    shares = G.sum(axis=0) * dx
    N = float(y1 * shares[0] + y2 * shares[1])
    return np.array([K, N, float(G[0, 0] * dx), float(G[0, 1] * dx)], dtype=float)

import numpy as np


def solve_equilibrium(gamma: float, psi: float, rho: float, y1: float, y2: float,
                              lam1: float, lam2: float, xlow: float, xbar: float, nint: int,
                              h: float, alpha: float, delta: float, tfp: float, r_lo: float,
                              r_hi: float, tol_r: float, cap: float) -> float:
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("y1", y1), ("y2", y2),
                      ("lam1", lam1), ("lam2", lam2), ("xlow", xlow), ("xbar", xbar),
                      ("h", h), ("alpha", alpha), ("delta", delta), ("tfp", tfp),
                      ("r_lo", r_lo), ("r_hi", r_hi), ("tol_r", tol_r), ("cap", cap)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    if isinstance(nint, (bool, np.bool_)) or not isinstance(nint, (int, np.integer)) or int(nint) < 1:
        raise ValueError("nint must be a positive integer")
    nint = int(nint)
    gamma, psi, rho = float(gamma), float(psi), float(rho)
    y1, y2, lam1, lam2 = float(y1), float(y2), float(lam1), float(lam2)
    xlow, xbar, h = float(xlow), float(xbar), float(h)
    alpha, delta, tfp = float(alpha), float(delta), float(tfp)
    r_lo, r_hi, tol_r, cap = float(r_lo), float(r_hi), float(tol_r), float(cap)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    if not tfp > 0.0:
        raise ValueError("tfp must be strictly positive")
    if not (0.0 < r_lo < r_hi < rho):
        raise ValueError("the bracket must satisfy 0 < r_lo < r_hi < rho")
    if not tol_r > 0.0:
        raise ValueError("tol_r must be strictly positive")

    tol_policy, tol_residual, max_outer, max_inner = 1e-7, 1e-12, 400, 100
    n = nint + 1
    x = xlow + (xbar - xlow) / nint * np.arange(n)

    def stage(r):
        constants = compute_model_constants(gamma, psi, rho, r)
        theta, b = float(constants[0]), float(constants[4])
        if theta < 1.0:
            raise ValueError("theta below one: the late-resolution policy iteration is not justified")
        barriers = compute_discrete_barriers(gamma, r, y1, y2, b, xlow, xbar, nint)
        lower, upper = barriers[:n, :], barriers[n:, :]
        sol = solve_value_function(gamma, psi, rho, r, y1, y2, lam1, lam2, xlow, xbar,
                                           lower, cap, tol_policy, tol_residual,
                                           max_outer, max_inner)
        V, s = sol[:n, :], sol[n:, :]
        if np.any(V < lower - 1e-8) or np.any(V > upper + 1e-8):
            raise ValueError("the converged value function left the discrete barriers")
        policies = compute_candidate_consumptions(V, gamma, psi, rho, r, y1, y2,
                                                          xlow, xbar, cap)
        czero = r * x[:, None] + np.array([y1, y2])[None, :]
        sF = np.maximum(czero - policies[:n, :], 0.0)
        sF[nint, :] = 0.0
        sB = np.minimum(czero - policies[n:, :], 0.0)
        sB[0, :] = 0.0
        if float(np.max(np.abs(sF + sB - s))) > 1e-9:
            raise ValueError("re-derived candidates do not reproduce the returned saving policy")
        V_again = evaluate_fixed_policy(policies, V, gamma, psi, rho, r, y1, y2,
                                                lam1, lam2, xlow, xbar, tol_residual, max_inner)
        if float(np.max(np.abs(V_again - V))) > 1e-9:
            raise ValueError("re-evaluating the converged policy does not reproduce the value function")
        G = stationary_distribution(s, lam1, lam2, h, xlow, xbar)
        agg = compute_aggregates(G, y1, y2, xlow, xbar)
        K, N = float(agg[0]), float(agg[1])
        if not K > 0.0:
            raise ValueError("aggregate capital must be strictly positive")
        residual = tfp * alpha * (K / N) ** (alpha - 1.0) - delta - r
        return residual, float(agg[2] + agg[3])

    f_lo = stage(r_lo)[0]
    f_hi = stage(r_hi)[0]
    if f_lo * f_hi > 0.0:
        raise ValueError("the residuals at the bracket endpoints must have opposite signs")

    lo, hi = (r_lo, r_hi) if f_lo > 0.0 else (r_hi, r_lo)
    while abs(hi - lo) > tol_r:
        mid = 0.5 * (lo + hi)
        if stage(mid)[0] > 0.0:
            lo = mid
        else:
            hi = mid
    return float(stage(0.5 * (lo + hi))[1])
SCICODE_GOLD_EOF
