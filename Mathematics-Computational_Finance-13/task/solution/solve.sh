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


def _node_value(z_root, sigma, lam, h, j):
    chi = 2.0 * np.sqrt(z_root) / sigma + j * lam * np.sqrt(h)
    return (sigma / 2.0 * max(chi, 0.0)) ** 2


def _cir_moments(z, kappa, theta, sigma, h):
    e = np.exp(-kappa * h)
    m = theta + (z - theta) * e
    v = sigma ** 2 * z / kappa * e * (1.0 - e) + theta * sigma ** 2 / (2.0 * kappa) * (1.0 - e) ** 2
    return m, v


def _triplet_probs(Zs, m, v):
    d1, d2, d3 = (np.asarray(Zs, float) - m)
    den = np.array([(d1 - d2) * (d1 - d3), (d2 - d1) * (d2 - d3), (d3 - d1) * (d3 - d2)])
    if np.min(np.abs(den)) < 1e-300:
        return None
    return np.array([v + d2 * d3, v + d1 * d3, v + d1 * d2]) / den


def cir_transition_stencil(z_root: float, kappa: float, theta: float, sigma: float,
                                   lam: float, h: float, j: int) -> np.ndarray:
    for name, val in (("z_root", z_root), ("kappa", kappa), ("theta", theta),
                      ("sigma", sigma), ("h", h)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not float(val) > 0.0:
            raise ValueError(f"{name} must be a positive real number")
    if not isinstance(lam, (int, float, np.integer, np.floating)) or not 0.0 < float(lam) < 2.0:
        raise ValueError("lam must satisfy 0 < lam < 2")
    if not isinstance(j, (int, np.integer)):
        raise ValueError("j must be an integer node index")

    z_root, kappa, theta = float(z_root), float(kappa), float(theta)
    sigma, lam, h, j = float(sigma), float(lam), float(h), int(j)

    j_min = int(np.ceil(-2.0 * np.sqrt(z_root) / (sigma * lam * np.sqrt(h))))
    z = _node_value(z_root, sigma, lam, h, j)
    if z <= 0.0:
        raise ValueError("parent node lies at or below the lattice floor")
    m, v = _cir_moments(z, kappa, theta, sigma, h)

    for span in range(2, 8):
        cands = [(lo, mid, lo + span)
                 for lo in range(j - span, j + 1)
                 for mid in range(lo + 1, lo + span)]
        cands.sort(key=lambda t: abs((t[0] + t[2]) / 2.0 - j))
        for idx in cands:
            Zs = [_node_value(z_root, sigma, lam, h, i) for i in idx]
            if len(set(np.round(Zs, 15))) < 3:
                continue
            p = _triplet_probs(Zs, m, v)
            if p is None or p.min() < -1e-12:
                continue
            p = np.clip(p, 0.0, None)
            p = p / p.sum()
            rel = np.array([max(i, j_min) - j for i in idx], dtype=float)
            return np.vstack([rel, p])

    raise ValueError("no admissible stencil exists at the requested node")

import numpy as np


def joint_factor_coupling(p_v: np.ndarray, z_v: np.ndarray,
                                  p_x: np.ndarray, z_x: np.ndarray, rho: float) -> np.ndarray:
    arrs = []
    for a in (p_v, z_v, p_x, z_x):
        b = np.asarray(a, dtype=float)
        if b.ndim != 1 or b.shape[0] != 3 or not np.all(np.isfinite(b)):
            raise ValueError("each marginal input must be a finite 1-D array of length 3")
        arrs.append(b)
    pv, zv, px, zx = arrs

    if not isinstance(rho, (int, float, np.integer, np.floating)) or not np.isfinite(rho):
        raise ValueError("rho must be a finite real number")
    rho = float(rho)
    if abs(rho) > 1.0:
        raise ValueError("rho must lie in [-1, 1]")

    for p, z in ((pv, zv), (px, zx)):
        if p.min() < 0.0 or abs(p.sum() - 1.0) > 1e-12:
            raise ValueError("marginal probabilities must be non-negative and sum to one")
        if abs(float((p * z).sum())) > 1e-9 or abs(float((p * z * z).sum()) - 1.0) > 1e-9:
            raise ValueError("innovations must be centred with unit variance under their marginal")

    table = pv[:, None] * px[None, :] * (1.0 + rho * zv[:, None] * zx[None, :])
    active = (pv[:, None] * px[None, :]) > 0.0
    if np.any(table[active] < 0.0):
        raise ValueError("the requested correlation is not admissible on this pair of stencils")
    return table

import numpy as np


def _node(z_root, sigma, lam, h, j):
    chi = 2.0 * np.sqrt(z_root) / sigma + j * lam * np.sqrt(h)
    return (sigma / 2.0 * max(chi, 0.0)) ** 2


def _moments(z, kappa, theta, sigma, h):
    e = np.exp(-kappa * h)
    return (theta + (z - theta) * e,
            sigma ** 2 * z / kappa * e * (1.0 - e) + theta * sigma ** 2 / (2.0 * kappa) * (1.0 - e) ** 2)


def _probs(Zs, m, v):
    d1, d2, d3 = (np.asarray(Zs, float) - m)
    den = np.array([(d1 - d2) * (d1 - d3), (d2 - d1) * (d2 - d3), (d3 - d1) * (d3 - d2)])
    if np.min(np.abs(den)) < 1e-300:
        return None
    return np.array([v + d2 * d3, v + d1 * d3, v + d1 * d2]) / den


def _stencil(x_root, kappa, theta, sigma, lam, h, j, j_min):
    z = _node(x_root, sigma, lam, h, j)
    m, v = _moments(z, kappa, theta, sigma, h)
    for span in range(2, 8):
        cands = [(lo, mid, lo + span) for lo in range(j - span, j + 1)
                 for mid in range(lo + 1, lo + span)]
        cands.sort(key=lambda t: abs((t[0] + t[2]) / 2.0 - j))
        for idx in cands:
            Zs = [_node(x_root, sigma, lam, h, i) for i in idx]
            if len(set(np.round(Zs, 15))) < 3:
                continue
            p = _probs(Zs, m, v)
            if p is None or p.min() < -1e-12:
                continue
            p = np.clip(p, 0.0, None)
            return np.array([max(i, j_min) for i in idx]), p / p.sum()
    raise ValueError("no admissible stencil exists on the rate lattice")


def calibrate_lattice_shift(x_root: float, kappa: float, theta: float, sigma: float,
                                    lam: float, h: float, market_disc: np.ndarray) -> np.ndarray:
    for name, val in (("x_root", x_root), ("kappa", kappa), ("theta", theta),
                      ("sigma", sigma), ("h", h)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not float(val) > 0.0:
            raise ValueError(f"{name} must be a positive real number")
    if not isinstance(lam, (int, float, np.integer, np.floating)) or not 0.0 < float(lam) < 2.0:
        raise ValueError("lam must satisfy 0 < lam < 2")
    P = np.asarray(market_disc, dtype=float)
    if P.ndim != 1 or P.size < 1 or not np.all(np.isfinite(P)) or P.min() <= 0.0:
        raise ValueError("market_disc must be a non-empty 1-D array of positive discount factors")

    x_root, kappa, theta = float(x_root), float(kappa), float(theta)
    sigma, lam, h = float(sigma), float(lam), float(h)
    j_min = int(np.ceil(-2.0 * np.sqrt(x_root) / (sigma * lam * np.sqrt(h))))

    shift = np.zeros(P.size, dtype=float)
    ad = {0: 1.0}
    for n in range(P.size):
        raw, total = {}, 0.0
        for k, mass in ad.items():
            xk = _node(x_root, sigma, lam, h, k)
            child, prob = _stencil(x_root, kappa, theta, sigma, lam, h, k, j_min)
            for c, p in zip(child, prob):
                w = mass * p * np.exp(-0.5 * h * (xk + _node(x_root, sigma, lam, h, int(c))))
                raw[int(c)] = raw.get(int(c), 0.0) + w
                total += w
        s = float(np.log(total / P[n]))
        shift[n] = s
        ad = {c: w * np.exp(-s) for c, w in raw.items()}
    return shift

import numpy as np


def equity_innovation_weights(rho_sv: float, rho_sr: float, rho_vr: float) -> np.ndarray:
    for name, val in (("rho_sv", rho_sv), ("rho_sr", rho_sr), ("rho_vr", rho_vr)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite real number")
        if abs(float(val)) > 1.0:
            raise ValueError(f"{name} must lie in [-1, 1]")
    a, b, c = float(rho_sv), float(rho_sr), float(rho_vr)

    den = 1.0 - c * c
    if den <= 1e-14:
        raise ValueError("the two factor innovations must not be perfectly correlated")

    det = 1.0 + 2.0 * a * b * c - a * a - b * b - c * c
    if det < 0.0:
        raise ValueError("the three correlations do not form a positive semidefinite matrix")

    w_v = (a - b * c) / den
    w_r = (b - a * c) / den
    w_perp = np.sqrt(max(det / den, 0.0))
    return np.array([w_v, w_r, w_perp], dtype=float)

import numpy as np


def fund_branch_multipliers(v_parent: float, x_parent: float, x_children: np.ndarray,
                                    joint: np.ndarray, z_v: np.ndarray, z_x: np.ndarray,
                                    weights: np.ndarray, shift: float, q: float, h: float) -> np.ndarray:
    for name, val in (("v_parent", v_parent), ("x_parent", x_parent),
                      ("shift", shift), ("q", q), ("h", h)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite real number")
    if float(h) <= 0.0:
        raise ValueError("h must be a positive time step")
    if float(v_parent) < 0.0:
        raise ValueError("v_parent must be non-negative")

    xc = np.asarray(x_children, dtype=float)
    zv = np.asarray(z_v, dtype=float)
    zx = np.asarray(z_x, dtype=float)
    w = np.asarray(weights, dtype=float)
    J = np.asarray(joint, dtype=float)
    for arr, n in ((xc, 3), (zv, 3), (zx, 3), (w, 3)):
        if arr.ndim != 1 or arr.shape[0] != n or not np.all(np.isfinite(arr)):
            raise ValueError("x_children, z_v, z_x and weights must be finite 1-D arrays of length 3")
    if J.shape != (3, 3) or not np.all(np.isfinite(J)):
        raise ValueError("joint must be a finite 3-by-3 array")
    if J.min() < 0.0 or abs(J.sum() - 1.0) > 1e-12:
        raise ValueError("joint must be a probability table summing to one")
    if w[2] < 0.0:
        raise ValueError("the residual weight must be non-negative")

    v_parent, x_parent = float(v_parent), float(x_parent)
    shift, q, h = float(shift), float(q), float(h)

    eta = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
    wq = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])

    R = 0.5 * h * (x_parent + xc) + shift
    R9 = np.repeat(R[None, :], 3, axis=0).ravel()
    base = (w[0] * zv[:, None] + w[1] * zx[None, :]).ravel()
    prob9 = J.ravel()

    sig = np.sqrt(v_parent * h)
    m0 = np.exp(R9[:, None] - q * h - 0.5 * v_parent * h
                + sig * (base[:, None] + w[2] * eta[None, :]))
    pr = prob9[:, None] * wq[None, :]
    disc = np.repeat(np.exp(-R9)[:, None], 3, axis=1)

    denom = float((pr * disc * m0).sum())
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("the branch set does not admit a positive martingale correction")
    m = m0 * (np.exp(-q * h) / denom)

    return np.vstack([pr.ravel(), disc.ravel(), m.ravel()])

import numpy as np


def guarantee_and_mortality_schedules(contribution: float, g_maturity: float,
                                              g_surrender: float, n_years: int, issue_age: int,
                                              mk_a: float, mk_b: float, mk_c: float) -> np.ndarray:
    if not isinstance(contribution, (int, float, np.integer, np.floating)) or not float(contribution) > 0.0:
        raise ValueError("contribution must be a positive real number")
    for name, val in (("g_maturity", g_maturity), ("g_surrender", g_surrender)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite real number")
    if not isinstance(n_years, (int, np.integer)) or int(n_years) < 1:
        raise ValueError("n_years must be an integer of at least one")
    if not isinstance(issue_age, (int, np.integer)) or int(issue_age) < 0:
        raise ValueError("issue_age must be a non-negative integer")
    if not isinstance(mk_a, (int, float, np.integer, np.floating)) or float(mk_a) < 0.0:
        raise ValueError("mk_a must be non-negative")
    if not isinstance(mk_b, (int, float, np.integer, np.floating)) or not float(mk_b) > 0.0:
        raise ValueError("mk_b must be positive")
    if not isinstance(mk_c, (int, float, np.integer, np.floating)) or not float(mk_c) > 1.0:
        raise ValueError("mk_c must exceed one")

    D = float(contribution)
    gm, gs = float(g_maturity), float(g_surrender)
    T, a0 = int(n_years), int(issue_age)
    A, B, C = float(mk_a), float(mk_b), float(mk_c)

    out = np.zeros((3, T + 1), dtype=float)
    for i in range(T + 1):
        if i == 0:
            continue
        j = np.arange(0, i)
        out[0, i] = float((D * np.exp(gm * (i - j))).sum())
        out[1, i] = float((D * np.exp(gs * (i - j))).sum())

    out[2, 0] = 1.0
    for i in range(T):
        hazard = A + B * C ** (a0 + i)
        if not np.isfinite(hazard):
            raise ValueError("the mortality parameters produce a non-finite hazard")
        out[2, i + 1] = out[2, i] * np.exp(-hazard)
    return out

import numpy as np


def _node(z_root, sigma, lam, h, j):
    chi = 2.0 * np.sqrt(z_root) / sigma + j * lam * np.sqrt(h)
    return (sigma / 2.0 * max(chi, 0.0)) ** 2


def _moments(z, kappa, theta, sigma, h):
    e = np.exp(-kappa * h)
    return (theta + (z - theta) * e,
            sigma ** 2 * z / kappa * e * (1.0 - e) + theta * sigma ** 2 / (2.0 * kappa) * (1.0 - e) ** 2)


def _probs(Zs, m, v):
    d1, d2, d3 = (np.asarray(Zs, float) - m)
    den = np.array([(d1 - d2) * (d1 - d3), (d2 - d1) * (d2 - d3), (d3 - d1) * (d3 - d2)])
    if np.min(np.abs(den)) < 1e-300:
        return None
    return np.array([v + d2 * d3, v + d1 * d3, v + d1 * d2]) / den


def _children(z_root, kappa, theta, sigma, lam, h, j, j_min):
    z = _node(z_root, sigma, lam, h, j)
    m, v = _moments(z, kappa, theta, sigma, h)
    for span in range(2, 8):
        cands = [(lo, mid, lo + span) for lo in range(j - span, j + 1)
                 for mid in range(lo + 1, lo + span)]
        cands.sort(key=lambda t: abs((t[0] + t[2]) / 2.0 - j))
        for idx in cands:
            Zs = [_node(z_root, sigma, lam, h, i) for i in idx]
            if len(set(np.round(Zs, 15))) < 3:
                continue
            p = _probs(Zs, m, v)
            if p is None or p.min() < -1e-12:
                continue
            return [max(i, j_min) for i in idx]
    raise ValueError("no admissible stencil exists at a reachable node")


def reachable_factor_nodes(z_root: float, kappa: float, theta: float, sigma: float,
                                   lam: float, h: float, n_steps: int) -> np.ndarray:
    for name, val in (("z_root", z_root), ("kappa", kappa), ("theta", theta),
                      ("sigma", sigma), ("h", h)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not float(val) > 0.0:
            raise ValueError(f"{name} must be a positive real number")
    if not isinstance(lam, (int, float, np.integer, np.floating)) or not 0.0 < float(lam) < 2.0:
        raise ValueError("lam must satisfy 0 < lam < 2")
    if not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 0:
        raise ValueError("n_steps must be a non-negative integer")

    z_root, kappa, theta = float(z_root), float(kappa), float(theta)
    sigma, lam, h, N = float(sigma), float(lam), float(h), int(n_steps)
    j_min = int(np.ceil(-2.0 * np.sqrt(z_root) / (sigma * lam * np.sqrt(h))))

    span = np.zeros((2, N + 1), dtype=float)
    current = {0}
    span[0, 0] = span[1, 0] = 0.0
    for n in range(N):
        nxt = set()
        for j in sorted(current):
            nxt.update(_children(z_root, kappa, theta, sigma, lam, h, j, j_min))
        current = nxt
        span[0, n + 1] = float(min(current))
        span[1, n + 1] = float(max(current))
    return span

import numpy as np


def _interp(grid, vals, F):
    F = np.asarray(F, dtype=float)
    out = np.interp(F, grid, vals)
    lo = F < grid[0]
    if lo.any():
        s = (vals[1] - vals[0]) / (grid[1] - grid[0])
        out = np.where(lo, vals[0] + s * (F - grid[0]), out)
    hi = F > grid[-1]
    if hi.any():
        s = (vals[-1] - vals[-2]) / (grid[-1] - grid[-2])
        out = np.where(hi, vals[-1] + s * (F - grid[-1]), out)
    return out


def backward_step_with_obstacle(fund_grid: np.ndarray, child_values: np.ndarray,
                                        child_index: np.ndarray, branches: np.ndarray,
                                        contribution: float, premium: float, death_prob: float,
                                        g_death: float, g_surrender: float, alpha_s: float,
                                        apply_obstacle: bool) -> np.ndarray:
    F = np.asarray(fund_grid, dtype=float)
    U = np.asarray(child_values, dtype=float)
    idx = np.asarray(child_index)
    B = np.asarray(branches, dtype=float)

    if F.ndim != 1 or F.size < 2 or not np.all(np.isfinite(F)) or np.any(np.diff(F) <= 0.0):
        raise ValueError("fund_grid must be a strictly increasing finite 1-D array")
    if U.ndim != 2 or U.shape[1] != F.size or not np.all(np.isfinite(U)):
        raise ValueError("child_values must be a finite 2-D array with one column per fund node")
    if B.ndim != 2 or B.shape[0] != 3 or not np.all(np.isfinite(B)):
        raise ValueError("branches must be a finite array with three rows")
    if idx.ndim != 1 or idx.size != B.shape[1] or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("child_index must be an integer array with one entry per branch")
    if idx.min() < 0 or idx.max() >= U.shape[0]:
        raise ValueError("child_index refers to a value function that was not supplied")
    for name, val in (("contribution", contribution), ("premium", premium),
                      ("death_prob", death_prob), ("g_death", g_death),
                      ("g_surrender", g_surrender), ("alpha_s", alpha_s)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite real number")
    if not 0.0 <= float(death_prob) <= 1.0:
        raise ValueError("death_prob must lie in [0, 1]")
    if not 0.0 < float(alpha_s) <= 1.0:
        raise ValueError("alpha_s must lie in (0, 1]")
    if B.shape[1] < 1 or B[0].min() < 0.0:
        raise ValueError("branch probabilities must be non-negative")

    D, P = float(contribution), float(premium)
    qd, gd = float(death_prob), float(g_death)
    gs, a_s = float(g_surrender), float(alpha_s)

    if qd > 0.0:
        benefit = np.maximum(F, gd)
        Um = (1.0 - qd) * U + qd * benefit[None, :]
    else:
        Um = U

    cont = np.zeros_like(F)
    for b in range(B.shape[1]):
        cont += B[0, b] * B[1, b] * _interp(F, Um[int(idx[b])], (F + D) * B[2, b])
    cont -= P

    if apply_obstacle:
        cont = np.maximum(cont, np.maximum(a_s * F, gs))
    return cont

import numpy as np
from scipy.optimize import brentq


def fair_annual_premium(n_years: int, steps_per_year: int, contribution: float,
                                g_maturity: float, g_surrender: float, alpha_s: float, q: float,
                                issue_age: int, mk_a: float, mk_b: float, mk_c: float,
                                v0: float, kappa_v: float, theta_v: float, sigma_v: float,
                                x0: float, kappa_r: float, theta_r: float, sigma_r: float,
                                rho_sv: float, rho_sr: float, rho_vr: float,
                                lam_v: float, lam_x: float,
                                market_disc: np.ndarray, n_fund: int, fund_max: float) -> float:
    if not isinstance(n_years, (int, np.integer)) or int(n_years) < 1:
        raise ValueError("n_years must be an integer of at least one")
    if not isinstance(steps_per_year, (int, np.integer)) or int(steps_per_year) < 1:
        raise ValueError("steps_per_year must be an integer of at least one")
    if not isinstance(n_fund, (int, np.integer)) or int(n_fund) < 3:
        raise ValueError("n_fund must be an integer of at least three")
    if not isinstance(fund_max, (int, float, np.integer, np.floating)) or not float(fund_max) > 0.0:
        raise ValueError("fund_max must be a positive real number")
    P = np.asarray(market_disc, dtype=float)
    T, Nyr, nF = int(n_years), int(steps_per_year), int(n_fund)
    N = T * Nyr
    if P.ndim != 1 or P.size != N or P.min() <= 0.0:
        raise ValueError("market_disc must hold one positive discount factor per numerical step")

    h = 1.0 / Nyr
    D, gm, gs = float(contribution), float(g_maturity), float(g_surrender)
    a_s, qy = float(alpha_s), float(q)

    sched = guarantee_and_mortality_schedules(D, gm, gs, T, int(issue_age),
                                                      float(mk_a), float(mk_b), float(mk_c))
    G_mat, G_sur, surv = sched[0], sched[1], sched[2]
    death = np.array([1.0 - surv[i + 1] / surv[i] for i in range(T)])

    weights = equity_innovation_weights(float(rho_sv), float(rho_sr), float(rho_vr))
    shift = calibrate_lattice_shift(float(x0), float(kappa_r), float(theta_r),
                                            float(sigma_r), float(lam_x), h, P)

    def _lattice_node(root, sigma, lam, j):
        chi = 2.0 * np.sqrt(root) / sigma + j * lam * np.sqrt(h)
        return (sigma / 2.0 * max(chi, 0.0)) ** 2

    vv = float(v0), float(kappa_v), float(theta_v), float(sigma_v), float(lam_v)
    xx = float(x0), float(kappa_r), float(theta_r), float(sigma_r), float(lam_x)

    stv, stx = {}, {}
    def _cached_stencil(par, j, cache):
        if j not in cache:
            cache[j] = cir_transition_stencil(par[0], par[1], par[2], par[3], par[4], h, j)
        return cache[j]

    span_v = reachable_factor_nodes(vv[0], vv[1], vv[2], vv[3], vv[4], h, N)
    span_x = reachable_factor_nodes(xx[0], xx[1], xx[2], xx[3], xx[4], h, N)
    reach_v = [list(range(int(span_v[0, n]), int(span_v[1, n]) + 1)) for n in range(N + 1)]
    reach_x = [list(range(int(span_x[0, n]), int(span_x[1, n]) + 1)) for n in range(N + 1)]

    def _step_moments(z, k, th, s):
        e = np.exp(-k * h)
        return (th + (z - th) * e,
                s * s * z / k * e * (1.0 - e) + th * s * s / (2.0 * k) * (1.0 - e) ** 2)

    def _node_innovations(par, j, cache):
        st = _cached_stencil(par, j, cache)
        Z = np.array([_lattice_node(par[0], par[3], par[4], int(j + r)) for r in st[0]])
        m, v = _step_moments(_lattice_node(par[0], par[3], par[4], j), par[1], par[2], par[3])
        return st[0].astype(int), st[1], Z, (Z - m) / np.sqrt(v)

    branch_cache = {}
    def _node_branches(n, jv, jx):
        key = (n, jv, jx)
        if key not in branch_cache:
            rv, pv, _, zv = _node_innovations(vv, jv, stv)
            rx, px, Zx, zx = _node_innovations(xx, jx, stx)
            J = joint_factor_coupling(pv, zv, px, zx, float(rho_vr))
            B = fund_branch_multipliers(_lattice_node(vv[0], vv[3], vv[4], jv),
                                                _lattice_node(xx[0], xx[3], xx[4], jx),
                                                Zx, J, zv, zx, weights, float(shift[n]), qy, h)
            cv = np.repeat(jv + rv, 3).repeat(3)
            cx = np.tile(jx + rx, 3).repeat(3)
            branch_cache[key] = (B, cv, cx)
        return branch_cache[key]

    def _net_value(prem, nfund):
        F = np.linspace(0.0, float(fund_max), nfund)
        nodes = [(jv, jx) for jv in sorted(reach_v[N]) for jx in sorted(reach_x[N])]
        pos = {kk: i for i, kk in enumerate(nodes)}
        U = np.repeat(np.maximum(F, G_mat[T])[None, :], len(nodes), axis=0)

        for n in range(N - 1, -1, -1):
            closes = ((n + 1) % Nyr == 0)
            i_next = (n + 1) // Nyr
            qd = float(death[i_next - 1]) if closes else 0.0
            gd = float(G_mat[i_next]) if closes else 0.0
            anniv = (n % Nyr == 0)
            i = n // Nyr
            pays = anniv and i <= T - 1
            surr = anniv and 1 <= i <= T - 1

            new_nodes = [(jv, jx) for jv in sorted(reach_v[n]) for jx in sorted(reach_x[n])]
            grid = np.array([0.0]) if n == 0 else F
            out = np.zeros((len(new_nodes), grid.size))
            for r, (jv, jx) in enumerate(new_nodes):
                B, cv, cx = _node_branches(n, jv, jx)
                ci = np.array([pos[(int(a), int(b))] for a, b in zip(cv, cx)], dtype=int)
                val = backward_step_with_obstacle(
                    F, U, ci, B, D if pays else 0.0, prem if pays else 0.0,
                    qd, gd, float(G_sur[i]) if surr else 0.0, a_s, surr)
                if n == 0:
                    out[r, 0] = float(val[0])
                else:
                    out[r] = val
            U, nodes, pos = out, new_nodes, {kk: i2 for i2, kk in enumerate(new_nodes)}
        return float(U[pos[(0, 0)], 0])

    def _solve_premium(nfund):
        return brentq(lambda p: _net_value(p, nfund), 1.0, 10.0 * max(D, 1.0),
                      xtol=1e-10, rtol=1e-14)

    coarse = _solve_premium(nF)
    fine = _solve_premium(2 * nF - 1)
    return float((4.0 * fine - coarse) / 3.0)
SCICODE_GOLD_EOF
