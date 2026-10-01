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


def build_axis_constraint_matrix(obs: dict, axis: int) -> np.ndarray:
    if not obs:
        raise ValueError("obs must be a non-empty dict of observations")
    n_dims = len(next(iter(obs)))
    if not (0 <= axis < n_dims):
        raise ValueError(f"axis must be in [0, {n_dims}), got {axis}")
    if any(len(idx) != n_dims for idx in obs):
        raise ValueError("all index tuples in obs must have the same length")
    complementary = sorted({idx[:axis] + idx[axis + 1:] for idx in obs})
    pos = {v: n for n, v in enumerate(complementary)}
    groups: dict = {}
    for idx, val in obs.items():
        groups.setdefault(idx[axis], []).append((idx[:axis] + idx[axis + 1:], val))
    rows = []
    for key in sorted(groups):
        entries = sorted(groups[key])
        for a in range(len(entries)):
            for b in range(a + 1, len(entries)):
                (idx_a, val_a), (idx_b, val_b) = entries[a], entries[b]
                row = np.zeros(len(complementary))
                row[pos[idx_b]] += val_a
                row[pos[idx_a]] -= val_b
                rows.append(row)
    return np.array(rows)

import numpy as np


def select_recursion_axis(obs: dict) -> int:
    if not obs:
        raise ValueError("obs must be a non-empty dict of observations")
    n_dims = len(next(iter(obs)))
    if n_dims < 2:
        raise ValueError("index tuples must have at least two entries")
    cand = []
    s_max = 0.0
    for kp in range(n_dims):
        B = build_axis_constraint_matrix(obs, kp)
        if B.ndim != 2 or B.shape[0] == 0:
            continue
        s = np.linalg.svd(B, compute_uv=False)
        s_min = float(s[-1])
        gap = float(s[-2] - s[-1]) if len(s) >= 2 else float(s[-1])
        s_max = max(s_max, float(s[0]))
        cand.append((s_min, gap, kp))
    if not cand:
        raise ValueError("no axis has a constraint matrix with at least one row")
    tol = 1e-10 * max(1.0, s_max)
    smallest = min(c[0] for c in cand)
    tied = [c for c in cand if c[0] <= smallest + tol]
    best_gap = max(c[1] for c in tied)
    winners = [c[2] for c in tied if c[1] >= best_gap - tol]
    return int(min(winners))

import numpy as np


def extract_complementary_pattern(obs: dict, axis: int) -> np.ndarray:
    B = build_axis_constraint_matrix(obs, axis)
    if B.ndim != 2 or B.shape[0] == 0:
        raise ValueError("no constraint rows: some removed-axis value needs two or more observations")
    Vt = np.linalg.svd(B, full_matrices=True)[2]
    x = np.array(Vt[-1], dtype=float)
    x = x / np.linalg.norm(x)
    nz = np.flatnonzero(np.abs(x) > 1e-12)
    if nz.size and x[nz[0]] < 0:
        x = -x
    return x

import numpy as np


def fit_resolved_factor(obs: dict, axis: int, other_factors: list, size: int) -> np.ndarray:
    if not obs:
        raise ValueError("obs must be a non-empty dict of observations")
    n_dims = len(next(iter(obs)))
    if not (0 <= axis < n_dims):
        raise ValueError(f"axis must be in [0, {n_dims}), got {axis}")
    if len(other_factors) != n_dims - 1:
        raise ValueError("other_factors must hold one vector per non-resolved axis")
    others = [np.asarray(f, dtype=float) for f in other_factors]
    num = np.zeros(size)
    den = np.zeros(size)
    for idx, val in obs.items():
        c = 1.0
        o = 0
        for i in range(n_dims):
            if i == axis:
                continue
            c *= others[o][idx[i]]
            o += 1
        num[idx[axis]] += c * val
        den[idx[axis]] += c * c
    if np.any(den == 0.0):
        raise ValueError("every index of the resolved axis must be observed with a nonzero complementary product")
    return num / den

import numpy as np


def complete_separable_landscape(obs: dict, dims: tuple) -> np.ndarray:
    if not obs:
        raise ValueError("obs must be a non-empty dict of observations")
    if len(dims) < 2:
        raise ValueError("dims must have at least two axes")
    if any(d <= 0 for d in dims):
        raise ValueError("all entries of dims must be positive")
    for idx in obs:
        if len(idx) != len(dims):
            raise ValueError("index tuple length must match len(dims)")
        if any(not (0 <= i < d) for i, d in zip(idx, dims)):
            raise ValueError("observed index out of bounds for dims")

    labels = list(range(len(dims)))
    dim_map = {lbl: dims[lbl] for lbl in labels}

    def recurse(obs_: dict, labels_: list):
        kp = select_recursion_axis(obs_)
        x = extract_complementary_pattern(obs_, kp)
        complementary = sorted({idx[:kp] + idx[kp + 1:] for idx in obs_})
        lab = labels_[kp]
        rest = [l for l in labels_ if l != lab]
        obs2 = {complementary[i]: float(x[i]) for i in range(len(complementary))}
        if len(rest) >= 2:
            us = recurse(obs2, rest)
        else:
            v = np.zeros(dim_map[rest[0]])
            for idx, val in obs2.items():
                v[idx[0]] = val
            us = {rest[0]: v}
        others = [us[l] for l in labels_ if l != lab]
        us[lab] = fit_resolved_factor(obs_, kp, others, dim_map[lab])
        return us

    factors = recurse(obs, labels)
    result = factors[labels[0]]
    for lbl in labels[1:]:
        result = np.multiply.outer(result, factors[lbl])
    return result

import numpy as np


def _svt(X: np.ndarray, tau: float) -> np.ndarray:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return (U * np.maximum(s - tau, 0.0)) @ Vt


def recover_coupled_matrices(
    landscape: np.ndarray,
    health_obs: dict,
    health_dims: tuple,
    lam_landscape: float,
    lam_health: float,
    lam_coupling: float,
    delta: float,
    n_iters: int,
) -> np.ndarray:
    if landscape.shape[0] != health_dims[0]:
        raise ValueError("health_dims[0] must match landscape.shape[0]")
    if lam_landscape < 0 or lam_health < 0 or lam_coupling < 0:
        raise ValueError("lam_landscape, lam_health, lam_coupling must be non-negative")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if n_iters < 0:
        raise ValueError("n_iters must be non-negative")
    for idx in health_obs:
        if not (0 <= idx[0] < health_dims[0] and 0 <= idx[1] < health_dims[1]):
            raise ValueError("health_obs index out of bounds for health_dims")

    n_rows = landscape.shape[0]
    flat_dim = int(np.prod(landscape.shape[1:]))
    n_cols = health_dims[1]
    yM = np.zeros(health_dims)
    OmM = np.zeros(health_dims)
    for idx, val in health_obs.items():
        yM[idx] = val
        OmM[idx] = 1.0
    T = landscape.copy()
    M = yM * OmM
    G = np.zeros((flat_dim, n_cols))
    tT = 0.9 / (1.0 + lam_coupling)
    for _ in range(n_iters):
        grad_T = (T - landscape) + lam_coupling * (
            T.reshape(n_rows, flat_dim) - M @ G.T
        ).reshape(landscape.shape)
        T = _svt((T - tT * grad_T).reshape(n_rows, flat_dim), tT * lam_landscape).reshape(landscape.shape)
        U1 = T.reshape(n_rows, flat_dim)
        grad_M = OmM * (M - yM) + lam_coupling * (M @ G.T - U1) @ G
        step_M = 0.9 / (1.0 + lam_coupling * np.linalg.norm(G, 2) ** 2)
        M = _svt(M - step_M * grad_M, step_M * lam_health)
        G = np.linalg.solve(
            lam_coupling * M.T @ M + delta * np.eye(n_cols), lam_coupling * M.T @ U1
        ).T
    return M

import numpy as np
from scipy.optimize import linprog


def _schedule_constraints(S0, decay_rates, threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget):
    n_workers, n_skills = S0.shape
    n_vars = n_workers * n_skills * n_shifts

    def ix(w, k, t):
        return (w * n_skills + k) * n_shifts + t

    rows_A = []
    rows_b = []
    for w in range(n_workers):
        for k in range(n_skills):
            d = decay_rates[k]
            retention = [(1 - d) ** e for e in range(n_shifts + 1)]
            for t in range(mandate_start, n_shifts + 1):
                row = np.zeros(n_vars)
                for s in range(t):
                    row[ix(w, k, s)] = -train_rate * retention[t - 1 - s]
                rows_A.append(row)
                rows_b.append(retention[t] * S0[w, k] - threshold)
            for t in range(1, n_shifts + 1):
                row = np.zeros(n_vars)
                for s in range(t):
                    row[ix(w, k, s)] = train_rate * retention[t - 1 - s]
                rows_A.append(row)
                rows_b.append(1.0 - retention[t] * S0[w, k])
    for w in range(n_workers):
        for t in range(n_shifts):
            row = np.zeros(n_vars)
            for k in range(n_skills):
                row[ix(w, k, t)] = 1.0
            rows_A.append(row)
            rows_b.append(worker_budget)
    for k in range(n_skills):
        for t in range(n_shifts):
            row = np.zeros(n_vars)
            for w in range(n_workers):
                row[ix(w, k, t)] = 1.0
            rows_A.append(row)
            rows_b.append(skill_budget)
    return np.array(rows_A), np.array(rows_b), n_vars


def solve_training_schedule(
    health_matrix: np.ndarray,
    decay_rates: list,
    threshold: float,
    train_rate: float,
    n_shifts: int,
    mandate_start: int,
    worker_budget: float,
    skill_budget: float,
) -> float:
    S0 = np.asarray(health_matrix, dtype=float)
    if S0.ndim != 2:
        raise ValueError("health_matrix must be a 2D array")
    n_workers, n_skills = S0.shape
    if len(decay_rates) != n_skills:
        raise ValueError("decay_rates length must match the number of skills")
    if any(d < 0 or d >= 1 for d in decay_rates):
        raise ValueError("decay_rates entries must lie in [0, 1)")
    if train_rate <= 0:
        raise ValueError("train_rate must be positive")
    if n_shifts <= 0:
        raise ValueError("n_shifts must be positive")
    if not (1 <= mandate_start <= n_shifts):
        raise ValueError("mandate_start must be in [1, n_shifts]")
    if worker_budget < 0 or skill_budget < 0:
        raise ValueError("worker_budget and skill_budget must be non-negative")
    S0 = np.clip(S0, 0.0, 1.0)
    A, b, n_vars = _schedule_constraints(
        S0, list(decay_rates), threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget
    )
    result = linprog(np.ones(n_vars), A_ub=A, b_ub=b, bounds=[(0, None)] * n_vars, method="highs")
    if not result.success:
        raise ValueError(f"the mandate cannot be met under the limits: {result.message}")
    return float(result.fun)

def compute_minimum_training_hours(
    landscape_obs: dict,
    landscape_dims: tuple,
    health_obs: dict,
    health_dims: tuple,
    lam_landscape: float,
    lam_health: float,
    lam_coupling: float,
    delta: float,
    n_iters_palm: int,
    decay_rates: list,
    threshold: float,
    train_rate: float,
    n_shifts: int,
    mandate_start: int,
    worker_budget: float,
    skill_budget: float,
) -> float:
    landscape = complete_separable_landscape(landscape_obs, landscape_dims)
    health_matrix = recover_coupled_matrices(
        landscape, health_obs, health_dims, lam_landscape, lam_health, lam_coupling, delta, n_iters_palm
    )
    return solve_training_schedule(
        health_matrix, decay_rates, threshold, train_rate, n_shifts, mandate_start, worker_budget, skill_budget
    )
SCICODE_GOLD_EOF
