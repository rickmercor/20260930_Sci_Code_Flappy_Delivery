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


def _xlogx(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0.0, x * np.log(np.where(x > 0.0, x, 1.0)), 0.0)


def _niche_weights(matrix, n_occupied):
    n = np.asarray(matrix, dtype=np.float64)
    total = n.sum()
    y = n.sum(axis=1)
    x = n.sum(axis=0)
    p = np.divide(n, y[:, None], out=np.zeros_like(n), where=y[:, None] > 0.0)
    pi = n / total
    q = y / total
    P = x / total
    hx = -_xlogx(P).sum()
    hy = -_xlogx(q).sum()
    hxy = -_xlogx(pi).sum()
    conditional = hxy - hy
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(p > 0.0, pi * np.log(np.where(p > 0.0, p, 1.0)), 0.0).sum(axis=0)
    delta = (first - _xlogx(P)) / hx
    exponent = delta * float(n_occupied) / float(n.shape[1])
    z = np.exp(exponent - exponent.max())
    return z / z.sum(), np.array([hx, hy, hxy, conditional], dtype=np.float64), delta


def noncircular_niche_profile(resource_matrix: "np.ndarray", k: float) -> "np.ndarray":
    n = np.asarray(resource_matrix, dtype=np.float64)
    if n.ndim != 2 or n.shape[0] < 3 or n.shape[1] < 2 or not np.isfinite(n).all() or np.any(n < 0.0):
        raise ValueError("resource_matrix must be finite, nonnegative, and at least 3 by 2")
    if not np.isfinite(k) or k <= 1.0 or np.any(n.sum(axis=1) <= 0.0):
        raise ValueError("each species must occur and k must exceed one")
    occupied = int(np.count_nonzero(n.sum(axis=0) > 0.0))
    s = n.shape[0]
    breadth = np.empty(s, dtype=np.float64)
    for i in range(s):
        w, _, _ = _niche_weights(np.delete(n, i, axis=0), occupied)
        ystar = float((w * k * n[i]).sum())
        ps = n[i] / ystar
        breadth[i] = -(k / np.log(k)) * float((w * _xlogx(ps)).sum())
    overlap = np.eye(s, dtype=np.float64)
    for i in range(s):
        for j in range(i):
            w, _, _ = _niche_weights(np.delete(n, [j, i], axis=0), occupied)
            pair = n[[j, i]]
            ystar = (w[None, :] * k * pair).sum(axis=1)
            ps = pair / ystar[:, None]
            val = -(w * k * (_xlogx(ps[0]) + _xlogx(ps[1]) - _xlogx(ps[0] + ps[1]))).sum() / (2.0 * np.log(2.0))
            overlap[i, j] = overlap[j, i] = float(val)
    return np.concatenate([breadth, overlap.ravel()]).astype(np.float64)

import numpy as np


def _bc_rows(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    num = np.abs(x[:, None, :] - y[None, :, :]).sum(axis=2)
    den = (x[:, None, :] + y[None, :, :]).sum(axis=2)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0.0)


def _rank(reference, target, k, eps):
    r = np.asarray(reference, dtype=np.float64)
    flat = r.reshape(-1, r.shape[2])
    d = _bc_rows(flat, np.asarray(target, dtype=np.float64)[None, :])[:, 0]
    ids = np.column_stack(np.unravel_index(np.arange(flat.shape[0]), r.shape[:2]))
    order = np.lexsort((ids[:, 1], ids[:, 0], d))
    order = order[d[order] < eps][:k]
    return ids[order], d[order]


def _kernel_weights(reference, target, idx, kernel, alpha, eps):
    r = np.asarray(reference, dtype=np.float64)
    d = _bc_rows(r.reshape(-1, r.shape[2]), np.asarray(target)[None, :])[:, 0].reshape(r.shape[:2])
    td = np.min(d, axis=1)[idx[:, 0]]
    q = td / max(float(eps), float(td.max()))
    if kernel == "linear":
        return 1.0 - q
    if kernel == "power":
        return 1.0 - q ** alpha
    if kernel == "exponential":
        return np.exp(-alpha * q)
    if kernel == "Gaussian":
        return np.exp(-(q ** 2.0))
    if kernel == "hyperbolic":
        return ((1.0 + q) ** (-alpha) - 2.0 ** (-alpha)) / (1.0 - 2.0 ** (-alpha))
    if kernel == "spherical":
        return 1.0 - 1.5 * q + 0.5 * q ** 3.0
    raise ValueError("unknown kernel")


def _sweep(reference, idx, weights, direction, min_pts, max_steps):
    r = np.asarray(reference, dtype=np.float64)
    out = []
    for shift in range(1, max_steps + 1):
        moved = idx[:, 1] + direction * shift
        valid = (moved >= 0) & (moved < r.shape[1]) & (weights > 0.0)
        if np.count_nonzero(valid) < min_pts:
            break
        w = weights[valid]
        out.append((r[idx[valid, 0], moved[valid]] * w[:, None]).sum(axis=0) / w.sum())
    return np.asarray(out, dtype=np.float64).reshape(-1, r.shape[2])


def _forecast(reference, target, candidate, max_steps):
    k, eps, min_pts, kernel, alpha = candidate
    idx, _ = _rank(reference, target, int(k), float(eps))
    if len(idx) < min_pts:
        raise ValueError("insufficient neighbours")
    w = _kernel_weights(reference, target, idx, kernel, float(alpha), float(eps))
    back = _sweep(reference, idx, w, -1, int(min_pts), int(max_steps))
    fwd = _sweep(reference, idx, w, 1, int(min_pts), int(max_steps))
    offsets = np.arange(-len(back), len(fwd) + 1, dtype=np.float64)
    states = np.vstack([back[::-1], np.asarray(target, dtype=np.float64)[None, :], fwd])
    return offsets, states, idx, w


def _mpd(reference, target, candidate, max_steps):
    offsets, states, _, _ = _forecast(reference, target, candidate, max_steps)
    values = []
    for off, state in zip(offsets, states):
        if off == 0:
            continue
        k, eps, min_pts, kernel, alpha = candidate
        idx, _ = _rank(reference, state, int(k), float(eps))
        if len(idx) < min_pts:
            continue
        w = _kernel_weights(reference, state, idx, kernel, float(alpha), float(eps))
        path = _sweep(reference, idx, w, -1 if off > 0 else 1, int(min_pts), abs(int(off)))
        if len(path):
            repath = np.vstack([state[None, :], path])
            values.append(float(_bc_rows(np.asarray(target)[None, :], repath).min()))
    if not values:
        raise ValueError("candidate has no usable re-prediction")
    return float(np.mean(values))


def petra_calibration(reference: "np.ndarray", calibration_targets: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int) -> "np.ndarray":
    r = np.asarray(reference, dtype=np.float64)
    x = np.asarray(calibration_targets, dtype=np.float64)
    if r.ndim != 3 or x.ndim != 2 or x.shape[1] != r.shape[2] or len(candidates) < 2:
        raise ValueError("unaligned PETRA inputs")
    scores = np.array([np.mean([_mpd(r, row, c, int(max_steps)) for row in x]) for c in candidates], dtype=np.float64)
    return np.concatenate([[float(np.argmin(scores))], scores]).astype(np.float64)

import numpy as np


def petra_residual_signal(reference: "np.ndarray", disturbed: "np.ndarray", candidate: tuple[int, float, int, str, float], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray") -> "np.ndarray":
    r = np.asarray(reference, dtype=np.float64)
    obs = np.asarray(disturbed, dtype=np.float64)
    mapping = np.asarray(guild_map, dtype=np.float64)
    ids = np.asarray(indices, dtype=np.int64)
    if r.ndim != 3 or obs.ndim != 2 or mapping.ndim != 2 or mapping.shape[1] != obs.shape[1] or ids.shape != (3,) or not (0 <= ids[0] < ids[1] < ids[2] < len(obs)):
        raise ValueError("unaligned disturbance inputs")
    _, path, _, _ = _forecast(r, obs[ids[0]], candidate, int(max_steps))
    dref = _bc_rows(obs[ids], path).min(axis=1)
    direct = float(_bc_rows(obs[[ids[0]]], obs[[ids[1]]])[0, 0])
    profile = np.array([1.0 - direct, dref[1] - dref[0], dref[1] - dref[2], dref[2] - dref[0]], dtype=np.float64)
    closest = path[int(np.argmin(_bc_rows(obs[[ids[2]]], path)[0]))]
    denom = float((obs[ids[2]] + closest).sum())
    signed = (obs[ids[2]] - closest) / denom
    exposure = mapping @ signed
    return np.concatenate([profile, exposure]).astype(np.float64)

import numpy as np


def assemble_coupled_system(
    base_competition: "np.ndarray",
    base_designs: "np.ndarray",
    base_loadings: "np.ndarray",
    base_reserves: "np.ndarray",
    niche_profile: "np.ndarray",
    petra_signal: "np.ndarray",
    coefficients: "np.ndarray",
) -> "np.ndarray":
    b = np.asarray(base_competition, dtype=np.float64)
    d = np.asarray(base_designs, dtype=np.float64)
    u = np.asarray(base_loadings, dtype=np.float64)
    ell = np.asarray(base_reserves, dtype=np.float64)
    n = np.asarray(niche_profile, dtype=np.float64)
    p = np.asarray(petra_signal, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64)

    if (
        b.ndim != 3
        or d.ndim != 2
        or u.ndim != 3
        or ell.shape != (b.shape[1],)
        or d.shape[1] != b.shape[1]
        or u.shape[:2] != b.shape[:2]
        or n.size != b.shape[1] + b.shape[1] ** 2
        or p.size != 4 + b.shape[1]
        or coef.shape != (8,)
    ):
        raise ValueError("unaligned coupled-system inputs")

    s, q = b.shape[:2]
    breadth = n[:q]
    overlap = n[q:].reshape(q, q)
    exposure = p[4:]
    resistance, amplitude, recovery, net = p[:4]
    tau0, tau1, chi, db, de, du, dl, dr = coef

    tau = tau0 + tau1 * np.array(
        [net, amplitude, recovery],
        dtype=np.float64,
    )
    direction = exposure[:, None] - exposure[None, :]

    B = b * np.clip(
        1.0
        + tau[:, None, None] * overlap[None, :, :]
        + chi * direction[None, :, :],
        0.08,
        None,
    )

    for z in B:
        np.fill_diagonal(z, 0.0)

    D = (
        d
        * (1.0 + db * (1.0 - breadth)[None, :])
        * (1.0 + de * np.abs(exposure)[None, :])
    )

    U = u.copy()
    U[:, :, 0] += du * exposure[None, :]
    U[:, :, 1] -= 0.7 * du * exposure[None, :]

    reserve = (
        ell
        * (1.0 + dl * np.abs(exposure))
        * (1.0 + dr * (1.0 - resistance))
    )

    return np.concatenate(
        [
            B.astype(np.float64).ravel(),
            D.astype(np.float64).ravel(),
            U.astype(np.float64).ravel(),
            reserve.astype(np.float64).ravel(),
        ]
    )

import numpy as np


def _canonical(k):
    m = k.mean(axis=0)
    return np.vstack([k - m[None, :], m])


def _feasibility(p):
    c, m = p[:-1], p[-1]
    n = c.shape[0]
    # Finite poles reached from uniform forcing only.
    v = np.ones(n, dtype=np.float64)
    krylov = []
    for _ in range(n):
        for q in krylov:
            v = v - q * np.dot(q, v)
        norm = np.linalg.norm(v)
        if norm <= 1e-11:
            break
        q = v / norm
        krylov.append(q)
        v = -c @ q
    Q = np.column_stack(krylov)
    A = Q.T @ (-c) @ Q
    eig = np.linalg.eigvals(A)
    poles = eig.real[np.abs(eig.imag) <= 1e-8]
    bounds = [0.0]
    for pole in poles:
        if pole > 0.0:
            bounds.append(float(pole))
    grid = np.unique(np.array(bounds, dtype=np.float64))
    # Locate the last sign boundary of 1 - (aI+C)^-1 1 dot m.
    def _den(a):
        return float(1.0 - m @ np.linalg.solve(a * np.eye(n) + c, np.ones(n)))
    candidates = list(grid)
    test = np.geomspace(max(1e-8, grid.max(initial=0.0) + 1e-7), 100.0, 300)
    prev_a, prev_v = test[0], _den(test[0])
    for a in test[1:]:
        val = _den(a)
        if np.isfinite(val) and np.isfinite(prev_v) and val * prev_v < 0.0:
            lo, hi = prev_a, a
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if _den(lo) * _den(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            candidates.append(0.5 * (lo + hi))
        prev_a, prev_v = a, val
    return float(max(candidates))


def plan_certificates(competition: "np.ndarray", designs: "np.ndarray", buffer: float, minimum_effort: float) -> "np.ndarray":
    B = np.asarray(competition, dtype=np.float64)
    D = np.asarray(designs, dtype=np.float64)
    if B.ndim != 3 or D.ndim != 2 or D.shape[1] != B.shape[1] or buffer <= 1.0 or minimum_effort <= 0.0:
        raise ValueError("invalid plan-certificate inputs")
    rows = []
    for d in D:
        f, g = [], []
        for b in B:
            k = b / d[None, :]
            f.append(_feasibility(_canonical(k)))
            g.append(max(0.0, -float(np.linalg.eigvalsh((k + k.T) / 2.0)[0])))
        lo = max(float(minimum_effort), float(buffer) * max(max(f), max(g)))
        rows.append(f + g + [lo])
    return np.asarray(rows, dtype=np.float64)

import numpy as np


def _canonical(k):
    m = k.mean(axis=0)
    return np.vstack([k - m[None, :], m])


def _response(p, d, u, a):
    c, m = p[:-1], p[-1]
    A = a * np.eye(len(d)) + c
    z = np.linalg.solve(A, np.ones(len(d)))
    den = 1.0 - m @ z
    x = (z / den) / d
    out = np.empty((len(d), u.shape[1] + 1), dtype=np.float64)
    out[:, 0] = x
    for j in range(u.shape[1]):
        q = np.linalg.solve(A, u[:, j])
        out[:, j + 1] = -(q + z * (m @ q) / den) / d
    return out


def population_radius_panel(competition: "np.ndarray", slopes: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", effort: float) -> "np.ndarray":
    B = np.asarray(competition, dtype=np.float64)
    d = np.asarray(slopes, dtype=np.float64)
    U = np.asarray(loadings, dtype=np.float64)
    ell = np.asarray(reserves, dtype=np.float64)
    if B.ndim != 3 or B.shape[1] != B.shape[2] or d.shape != (B.shape[1],) or U.ndim != 3 or U.shape[:2] != B.shape[:2] or ell.shape != d.shape or not np.isfinite(effort) or effort <= 0.0:
        raise ValueError("invalid population-radius inputs")
    rows = []
    for b, loading in zip(B, U):
        p = _canonical(b / d[None, :])
        response = _response(p, d, loading, float(effort))
        norm = np.linalg.norm(response[:, 1:], axis=1)
        if np.any(norm <= 0.0):
            raise ValueError("a disturbance-response row is zero")
        rows.append((response[:, 0] - ell) / norm)
    return np.asarray(rows, dtype=np.float64)

import numpy as np


def _min_radius(panel, d, u, ell, a):
    vals = []
    for p, loading in zip(panel, u):
        response = _response(p, d, loading, a)
        vals.extend(((response[:, 0] - ell) / np.linalg.norm(response[:, 1:], axis=1)).tolist())
    return float(np.min(vals))


def continuous_plan_optimum(canonical_panel: "np.ndarray", slopes: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", interval: "np.ndarray", policy: "np.ndarray") -> "np.ndarray":
    P = np.asarray(canonical_panel, dtype=np.float64)
    d = np.asarray(slopes, dtype=np.float64)
    u = np.asarray(loadings, dtype=np.float64)
    ell = np.asarray(reserves, dtype=np.float64)
    limits = np.asarray(interval, dtype=np.float64)
    rule = np.asarray(policy, dtype=np.float64)
    if P.ndim != 3 or P.shape[1] != P.shape[2] + 1 or d.shape != (P.shape[2],) or u.shape[:2] != (P.shape[0], P.shape[2]) or ell.shape != d.shape or limits.shape != (2,) or limits[0] <= 0.0 or limits[1] < limits[0] or rule.shape != (2,) or np.any(rule < 0.0):
        raise ValueError("invalid continuous optimization inputs")
    gain, penalty = map(float, rule)
    lo, hi = map(float, limits)
    # Deterministic global search followed by bounded golden refinement on every local peak.
    grid = np.linspace(lo, hi, 1025)
    score = np.array([_min_radius(P, d, u, ell, a) + gain * np.log1p(a) - penalty * a for a in grid])
    candidates = [0, len(grid) - 1]
    candidates.extend((np.flatnonzero((score[1:-1] >= score[:-2]) & (score[1:-1] >= score[2:])) + 1).tolist())
    best_a, best_score = lo, -np.inf
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    for idx in candidates:
        left = grid[max(0, idx - 1)]
        right = grid[min(len(grid) - 1, idx + 1)]
        for _ in range(70):
            c = right - (right - left) / phi
            e = left + (right - left) / phi
            fc = _min_radius(P, d, u, ell, c) + gain * np.log1p(c) - penalty * c
            fe = _min_radius(P, d, u, ell, e) + gain * np.log1p(e) - penalty * e
            if fc >= fe:
                right = e
            else:
                left = c
        a = 0.5 * (left + right)
        val = _min_radius(P, d, u, ell, a) + gain * np.log1p(a) - penalty * a
        if val > best_score + 1e-12 or (abs(val - best_score) <= 1e-12 and a < best_a):
            best_a, best_score = a, val
    return np.array([best_a, _min_radius(P, d, u, ell, best_a), best_score], dtype=np.float64)

import numpy as np


def management_score_vector(competition: "np.ndarray", designs: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", upper_efforts: "np.ndarray", costs: "np.ndarray", penalties: "np.ndarray", buffer: float, minimum_effort: float) -> "np.ndarray":
    B = np.asarray(competition, dtype=np.float64)
    D = np.asarray(designs, dtype=np.float64)
    U = np.asarray(loadings, dtype=np.float64)
    ell = np.asarray(reserves, dtype=np.float64)
    upper = np.asarray(upper_efforts, dtype=np.float64)
    cost = np.asarray(costs, dtype=np.float64)
    penalty = np.asarray(penalties, dtype=np.float64)
    if D.ndim != 2 or upper.shape != (len(D),) or cost.shape != upper.shape or penalty.shape != (len(D), 2) or np.any(cost <= 0.0) or np.any(penalty < 0.0):
        raise ValueError("invalid plan-score inputs")
    cert = plan_certificates(B, D, float(buffer), float(minimum_effort))
    rows = np.full((len(D), 5), np.nan, dtype=np.float64)
    for i, d in enumerate(D):
        lo = cert[i, -1]
        if lo > upper[i]:
            continue
        panel = np.array([_canonical(b / d[None, :]) for b in B])
        opt = continuous_plan_optimum(panel, d, U, ell, np.array([lo, upper[i]]), penalty[i])
        radius = float(np.min(population_radius_panel(B, d, U, ell, opt[0])))
        raw = radius + penalty[i, 0] * np.log1p(opt[0]) - penalty[i, 1] * opt[0]
        score = raw / cost[i]
        rows[i] = [lo, opt[0], radius, raw, score]
    return rows

import numpy as np


def select_management_plan(score_table: "np.ndarray", tie_tolerance: float) -> "np.ndarray":
    table = np.asarray(score_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] != 5 or not np.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("score_table must have five columns and a nonnegative tie tolerance")
    valid = np.isfinite(table[:, -1]) & (table[:, -1] >= 0.0)
    if not np.any(valid):
        return np.array([-1.0, 0.0], dtype=np.float64)
    best = np.max(table[valid, -1])
    chosen = int(np.flatnonzero(valid & (table[:, -1] >= best - tie_tolerance))[0])
    return np.array([float(chosen), float(table[chosen, -1])], dtype=np.float64)

import numpy as np


def select_coupled_management(resource_matrix: "np.ndarray", k: float, reference: "np.ndarray", calibration_targets: "np.ndarray", disturbed: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray", base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", upper_efforts: "np.ndarray", costs: "np.ndarray", penalties: "np.ndarray", coefficients: "np.ndarray", buffer: float, minimum_effort: float) -> float:
    niche = noncircular_niche_profile(resource_matrix, k)
    calibration = petra_calibration(reference, calibration_targets, candidates, max_steps)
    candidate = candidates[int(calibration[0])]
    signal = petra_residual_signal(reference, disturbed, candidate, max_steps, indices, guild_map)
    packed = assemble_coupled_system(base_competition, base_designs, base_loadings,
                                     base_reserves, niche, signal, coefficients)
    s, q = base_competition.shape[:2]
    n_b = s * q * q
    n_d = base_designs.size
    n_u = base_loadings.size
    B = packed[:n_b].reshape(base_competition.shape)
    D = packed[n_b:n_b + n_d].reshape(base_designs.shape)
    U = packed[n_b + n_d:n_b + n_d + n_u].reshape(base_loadings.shape)
    ell = packed[n_b + n_d + n_u:].reshape(base_reserves.shape)
    scores = management_score_vector(B, D, U, ell, upper_efforts, costs, penalties,
                                     buffer, minimum_effort)
    return float(select_management_plan(scores, 1e-9)[1])
SCICODE_GOLD_EOF
