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
 
 
def assemble_periodic_system(base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", pulse_decay_base: "np.ndarray", climate_covariance_base: "np.ndarray", idiosyncratic_noise_base: "np.ndarray", niche_profile: "np.ndarray", petra_signal: "np.ndarray", coefficients: "np.ndarray") -> "np.ndarray":
    b = np.asarray(base_competition, dtype=np.float64)
    d = np.asarray(base_designs, dtype=np.float64)
    u = np.asarray(base_loadings, dtype=np.float64)
    reserve0 = np.asarray(base_reserves, dtype=np.float64)
    pulse0 = np.asarray(pulse_decay_base, dtype=np.float64)
    climate0 = np.asarray(climate_covariance_base, dtype=np.float64)
    idio0 = np.asarray(idiosyncratic_noise_base, dtype=np.float64)
    n = np.asarray(niche_profile, dtype=np.float64)
    p = np.asarray(petra_signal, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64)
    if b.ndim != 3 or b.shape[1] != b.shape[2] or d.ndim != 2 or d.shape[1] != b.shape[1]:
        raise ValueError("unaligned competition and design arrays")
    s, q = b.shape[:2]
    plans = d.shape[0]
    if u.shape != (s, q, 2) or reserve0.shape != (q,) or pulse0.shape != (plans, q):
        raise ValueError("unaligned loading, reserve, or pulse arrays")
    if climate0.shape != (s, 2, 2) or idio0.shape != (s, q):
        raise ValueError("unaligned climate arrays")
    if n.size != q + q * q or p.size != 4 + q or coef.shape != (11,):
        raise ValueError("unaligned niche, PETRA, or coefficient arrays")
    arrays = (b, d, u, reserve0, pulse0, climate0, idio0, n, p, coef)
    if not all(np.isfinite(x).all() for x in arrays) or np.any(d <= 0.0) or np.any(reserve0 < 0.0) or np.any(pulse0 <= 0.0) or np.any(idio0 <= 0.0):
        raise ValueError("periodic-system inputs must be finite with positive scales")
    breadth = n[:q]
    overlap = n[q:].reshape(q, q)
    resistance, amplitude, recovery, net = p[:4]
    exposure = p[4:]
    tau0, tau1, chi, db, de, du, dl, dr, dp, dc, di = coef
    tau = tau0 + tau1 * np.array([net, amplitude, recovery], dtype=np.float64)
    if s != 3:
        raise ValueError("the periodic task requires three climate conditions")
    direction = exposure[:, None] - exposure[None, :]
    B = b * np.clip(1.0 + tau[:, None, None] * overlap[None, :, :] + chi * direction[None, :, :], 0.08, None)
    for matrix in B:
        np.fill_diagonal(matrix, 0.0)
    D = d * (1.0 + db * (1.0 - breadth)[None, :]) * (1.0 + de * np.abs(exposure)[None, :])
    U = u.copy()
    U[:, :, 0] += du * exposure[None, :]
    U[:, :, 1] -= 0.7 * du * exposure[None, :]
    reserve = reserve0 * (1.0 + dl * np.abs(exposure)) * (1.0 + dr * (1.0 - resistance))
    pulse = pulse0 * (1.0 + dp * (1.0 - breadth)[None, :]) * (1.0 + 0.5 * dp * np.abs(exposure)[None, :])
    climate = climate0.copy()
    climate[:, 0, 1] += dc * np.array([net, -amplitude, recovery], dtype=np.float64)
    climate[:, 1, 0] = climate[:, 0, 1]
    if np.any(np.linalg.eigvalsh(climate) <= 0.0):
        raise ValueError("adjusted climate covariance must be positive definite")
    idio = idio0 * (1.0 + di * np.abs(exposure)[None, :])
    return np.concatenate([B.ravel(), D.ravel(), U.ravel(), reserve.ravel(), pulse.ravel(), climate.ravel(), idio.ravel()]).astype(np.float64)

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
from scipy.linalg import expm, solve_discrete_lyapunov
 
 
def periodic_risk_panel(competition: "np.ndarray", slope: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", effort: float, contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    B = np.asarray(competition, dtype=np.float64)
    d = np.asarray(slope, dtype=np.float64)
    U = np.asarray(loadings, dtype=np.float64)
    reserve = np.asarray(reserves, dtype=np.float64)
    pulse = np.asarray(pulse_decay, dtype=np.float64)
    climate = np.asarray(climate_covariance, dtype=np.float64)
    idio = np.asarray(idiosyncratic_noise, dtype=np.float64)
    duration = np.asarray(durations, dtype=np.float64)
    e = np.asarray(exposure, dtype=np.float64)
    if B.ndim != 3 or B.shape[1] != B.shape[2]:
        raise ValueError("competition must contain square condition matrices")
    s, q = B.shape[:2]
    if d.shape != (q,) or U.shape != (s, q, 2) or reserve.shape != (q,) or pulse.shape != (q,):
        raise ValueError("unaligned plan arrays")
    if climate.shape != (s, 2, 2) or idio.shape != (s, q) or duration.shape != (s, 2) or e.shape != (q,):
        raise ValueError("unaligned climate arrays")
    arrays = (B, d, U, reserve, pulse, climate, idio, duration, e)
    if not all(np.isfinite(x).all() for x in arrays) or np.any(d <= 0.0) or np.any(pulse <= 0.0) or np.any(idio <= 0.0) or np.any(duration <= 0.0):
        raise ValueError("periodic-risk inputs must be finite with positive scales")
    if not np.isfinite(effort) or effort <= 0.0 or contraction_gain < 0.0 or not (0.0 < stability_limit < 1.0) or chi_radius <= 0.0:
        raise ValueError("invalid periodic-risk scalars")
    rows = []
    for c in range(s):
        M = float(effort) * np.diag(d) + B[c]
        x = np.linalg.solve(M, np.ones(q, dtype=np.float64))
        scale0 = 0.78 + 0.55 * np.abs(e) + 0.08 * c
        scale1 = 1.22 - 0.35 * np.abs(e) - 0.05 * c
        J0 = -np.diag(x * scale0) @ M
        J1 = -np.diag(x * scale1) @ M
        E0 = expm(J0 * duration[c, 0])
        E1 = expm(J1 * duration[c, 1])
        P = np.diag(np.exp(-(pulse + contraction_gain * float(effort))))
        F = E1 @ P @ E0
        rho = float(np.max(np.abs(np.linalg.eigvals(F))))
        G0 = np.diag(x) @ U[c]
        rotation = np.array([[0.82 + 0.03 * c, -0.21], [0.17, 0.91 - 0.02 * c]], dtype=np.float64)
        G1 = np.diag(x) @ (U[c] @ rotation)
        Q0 = G0 @ climate[c] @ G0.T + np.diag(idio[c] * x * x)
        Q1 = G1 @ climate[c] @ G1.T + np.diag((1.15 - 0.08 * c) * idio[c] * x * x)
        Qcycle = E1 @ P @ Q0 @ P.T @ E1.T + Q1
        usable = bool(np.all(x > reserve) and rho < stability_limit)
        if usable:
            C = solve_discrete_lyapunov(F, Qcycle)
            C = (C + C.T) / 2.0
            diagonal = np.diag(C)
            if np.any(diagonal <= 0.0) or not np.isfinite(C).all():
                raise ValueError("stationary covariance is not positive on the diagonal")
            standardized = (x - reserve) / (float(chi_radius) * np.sqrt(diagonal))
            limiting = int(np.argmin(standardized))
            trace_ratio = float(np.trace(C) / np.dot(x, x))
            security = float(standardized[limiting])
        else:
            trace_ratio = 1.0e12
            security = -1.0e12
            limiting = -1
        rows.append([float(np.min(x)), rho, trace_ratio, security, float(limiting), float(usable)])
    return np.asarray(rows, dtype=np.float64)

import numpy as np
from scipy.optimize import minimize_scalar
 
 
def continuous_plan_profile(competition: "np.ndarray", slope: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", interval: "np.ndarray", policy: "np.ndarray", cost: float, contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    bounds = np.asarray(interval, dtype=np.float64)
    rule = np.asarray(policy, dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] <= 0.0 or bounds[1] < bounds[0]:
        raise ValueError("interval must be a positive closed interval")
    if rule.shape != (5,) or not np.isfinite(rule).all() or not np.isfinite(cost) or cost <= 0.0:
        raise ValueError("invalid policy or cost")
    gain, effort_penalty, covariance_penalty, floquet_penalty, curvature = rule
 
    def _evaluate(a):
        panel = periodic_risk_panel(competition, slope, loadings, reserves, pulse_decay, climate_covariance, idiosyncratic_noise, durations, exposure, float(a), contraction_gain, stability_limit, chi_radius)
        if not np.all(panel[:, 5] == 1.0):
            return None
        shared = gain * np.log1p(float(a)) - effort_penalty * float(a) - curvature * (float(a) - 6.1) ** 2
        scores = (panel[:, 3] - covariance_penalty * np.log1p(panel[:, 2]) - floquet_penalty * panel[:, 1] + shared) / float(cost)
        return scores, panel
 
    grid = np.linspace(bounds[0], bounds[1], 129, dtype=np.float64)
    values = np.full(129, -1.0e12, dtype=np.float64)
    for index, effort in enumerate(grid):
        outcome = _evaluate(float(effort))
        if outcome is not None:
            values[index] = float(np.min(outcome[0]))
    peak_indices = [index for index in range(1, 128) if values[index] > -1.0e11 and values[index] >= values[index - 1] and values[index] >= values[index + 1]]
    candidates = [float(grid[index]) for index in peak_indices]
    candidates.extend([float(grid[0]), float(grid[-1])])
    for index in peak_indices:
        lo, hi = float(grid[index - 1]), float(grid[index + 1])
 
        def _objective(a):
            outcome = _evaluate(float(a))
            return 1.0e12 if outcome is None else -float(np.min(outcome[0]))
 
        fitted = minimize_scalar(_objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-11, "maxiter": 120})
        if fitted.success and fitted.fun < 1.0e11:
            candidates.append(float(fitted.x))
    best = None
    for effort in candidates:
        outcome = _evaluate(effort)
        if outcome is None:
            continue
        scores, panel = outcome
        robust = float(np.min(scores))
        if best is None or robust > best[0] + 1e-12 or (abs(robust - best[0]) <= 1e-12 and effort < best[1]):
            best = (robust, effort, scores, panel)
    if best is None:
        unavailable = np.full(8, -1.0e12, dtype=np.float64)
        unavailable[[0, 5, 6, 7]] = -1.0
        return unavailable
    robust, effort, scores, panel = best
    active = int(np.argmin(scores))
    return np.array([effort, scores[0], scores[1], scores[2], robust, float(active), panel[active, 4], float(np.max(panel[:, 1]))], dtype=np.float64)

import numpy as np
 
 
def management_profile_table(competition: "np.ndarray", slopes: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", certificates: "np.ndarray", upper_efforts: "np.ndarray", policies: "np.ndarray", costs: "np.ndarray", contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    D = np.asarray(slopes, dtype=np.float64)
    P = np.asarray(pulse_decay, dtype=np.float64)
    cert = np.asarray(certificates, dtype=np.float64)
    upper = np.asarray(upper_efforts, dtype=np.float64)
    rules = np.asarray(policies, dtype=np.float64)
    price = np.asarray(costs, dtype=np.float64)
    if D.ndim != 2 or P.shape != D.shape or cert.ndim != 2 or cert.shape[0] != D.shape[0]:
        raise ValueError("unaligned plan table inputs")
    plans = D.shape[0]
    if upper.shape != (plans,) or rules.shape != (plans, 5) or price.shape != (plans,) or np.any(price <= 0.0):
        raise ValueError("unaligned plan policy inputs")
    rows = []
    for plan in range(plans):
        interval = np.array([cert[plan, -1], upper[plan]], dtype=np.float64)
        rows.append(continuous_plan_profile(competition, D[plan], loadings, reserves, P[plan], climate_covariance, idiosyncratic_noise, durations, exposure, interval, rules[plan], price[plan], contraction_gain, stability_limit, chi_radius))
    return np.asarray(rows, dtype=np.float64)

import numpy as np
 
 
def select_regret_controlled_plan(profile_table: "np.ndarray", regret_weight: float, regret_limit: float, tie_tolerance: float) -> "np.ndarray":
    table = np.asarray(profile_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 8:
        raise ValueError("profile_table must contain at least two plans and three conditions")
    if regret_weight < 0.0 or regret_limit < 0.0 or tie_tolerance < 0.0 or not np.isfinite([regret_weight, regret_limit, tie_tolerance]).all():
        raise ValueError("invalid regret-selection scalars")
    conditions = table.shape[1] - 5
    scenario = table[:, 1:1 + conditions]
    robust = table[:, 1 + conditions]
    available = np.isfinite(scenario).all(axis=1) & (robust > -1.0e11)
    if not np.any(available):
        raise ValueError("no plan has a usable common-effort profile")
    best_by_condition = np.max(scenario[available], axis=0)
    audit = np.full((table.shape[0], 4), -1.0e12, dtype=np.float64)
    for plan in np.where(available)[0]:
        regrets = best_by_condition - scenario[plan]
        tail = float(np.sort(regrets)[-2:].mean())
        decision = float(robust[plan] - regret_weight * tail) if tail <= regret_limit else -1.0e12
        audit[plan] = [decision, robust[plan], tail, table[plan, 0]]
    best = float(np.max(audit[:, 0]))
    eligible = np.where(audit[:, 0] >= best - tie_tolerance)[0]
    if eligible.size == 0 or best <= -1.0e11:
        raise ValueError("no plan satisfies the regret limit")
    winner = int(eligible[0])
    head = np.array([float(winner), audit[winner, 0], audit[winner, 1], audit[winner, 2], audit[winner, 3]], dtype=np.float64)
    return np.concatenate([head, audit.ravel()]).astype(np.float64)

import numpy as np
 
 
def select_periodic_management(resource_matrix: "np.ndarray", k: float, reference: "np.ndarray", calibration_targets: "np.ndarray", disturbed: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray", base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", pulse_decay_base: "np.ndarray", climate_covariance_base: "np.ndarray", idiosyncratic_noise_base: "np.ndarray", durations: "np.ndarray", upper_efforts: "np.ndarray", costs: "np.ndarray", policies: "np.ndarray", coefficients: "np.ndarray", buffer: float, minimum_effort: float, contraction_gain: float, stability_limit: float, chi_radius: float, regret_weight: float, regret_limit: float, tie_tolerance: float) -> float:
    niche = noncircular_niche_profile(resource_matrix, k)
    calibration = petra_calibration(reference, calibration_targets, candidates, max_steps)
    candidate = candidates[int(calibration[0])]
    signal = petra_residual_signal(reference, disturbed, candidate, max_steps, indices, guild_map)
    packed = assemble_periodic_system(base_competition, base_designs, base_loadings, base_reserves, pulse_decay_base, climate_covariance_base, idiosyncratic_noise_base, niche, signal, coefficients)
    B0 = np.asarray(base_competition, dtype=np.float64)
    D0 = np.asarray(base_designs, dtype=np.float64)
    U0 = np.asarray(base_loadings, dtype=np.float64)
    reserve0 = np.asarray(base_reserves, dtype=np.float64)
    pulse0 = np.asarray(pulse_decay_base, dtype=np.float64)
    climate0 = np.asarray(climate_covariance_base, dtype=np.float64)
    idio0 = np.asarray(idiosyncratic_noise_base, dtype=np.float64)
    sizes = [B0.size, D0.size, U0.size, reserve0.size, pulse0.size, climate0.size, idio0.size]
    cuts = np.cumsum([0] + sizes)
    B = packed[cuts[0]:cuts[1]].reshape(B0.shape)
    D = packed[cuts[1]:cuts[2]].reshape(D0.shape)
    U = packed[cuts[2]:cuts[3]].reshape(U0.shape)
    reserve = packed[cuts[3]:cuts[4]].reshape(reserve0.shape)
    pulse = packed[cuts[4]:cuts[5]].reshape(pulse0.shape)
    climate = packed[cuts[5]:cuts[6]].reshape(climate0.shape)
    idio = packed[cuts[6]:cuts[7]].reshape(idio0.shape)
    certificates = plan_certificates(B, D, buffer, minimum_effort)
    profiles = management_profile_table(B, D, U, reserve, pulse, climate, idio, durations, signal[4:], certificates, upper_efforts, policies, costs, contraction_gain, stability_limit, chi_radius)
    selected = select_regret_controlled_plan(profiles, regret_weight, regret_limit, tie_tolerance)
    return float(selected[1])
SCICODE_GOLD_EOF
