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

import numpy as np


def _reaches_all(adjacency: np.ndarray, start: int) -> bool:
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in np.nonzero(adjacency[node])[0]:
            nxt = int(nxt)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == adjacency.shape[0]


def build_dispersal_landscape(
    downstream: tuple,
    downstream_weight: float,
    upstream_weight: float,
    extra_links: tuple,
    sites_per_patch_drained: float,
) -> dict:
    """Reference implementation."""
    try:
        parent = [int(p) for p in downstream]
    except (TypeError, ValueError):
        raise ValueError("downstream must be a sequence of integers")
    for raw, p in zip(downstream, parent):
        if isinstance(raw, bool) or float(raw) != p:
            raise ValueError("downstream must be a sequence of integers")
    n = len(parent)
    if n < 2:
        raise ValueError("the landscape needs at least two patches")
    if sum(1 for p in parent if p == -1) != 1:
        raise ValueError("the drainage map must have exactly one outlet")
    for i, p in enumerate(parent):
        if p != -1 and not 0 <= p < n:
            raise ValueError("downstream index outside the patch set")
        if p == i:
            raise ValueError("a patch cannot drain into itself")
    wd = float(downstream_weight)
    wu = float(upstream_weight)
    rho = float(sites_per_patch_drained)
    if not (math.isfinite(wd) and wd > 0.0):
        raise ValueError("downstream_weight must be finite and above zero")
    if not (math.isfinite(wu) and wu >= 0.0):
        raise ValueError("upstream_weight must be finite and not below zero")
    if not (math.isfinite(rho) and rho > 0.0):
        raise ValueError("sites_per_patch_drained must be finite and above zero")

    drainage = np.zeros(n, dtype=int)
    for i in range(n):
        node, steps = i, 0
        while node != -1:
            drainage[node] += 1
            node = parent[node]
            steps += 1
            if steps > n:
                raise ValueError("the drainage map contains a cycle")

    weights = np.zeros((n, n))
    for i, p in enumerate(parent):
        if p != -1:
            weights[i, p] = wd
            weights[p, i] = wu
    river = weights > 0.0
    seen = set()
    for link in extra_links:
        if len(link) != 3:
            raise ValueError("each extra link must be a triple (i, j, w)")
        i, j, w = link
        if isinstance(i, bool) or isinstance(j, bool) or float(i) != int(i) or float(j) != int(j):
            raise ValueError("extra link endpoints must be integers")
        i, j, w = int(i), int(j), float(w)
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError("extra link endpoints must be distinct patches")
        if not (math.isfinite(w) and w > 0.0):
            raise ValueError("extra link weights must be finite and above zero")
        if (i, j) in seen:
            raise ValueError("extra links repeat an ordered pair")
        if river[i, j]:
            raise ValueError("extra link duplicates a river link")
        seen.add((i, j))
        weights[i, j] = w

    adjacency = weights > 0.0
    connected = all(_reaches_all(adjacency, s) for s in range(n))
    return {
        "weights": weights,
        "sites": rho * drainage.astype(float),
        "drainage": drainage,
        "out_strength": weights.sum(axis=1),
        "n_links": int(np.count_nonzero(weights)),
        "strongly_connected": int(connected),
    }

import math

import numpy as np


def _check_network(weights, sites):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError("weights must be a square array of at least two patches")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or np.any(np.diag(w) != 0.0):
        raise ValueError("weights must be finite, non-negative and zero on the diagonal")
    m = np.asarray(sites, dtype=float)
    if m.shape != (w.shape[0],) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch")
    return w, m


def explorer_resolvent(
    weights: np.ndarray,
    sites: np.ndarray,
    exploration_efficiency: float,
    mortality_ratio: float,
) -> dict:
    """Reference implementation."""
    w, m = _check_network(weights, sites)
    f = float(exploration_efficiency)
    g = float(mortality_ratio)
    if not (math.isfinite(f) and f >= 0.0 and math.isfinite(g) and g >= 0.0):
        raise ValueError("exploration_efficiency and mortality_ratio must be finite and not below zero")
    n = w.shape[0]
    q = w.sum(axis=1)
    operator = np.diag(q) - (m[None, :] / m[:, None]) * w.T

    omega, vectors = np.linalg.eig(operator)
    if np.linalg.cond(vectors) > 1e10:
        raise ValueError("the explorer operator is not diagonalisable to working precision")
    inverse_vectors = np.linalg.inv(vectors)
    summed = (vectors / (1.0 + g + f * omega)[None, :]) @ inverse_vectors
    resolvent = summed.real

    full = (1.0 + g) * np.eye(n) + f * operator
    order = np.lexsort((omega.imag, np.round(omega.real, 12)))
    return {
        "operator": operator,
        "eigenvalues_real": omega.real[order],
        "eigenvalues_imag": omega.imag[order],
        "resolvent": resolvent,
        "conservation_residual": float(np.max(np.abs(m @ operator)) / np.max(m * np.maximum(q, 1e-300))),
        "inversion_residual": float(np.max(np.abs(full @ resolvent - np.eye(n)))),
        "discarded_imaginary": float(np.max(np.abs(summed.imag))),
    }

import math

import numpy as np


def _check_network(weights, sites):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError("weights must be a square array of at least two patches")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or np.any(np.diag(w) != 0.0):
        raise ValueError("weights must be finite, non-negative and zero on the diagonal")
    m = np.asarray(sites, dtype=float)
    if m.shape != (w.shape[0],) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch")
    return w, m


def effective_colonisation_kernel(
    weights: np.ndarray,
    sites: np.ndarray,
    max_explorability: float,
    exploration_rate: float,
    colonisation_rate: float,
    explorer_death_rate: float,
) -> dict:
    """Reference implementation."""
    w, m = _check_network(weights, sites)
    xi = float(max_explorability)
    d = float(exploration_rate)
    lam = float(colonisation_rate)
    gamma = float(explorer_death_rate)
    for value in (xi, d, lam):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError("max_explorability, exploration_rate and colonisation_rate must be finite and above zero")
    if not (math.isfinite(gamma) and gamma >= 0.0):
        raise ValueError("explorer_death_rate must be finite and not below zero")
    f = d / lam
    g = gamma / lam

    resolvent = explorer_resolvent(w, m, f, g)["resolvent"]  # noqa: F821
    feasibility = xi * w / (1.0 + 1.0 / f)
    release = feasibility * (m[:, None] / m[None, :])
    kernel = resolvent @ release.T
    count_kernel = kernel * (m[:, None] / m[None, :])
    budget = count_kernel.sum(axis=0)
    expected = feasibility.sum(axis=1) / (1.0 + g)
    return {
        "kernel": kernel,
        "count_kernel": count_kernel,
        "release_feasibility": feasibility,
        "colonisation_budget": budget,
        "exploration_efficiency": f,
        "mortality_ratio": g,
        "budget_residual": float(np.max(np.abs(budget - expected))),
    }

import numpy as np


def _perron(matrix):
    values, vectors = np.linalg.eig(matrix)
    lead = int(np.argmax(values.real))
    vector = vectors[:, lead]
    vector = vector * np.exp(-1j * np.angle(vector[np.argmax(np.abs(vector))]))
    vector = vector.real
    return float(values[lead].real), vector / vector.sum()


def _is_irreducible(matrix):
    n = matrix.shape[0]
    reach = (matrix > 0.0).astype(float) + np.eye(n)
    power = np.eye(n)
    for _ in range(n - 1):
        power = np.minimum(power @ reach, 1.0)
    return bool(np.all(power > 0.0))


def generalised_capacity(
    kernel: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
) -> dict:
    """Reference implementation."""
    k = np.asarray(kernel, dtype=float)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] < 2:
        raise ValueError("kernel must be a square array of at least two patches")
    if not np.all(np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError("kernel must be finite and non-negative")
    n = k.shape[0]
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    for arr in (c, e):
        if arr.shape != (n,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("fecundity and extinction must be finite, above zero and one per patch")
    if not _is_irreducible(k):
        raise ValueError("the kernel is not irreducible")

    landscape = k * (c / e)[None, :]
    capacity, right = _perron(landscape)
    _, left = _perron(landscape.T)
    residual = max(float(np.max(np.abs(landscape @ right - capacity * right))),
                   float(np.max(np.abs(left @ landscape - capacity * left))))
    return {
        "capacity": capacity,
        "kernel_radius": float(np.max(np.abs(np.linalg.eigvals(k)))),
        "perron_right": right,
        "perron_left": left,
        "perron_residual": residual,
        "irreducible": 1,
    }

import numpy as np


def equilibrium_occupancy(
    kernel: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Reference implementation."""
    k = np.asarray(kernel, dtype=float)
    m = np.asarray(sites, dtype=float)
    if m.ndim != 1 or k.shape != (m.size, m.size) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch of the kernel")
    capacity = generalised_capacity(k, fecundity, extinction)["capacity"]  # noqa: F821
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    pressure_matrix = k * c[None, :]
    n = m.size

    p = np.ones(n)
    if capacity > 1.0:
        for _ in range(1000000):
            s = pressure_matrix @ p
            nxt = s / (e + s)
            if np.max(np.abs(nxt - p)) < 1e-15:
                p = nxt
                break
            p = nxt
        else:
            raise ValueError("the fixed-point iteration failed to converge")
        for _ in range(20):
            s = pressure_matrix @ p
            g = -e * p + (1.0 - p) * s
            jac = -np.diag(e + s) + (1.0 - p)[:, None] * pressure_matrix
            step = np.linalg.solve(jac, -g)
            p = p + step
            if np.max(np.abs(step)) < 1e-16:
                break
        if np.any(p <= 0.0) or np.any(p >= 1.0):
            raise ValueError("the steady state left the unit cube")
    else:
        p = np.zeros(n)

    s = pressure_matrix @ p
    residual = float(np.max(np.abs(-e * p + (1.0 - p) * s)))
    jac = -np.diag(e + s) + (1.0 - p)[:, None] * pressure_matrix
    return {
        "occupancy": p,
        "occupied_sites": float(m @ p),
        "mean_occupancy": float(p.mean()),
        "residual": residual,
        "stability_abscissa": float(np.max(np.linalg.eigvals(jac).real)),
        "persistent": int(capacity > 1.0),
    }

import numpy as np


def establishment_multiplier(
    resident_kernel: np.ndarray,
    invader_kernel: np.ndarray,
    resident_fecundity: np.ndarray,
    resident_extinction: np.ndarray,
    invader_fecundity: np.ndarray,
    invader_extinction: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Reference implementation."""
    k1 = np.asarray(resident_kernel, dtype=float)
    k2 = np.asarray(invader_kernel, dtype=float)
    if k1.shape != k2.shape:
        raise ValueError("the resident and invader kernels must have the same size")
    resident = equilibrium_occupancy(k1, resident_fecundity, resident_extinction, sites)  # noqa: F821
    if resident["persistent"] != 1:
        raise ValueError("the resident cannot persist on its own")
    p1 = resident["occupancy"]
    alone = generalised_capacity(k2, invader_fecundity, invader_extinction)  # noqa: F821
    thinned = (1.0 - p1)[:, None] * k2
    invasion = generalised_capacity(thinned, invader_fecundity, invader_extinction)  # noqa: F821
    resident_capacity = generalised_capacity(k1, resident_fecundity, resident_extinction)  # noqa: F821
    return {
        "resident_occupancy": p1,
        "resident_sites": resident["occupied_sites"],
        "resident_capacity": resident_capacity["capacity"],
        "invader_capacity": alone["capacity"],
        "invasion_capacity": invasion["capacity"],
        "multiplier": 1.0 / invasion["capacity"],
    }

import numpy as np
from scipy.integrate import solve_ivp


def _check_species_arrays(kernels, fecundity, extinction, sites):
    k = np.asarray(kernels, dtype=float)
    if k.ndim != 3 or k.shape[1] != k.shape[2] or k.shape[0] < 1 or k.shape[1] < 2:
        raise ValueError("kernels must be an S by N by N array")
    if not np.all(np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError("kernels must be finite and non-negative")
    s, n = k.shape[0], k.shape[1]
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    m = np.asarray(sites, dtype=float)
    for arr in (c, e):
        if arr.shape != (s, n) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("fecundity and extinction must be finite, above zero and S by N")
    if m.shape != (n,) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch")
    return k, c, e, m


def _occupancy_drift(p, pressure, e):
    free = 1.0 - p.sum(axis=0)
    return -e * p + free[None, :] * np.einsum("aij,aj->ai", pressure, p)


def _occupancy_jacobian(p, pressure, e):
    s, n = p.shape
    free = 1.0 - p.sum(axis=0)
    gain = np.einsum("aij,aj->ai", pressure, p)
    jac = np.zeros((s * n, s * n))
    for a in range(s):
        for b in range(s):
            block = -np.diag(gain[a])
            if a == b:
                block = block - np.diag(e[a]) + free[:, None] * pressure[a]
            jac[a * n:(a + 1) * n, b * n:(b + 1) * n] = block
    return jac


def coexistence_state(
    kernels: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    sites: np.ndarray,
    n_starts: int,
) -> dict:
    """Reference implementation."""
    k, c, e, m = _check_species_arrays(kernels, fecundity, extinction, sites)
    if isinstance(n_starts, bool) or int(n_starts) != n_starts or int(n_starts) < 2:
        raise ValueError("n_starts must be an integer of at least two")
    for a in range(k.shape[0]):
        generalised_capacity(k[a], c[a], e[a])  # noqa: F821
    s, n = c.shape
    pressure = k * c[:, None, :]

    finals = []
    for start in range(int(n_starts)):
        phase = np.arange(s * n).reshape(s, n) + 1.0
        weights = 0.2 + 0.8 * np.mod(phase * (0.6180339887 + 0.173 * start), 1.0)
        p0 = 0.9 * weights / weights.sum(axis=0)[None, :]
        sol = solve_ivp(lambda t, y: _occupancy_drift(y.reshape(s, n), pressure, e).ravel(),
                        (0.0, 6000.0), p0.ravel(), method="LSODA", rtol=1e-11, atol=1e-13)
        if not sol.success:
            raise ValueError("the relaxation of the occupancy dynamics failed")
        p = sol.y[:, -1].reshape(s, n)
        for _ in range(30):
            step = np.linalg.solve(_occupancy_jacobian(p, pressure, e), -_occupancy_drift(p, pressure, e).ravel())
            p = p + step.reshape(s, n)
            if np.max(np.abs(step)) < 1e-15:
                break
        if np.any(p <= 1e-9) or np.any(p.sum(axis=0) >= 1.0):
            raise ValueError("a start left a species extinct in some patch or the sites overfilled")
        finals.append(p)
    spread = max(float(np.max(np.abs(f - finals[0]))) for f in finals)
    if spread > 1e-8:
        raise ValueError("the starts reached different states")
    p = finals[0]
    eig = np.linalg.eigvals(_occupancy_jacobian(p, pressure, e)).real
    if np.max(eig) >= 0.0:
        raise ValueError("the state reached is not stable")
    held = p @ m
    return {
        "occupancy": p,
        "held_sites": held,
        "total_held": float(held.sum()),
        "start_spread": spread,
        "residual": float(np.max(np.abs(_occupancy_drift(p, pressure, e)))),
        "stability_abscissa": float(np.max(eig)),
        "slowest_rate": float(-np.max(eig)),
    }

import numpy as np


def _check_network(weights, sites):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError("weights must be a square array of at least two patches")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or np.any(np.diag(w) != 0.0):
        raise ValueError("weights must be finite, non-negative and zero on the diagonal")
    m = np.asarray(sites, dtype=float)
    if m.shape != (w.shape[0],) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch")
    return w, m


def _species_rates(n_species, *rates, allow_zero_last=True):
    out = []
    for k, r in enumerate(rates):
        arr = np.asarray(r, dtype=float)
        floor_ok = (k == len(rates) - 1) and allow_zero_last
        if arr.shape != (n_species,) or not np.all(np.isfinite(arr)) or (np.any(arr < 0.0) if floor_ok else np.any(arr <= 0.0)):
            raise ValueError("species rates must be finite, one per species, and above zero (gamma not below zero)")
        out.append(arr)
    return out


def chain_linearisation(
    weights: np.ndarray,
    sites: np.ndarray,
    settled: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    max_explorability: np.ndarray,
    exploration_rate: np.ndarray,
    colonisation_rate: np.ndarray,
    explorer_death_rate: np.ndarray,
) -> dict:
    """Reference implementation."""
    w, m = _check_network(weights, sites)
    n = m.size
    s_mat = np.asarray(settled, dtype=float)
    if s_mat.ndim != 2 or s_mat.shape[1] != n or s_mat.shape[0] < 1:
        raise ValueError("settled must be an S by N array")
    s = s_mat.shape[0]
    if not np.all(np.isfinite(s_mat)) or np.any(s_mat < 0.0) or np.any(s_mat.sum(axis=0) > m * (1.0 + 1e-12)):
        raise ValueError("settled numbers must be finite, not below zero and fit their sites")
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    for arr in (c, e):
        if arr.shape != (s, n) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("fecundity and extinction must be finite, above zero and S by N")
    xi, d, lam, gamma = _species_rates(s, max_explorability, exploration_rate, colonisation_rate, explorer_death_rate)

    q = w.sum(axis=1)
    occupied = s_mat.sum(axis=0)
    free = 1.0 - occupied / m
    dim = 2 * s * n
    jac = np.zeros((dim, dim))
    cov = np.zeros((dim, dim))
    explorers = np.zeros((s, n))
    residual = 0.0

    def _sl(block, a):
        start = (block * s + a) * n
        return slice(start, start + n)

    for a in range(s):
        feasibility = effective_colonisation_kernel(  # noqa: F821
            w, m, xi[a], d[a], lam[a], gamma[a])["release_feasibility"]
        outflow = d[a] * q + gamma[a] + lam[a]
        operator = np.diag(outflow) - d[a] * w.T
        births = feasibility.T @ (c[a] * s_mat[a])
        x = np.linalg.solve(operator, births)
        explorers[a] = x
        residual = max(residual, float(np.max(np.abs(operator @ x - births))))
        success = lam[a] * x * free

        for b in range(s):
            block = -np.diag(lam[a] * x / m)
            if a == b:
                block = block - np.diag(e[a])
            jac[_sl(0, a), _sl(0, b)] = block
        jac[_sl(0, a), _sl(1, a)] = np.diag(lam[a] * free)
        jac[_sl(1, a), _sl(0, a)] = feasibility.T * c[a][None, :]
        jac[_sl(1, a), _sl(1, a)] = d[a] * w.T - np.diag(outflow)

        cov[_sl(0, a), _sl(0, a)] = np.diag(e[a] * s_mat[a] + success)
        cov[_sl(0, a), _sl(1, a)] = np.diag(-success)
        cov[_sl(1, a), _sl(0, a)] = np.diag(-success)
        moves = w * x[:, None]
        xx = -d[a] * (moves + moves.T)
        xx[np.diag_indices(n)] = births + d[a] * q * x + d[a] * (w.T @ x) + (gamma[a] + lam[a]) * x
        cov[_sl(1, a), _sl(1, a)] = xx

    return {
        "explorers": explorers,
        "jacobian": jac,
        "jump_covariance": cov,
        "drift_residual": residual,
    }

import numpy as np
from scipy.linalg import solve_continuous_lyapunov


def stationary_covariance(
    jacobian: np.ndarray,
    jump_covariance: np.ndarray,
    n_species: int,
    n_patches: int,
) -> dict:
    """Reference implementation."""
    for value in (n_species, n_patches):
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError("n_species and n_patches must be integers of at least one")
    s, n = int(n_species), int(n_patches)
    dim = 2 * s * n
    j = np.asarray(jacobian, dtype=float)
    b = np.asarray(jump_covariance, dtype=float)
    for arr in (j, b):
        if arr.shape != (dim, dim) or not np.all(np.isfinite(arr)):
            raise ValueError("jacobian and jump_covariance must be finite and 2SN by 2SN")
    scale = max(1.0, float(np.max(np.abs(b))))
    if float(np.max(np.abs(b - b.T))) > 1e-10 * scale:
        raise ValueError("jump_covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(0.5 * (b + b.T)))) < -1e-9 * scale:
        raise ValueError("jump_covariance must be positive semi-definite")
    abscissa = float(np.max(np.linalg.eigvals(j).real))
    if abscissa >= 0.0:
        raise ValueError("the Jacobian is not stable")

    sigma = solve_continuous_lyapunov(j, -b)
    sigma = 0.5 * (sigma + sigma.T)
    totals = np.zeros((s, s))
    for a in range(s):
        for c in range(s):
            totals[a, c] = sigma[a * n:(a + 1) * n, c * n:(c + 1) * n].sum()
    sd = np.sqrt(np.diag(totals))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = totals / np.outer(sd, sd)
    patch = np.sqrt(np.clip(np.diag(sigma)[:s * n], 0.0, None)).reshape(s, n)
    return {
        "covariance": sigma,
        "total_sd": sd,
        "total_correlation": corr,
        "patch_sd": patch,
        "relaxation_rate": -abscissa,
    }

import math

import numpy as np


def mean_shift(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    colonisation_rate: np.ndarray,
    sites: np.ndarray,
) -> dict:
    """Reference implementation."""
    lam = np.asarray(colonisation_rate, dtype=float)
    m = np.asarray(sites, dtype=float)
    if lam.ndim != 1 or lam.size < 1 or not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        raise ValueError("colonisation_rate must be finite, above zero and one per species")
    if m.ndim != 1 or m.size < 1 or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite and above zero")
    s, n = lam.size, m.size
    dim = 2 * s * n
    j = np.asarray(jacobian, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    for arr in (j, sigma):
        if arr.shape != (dim, dim) or not np.all(np.isfinite(arr)):
            raise ValueError("jacobian and covariance must be finite and 2SN by 2SN")
    if float(np.max(np.abs(sigma - sigma.T))) > 1e-8 * max(1.0, float(np.max(np.abs(sigma)))):
        raise ValueError("covariance must be symmetric")
    if not math.isfinite(np.linalg.cond(j)) or np.linalg.cond(j) > 1e14:
        raise ValueError("the Jacobian is singular")

    forcing = np.zeros(dim)
    for a in range(s):
        for i in range(n):
            x_index = (s + a) * n + i
            coupled = sum(sigma[x_index, b * n + i] for b in range(s))
            forcing[a * n + i] = -(lam[a] / m[i]) * coupled
    shift = -np.linalg.solve(j, forcing)
    total = np.array([shift[a * n:(a + 1) * n].sum() for a in range(s)])
    return {"shift": shift, "total_shift": total, "forcing": forcing}

import numpy as np


def two_species_fluctuation_report(
    downstream: tuple,
    downstream_weight: float,
    upstream_weight: float,
    extra_links: tuple,
    sites_per_patch_drained: float,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    max_explorability: np.ndarray,
    exploration_rate: np.ndarray,
    colonisation_rate: np.ndarray,
    explorer_death_rate: np.ndarray,
    n_starts: int,
) -> dict:
    """Reference implementation."""
    landscape = build_dispersal_landscape(  # noqa: F821
        downstream, downstream_weight, upstream_weight, extra_links, sites_per_patch_drained)
    if landscape["strongly_connected"] != 1:
        raise ValueError("the dispersal network is not strongly connected")
    weights = landscape["weights"]
    sites = landscape["sites"]
    n = sites.size
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    rates = [np.asarray(r, dtype=float) for r in (max_explorability, exploration_rate, colonisation_rate, explorer_death_rate)]
    if c.shape != (2, n) or e.shape != (2, n) or any(r.shape != (2,) for r in rates):
        raise ValueError("fecundity and extinction must be 2 by N and the explorer rates must be given for two species")
    xi, d, lam, gamma = rates

    kernels = np.zeros((2, n, n))
    max_imag = np.zeros(2)
    for a in range(2):
        out = effective_colonisation_kernel(  # noqa: F821
            weights, sites, xi[a], d[a], lam[a], gamma[a])
        kernels[a] = out["kernel"]
        spectrum = explorer_resolvent(  # noqa: F821
            weights, sites, out["exploration_efficiency"], out["mortality_ratio"])
        max_imag[a] = float(np.max(np.abs(spectrum["eigenvalues_imag"])))

    capacity = generalised_capacity(kernels[0], c[0], e[0])  # noqa: F821
    invader_alone = equilibrium_occupancy(kernels[1], c[1], e[1], sites)  # noqa: F821
    establish = establishment_multiplier(  # noqa: F821
        kernels[0], kernels[1], c[0], e[0], c[1], e[1], sites)
    coexist = coexistence_state(kernels, c, e, sites, n_starts)  # noqa: F821
    settled = coexist["occupancy"] * sites[None, :]
    chain = chain_linearisation(  # noqa: F821
        weights, sites, settled, c, e, xi, d, lam, gamma)
    fluct = stationary_covariance(chain["jacobian"], chain["jump_covariance"], 2, n)  # noqa: F821
    shift = mean_shift(chain["jacobian"], fluct["covariance"], lam, sites)  # noqa: F821

    return {
        "species2_total_sd": float(fluct["total_sd"][1]),
        "species1_capacity": capacity["capacity"],
        "establishment_multiplier": establish["multiplier"],
        "held_sites": coexist["held_sites"],
        "species1_total_sd": float(fluct["total_sd"][0]),
        "species2_mean_shift": float(shift["total_shift"][1]),
        "species1_mean_shift": float(shift["total_shift"][0]),
        "total_correlation": float(fluct["total_correlation"][0, 1]),
        "relaxation_rate": fluct["relaxation_rate"],
        "explorer_totals": chain["explorers"].sum(axis=1),
        "invader_capacity": establish["invader_capacity"],
        "invasion_capacity": establish["invasion_capacity"],
        "resident_sites": establish["resident_sites"],
        "invader_alone_sites": invader_alone["occupied_sites"],
        "operator_max_imag": max_imag,
        "sites": sites,
    }
SCICODE_GOLD_EOF
