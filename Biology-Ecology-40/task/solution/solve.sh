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


def _check_strain(phi, preference, present):
    """Validate a strain's trait, preference order and the presence vector."""
    phi = float(phi)
    if not math.isfinite(phi) or phi < 0.0 or phi > 1.0:
        raise ValueError("phi must be a finite number in [0, 1]")
    pref = np.asarray(preference)
    if pref.ndim != 1 or pref.size < 2 or not np.issubdtype(pref.dtype, np.integer):
        raise ValueError("preference must be a one-dimensional integer array of length at least two")
    if sorted(pref.tolist()) != list(range(pref.size)):
        raise ValueError("preference must be a permutation of 0, ..., n - 1")
    pres = np.asarray(present)
    if pres.shape != pref.shape or pres.dtype != bool:
        raise ValueError("present must be a Boolean array matching the preference order")
    return phi, pref, pres


def proteome_allocation(phi: float, preference: np.ndarray, present: np.ndarray,
                                max_rate: float = 0.8, other_factor: float = 0.5) -> dict:
    """Reference implementation."""
    phi, pref, pres = _check_strain(phi, preference, present)
    max_rate = float(max_rate)
    other_factor = float(other_factor)
    if not math.isfinite(max_rate) or max_rate <= 0.0:
        raise ValueError("max_rate must be positive and finite")
    if not math.isfinite(other_factor) or other_factor <= 0.0 or other_factor > 1.0:
        raise ValueError("other_factor must lie in (0, 1]")

    n = pref.size
    rates = np.full(n, max_rate * other_factor)
    rates[pref[0]] = max_rate
    allocation = np.zeros(n)
    count = int(pres.sum())
    if count == 1:
        allocation[pres] = 1.0
    elif count > 1:
        allocation[pres] = phi / count
        primary = next(int(k) for k in pref if pres[k])
        allocation[primary] = 1.0 - (count - 1) * phi / count
    uptake = allocation * rates
    return {"allocation": allocation,
            "potential_rates": rates,
            "uptake": uptake,
            "growth_rate": float(uptake.sum())}

import math

import numpy as np


def reallocation_lag(phi: float, preference: np.ndarray, depleted: int, present_after: np.ndarray,
                             lag_scale: float = 0.3) -> dict:
    """Reference implementation."""
    lag_scale = float(lag_scale)
    if not math.isfinite(lag_scale) or lag_scale <= 0.0:
        raise ValueError("lag_scale must be positive and finite")
    after = np.asarray(present_after)
    if isinstance(depleted, bool) or not isinstance(depleted, (int, np.integer)):
        raise ValueError("depleted must be an integer index")
    depleted = int(depleted)
    if after.ndim != 1 or depleted < 0 or depleted >= after.size:
        raise ValueError("depleted must index a resource")
    if after.dtype == bool and after[depleted]:
        raise ValueError("the exhausted resource cannot be present afterwards")
    before = after.copy()
    before[depleted] = True
    alloc = proteome_allocation(phi, preference, before)["allocation"]  # noqa: F821
    if alloc[depleted] <= 0.0 or not after.any():
        return {"lag": 0.0, "retained_fraction": 1.0}
    retained = float(alloc[after].sum())
    if retained <= 0.0:
        return {"lag": math.inf, "retained_fraction": 0.0}
    return {"lag": lag_scale * math.log(1.0 / retained), "retained_fraction": retained}

import math

import numpy as np


def _exhaustion_root(stock, coefficients, rates, horizon):
    """Root of stock - sum c_i expm1(r_i t) on (0, horizon], or None when it lies beyond."""
    total = sum(coefficients)
    if total <= 0.0:
        return None
    if stock - sum(c * math.expm1(r * horizon) for c, r in zip(coefficients, rates)) > 0.0:
        return None
    lo = math.log1p(stock / total) / max(rates)
    hi = min(horizon, math.log1p(stock / total) / min(rates))
    t = hi
    for _ in range(200):
        value = stock - sum(c * math.expm1(r * t) for c, r in zip(coefficients, rates))
        if value > 0.0:
            lo = t
        else:
            hi = t
        slope = -sum(c * r * math.exp(r * t) for c, r in zip(coefficients, rates))
        step = t - value / slope
        if not (lo <= step <= hi):
            step = 0.5 * (lo + hi)
        if abs(step - t) <= 1e-15 * max(t, 1e-300):
            return step
        t = step
    return t


def first_exhaustion(stock: np.ndarray, biomass: np.ndarray, rates: np.ndarray, uptake: np.ndarray,
                             horizon: float) -> dict:
    """Reference implementation."""
    s = np.asarray(stock, dtype=float)
    b = np.asarray(biomass, dtype=float)
    r = np.asarray(rates, dtype=float)
    u = np.asarray(uptake, dtype=float)
    horizon = float(horizon)
    if s.ndim != 1 or b.ndim != 1 or r.shape != b.shape or u.shape != (b.size, s.size):
        raise ValueError("stock (n,), biomass (m,), rates (m,) and uptake (m, n) must agree in shape")
    for name, a in (("stock", s), ("biomass", b), ("rates", r), ("uptake", u)):
        if not np.all(np.isfinite(a)) or np.any(a < 0.0):
            raise ValueError(name + " must be finite and nonnegative")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if np.any(np.abs(u.sum(axis=1) - r) > 1e-12 * np.maximum(1.0, r)):
        raise ValueError("each growth rate must equal the sum of its uptake rates")

    best_time, best_resource = horizon, -1
    for k in range(s.size):
        if s[k] <= 0.0:
            continue
        coeffs, rs = [], []
        for i in range(b.size):
            if r[i] > 0.0 and u[i, k] > 0.0 and b[i] > 0.0:
                coeffs.append(float(b[i] * u[i, k] / r[i]))
                rs.append(float(r[i]))
        root = _exhaustion_root(float(s[k]), coeffs, rs, best_time)
        if root is not None and (root < best_time or best_resource < 0):
            best_time, best_resource = root, k
    return {"time": float(best_time), "resource": int(best_resource)}

import math

import numpy as np


def grow_one_cycle(phi: np.ndarray, preference: np.ndarray, biomass: np.ndarray, supply: np.ndarray,
                           lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Reference implementation."""
    phis = np.asarray(phi, dtype=float)
    prefs = np.asarray(preference)
    b = np.array(biomass, dtype=float)
    stock = np.array(supply, dtype=float)
    supplied = stock.copy()
    lag_scale = float(lag_scale)
    cycle_length = float(cycle_length)
    if phis.ndim != 1 or phis.size < 1 or prefs.shape != (phis.size, 2) or b.shape != phis.shape:
        raise ValueError("phi (m,), preference (m, 2) and biomass (m,) must agree in shape")
    if stock.shape != (2,) or not np.all(np.isfinite(stock)) or np.any(stock <= 0.0):
        raise ValueError("supply must hold two positive finite amounts")
    if not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError("biomass must be positive and finite")
    for name, v in (("lag_scale", lag_scale), ("cycle_length", cycle_length)):
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(name + " must be positive and finite")

    m = phis.size
    present = np.array([True, True])
    lag_left = np.zeros(m)
    lags = np.zeros(m)
    times = np.full(2, math.inf)
    first = -1
    t = 0.0
    while t < cycle_length and present.any():
        uptake = np.zeros((m, 2))
        rates = np.zeros(m)
        for i in range(m):
            if lag_left[i] <= 0.0:
                d = proteome_allocation(phis[i], prefs[i], present)  # noqa: F821
                uptake[i] = d["uptake"]
                rates[i] = d["growth_rate"]
        waiting = lag_left[lag_left > 0.0]
        horizon = min(cycle_length - t, float(waiting.min()) if waiting.size else math.inf)
        if math.isinf(horizon):
            break
        event = first_exhaustion(stock * present, b, rates, uptake, horizon)  # noqa: F821
        dt = event["time"]
        k = event["resource"]
        grow = np.expm1(rates * dt)
        for j in range(2):
            if present[j]:
                used = float(np.sum(np.where(rates > 0.0, b * uptake[:, j] / np.where(rates > 0.0, rates, 1.0) * grow, 0.0)))
                stock[j] = max(stock[j] - used, 0.0)
        b = b * np.exp(rates * dt)
        t += dt
        in_lag = lag_left > 0.0
        lag_left[in_lag] = np.where(lag_left[in_lag] - dt <= 1e-12 * max(1.0, t), 0.0, lag_left[in_lag] - dt)
        exhausted = [k] if k >= 0 else []
        for j in range(2):
            if present[j] and j != k and stock[j] <= 1e-12 * supplied[j]:
                exhausted.append(j)
        if exhausted:
            for j in exhausted:
                present[j] = False
                stock[j] = 0.0
                times[j] = t
                if first < 0:
                    first = j
            if not present.any():
                break
            for i in range(m):
                for j in exhausted:
                    if not in_lag[i] and uptake[i, j] > 0.0:
                        lag = reallocation_lag(phis[i], prefs[i], j, present.copy(), lag_scale)["lag"]  # noqa: F821
                        lag_left[i] = lag
                        lags[i] += lag
    return {"biomass": b, "exhaustion_times": times, "first_exhausted": int(first), "lags": lags}

import math

import numpy as np


def _strain_table(phi, lag_scale):
    """Rates and lags of a strain of trait phi for both preference orders.

    Entry j holds the two-resource growth rate, the single-resource rate on each resource and
    the lag after the exhaustion of each resource, for the strain preferring resource j.
    """
    _PREFS = (np.array([0, 1]), np.array([1, 0]))
    _BOTH = np.array([True, True])
    _ALONE = (np.array([True, False]), np.array([False, True]))
    table = []
    for pref in _PREFS:
        both = proteome_allocation(phi, pref, _BOTH)  # noqa: F821
        single = [proteome_allocation(phi, pref, _ALONE[k])["growth_rate"] for k in (0, 1)]  # noqa: F821
        lags = []
        for k in (0, 1):
            if both["uptake"][k] > 0.0:
                lags.append(reallocation_lag(phi, pref, k, _ALONE[1 - k], lag_scale)["lag"])  # noqa: F821
            else:
                lags.append(0.0)
        table.append((both["growth_rate"], single, lags))
    return table


def _replay(entry, first, t1, t2, cycle_length):
    """Log fold change of a rare strain under the exhaustion schedule (first, t1, t2)."""
    rate_both, single, lags = entry
    if first < 0 or t1 >= cycle_length:
        return rate_both * cycle_length
    remaining = 1 - first
    start = t1 + lags[first]
    stop = min(t2, cycle_length)
    return rate_both * t1 + (single[remaining] * (stop - start) if stop > start else 0.0)


def invasion_growth_rates(phi_resident: float, phi_mutants: np.ndarray, supply: np.ndarray, burn_in: int,
                                  dilution: float = 100.0, lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Reference implementation."""
    phi_resident = float(phi_resident)
    if not math.isfinite(phi_resident) or phi_resident <= 0.0 or phi_resident > 1.0:
        raise ValueError("phi_resident must lie in (0, 1]")
    mutants = np.asarray(phi_mutants, dtype=float)
    if mutants.ndim != 1 or mutants.size < 1 or not np.all(np.isfinite(mutants)) \
            or np.any(mutants < 0.0) or np.any(mutants > 1.0):
        raise ValueError("phi_mutants must be a nonempty one-dimensional array in [0, 1]")
    s = np.asarray(supply, dtype=float)
    if s.ndim != 2 or s.shape[1] != 2 or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("supply must be a positive finite array of shape (n_cycles, 2)")
    if isinstance(burn_in, bool) or not isinstance(burn_in, (int, np.integer)) or burn_in < 0 or burn_in >= s.shape[0]:
        raise ValueError("burn_in must be an integer in [0, n_cycles)")
    dilution = float(dilution)
    if not math.isfinite(dilution) or dilution <= 1.0:
        raise ValueError("dilution must exceed one")

    resident_table = _strain_table(phi_resident, lag_scale)
    mutant_tables = [_strain_table(p, lag_scale) for p in mutants]
    phis = np.array([phi_resident, phi_resident])
    prefs = np.array([[0, 1], [1, 0]])
    b = np.full(2, 1.0 / (2.0 * (dilution - 1.0)))
    log_d = math.log(dilution)
    sums = np.zeros((2, mutants.size))
    res_sums = np.zeros(2)
    share = 0.0
    error = 0.0
    niche_one = 0.0
    niche_two = 0.0
    kept = 0
    for c in range(s.shape[0]):
        start = b
        out = grow_one_cycle(phis, prefs, start, s[c], lag_scale, cycle_length)  # noqa: F821
        first = out["first_exhausted"]
        times = out["exhaustion_times"]
        t1 = float(times[first]) if first >= 0 else math.inf
        t2 = float(times[1 - first]) if first >= 0 else math.inf
        if c >= burn_in:
            kept += 1
            share += start[0] / start.sum()
            niche_one += min(t1, cycle_length)
            niche_two += min(t2, cycle_length) - min(t1, cycle_length)
            for j in (0, 1):
                g = _replay(resident_table[j], first, t1, t2, cycle_length)
                res_sums[j] += g - log_d
                error = max(error, abs(g - math.log(out["biomass"][j] / start[j])))
                for q, table in enumerate(mutant_tables):
                    sums[j, q] += _replay(table[j], first, t1, t2, cycle_length) - log_d
        b = out["biomass"] / dilution
    return {"invasion_rates": sums / kept,
            "resident_rates": res_sums / kept,
            "replay_error": float(error),
            "mean_share": float(share / kept),
            "primary_niche": float(niche_one / kept),
            "secondary_niche": float(niche_two / kept)}

import math

import numpy as np


def selection_gradient(phi: float, supply: np.ndarray, burn_in: int, step: float = 0.05,
                               dilution: float = 100.0, lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Reference implementation."""
    phi = float(phi)
    step = float(step)
    if not math.isfinite(step) or step <= 0.0 or step > 0.5:
        raise ValueError("step must lie in (0, 0.5]")
    if not math.isfinite(phi) or phi <= 0.0 or phi * math.exp(step) > 1.0 + 1e-12:
        raise ValueError("phi must be positive with phi exp(step) at most one")
    mutants = np.array([phi * math.exp(-step), phi, min(phi * math.exp(step), 1.0)])
    rates = invasion_growth_rates(phi, mutants, supply, burn_in, dilution, lag_scale, cycle_length)["invasion_rates"]  # noqa: F821
    gradient = float(np.mean((rates[:, 2] - rates[:, 0]) / (2.0 * step)))
    curvature = float(np.mean((rates[:, 2] - 2.0 * rates[:, 1] + rates[:, 0]) / step ** 2))
    return {"gradient": gradient, "curvature": curvature, "rates": rates}

import math

import numpy as np


def singular_strategy(supply: np.ndarray, burn_in: int, lower: float = 0.01, upper: float = 0.9,
                              tolerance: float = 1e-9) -> dict:
    """Reference implementation."""
    lower = float(lower)
    upper = float(upper)
    tolerance = float(tolerance)
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower <= 0.0 or upper <= lower \
            or upper > math.exp(-0.07):
        raise ValueError("the bracket must satisfy 0 < lower < upper <= exp(-0.07)")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    def _gradient(u):
        return selection_gradient(math.exp(u), supply, burn_in)["gradient"]  # noqa: F821

    a, b = math.log(lower), math.log(upper)
    ga, gb = _gradient(a), _gradient(b)
    g_lower, g_upper = ga, gb
    if not (ga > 0.0 > gb):
        raise ValueError("the selection gradient must be positive at lower and negative at upper")
    side = 0
    x = a
    evaluations = 0
    for _ in range(100):
        x_new = b - gb * (b - a) / (gb - ga)
        gx = _gradient(x_new)
        evaluations += 1
        moved = abs(x_new - x)
        x = x_new
        if gx == 0.0:
            a = b = x
            break
        if gx > 0.0:
            a, ga = x, gx
            if side == 1:
                gb *= 0.5
            side = 1
        else:
            b, gb = x, gx
            if side == -1:
                ga *= 0.5
            side = -1
        if b - a < tolerance or moved < tolerance:
            break
    at_star = selection_gradient(math.exp(x), supply, burn_in)  # noqa: F821
    left = selection_gradient(math.exp(x - 0.02), supply, burn_in)["gradient"]  # noqa: F821
    right = selection_gradient(math.exp(x + 0.02), supply, burn_in)["gradient"]  # noqa: F821
    return {"phi_star": math.exp(x),
            "gradient_lower": float(g_lower),
            "gradient_upper": float(g_upper),
            "gradient_star": float(at_star["gradient"]),
            "curvature": float(at_star["curvature"]),
            "convergence_slope": float((right - left) / 0.04),
            "evaluations": int(evaluations)}

import math

import numpy as np


def _dirichlet_supply(alpha, n_cycles, seed):
    """Per-cycle supplies of two resources, one unit in total, split by Dir(alpha, alpha)."""
    rng = np.random.default_rng(seed)
    draws = rng.gamma(alpha, 1.0, size=(n_cycles, 2))
    draws = np.maximum(draws, 1e-300)
    return draws / draws.sum(axis=1, keepdims=True)


def evolutionarily_stable_allocation(alpha: float, n_cycles: int, burn_in: int, seed: int) -> dict:
    """Reference implementation."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.01:
        raise ValueError("alpha must be finite and at least 0.01")
    for name, v in (("n_cycles", n_cycles), ("burn_in", burn_in), ("seed", seed)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    if n_cycles < 100 or burn_in < 0 or 2 * burn_in >= n_cycles or seed < 0:
        raise ValueError("need n_cycles >= 100, 0 <= burn_in < n_cycles / 2 and seed >= 0")

    supply = _dirichlet_supply(alpha, int(n_cycles), int(seed))
    star = singular_strategy(supply, int(burn_in))  # noqa: F821
    phi_star = star["phi_star"]
    grid = np.concatenate([[0.0], phi_star * np.exp(np.linspace(-4.0, math.log(1.0 / phi_star), 25))])
    grid = np.minimum(grid, 1.0)
    check = invasion_growth_rates(phi_star, np.concatenate([[phi_star], grid]), supply, int(burn_in))  # noqa: F821
    rates = check["invasion_rates"]
    excess = rates[:, 1:] - rates[:, :1]
    j, q = np.unravel_index(int(np.argmax(excess)), excess.shape)
    pref = np.array([[0, 1], [1, 0]])
    both = np.array([True, True])
    growth = proteome_allocation(phi_star, pref[0], both)["growth_rate"]  # noqa: F821
    lag_primary = reallocation_lag(phi_star, pref[0], 0, np.array([False, True]))["lag"]  # noqa: F821
    lag_secondary = reallocation_lag(phi_star, pref[0], 1, np.array([True, False]))["lag"]  # noqa: F821
    below = selection_gradient(phi_star * math.exp(-0.5), supply, int(burn_in))["gradient"]  # noqa: F821
    above = selection_gradient(min(phi_star * math.exp(0.5), math.exp(-0.05)), supply, int(burn_in))["gradient"]  # noqa: F821

    balanced = np.array([0.5, 0.5])
    share = check["mean_share"]
    start = np.array([share, 1.0 - share]) / (100.0 - 1.0)
    tables = [proteome_allocation(phi_star, pref[i], both) for i in range(2)]  # noqa: F821
    first = first_exhaustion(balanced, start, np.array([t["growth_rate"] for t in tables]),  # noqa: F821
                                     np.array([t["uptake"] for t in tables]), 24.0)["time"]
    cycle = grow_one_cycle(np.array([phi_star, phi_star]), pref, start, balanced)  # noqa: F821
    ends = np.minimum(cycle["exhaustion_times"], 24.0)
    return {"phi_star": float(phi_star),
            "invasion_excess": float(excess[j, q]),
            "excess_trait": float(grid[q]),
            "curvature": star["curvature"],
            "convergence_slope": star["convergence_slope"],
            "resident_share": check["mean_share"],
            "replay_error": check["replay_error"],
            "growth_rate": float(growth),
            "lag_primary": float(lag_primary),
            "lag_secondary": float(lag_secondary),
            "gradient_below": float(below),
            "gradient_above": float(above),
            "primary_niche": check["primary_niche"],
            "secondary_niche": check["secondary_niche"],
            "balanced_primary_niche": float(first),
            "balanced_secondary_niche": float(ends.max() - ends.min())}
SCICODE_GOLD_EOF
