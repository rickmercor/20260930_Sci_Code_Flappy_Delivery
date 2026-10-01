"""
Given the remaining stock of every resource, the biomass, growth rate and per-resource uptake rates of every strain, and a positive horizon, return, resolved to near machine precision, the earliest time within the horizon at which a resource with positive stock is exhausted and the index of that resource. If no resource runs out within the horizon, return the horizon itself and the index -1. Resources with zero stock are already gone and are ignored. The growth rate of each strain must equal the sum of its uptake rates.

Resources in a serial-dilution cycle are supplied far above the half-saturation constants of the strains, so every strain that is growing does so exponentially at a constant rate until something changes, and the only events that change a rate are the exhaustion of a resource and the end of a lag. Between two events the biomass of strain i is b_i exp(r_i t) and, with a yield of one, the amount of resource k it has consumed is b_i (u_ik / r_i)(exp(r_i t) - 1), where u_ik is its uptake rate of resource k per unit biomass and r_i = sum_k u_ik its growth rate. A strain in a lag has r_i = 0 and consumes nothing.

The time at which resource k runs out is therefore the root of S_k(t) = R_k - sum_i c_ik (exp(r_i t) - 1) with c_ik = b_i u_ik / r_i, a function that is decreasing and concave in t. Concavity decides how to solve it robustly: a Newton step taken from a point to the right of the root lands again to the right of the root and closer to it, so a Newton iteration started from the right converges monotonically, and a bracket makes it safe. A tight bracket is available in closed form, because the consumption lies between that of all consumers growing at the slowest and at the fastest of their rates. The event that ends the interval is the earliest root over all present resources, provided it falls within the horizon of the interval, which is the time to the next lag end or to the end of the cycle; otherwise no resource runs out in the interval. The root must be resolved to full precision because resources supplied in trace amounts, far below the biomass present, run out within a tiny fraction of an hour and this timing sets which strains pay a lag.

Returns
-------
dict holding the float time of the first exhaustion (or the horizon) and the int resource index (or -1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def first_exhaustion(stock: np.ndarray, biomass: np.ndarray, rates: np.ndarray, uptake: np.ndarray,
                     horizon: float) -> dict:
    """Earliest exhaustion of a resource under exponential growth within a horizon.

    Parameters
    ----------
    stock : np.ndarray
        Remaining resource amounts, shape (n,).
    biomass : np.ndarray
        Strain biomasses, shape (m,).
    rates : np.ndarray
        Strain growth rates per hour, shape (m,).
    uptake : np.ndarray
        Per-biomass uptake rates, shape (m, n).
    horizon : float
        Length of the interval, hours.

    Returns
    -------
    dict
        Under the keys time and resource.

    Raises
    ------
    ValueError
        When an array has the wrong shape, holds a negative or non-finite value, an uptake row
        does not sum to its growth rate, or the horizon is not positive and finite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_first_exhaustion(stock: np.ndarray, biomass: np.ndarray, rates: np.ndarray, uptake: np.ndarray,
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the declared test cases for this step."""
    SETUP = """import math
import numpy as np

def isolated(fn, *args, **kwargs):
    return fn(*(x.copy() if isinstance(x, np.ndarray) else x for x in args),
              **{k: (x.copy() if isinstance(x, np.ndarray) else x) for k, x in kwargs.items()})

def num(v, k=9):
    return float(round(float(v), k))

def run(fn, stock, biomass, rates, uptake, horizon):
    d = fn(np.array(stock, float), np.array(biomass, float), np.array(rates, float), np.array(uptake, float), horizon)
    return (num(d["time"]), int(d["resource"]))
"""

    return [
        {
            "setup": SETUP + """
stock = np.array([math.expm1(1.0)])
biomass = np.array([1.0])
rates = np.array([0.5])
uptake = np.array([[0.5]])
""",
            "call": "run(first_exhaustion, stock, biomass, rates, uptake, 2.0)",
            "gold_call": "run(_oracle_first_exhaustion, stock, biomass, rates, uptake, 2.0)",
        },
        {
            "setup": SETUP + """
def digest(fn):
    up1 = [0.8 * 0.85, 0.4 * 0.15]
    up2 = [0.4 * 0.15, 0.8 * 0.85]
    uptake = [up1, up2]
    rates = [sum(up1), sum(up2)]
    out = run(fn, [0.3, 0.7], [0.004, 0.006], rates, uptake, 24.0)
    out += run(fn, [0.3, 0.7], [0.004, 0.006], [0.0, rates[1]], [[0.0, 0.0], up2], 24.0)
    out += run(fn, [1e-11, 1.0], [0.005, 0.005], rates, uptake, 24.0)
    out += run(fn, [0.0, 0.5], [0.01], [0.4], [[0.0, 0.4]], 24.0)
    return out
""",
            "call": "digest(first_exhaustion)",
            "gold_call": "digest(_oracle_first_exhaustion)",
        },
        {
            "setup": SETUP + """
def closed(fn):
    single = isolated(fn, np.array([0.9]), np.array([0.01]), np.array([0.5]), np.array([[0.5]]), 24.0)
    exact = math.log1p(0.9 / 0.01) / 0.5
    b = np.array([0.003, 0.007, 0.002]); r = np.array([0.8, 0.46, 0.6])
    u = np.array([[0.8, 0.0], [0.34, 0.12], [0.3, 0.3]])
    d = isolated(fn, np.array([0.45, 0.55]), b, r, u, 24.0)
    k, t = d["resource"], d["time"]
    left = 0.45 if k == 0 else 0.55
    residual = left - float(np.sum(b * u[:, k] / r * np.expm1(r * t)))
    short = isolated(fn, np.array([0.45, 0.55]), b, r, u, 1.0)
    return (num(single["time"] - exact) + 0.0, int(single["resource"]), num(t), int(k),
            num(residual, 12) + 0.0, num(short["time"]), int(short["resource"]))
""",
            "call": "closed(first_exhaustion)",
            "gold_call": "closed(_oracle_first_exhaustion)",
        },
        {
            "setup": SETUP + """
def idle(fn):
    out = run(fn, [0.4, 0.6], [0.01, 0.01], [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]], 5.5)
    out += run(fn, [0.4, 0.6], [0.0, 0.0], [0.8, 0.8], [[0.8, 0.0], [0.0, 0.8]], 5.5)
    return out
""",
            "call": "idle(first_exhaustion)",
            "gold_call": "idle(_oracle_first_exhaustion)",
        },
        {
            "setup": SETUP + """
def rejects(fn):
    s = np.array([0.5, 0.5]); b = np.array([0.01]); r = np.array([0.8]); u = np.array([[0.8, 0.0]])
    bad = [
        (np.array([-0.1, 0.5]), b, r, u, 24.0),
        (s, np.array([np.nan]), r, u, 24.0),
        (s, b, np.array([0.8, 0.8]), u, 24.0),
        (s, b, r, np.array([[0.8, 0.0, 0.0]]), 24.0),
        (s, b, np.array([0.5]), u, 24.0),
        (s, b, r, u, 0.0),
        (s, b, r, u, math.inf),
    ]
    count = 0
    for args in bad:
        try:
            isolated(fn, *args)
        except ValueError:
            count += 1
    return count
""",
            "call": "rejects(first_exhaustion)",
            "gold_call": "rejects(_oracle_first_exhaustion)",
        },
    ]
