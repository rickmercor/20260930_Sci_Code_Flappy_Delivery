"""
Given the secondary allocations, preference orders and starting biomasses of a community of strains and the amounts of the two resources supplied, integrate one cycle and return the final biomass of every strain, the time at which each resource was exhausted (infinity if it was not exhausted within the cycle), the index of the resource exhausted first (-1 if none) and the lag each strain incurred during the cycle (zero if none, infinity for a strain that can never switch). A resource is exhausted when the root finder of the preceding step returns it or, at the end of any interval, when its remaining stock has fallen to one part in 10^12 of the amount supplied or below; resources exhausted at the same instant are exhausted together, and when none remains no lag is incurred. The first exhausted index is the one returned by the root finder when there is one.

One boom phase of the serial-dilution cycle, with two substitutable resources supplied at its start, is a sequence of intervals of exponential growth separated by events. At the start of the cycle both resources are present, however small the amount supplied of either, and no strain is in a lag, so every strain grows at the rate fixed by its allocation over the two resources. The first event is the exhaustion of one resource. Every strain that was consuming it, which with two resources is every strain whose allocation to it is positive, then enters a reallocation lag: a long one if it has lost its primary resource and a short one if it has lost its secondary resource. Strains in a lag neither grow nor consume. A strain whose lag ends while the other resource is still present resumes growth on that resource alone, with its whole metabolic proteome devoted to it. The second exhaustion ends all growth for the cycle, as does the end of the cycle itself, 24 hours after the supply.

The dynamics are integrated exactly from event to event, because between events every biomass is a pure exponential. The events are the resource exhaustions and the lag ends; the earliest exhaustion inside an interval is found by the root finder of the preceding step, with the horizon of the interval set to the time remaining before the next lag end or the end of the cycle. Order matters at an exhaustion: the lag a strain incurs is computed from the allocation it held while the exhausted resource was still present. Two resources can also run out at the same instant. This is not a coincidence: under a constant supply a pair of strains with opposite preferences converges within a few dozen cycles to the composition at which both resources run out together, and mirror-image strains under a balanced supply do so from the start, and rounding then leaves one of them with a residue of a few units in the last place; that resource must be treated as exhausted too, or the strains would go on growing on nothing until the end of the cycle.

Resources supplied in trace amounts matter. When one resource is supplied at, say, one part in ten to the twelve, it runs out within microseconds, and every strain that allocated to it pays its lag at the very start of the cycle. Under strongly fluctuating supply this happens in a large share of cycles, and it is where the secondary allocation is both rewarded and penalised.

Returns
-------
dict holding the np.ndarray biomass of shape (m,), the np.ndarray exhaustion_times of shape (2,), the int first_exhausted and the np.ndarray lags of shape (m,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def grow_one_cycle(phi: np.ndarray, preference: np.ndarray, biomass: np.ndarray, supply: np.ndarray,
                   lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Event-driven integration of one growth cycle on two substitutable resources.

    Parameters
    ----------
    phi : np.ndarray
        Secondary allocations, shape (m,).
    preference : np.ndarray
        Preference orders, shape (m, 2).
    biomass : np.ndarray
        Starting biomasses, shape (m,).
    supply : np.ndarray
        Amounts of the two resources supplied, shape (2,).
    lag_scale : float
        Autocatalytic timescale tau0, hours.
    cycle_length : float
        Cycle duration, hours.

    Returns
    -------
    dict
        Under the keys biomass, exhaustion_times, first_exhausted and lags.

    Raises
    ------
    ValueError
        When the arrays disagree in shape, a trait lies outside [0, 1], a preference row is not
        an integer permutation of (0, 1), a biomass or a supply is not positive and finite, or a
        time parameter is not positive and finite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_grow_one_cycle(phi: np.ndarray, preference: np.ndarray, biomass: np.ndarray, supply: np.ndarray,
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
                d = _oracle_proteome_allocation(phis[i], prefs[i], present)  # noqa: F821
                uptake[i] = d["uptake"]
                rates[i] = d["growth_rate"]
        waiting = lag_left[lag_left > 0.0]
        horizon = min(cycle_length - t, float(waiting.min()) if waiting.size else math.inf)
        if math.isinf(horizon):
            break
        event = _oracle_first_exhaustion(stock * present, b, rates, uptake, horizon)  # noqa: F821
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
                        lag = _oracle_reallocation_lag(phis[i], prefs[i], j, present.copy(), lag_scale)["lag"]  # noqa: F821
                        lag_left[i] = lag
                        lags[i] += lag
    return {"biomass": b, "exhaustion_times": times, "first_exhausted": int(first), "lags": lags}

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

def num(v, k=10):
    v = float(v)
    return v if math.isinf(v) else float(round(v, k))

def pack(d):
    return (tuple(num(x) for x in d["biomass"]) + tuple(num(x) for x in d["exhaustion_times"])
            + (int(d["first_exhausted"]),) + tuple(num(x) for x in d["lags"]))
"""

    return [
        {
            "setup": SETUP + """
def digest(fn):
    phi = np.array([0.08, 0.3, 1.0])
    pref = np.array([[0, 1], [1, 0], [0, 1]])
    b0 = np.array([0.004, 0.003, 0.003])
    a = isolated(fn, phi, pref, b0, np.array([0.8, 0.2]))
    c = isolated(fn, phi, pref, b0, np.array([1.0 - 1e-12, 1e-12]))
    return pack(a) + pack(c)
""",
            "call": "digest(grow_one_cycle)",
            "gold_call": "digest(_oracle_grow_one_cycle)",
        },
        {
            "setup": SETUP + """
def identities(fn):
    phi = np.array([0.2, 0.2]); pref = np.array([[0, 1], [1, 0]]); b0 = np.array([0.005, 0.005])
    s = np.array([0.35, 0.65])
    d = isolated(fn, phi, pref, b0, s)
    gain = float(d["biomass"].sum() - b0.sum() - s.sum())
    co = isolated(fn, np.array([1.0]), np.array([[0, 1]]), np.array([0.01]), np.array([0.5, 0.5]))
    t1 = float(np.min(co["exhaustion_times"]))
    closed = math.log1p(0.5 / (0.01 * 0.5 * 0.8 / 0.6)) / 0.6
    spec = isolated(fn, np.array([0.0]), np.array([[0, 1]]), np.array([0.01]), np.array([0.3, 0.7]))
    tie = isolated(fn, np.array([0.05, 0.05]), pref, np.array([0.005, 0.005]), np.array([0.5, 0.5]))
    tie_gain = float(tie["biomass"].sum() - 0.01 - 1.0)
    return (num(tie_gain, 9) + 0.0, num(abs(tie["exhaustion_times"][0] - tie["exhaustion_times"][1]), 9) + 0.0,
            num(gain, 12) + 0.0, num(t1 - min(closed, math.log1p(0.5 / (0.01 * 0.5 * 0.4 / 0.6)) / 0.6), 12) + 0.0,
            num(spec["biomass"][0] - 0.31, 12) + 0.0, num(spec["lags"][0]), num(spec["exhaustion_times"][1]))
""",
            "call": "identities(grow_one_cycle)",
            "gold_call": "identities(_oracle_grow_one_cycle)",
        },
        {
            "setup": SETUP + """
def boundary(fn):
    phi = np.array([0.02, 0.5]); pref = np.array([[0, 1], [1, 0]]); b0 = np.array([0.005, 0.005])
    short = isolated(fn, phi, pref, b0, np.array([0.5, 0.5]), 0.3, 2.0)
    late = isolated(fn, phi, pref, b0, np.array([0.05, 0.95]))
    return pack(short) + pack(late)
""",
            "call": "boundary(grow_one_cycle)",
            "gold_call": "boundary(_oracle_grow_one_cycle)",
        },
        {
            "setup": SETUP + """
def rejects(fn):
    phi = np.array([0.2]); pref = np.array([[0, 1]]); b0 = np.array([0.01]); s = np.array([0.5, 0.5])
    bad = [
        (np.array([1.5]), pref, b0, s),
        (phi, np.array([[0, 0]]), b0, s),
        (phi, pref, np.array([0.0]), s),
        (phi, pref, b0, np.array([0.0, 1.0])),
        (phi, pref, b0, np.array([0.3, 0.3, 0.4])),
        (phi, np.array([0, 1]), b0, s),
        (np.array([0.2, 0.3]), pref, b0, s),
    ]
    count = 0
    for args in bad:
        try:
            isolated(fn, *args)
        except ValueError:
            count += 1
    return count
""",
            "call": "rejects(grow_one_cycle)",
            "gold_call": "rejects(_oracle_grow_one_cycle)",
        },
    ]
