"""
Given a strain's secondary allocation phi and preference order, the index of the resource that has just been exhausted and the Boolean presence vector just after the exhaustion, return the reallocation lag in hours and the fraction f0 of the metabolic proteome that was allocated, just before the exhaustion, to the resources still present. Use the allocation rule of the preceding step with the exhausted resource counted as present. Return a lag of zero, with f0 equal to one, when the strain was not consuming the exhausted resource or when no resource remains; return an infinite lag when f0 is zero.

When a resource that a strain is consuming runs out, the enzymes it holds for that resource become useless, and the strain must rebuild its metabolic proteome around the resources still present before it can grow again. The rebuilding is autocatalytic: the enzymes already allocated to the remaining resources supply the energy and precursors from which further enzymes for those resources are made, so the fraction of the metabolic proteome devoted to the remaining resources, call it f, grows as df/dt = f / tau0 until it reaches one. Growth is suspended while this reallocation is under way, so its duration is a lag. Integrating from the fraction f0 held at the moment of exhaustion to one gives the lag tau = tau0 ln(1 / f0), with tau0 = 0.3 hours.

The fraction f0 is the sum of the allocations, just before the exhaustion, to the resources that remain. Two cases arise, and both carry a lag. If the exhausted resource was the primary one, f0 is the secondary share (n - 1) phi / n, which is small for a hierarchical utiliser and zero for a specialist, whose lag is infinite: a specialist never switches. If the exhausted resource was a secondary one, f0 is one minus that resource's share phi / n, and the lag, tau0 ln(n / (n - phi)), is short but not zero, and it grows with phi. The trait therefore buys a shorter lag when the primary resource runs out at the price of a longer one when a secondary resource runs out, on top of the lower growth rate while the primary resource lasts. A strain that was not consuming the exhausted resource suffers no lag, and when no resource remains there is nothing to switch to and no lag is defined; growth simply stops.

Returns
-------
dict holding the float lag, in hours and possibly infinite, and the float retained_fraction, the fraction f0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reallocation_lag(phi: float, preference: np.ndarray, depleted: int, present_after: np.ndarray,
                     lag_scale: float = 0.3) -> dict:
    """Lag after a consumed resource is exhausted, from autocatalytic proteome reallocation.

    Parameters
    ----------
    phi : float
        Secondary allocation in [0, 1].
    preference : np.ndarray
        Permutation of the resource indices, most preferred first.
    depleted : int
        Index of the resource just exhausted.
    present_after : np.ndarray
        Boolean presence of each resource just after the exhaustion.
    lag_scale : float
        Autocatalytic timescale tau0, hours.

    Returns
    -------
    dict
        Under the keys lag and retained_fraction.

    Raises
    ------
    ValueError
        When phi lies outside [0, 1], the preference is not an integer-typed permutation, the
        presence vector is not a Boolean array, the exhausted index is not an integer, is out of
        range or is still marked present, or lag_scale is not positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_reallocation_lag(phi: float, preference: np.ndarray, depleted: int, present_after: np.ndarray,
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
    alloc = _oracle_proteome_allocation(phi, preference, before)["allocation"]  # noqa: F821
    if alloc[depleted] <= 0.0 or not after.any():
        return {"lag": 0.0, "retained_fraction": 1.0}
    retained = float(alloc[after].sum())
    if retained <= 0.0:
        return {"lag": math.inf, "retained_fraction": 0.0}
    return {"lag": lag_scale * math.log(1.0 / retained), "retained_fraction": retained}

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

def num(v, k=12):
    v = float(v)
    return v if math.isinf(v) else float(round(v, k))

def pair(d):
    return (num(d["lag"]), num(d["retained_fraction"]))
"""

    return [
        {
            "setup": SETUP + """
def digest(fn):
    out = ()
    for pref in (np.array([0, 1]), np.array([1, 0])):
        for phi in (1e-4, 0.05, 0.3, 1.0):
            for dep in (0, 1):
                after = np.array([dep != 0, dep != 1])
                out += pair(isolated(fn, phi, pref, dep, after))
    return out
""",
            "call": "digest(reallocation_lag)",
            "gold_call": "digest(_oracle_reallocation_lag)",
        },
        {
            "setup": SETUP + """
def limits(fn):
    pref = np.array([0, 1])
    spec_primary = isolated(fn, 0.0, pref, 0, np.array([False, True]))
    spec_secondary = isolated(fn, 0.0, pref, 1, np.array([True, False]))
    nothing = isolated(fn, 0.4, pref, 1, np.array([False, False]))
    scaled = isolated(fn, 0.2, pref, 0, np.array([False, True]), 0.9)["lag"] / isolated(fn, 0.2, pref, 0, np.array([False, True]))["lag"]
    phi = 0.17
    closed_p = isolated(fn, phi, pref, 0, np.array([False, True]))["lag"] - 0.3 * math.log(2.0 / phi)
    closed_s = isolated(fn, phi, pref, 1, np.array([True, False]))["lag"] - 0.3 * math.log(2.0 / (2.0 - phi))
    return pair(spec_primary) + pair(spec_secondary) + pair(nothing) + (num(scaled), num(closed_p) + 0.0, num(closed_s) + 0.0)
""",
            "call": "limits(reallocation_lag)",
            "gold_call": "limits(_oracle_reallocation_lag)",
        },
        {
            "setup": SETUP + """
def three(fn):
    pref = np.array([1, 2, 0])
    out = pair(isolated(fn, 0.6, pref, 1, np.array([True, False, True])))
    out += pair(isolated(fn, 0.6, pref, 0, np.array([False, True, True])))
    out += pair(isolated(fn, 0.6, pref, 2, np.array([False, True, False])))
    out += pair(isolated(fn, 0.6, pref, 0, np.array([False, False, True])))
    return out
""",
            "call": "three(reallocation_lag)",
            "gold_call": "three(_oracle_reallocation_lag)",
        },
        {
            "setup": SETUP + """
def rejects(fn):
    pref = np.array([0, 1])
    bad = [
        (0.3, pref, 0, np.array([True, True]), 0.3),
        (0.3, pref, 2, np.array([False, True]), 0.3),
        (0.3, pref, -1, np.array([False, True]), 0.3),
        (0.3, pref, 0.0, np.array([False, True]), 0.3),
        (1.3, pref, 0, np.array([False, True]), 0.3),
        (0.3, np.array([1, 1]), 0, np.array([False, True]), 0.3),
        (0.3, pref, 0, np.array([False, True]), 0.0),
        (0.3, pref, 0, np.array([0, 1]), 0.3),
    ]
    count = 0
    for args in bad:
        try:
            isolated(fn, *args)
        except ValueError:
            count += 1
    return count
""",
            "call": "rejects(reallocation_lag)",
            "gold_call": "rejects(_oracle_reallocation_lag)",
        },
    ]
