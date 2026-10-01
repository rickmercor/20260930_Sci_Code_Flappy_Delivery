"""
Given the secondary allocation phi of a strain, its preference order over the resources and a Boolean vector marking which resources are currently present, return the metabolic proteome allocation over all resources, the potential growth rates, the per-resource uptake rates per unit biomass (allocation times potential rate) and the total growth rate (their sum). When no resource is present the allocation, the uptake and the growth rate are all zero.

A microbe growing on several substitutable resources divides the metabolic sector of its proteome among them, and its growth rate is set by how much enzyme it holds for each resource that is actually available. In the proteome-allocation picture the growth rate on a mixture is the allocation-weighted sum of the potential growth rates on the individual resources, where the potential rate on a resource is the rate the strain would reach if its whole metabolic proteome were devoted to that resource. With a biomass yield of one on every resource, the rate at which a unit of biomass consumes a resource equals the growth it derives from that resource.

The strain carries one heritable trait, the secondary allocation phi in [0, 1]. It has a fixed preference order over the resources; among those currently present, the highest ranked one is its primary resource and the others are secondary. Each present resource receives the fraction phi / n of the metabolic proteome, where n is the number of resources present, and the primary resource receives in addition the fraction 1 - phi, so that the primary share is 1 - (n - 1) phi / n and the shares of the present resources sum to one. An absent resource receives nothing, and when only one resource is present it receives the whole proteome. The limits are the strategies seen in nature: phi = 0 is a specialist that enzymatically ignores every resource but its primary one, a small positive phi is a hierarchical (diauxic) utiliser, and phi = 1 splits the proteome equally among all present resources, the idealised co-utiliser.

The potential growth rate of a strain on its most preferred resource is g0 = 0.8 per hour and on any other resource the fraction 0.5 of that, 0.4 per hour, so the most preferred resource is also the one the strain grows on fastest. Raising phi therefore lowers the growth rate while the most preferred resource is present, which is the cost side of the trade-off whose benefit, a shorter lag after a resource runs out, is the subject of the next step.

Returns
-------
dict holding the np.ndarray allocation of shape (n,), the np.ndarray potential_rates of shape (n,), the np.ndarray uptake of shape (n,) and the float growth_rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def proteome_allocation(phi: float, preference: np.ndarray, present: np.ndarray,
                        max_rate: float = 0.8, other_factor: float = 0.5) -> dict:
    """Metabolic proteome allocation, uptake and growth rate of a strain.

    Parameters
    ----------
    phi : float
        Secondary allocation in [0, 1].
    preference : np.ndarray
        Permutation of the resource indices, most preferred first, shape (n,).
    present : np.ndarray
        Boolean presence of each resource, shape (n,).
    max_rate : float
        Potential growth rate on the most preferred resource, per hour.
    other_factor : float
        Potential rate on any other resource relative to max_rate.

    Returns
    -------
    dict
        Under the keys allocation, potential_rates, uptake and growth_rate.

    Raises
    ------
    ValueError
        When phi lies outside [0, 1], the preference is not an integer-typed permutation of at least
        two indices, the presence vector is not a Boolean array matching it, or a rate parameter is
        invalid.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_proteome_allocation(phi: float, preference: np.ndarray, present: np.ndarray,
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the declared test cases for this step."""
    SETUP = """import numpy as np

def isolated(fn, *args, **kwargs):
    return fn(*(x.copy() if isinstance(x, np.ndarray) else x for x in args),
              **{k: (x.copy() if isinstance(x, np.ndarray) else x) for k, x in kwargs.items()})

def num(v, k=12):
    return float(round(float(v), k))

def pack(d):
    return (tuple(num(x) for x in d["allocation"]) + tuple(num(x) for x in d["potential_rates"])
            + tuple(num(x) for x in d["uptake"]) + (num(d["growth_rate"]),))
"""

    return [
        {
            "setup": SETUP + """
def digest(fn):
    out = ()
    for pref in (np.array([0, 1]), np.array([1, 0])):
        for pres in ([True, True], [True, False], [False, True]):
            out += pack(isolated(fn, 0.3, pref, np.array(pres)))
    return out
""",
            "call": "digest(proteome_allocation)",
            "gold_call": "digest(_oracle_proteome_allocation)",
        },
        {
            "setup": SETUP + """
def limits(fn):
    pref = np.array([0, 1])
    both = np.array([True, True])
    spec = isolated(fn, 0.0, pref, both)
    co = isolated(fn, 1.0, pref, both)
    slope = (isolated(fn, 0.6, pref, both)["growth_rate"] - isolated(fn, 0.2, pref, both)["growth_rate"]) / 0.4
    sums = [num(isolated(fn, p, pref, both)["allocation"].sum()) for p in (0.0, 0.05, 0.5, 1.0)]
    none = isolated(fn, 0.4, pref, np.array([False, False]))
    return pack(spec) + pack(co) + (num(slope),) + tuple(sums) + pack(none)
""",
            "call": "limits(proteome_allocation)",
            "gold_call": "limits(_oracle_proteome_allocation)",
        },
        {
            "setup": SETUP + """
def three(fn):
    pref = np.array([2, 0, 1])
    out = pack(isolated(fn, 0.45, pref, np.array([True, True, True])))
    out += pack(isolated(fn, 0.45, pref, np.array([True, True, False])))
    out += pack(isolated(fn, 0.45, pref, np.array([False, True, False])))
    out += pack(isolated(fn, 0.45, pref, np.array([True, False, True]), 1.2, 0.25))
    return out
""",
            "call": "three(proteome_allocation)",
            "gold_call": "three(_oracle_proteome_allocation)",
        },
        {
            "setup": SETUP + """
def rejects(fn):
    good_p = np.array([0, 1])
    good_r = np.array([True, True])
    bad = [
        (-0.1, good_p, good_r, 0.8, 0.5),
        (1.2, good_p, good_r, 0.8, 0.5),
        (float("nan"), good_p, good_r, 0.8, 0.5),
        (0.3, np.array([0, 0]), good_r, 0.8, 0.5),
        (0.3, np.array([0]), np.array([True]), 0.8, 0.5),
        (0.3, np.array([0.0, 1.0]), good_r, 0.8, 0.5),
        (0.3, good_p, np.array([1, 1]), 0.8, 0.5),
        (0.3, good_p, np.array([True, True, False]), 0.8, 0.5),
        (0.3, good_p, good_r, -0.8, 0.5),
        (0.3, good_p, good_r, 0.8, 1.5),
    ]
    count = 0
    for args in bad:
        try:
            isolated(fn, *args)
        except ValueError:
            count += 1
    return count
""",
            "call": "rejects(proteome_allocation)",
            "gold_call": "rejects(_oracle_proteome_allocation)",
        },
    ]
