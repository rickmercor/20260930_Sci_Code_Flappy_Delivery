"""
Given the resident trait phi, the supply sequence, the burn-in and a step h in ln phi, compute the invasion growth rates of mutants with traits phi exp(-h), phi and phi exp(h), for both preference orders, against the symmetric resident pair, and return the selection gradient (the central difference of the rates divided by 2h) and the curvature (the second central difference divided by h squared), each averaged over the two preference orders, together with the three-by-two table of rates. The largest mutant trait must not exceed one.

The direction in which the secondary allocation evolves is set by the selection gradient, the rate at which the invasion growth rate of a rare mutant changes with the mutant's trait, evaluated at the resident trait. Mutations act multiplicatively on phi, so the natural coordinate is ln phi and the gradient is D(phi) = d lambda(phi'; phi) / d ln phi' at phi' = phi. Where D is positive, mutants with slightly larger phi invade and phi rises; where it is negative, phi falls. A singular strategy, where D vanishes, is a candidate endpoint of evolution. It is evolutionarily stable, uninvadable by nearby mutants, when the invasion growth rate is at a maximum there, that is when the second derivative d2 lambda / d (ln phi')2 at phi' = phi is negative.

Both derivatives are estimated by central differences in ln phi' with a common resident trajectory and a common supply sequence for all mutants. Holding the environment fixed across the mutants being compared makes the difference of two invasion rates far more precise than either rate alone, because the cycle-to-cycle fluctuations, which are large when the supply ratio fluctuates strongly, cancel between them. The two mutant preference orders give two estimates of each derivative, which coincide in expectation for symmetric supply; their average is reported.

Returns
-------
dict holding the float gradient, the float curvature and the np.ndarray rates of shape (2, 3), row j for mutants preferring resource j, columns for phi exp(-h), phi and phi exp(h).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def selection_gradient(phi: float, supply: np.ndarray, burn_in: int, step: float = 0.05,
                       dilution: float = 100.0, lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Selection gradient and curvature of the invasion growth rate in ln phi.

    Parameters
    ----------
    phi : float
        Resident secondary allocation.
    supply : np.ndarray
        Per-cycle supplies of the two resources, shape (n_cycles, 2).
    burn_in : int
        Cycles discarded before averaging.
    step : float
        Central-difference step in ln phi.
    dilution : float
        Dilution factor.
    lag_scale : float
        Autocatalytic timescale tau0, hours.
    cycle_length : float
        Cycle duration, hours.

    Returns
    -------
    dict
        Under the keys gradient, curvature and rates.

    Raises
    ------
    ValueError
        When the step lies outside (0, 0.5], phi lies outside (0, 1] or phi exp(step) exceeds
        one, or the supply and burn-in are invalid.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_selection_gradient(phi: float, supply: np.ndarray, burn_in: int, step: float = 0.05,
                               dilution: float = 100.0, lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Reference implementation."""
    phi = float(phi)
    step = float(step)
    if not math.isfinite(step) or step <= 0.0 or step > 0.5:
        raise ValueError("step must lie in (0, 0.5]")
    if not math.isfinite(phi) or phi <= 0.0 or phi * math.exp(step) > 1.0 + 1e-12:
        raise ValueError("phi must be positive with phi exp(step) at most one")
    mutants = np.array([phi * math.exp(-step), phi, min(phi * math.exp(step), 1.0)])
    rates = _oracle_invasion_growth_rates(phi, mutants, supply, burn_in, dilution, lag_scale, cycle_length)["invasion_rates"]  # noqa: F821
    gradient = float(np.mean((rates[:, 2] - rates[:, 0]) / (2.0 * step)))
    curvature = float(np.mean((rates[:, 2] - 2.0 * rates[:, 1] + rates[:, 0]) / step ** 2))
    return {"gradient": gradient, "curvature": curvature, "rates": rates}

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

def supply_sequence(n, alpha, seed):
    rng = np.random.default_rng(seed)
    g = rng.gamma(alpha, 1.0, size=(n, 2))
    g = np.maximum(g, 1e-300)
    return g / g.sum(axis=1, keepdims=True)

def pack(d):
    return (num(d["gradient"]), num(d["curvature"])) + tuple(num(x) for x in d["rates"].ravel())
"""

    return [
        {
            "setup": SETUP + """
def digest(fn):
    s = supply_sequence(600, 0.03, 21)
    return pack(isolated(fn, 0.08, s, 100)) + pack(isolated(fn, 0.7, s, 100))
""",
            "call": "digest(selection_gradient)",
            "gold_call": "digest(_oracle_selection_gradient)",
        },
        {
            "setup": SETUP + """
def balanced(fn):
    s = np.full((60, 2), 0.5)
    lo = isolated(fn, 0.01, s, 20)
    hi = isolated(fn, 0.4, s, 20)
    t = supply_sequence(300, 0.3, 4)
    a = isolated(fn, 0.2, t, 50)
    b = isolated(fn, 0.2, t[:, ::-1].copy(), 50)
    return (num(lo["gradient"]), num(hi["gradient"]), num(a["gradient"] - b["gradient"], 12) + 0.0,
            num(a["curvature"] - b["curvature"], 10) + 0.0)
""",
            "call": "balanced(selection_gradient)",
            "gold_call": "balanced(_oracle_selection_gradient)",
        },
        {
            "setup": SETUP + """
def boundary(fn):
    s = supply_sequence(400, 0.1, 9)
    edge = isolated(fn, math.exp(-0.05) * (1.0 - 1e-9), s, 50)
    half = isolated(fn, 0.2, s, 50, 0.025)
    full = isolated(fn, 0.2, s, 50, 0.05)
    return pack(edge) + (num(half["gradient"]), num(full["gradient"]))
""",
            "call": "boundary(selection_gradient)",
            "gold_call": "boundary(_oracle_selection_gradient)",
        },
        {
            "setup": SETUP + """
def rejects(fn):
    s = np.full((10, 2), 0.5)
    bad = [
        (0.0, s, 0, 0.05),
        (0.99, s, 0, 0.05),
        (0.3, s, 0, 0.0),
        (0.3, s, 0, 0.8),
        (0.3, s, 10, 0.05),
        (0.3, np.full((10, 2), -0.5), 0, 0.05),
    ]
    count = 0
    for args in bad:
        try:
            isolated(fn, *args)
        except ValueError:
            count += 1
    return count
""",
            "call": "rejects(selection_gradient)",
            "gold_call": "rejects(_oracle_selection_gradient)",
        },
    ]
