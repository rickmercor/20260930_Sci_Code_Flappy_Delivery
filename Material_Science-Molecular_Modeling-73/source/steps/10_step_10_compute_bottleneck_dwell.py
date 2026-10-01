"""
Reduce the accumulated-time profile to the dwell time of the interface that limits the transition.

The accumulated-time profile answers a mechanistic question that no single sampled ensemble can: of the elapsed time of one transition, how much is spent in the neighbourhood of each interface. Its largest entry almost always belongs to the reactant boundary, because the trajectory waits there through many failed attempts, and that entry says nothing about the transition itself - it restates that the process is rare. The informative quantity is the largest entry among the remaining interfaces, which locates the region where the trajectory that has already left the reactant state actually loses its time. That region need not be the one with the smallest local progression probability, because a segment type can be visited often yet be short, or be visited rarely yet be long; only the product of occupation and duration decides.

Reporting the result as a share of the total profile and restoring the absolute scale from the independently computed rate keeps the two halves of the calculation honest. The share comes from the occupation decomposition, while the timescale comes from the flux multiplied by the crossing probability, which is the standard estimator and does not use the decomposition at all. Because the profile sums to the mean duration of one passage, and that duration is the reciprocal of the rate, multiplying the share by the reciprocal of the rate returns the dwell time in physical units and simultaneously checks that the two routes agree.


Formulas:

    share = max(dwell[1:]) / sum(dwell)

    bottleneck_dwell = share / rate_constant

Returns
-------
float: dwell time of the rate-limiting interface above the reactant boundary,  in the time unit implied by rate_constant, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_bottleneck_dwell(dwell_times, rate_constant: float) -> float:
    """Return the dwell time of the slowest interface above the reactant boundary.

    Parameters
    ----------
    dwell_times : array_like
        Float array of shape (N,) holding the mean time one passage accumulates
        in segments centred on lambda_0 ... lambda_(N-1), in any consistent
        unit. Entries must be non-negative and at least two are required.
    rate_constant : float
        Rate constant of the transition into the product state, in the
        reciprocal of the time unit in which the answer is wanted. Must be
        positive.

    Returns
    -------
    bottleneck_dwell : float
        Mean time one passage accumulates in segments centred on the interface
        above the reactant boundary that carries the largest such time,
        expressed in the time unit implied by rate_constant, as a native Python
        float.

    Raises
    ------
    ValueError
        If no positive dwell time is accumulated above the reactant boundary
        or if rate_constant is not positive.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_bottleneck_dwell(dwell_times, rate_constant: float) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    profile = np.asarray(dwell_times, dtype=float)
    if profile.ndim != 1:
        raise ValueError("dwell_times must be one dimensional")
    if profile.size < 2:
        raise ValueError("dwell_times must cover at least one interface above the boundary")
    if not np.all(np.isfinite(profile)):
        raise ValueError("dwell_times must be finite")
    if np.any(profile < 0.0):
        raise ValueError("dwell_times must be non-negative")
    if isinstance(rate_constant, bool) or not isinstance(
            rate_constant, (int, float, np.integer, np.floating)):
        raise ValueError("rate_constant must be a real number")
    if not np.isfinite(rate_constant) or float(rate_constant) <= 0.0:
        raise ValueError("rate_constant must be a positive finite number")

    total = float(profile.sum())
    if total <= 0.0:
        raise ValueError("a passage must accumulate a positive time")

    # The reactant boundary carries the waiting time of the rare event, which
    # is not part of the transit; the bottleneck is sought above it.
    above = profile[1:]
    largest = float(above.max())
    if largest <= 0.0:
        raise ValueError("no time is accumulated above the reactant boundary")

    share = largest / total

    # The profile sums to the mean duration of one passage, so dividing the
    # share by the rate restores the absolute timescale.
    return float(share / float(rate_constant))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the measured profile in phase points with the measured rate ---
        {
            "setup": """import numpy as np
profile = np.array([5296.271017, 320.792741, 912.711622, 470.985183,
                    345.400491, 248.714947, 143.465517])
rate_constant = 0.006461332817234334
""",
            "call": "compute_bottleneck_dwell(profile, rate_constant)",
            "gold_call": "_oracle_compute_bottleneck_dwell(profile, rate_constant)",
        },
        # --- Valid: the same shape with the bottleneck moved outwards ---
        {
            "setup": """import numpy as np
profile = np.array([5296.271017, 320.792741, 412.711622, 470.985183,
                    845.400491, 248.714947, 143.465517])
rate_constant = 0.006461332817234334
""",
            "call": "compute_bottleneck_dwell(profile, rate_constant)",
            "gold_call": "_oracle_compute_bottleneck_dwell(profile, rate_constant)",
        },
        # --- Boundary: the reactant entry dominates but must be excluded from
        #     the search while still counting towards the total ---
        {
            "setup": """import numpy as np
profile = np.array([99000.0, 12.0, 7.0])
rate_constant = 2.5
""",
            "call": "compute_bottleneck_dwell(profile, rate_constant)",
            "gold_call": "_oracle_compute_bottleneck_dwell(profile, rate_constant)",
        },
        # --- Edge: a flat profile, where the share is set by the interface count ---
        {
            "setup": """import numpy as np
profile = np.full(9, 41.5)
rate_constant = 0.125
""",
            "call": "compute_bottleneck_dwell(profile, rate_constant)",
            "gold_call": "_oracle_compute_bottleneck_dwell(profile, rate_constant)",
        },
        # --- Invalid: no time accumulated above the reactant boundary ---
        {
            "setup": """import numpy as np
profile = np.array([5296.271017, 0.0, 0.0, 0.0])
def run_model():
    try:
        compute_bottleneck_dwell(profile, 0.00646)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_bottleneck_dwell(profile, 0.00646)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive rate constant ---
        {
            "setup": """import numpy as np
profile = np.array([5296.271017, 320.792741, 912.711622])
def run_model():
    try:
        compute_bottleneck_dwell(profile, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_bottleneck_dwell(profile, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
