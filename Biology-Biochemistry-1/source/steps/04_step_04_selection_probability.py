"""
Step 4 - the rejection-free kMC first-event selection probability.

In one rejection-free kinetic Monte Carlo step each event (site i reacting by homolysis or hydrolysis) is chosen with probability proportional to its rate.

 The probability that the first event belongs to the target pathway is therefore the weighted ratio P(target) = sum_i R_target_i / ( sum_i R_target_i + sum_i R_other_i ). Either pathway can be the target: the rates passed first are the ones whose share is returned. 

This is the ensemble-level selection probability over all reactive sites; it is the probability the rejection-free kMC selection produces, and it is a single deterministic scalar. This step computes it from the per-site rates of the two competing pathways.

Returns
-------
p : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def selection_probability(r_target: "np.ndarray", r_other: "np.ndarray") -> float:
    """Weighted first-event probability of the target pathway.

    Parameters
    ----------
    r_target : np.ndarray
        Shape (n,), rate of the pathway being asked about at each site, in s^-1
        (non-negative).
    r_other : np.ndarray
        Shape (n,), rate of the competing pathway at each site, in s^-1 (non-negative).

    Returns
    -------
    float
        P(target) = sum(r_target) / (sum(r_target) + sum(r_other)).

    Raises
    ------
    ValueError
        If the arrays are not equal-length one-dimensional non-negative finite arrays, or
        if the total rate is zero.
    """
    return p  # placeholder to fill

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_selection_probability(r_target: "np.ndarray", r_other: "np.ndarray") -> float:
    """Reference implementation for selection_probability."""
    import numpy as np

    rh = np.asarray(r_target, dtype=float)
    ry = np.asarray(r_other, dtype=float)
    if rh.ndim != 1 or ry.ndim != 1 or rh.shape != ry.shape or rh.size == 0:
        raise ValueError("rates must be non-empty equal-length one-dimensional arrays")
    if not (np.all(np.isfinite(rh)) and np.all(np.isfinite(ry))):
        raise ValueError("rates must be finite")
    if np.any(rh < 0.0) or np.any(ry < 0.0):
        raise ValueError("rates must be non-negative")
    tot = float(np.sum(rh) + np.sum(ry))
    if tot <= 0.0:
        raise ValueError("total rate must be positive")
    return float(np.sum(rh) / tot)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    invalid = setup + (
        "def run_model(a, b):\n"
        "    try:\n"
        "        selection_probability(a, b)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(a, b):\n"
        "    try:\n"
        "        _oracle_selection_probability(a, b)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        {"setup": setup + "rh = np.array([1.0, 2.0, 3.0]); ry = np.array([4.0, 5.0, 6.0])\n",
         "call": "selection_probability(rh, ry)",
         "gold_call": "_oracle_selection_probability(rh, ry)"},
        {"setup": setup + "rh = np.array([1.0, 1.0]); ry = np.array([1.0, 1.0])\n",
         "call": "selection_probability(rh, ry)",
         "gold_call": "_oracle_selection_probability(rh, ry)"},
        {"setup": invalid,
         "call": "run_model(np.array([0.0, 0.0]), np.array([0.0, 0.0]))",
         "gold_call": "run_gold(np.array([0.0, 0.0]), np.array([0.0, 0.0]))"},
        {"setup": invalid,
         "call": "run_model(np.array([1.0, 2.0]), np.array([1.0]))",
         "gold_call": "run_gold(np.array([1.0, 2.0]), np.array([1.0]))"},
    ]
