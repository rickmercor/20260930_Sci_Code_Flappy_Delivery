"""
Step 5 - the first-event branch probabilities of the rejection-free kMC draw.

In one rejection-free kinetic Monte Carlo step over the system, each of the competing event channels (homolysis or hydrolysis at each site) is chosen with probability proportional to its rate. The branch probability w_i that the first event is a target pathway event specifically at site i is therefore:

    w_i = R_target_i / ( sum_j R_target_j + sum_j R_other_j ).

These shares sum to the single-event target-pathway probability P1 = sum(R_target)/R_total; they are the weights with which the possible first-break sites enter the cascade. This step returns the branch-probability vector, one entry per site.

Returns
-------
w : np.ndarray, shape (n,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def branch_probabilities(r_target: "np.ndarray", r_other: "np.ndarray") -> "np.ndarray":
    """First-event branch probability of the target pathway, per site.

    Parameters
    ----------
    r_target : np.ndarray
        Shape (n,), rate of the pathway being asked about at each site, in s^-1
        (non-negative).
    r_other : np.ndarray
        Shape (n,), rate of the competing pathway at each site, in s^-1 (non-negative).

    Returns
    -------
    np.ndarray
        Shape (n,); w_i = r_target_i / (sum(r_target) + sum(r_other)). The entries sum to
        the overall first-event target-pathway probability, not to one.

    Raises
    ------
    ValueError
        If the arrays are not equal-length one-dimensional non-negative finite arrays,
        or if the total rate is zero.
    """
    return w  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_branch_probabilities(r_target: "np.ndarray", r_other: "np.ndarray") -> "np.ndarray":
    """Reference implementation for branch_probabilities."""
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
    return rh / tot

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    invalid = setup + (
        "def run_model(a, b):\n"
        "    try:\n"
        "        branch_probabilities(a, b)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(a, b):\n"
        "    try:\n"
        "        _oracle_branch_probabilities(a, b)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        # Normal: branch shares of a three-site system; they sum to 1/2, not 1.
        {"setup": setup + "rh = np.array([1.0, 2.0, 3.0]); ry = np.array([4.0, 5.0, 6.0])\n",
         "call": "branch_probabilities(rh, ry)",
         "gold_call": "_oracle_branch_probabilities(rh, ry)"},
        # Boundary: hydrolysis-free system, branch shares sum to one.
        {"setup": setup + "rh = np.array([1.0, 2.0]); ry = np.array([0.0, 0.0])\n",
         "call": "branch_probabilities(rh, ry)",
         "gold_call": "_oracle_branch_probabilities(rh, ry)"},
        # Edge: a single-site system.
        {"setup": setup + "rh = np.array([3.0]); ry = np.array([7.0])\n",
         "call": "branch_probabilities(rh, ry)",
         "gold_call": "_oracle_branch_probabilities(rh, ry)"},
        # Invalid: negative rate.
        {"setup": invalid,
         "call": "run_model(np.array([1.0, -2.0]), np.array([1.0, 1.0]))",
         "gold_call": "run_gold(np.array([1.0, -2.0]), np.array([1.0, 1.0]))"},
        # Invalid: mismatched array lengths.
        {"setup": invalid,
         "call": "run_model(np.array([1.0, 2.0]), np.array([1.0]))",
         "gold_call": "run_gold(np.array([1.0, 2.0]), np.array([1.0]))"},
    ]
