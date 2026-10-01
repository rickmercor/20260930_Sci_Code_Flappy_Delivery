"""
Build a self-similar bridge between two endpoints by recursively inserting states into the allowed region.

An admissible equation of state connecting two endpoints is constructed by repeated subdivision rather than by writing down a functional form. A single intermediate state is placed inside the allowed region; because it is admissible, at least one physical equation of state passes through it, and the original interval splits into two shorter ones whose own allowed regions are computed from the new state and the corresponding original endpoint.




Applying the same rule to each subinterval, and again to their subintervals, produces structure on every density scale. After a fixed number of levels the result is a sequence of states ordered in density, running from the low endpoint to the high one, whose piecewise-linear interpolation is a bridge carrying fluctuations down to the finest scale resolved.




The recursion is self-similar because the rule applied to a subinterval is identical to the rule applied to the whole: only the endpoints defining the allowed region change. Prescribing the quantiles rather than drawing them at random fixes one representative member of this family, so the bridge is reproducible while retaining the multi-scale structure.

Returns
-------
np.ndarray, shape (2**depth + 1, 3), rows [mu, n, p] ordered by increasing n
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def self_similar_refine(beta_low: "np.ndarray", beta_high: "np.ndarray", depth: int,
                        quantiles: "np.ndarray") -> "np.ndarray":
    '''Construct the ordered sequence of states forming a self-similar bridge.

    Parameters
    ----------
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.
    depth : int
        Integer number of refinement levels, at least 0 and at most 12. A depth
        of zero inserts no intermediate state.
    quantiles : np.ndarray
        Finite shape (3,) with every entry in [0, 1], applied unchanged at every
        insertion.

    Returns
    -------
    states : np.ndarray
        Finite shape (K, 3) with K = 2**depth + 1, the states (mu, n, p) ordered
        by increasing density, beginning with beta_low and ending with
        beta_high. At each level the interval is split by inserting the state at
        the prescribed quantiles of its own allowed region, and the two
        resulting subintervals are refined in the same way to the remaining
        depth, the lower one first.

    Raises
    ------
    ValueError
        If depth is not an integer in [0, 12], if a preceding step rejects its
        inputs, or if the constructed sequence is not finite or not strictly
        increasing in density.
    '''
    return states

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _refine_interval(beta_low, beta_high, depth, quantiles):
    """Return the interior states of one interval, ordered by increasing density."""
    if depth == 0:
        return []
    middle = _oracle_quantile_point_in_volume(quantiles, beta_low, beta_high)
    left = _refine_interval(beta_low, middle, depth - 1, quantiles)
    right = _refine_interval(middle, beta_high, depth - 1, quantiles)
    return left + [middle] + right


def _oracle_self_similar_refine(beta_low: "np.ndarray", beta_high: "np.ndarray", depth: int,
                                quantiles: "np.ndarray") -> "np.ndarray":
    low, high, _ = _validated_endpoints(beta_low, beta_high)
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)):
        raise ValueError("depth must be an integer")
    levels = int(depth)
    if not 0 <= levels <= 12:
        raise ValueError("depth must lie in [0, 12]")
    interior = _refine_interval(low, high, levels, quantiles)
    states = np.array([low] + [np.asarray(b, dtype=float) for b in interior] + [high],
                      dtype=float)
    if not np.all(np.isfinite(states)):
        raise ValueError("the constructed bridge must be finite")
    if not np.all(np.diff(states[:, 1]) > 0.0):
        raise ValueError("the bridge must be strictly increasing in density")
    return states

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    anchors = ("import numpy as np\n"
               "low = np.array([1.00, 0.32, 0.018])\n"
               "high = np.array([2.60, 4.80, 3.500])\n"
               "q = np.array([0.5, 0.5, 0.5])\n")
    guard = ('def run_model():\n'
             '    try:\n'
             '        self_similar_refine(low.copy(), high.copy(), depth, q.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n'
             'def run_oracle():\n'
             '    try:\n'
             '        _oracle_self_similar_refine(low.copy(), high.copy(), depth, q.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n')
    return [
        # --- Boundary: depth zero returns the two endpoints unchanged ---
        {"setup": anchors + "depth = 0\n",
         "call": "self_similar_refine(low.copy(), high.copy(), depth, q.copy())",
         "gold_call": "_oracle_self_similar_refine(low.copy(), high.copy(), depth, q.copy())", "tol": 1e-09},
        # --- Normal: a single insertion, the coarsest non-trivial bridge ---
        {"setup": anchors + "depth = 1\n",
         "call": "self_similar_refine(low.copy(), high.copy(), depth, q.copy())",
         "gold_call": "_oracle_self_similar_refine(low.copy(), high.copy(), depth, q.copy())", "tol": 1e-09},
        # --- Normal: three levels, where self-similar structure becomes visible ---
        {"setup": anchors + "depth = 3\n",
         "call": "self_similar_refine(low.copy(), high.copy(), depth, q.copy())",
         "gold_call": "_oracle_self_similar_refine(low.copy(), high.copy(), depth, q.copy())", "tol": 1e-09},
        # --- Normal: off-median quantiles bias every insertion in the same direction ---
        {"setup": anchors.replace("0.5, 0.5, 0.5", "0.3, 0.6, 0.4") + "depth = 3\n",
         "call": "self_similar_refine(low.copy(), high.copy(), depth, q.copy())",
         "gold_call": "_oracle_self_similar_refine(low.copy(), high.copy(), depth, q.copy())", "tol": 1e-09},
        # --- Normal: a different endpoint pair ---
        {"setup": ("import numpy as np\n"
                   "low = np.array([1.05, 0.40, 0.025])\n"
                   "high = np.array([2.40, 4.00, 2.800])\n"
                   "q = np.array([0.5, 0.5, 0.5])\ndepth = 2\n"),
         "call": "self_similar_refine(low.copy(), high.copy(), depth, q.copy())",
         "gold_call": "_oracle_self_similar_refine(low.copy(), high.copy(), depth, q.copy())", "tol": 1e-09},
        # --- Invalid: a negative depth must raise ValueError ---
        {"setup": anchors + "depth = -1\n" + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a non-integer depth must raise ValueError ---
        {"setup": anchors + "depth = 2.5\n" + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
