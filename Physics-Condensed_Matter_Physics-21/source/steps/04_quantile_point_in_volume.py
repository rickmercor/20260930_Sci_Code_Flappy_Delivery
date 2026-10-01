"""
Select a single thermodynamic state inside the allowed region at prescribed quantiles of the uniform measure, drawing the three variables in sequence.

A point is placed in the allowed region by exploiting the fact that a slice at fixed chemical potential is a triangle. The chemical potential is chosen first, from its own marginal; the number density is then chosen from the distribution conditional on that chemical potential; and the pressure is chosen last, from the distribution conditional on both.




Conditional on the chemical potential, the density marginal is proportional to the width of the pressure interval at that density, which is the height of the triangular slice. Conditional on both the chemical potential and the density, the pressure is uniform between the two pressure edges, so its quantile enters linearly.




Replacing random draws by prescribed quantiles at each of the three stages makes the selection deterministic while leaving the geometry of the construction untouched: the same marginals are inverted in the same order, and the resulting state lies in the allowed region for any quantiles in the unit interval.

Returns
-------
np.ndarray, shape (3,), [mu, n, p] in GeV, fm^-3, GeV fm^-3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quantile_point_in_volume(quantiles: "np.ndarray", beta_low: "np.ndarray",
                             beta_high: "np.ndarray") -> "np.ndarray":
    '''Place one state in the allowed region at prescribed quantiles.

    Parameters
    ----------
    quantiles : np.ndarray
        Finite shape (3,) with every entry in [0, 1], the quantiles used for the
        chemical potential, the density and the pressure in that order.
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.

    Returns
    -------
    point : np.ndarray
        Finite shape (3,), the state (mu, n, p) in GeV, fm^-3 and GeV fm^-3.
        The chemical potential is taken at its quantile of the induced marginal.
        The density is taken at its quantile of the conditional distribution
        whose density is proportional to the width of the pressure interval at
        that density. The pressure is taken at its quantile of the uniform
        distribution between the pressure edges, so it equals
        pmin + quantiles[2]*(pmax - pmin). Each conditional inversion has an
        absolute accuracy of 1e-10 in its own units. At an endpoint chemical
        potential, where the slice has zero area, use the midpoint of the
        density bounds and the common pressure edge.

    Raises
    ------
    ValueError
        If the quantiles are not a finite real shape (3,) array inside [0, 1],
        if a preceding step rejects its inputs, or if the selected state is not
        finite.
    '''
    return point

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_quantile_point_in_volume(quantiles: "np.ndarray", beta_low: "np.ndarray",
                                     beta_high: "np.ndarray") -> "np.ndarray":
    if not np.isrealobj(quantiles):
        raise ValueError("the quantiles must be real")
    try:
        q = np.asarray(quantiles, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the quantiles must be a finite real array") from exc
    if q.shape != (3,) or not np.all(np.isfinite(q)) or np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("the quantiles must be a shape (3,) array inside [0, 1]")
    mu = _oracle_chemical_potential_quantile(q[0], beta_low, beta_high)
    span = _oracle_allowed_density_bounds(mu, beta_low, beta_high)
    lo, hi = float(span[0]), float(span[1])
    if q[0] == 0.0 or q[0] == 1.0 or hi <= lo:
        n = 0.5 * (lo + hi)
    elif q[1] == 0.0:
        n = lo
    elif q[1] == 1.0:
        n = hi
    else:
        # The pressure width is a triangular density. Its mode is where the
        # two upper pressure edges meet; invert its normalized CDF exactly.
        low, high, _ = _validated_endpoints(beta_low, beta_high)
        mode = 2.0 * mu * (high[2] - low[2]) / ((high[0] - low[0]) * (high[0] + low[0]))
        mode = float(np.clip(mode, lo, hi))
        width = hi - lo
        if q[1] <= (mode - lo) / width:
            n = lo + np.sqrt(q[1] * width * (mode - lo))
        else:
            n = hi - np.sqrt((1.0 - q[1]) * width * (hi - mode))
    edges = _oracle_allowed_pressure_bounds(mu, n, beta_low, beta_high)
    point = np.array([mu, n, edges[0] + q[2] * (edges[1] - edges[0])], dtype=float)
    if not np.all(np.isfinite(point)):
        raise ValueError("the selected state must be finite")
    return point

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    anchors = ("import numpy as np\n"
               "low = np.array([1.00, 0.32, 0.018])\n"
               "high = np.array([2.60, 4.80, 3.500])\n")
    guard = ('def run_model():\n'
             '    try:\n'
             '        quantile_point_in_volume(q.copy(), low.copy(), high.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n'
             'def run_oracle():\n'
             '    try:\n'
             '        _oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n')
    return [
        # --- Normal: the median in all three variables, as used by the benchmark ---
        {"setup": anchors + "q = np.array([0.5, 0.5, 0.5])\n",
         "call": "quantile_point_in_volume(q.copy(), low.copy(), high.copy())",
         "gold_call": "_oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())", "tol": 1e-09},
        # --- Normal: off-median in the pressure only, sliding along the slice height ---
        {"setup": anchors + "q = np.array([0.5, 0.5, 0.2])\n",
         "call": "quantile_point_in_volume(q.copy(), low.copy(), high.copy())",
         "gold_call": "_oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())", "tol": 1e-09},
        # --- Normal: off-median in the density, moving along the slice ---
        {"setup": anchors + "q = np.array([0.35, 0.7, 0.5])\n",
         "call": "quantile_point_in_volume(q.copy(), low.copy(), high.copy())",
         "gold_call": "_oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())", "tol": 1e-09},
        # --- Boundary: the lowest corner of a slice that still has positive width ---
        {"setup": anchors + "q = np.array([0.02, 0.0, 0.0])\n",
         "call": "quantile_point_in_volume(q.copy(), low.copy(), high.copy())",
         "gold_call": "_oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())", "tol": 1e-09},
        # --- Boundary: the highest corner of a slice that still has positive width ---
        {"setup": anchors + "q = np.array([0.98, 1.0, 1.0])\n",
         "call": "quantile_point_in_volume(q.copy(), low.copy(), high.copy())",
         "gold_call": "_oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())", "tol": 1e-09},
        # --- Normal: a different endpoint pair ---
        {"setup": ("import numpy as np\n"
                   "low = np.array([1.05, 0.40, 0.025])\n"
                   "high = np.array([2.40, 4.00, 2.800])\n"
                   "q = np.array([0.5, 0.5, 0.5])\n"),
         "call": "quantile_point_in_volume(q.copy(), low.copy(), high.copy())",
         "gold_call": "_oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())", "tol": 1e-09},
        # --- Invalid: a quantile outside the unit interval must raise ValueError ---
        {"setup": anchors + "q = np.array([0.5, -0.1, 0.5])\n" + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Edge: a small conditional slice reached during recursive refinement ---
        {"setup": ("import numpy as np\n"
                   "low = np.array([1.775893133518957, 1.8966459940619773, 0.7901034144969973])\n"
                   "high = np.array([1.8749986667842924, 2.177831555471652, 0.9920739149035971])\n"
                   "q = np.array([0.5, 0.5, 0.5])\n"),
         "call": "quantile_point_in_volume(q.copy(), low.copy(), high.copy())",
         "gold_call": "_oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())", "tol": 1e-10},
        # --- Boundary: zero-area slice at the low chemical-potential endpoint ---
        {"setup": anchors + "q = np.array([0.0, 0.2, 0.8])\n",
         "call": "quantile_point_in_volume(q.copy(), low.copy(), high.copy())",
         "gold_call": "_oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())", "tol": 1e-10},
        # --- Boundary: zero-area slice at the high chemical-potential endpoint ---
        {"setup": anchors + "q = np.array([1.0, 0.8, 0.2])\n",
         "call": "quantile_point_in_volume(q.copy(), low.copy(), high.copy())",
         "gold_call": "_oracle_quantile_point_in_volume(q.copy(), low.copy(), high.copy())", "tol": 1e-10},
    ]
