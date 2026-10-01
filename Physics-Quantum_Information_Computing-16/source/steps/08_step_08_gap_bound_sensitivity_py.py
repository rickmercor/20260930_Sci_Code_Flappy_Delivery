"""
Step 8 (ORCHESTRATOR): d delta_h / dr of the level-3 certifiable bound at fixed s.

The derivative of the certifiable optimum includes the r-dependence of the certifiability projector P_2_perp = supp(h(r)). The unrestricted LTI gradient formula of the paper, with the projector frozen, contributes 0.31745729; the projected-gauge term contributes the remaining 3.61982417, for a total sensitivity of 3.93728145 at the target point. A symmetric finite difference of the full optimum reproduces this value and is the quantity returned.

Returns
-------
# float, d delta_h / dr at (r, s) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================
def gap_bound_sensitivity(r: float = 0.347, s: float = 0.783, step: float = 1e-4,
                          tol: float = 1e-10, solver_tol: float = 1e-11) -> float:
    '''Return d delta_h(r,s) / dr at fixed s by a symmetric finite difference.

    Parameters
    ----------
    r, s : float
        Positive finite deformation parameters at which the derivative is taken.
    step : float
        Finite positive finite-difference step, smaller than r.
    tol : float
        Positive eigenvalue threshold.
    solver_tol : float
        Positive finite SDP solver tolerance.

    Returns
    -------
    deriv : float
        The derivative as a native Python float.

    Raises
    ------
    ValueError
        If r or s is non-finite or <= 0, step is non-finite, <= 0 or >= r,
        or a downstream step raises.
    '''
    return deriv

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================
import numpy as np
def _oracle_gap_bound_sensitivity(r: float = 0.347, s: float = 0.783, step: float = 1e-4,
                                  tol: float = 1e-10, solver_tol: float = 1e-11) -> float:
    import numpy as np
    r = float(r); s = float(s); step = float(step)
    if not np.isfinite(r) or not np.isfinite(s) or r <= 0.0 or s <= 0.0:
        raise ValueError("r and s must be finite and > 0")
    if not np.isfinite(step) or step <= 0.0 or r - step <= 0.0:
        raise ValueError("step must be finite, > 0 and smaller than r")
    fn = lambda rr: _oracle_certified_bound_at_parameters(rr, s, tol, solver_tol)
    return float(_oracle_central_difference(fn, r, step))

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        # normal: target point, 3.93728
        {
            "setup": "import numpy as np\nr=0.347; s=0.783",
            "call": "gap_bound_sensitivity(r,s)",
            "gold_call": "_oracle_gap_bound_sensitivity(r,s)",
        },

        # boundary: larger step, O(step^2) change only (3.93728)
        {
            "setup": "import numpy as np\nr=0.347; s=0.783; step=1e-3",
            "call": "gap_bound_sensitivity(r,s,step)",
            "gold_call": "_oracle_gap_bound_sensitivity(r,s,step)",
        },

        # edge: undeformed point, derivative ~ 0
        {
            "setup": "import numpy as np\nr=1.0; s=1.0",
            "call": "gap_bound_sensitivity(r,s)",
            "gold_call": "_oracle_gap_bound_sensitivity(r,s)",
        },
    ]

# ORCHESTRATOR: YES
