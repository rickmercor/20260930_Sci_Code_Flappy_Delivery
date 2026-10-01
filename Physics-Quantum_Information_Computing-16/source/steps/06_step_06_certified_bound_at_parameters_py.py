"""
Step 6: The certified level-3 bound delta_h(r,s) as a function of the parameters.

Every ingredient of the SDP -- the coefficients, interaction, its kernel, and the projector P_2_perp equal to the support of h -- is rebuilt at each pair (r, s). This is the function whose derivative with respect to r is requested.

Returns
-------
# float, delta_h(r,s) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================
def certified_bound_at_parameters(r: float, s: float, tol: float = 1e-10, solver_tol: float = 1e-11) -> float:
    '''Return delta_h(r,s), the level-3 certifiable LTI bound for the original interaction.

    Parameters
    ----------
    r, s : float
        Positive finite deformation parameters.
    tol : float
        Positive eigenvalue threshold for kernels and projectors.
    solver_tol : float
        Positive finite SDP solver tolerance.

    Returns
    -------
    delta_h : float
        Certified bound as a native Python float.

    Raises
    ------
    ValueError
        If r, s, tol or solver_tol is non-finite or <= 0, or a downstream step raises.
    '''
    return delta_h  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================
import numpy as np
def _oracle_certified_bound_at_parameters(r: float, s: float, tol: float = 1e-10, solver_tol: float = 1e-11) -> float:
    import numpy as np
    r = float(r); s = float(s); tol = float(tol); solver_tol = float(solver_tol)
    if not np.isfinite(r) or not np.isfinite(s) or r <= 0.0 or s <= 0.0:
        raise ValueError("r and s must be finite and > 0")
    if not np.isfinite(tol) or tol <= 0.0 or not np.isfinite(solver_tol) or solver_tol <= 0.0:
        raise ValueError("tol and solver_tol must be finite and > 0")
    h, _ = _oracle_build_local_interaction(r, s)
    P2, _ = _oracle_excited_subspace_projector(h, tol)      # H_2 = h, so P_2^perp = supp(h)
    return float(_oracle_solve_certifiable_lti_bound(h, P2, solver_tol))

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        # normal
        {
            "setup": "import numpy as np\nr=0.347; s=0.783",
            "call": "certified_bound_at_parameters(r,s)",
            "gold_call": "_oracle_certified_bound_at_parameters(r,s)",
        },

        # boundary: undeformed point
        {
            "setup": "import numpy as np\nr=1.0; s=1.0",
            "call": "certified_bound_at_parameters(r,s)",
            "gold_call": "_oracle_certified_bound_at_parameters(r,s)",
        },

        # edge: inside the region where level 3 certifies only a small gap
        {
            "setup": "import numpy as np\nr=0.3; s=0.783",
            "call": "certified_bound_at_parameters(r,s)",
            "gold_call": "_oracle_certified_bound_at_parameters(r,s)",
        },
    ]
