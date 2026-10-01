"""
Combine two conditional risk values and the supplied splitting controls into the scalar risk terms used by the penalized model.

Two conditional risk values are supplied for one period, and the result records the scalar quantities needed by the penalized portfolio objective.

Returns
-------
The returned array records the two coefficients, the two branch values, and their active maximum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_dc_risk_terms(cvar_lower: float, cvar_alpha: float, alpha: float, gamma: float, tau: float) -> np.ndarray:
    """Return the scalar quantities defining one period's penalized risk contribution.

    Parameters
    ----------
    cvar_lower : float
        Conditional risk value evaluated at the lower tail level.
    cvar_alpha : float
        Conditional risk value evaluated at the requested confidence level.
    alpha : float
        Requested confidence level.
    gamma : float
        Positive splitting offset.
    tau : float
        VaR threshold.

    Returns
    -------
    terms : np.ndarray
        Five-entry vector containing the two splitting coefficients, the two
        transformed branch values, and their maximum.

    Raises
    ------
    ValueError
        If alpha is outside (0, 1) or gamma is outside (0, alpha).
    """
    return terms

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_dc_risk_terms(cvar_lower: float, cvar_alpha: float, alpha: float, gamma: float, tau: float) -> np.ndarray:
    a=float(alpha); g=float(gamma); t=float(tau)
    lo=float(cvar_lower); hi=float(cvar_alpha)
    if not (0.0 < a < 1.0 and 0.0 < g < a):
        raise ValueError("invalid confidence or gamma")
    A=(1.0-a+g)/g
    B=(1.0-a)/g
    left=A*lo-t
    right=B*hi
    return np.array([A,B,left,right,max(left,right)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent term-construction cases."""
    return [
        {"setup": "c1=0.02204225352112676; c2=0.022142857142857145; a=0.65; g=0.005; t=0.02", "call": "tuple(build_dc_risk_terms(c1,c2,a,g,t))", "gold_call": "tuple(_oracle_build_dc_risk_terms(c1,c2,a,g,t))", "tol": 1e-12},
        {"setup": "c1=0.10; c2=0.11; a=0.50; g=0.01; t=0.0", "call": "tuple(build_dc_risk_terms(c1,c2,a,g,t))", "gold_call": "tuple(_oracle_build_dc_risk_terms(c1,c2,a,g,t))", "tol": 1e-12},
        {"setup": "c1=0.02; c2=0.021; a=0.73; g=0.03; t=0.015", "call": "tuple(build_dc_risk_terms(c1,c2,a,g,t))", "gold_call": "tuple(_oracle_build_dc_risk_terms(c1,c2,a,g,t))", "tol": 1e-12},
        {"setup": "c1=0.03; c2=0.025; a=0.65; g=0.005; t=0.02", "call": "tuple(build_dc_risk_terms(c1,c2,a,g,t))", "gold_call": "tuple(_oracle_build_dc_risk_terms(c1,c2,a,g,t))", "tol": 1e-12},
    ]
