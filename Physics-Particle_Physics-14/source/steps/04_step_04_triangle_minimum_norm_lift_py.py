"""
Lift the source critical class into the degree-bounded benchmark system.

This step verifies the paper's explicit triangle representative in the assembled syzygy kernel, then solves the paper-style coefficient system with its critical normalization; minimum norm is only the deterministic benchmark gauge.

Returns
-------
np.ndarray shape-(43,) minimum-norm lift.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def triangle_minimum_norm_lift(s: float) -> "np.ndarray":
    '''Return the deterministic minimum-norm lift of the source critical class.

    Parameters
    ----------
    s : float
        Positive finite kinematic invariant.

    Returns
    -------
    np.ndarray
        Shape-(43,) minimum-Euclidean-norm coefficient vector satisfying the
        assembled syzygy system with the source critical normalization
        a0 = -2 s^2.

    Raises
    ------
    ValueError
        If s is non-finite or non-positive, or if the source triangle
        representative fails the assembled syzygy identity check.
    '''
    return coeffs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_triangle_minimum_norm_lift(s: float) -> "np.ndarray":
    s=float(s)
    b=_oracle_triangle_normalized_baikov(s)
    source=_oracle_triangle_source_nonprincipal_syzygy(s)
    M=_oracle_triangle_syzygy_matrix(b)
    if np.max(np.abs(M @ source)) > 1e-10:
        raise ValueError('source representative does not satisfy the assembled syzygy system')
    A=np.vstack([M,np.eye(1,43,0,dtype=float)])
    rhs=np.r_[np.zeros(35),source[0]]
    return np.linalg.lstsq(A,rhs,rcond=1e-12)[0]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'s=5.0','call':'triangle_minimum_norm_lift(s)','gold_call':'_oracle_triangle_minimum_norm_lift(s)','tol':3e-10},
        {'setup':'s=2.0','call':'triangle_minimum_norm_lift(s)','gold_call':'_oracle_triangle_minimum_norm_lift(s)','tol':3e-10},
        {'setup':'s=7.5','call':'triangle_minimum_norm_lift(s)','gold_call':'_oracle_triangle_minimum_norm_lift(s)','tol':5e-10},
    ]
