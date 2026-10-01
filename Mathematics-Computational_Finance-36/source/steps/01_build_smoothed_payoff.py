"""
Construct the spatial discretization and the corresponding regularized initial data required by the numerical formulation specified in the reference work. The construction must faithfully reproduce the reference treatment of the nonsmooth terminal payoff while remaining consistent with the supplied option parameters and computational domain.






The implementation should return the complete spatial grid together with the initial data evaluated on that grid. Preserve the ordering, numerical precision, and dimensionality required by the downstream numerical stages.

The terminal payoff of a European call is continuous but is not sufficiently smooth at the strike. Directly applying a spatial discretization to such data can introduce numerical difficulties near the nonsmooth point. The reference work addresses this issue by replacing the local nonsmooth behavior with a smooth approximation before the numerical evolution is performed.






The required construction is therefore not an arbitrary smoothing procedure: it must follow the specific regularization prescribed by the reference work and remain compatible with the finite computational domain used by the benchmark.

Returns
-------
Returns a single float64 NumPy array of shape (M+1, 2). Column 0 holds the uniform spatial grid in increasing order and column 1 holds the regularized initial value at the corresponding node.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_smoothed_payoff(K: float, S_max: float, epsilon: float, M: int) -> "np.ndarray":
    """Return the uniform asset grid and the reference regularized call payoff.

    Parameters
    ----------
    K : float
        Positive strike price.
    S_max : float
        Positive upper endpoint of the truncated asset domain.
    epsilon : float
        Positive half-width of the local payoff regularization.
    M : int
        Number of spatial intervals; must be at least 2.

    Returns
    -------
    "np.ndarray"
        A float64 array of shape `(M+1, 2)`. Column 0 is the uniform spatial grid in increasing order; column 1 is the regularized initial value at the corresponding node.

    Raises
    ------
    ValueError
        If an input violates the stated positivity or interval-count requirements.
    
    Notes
    -----
    The returned arrays are newly constructed and are not required to share storage with inputs.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_smoothed_payoff(K: float, S_max: float, epsilon: float, M: int) -> "np.ndarray":
    import numpy as np
    if not (isinstance(K, (int, float)) and float(K) > 0.0): raise ValueError("K must be positive")
    if not (isinstance(S_max, (int, float)) and float(S_max) > 0.0): raise ValueError("S_max must be positive")
    if not (isinstance(epsilon, (int, float)) and float(epsilon) > 0.0): raise ValueError("epsilon must be positive")
    if not (isinstance(M, (int, np.integer)) and int(M) >= 2): raise ValueError("M must be an integer >= 2")
    K=float(K); S_max=float(S_max); epsilon=float(epsilon); M=int(M)
    s_grid=np.arange(M+1,dtype=np.float64)*(S_max/M)
    x=s_grid-K
    e1=35.0*epsilon/256.0; e2=0.5; e3=35.0/(64.0*epsilon); e5=-35.0/(128.0*epsilon**3); e7=7.0/(64.0*epsilon**5); e9=5.0/(256.0*epsilon**7)
    polynomial=e1+e2*x+e3*x**2+e5*x**4+e7*x**6+e9*x**8
    payoff=np.where(x<=-epsilon,0.0,np.where(x>=epsilon,x,polynomial))
    return np.column_stack((s_grid,payoff.astype(np.float64)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"K=1.2; S_max=8.4; epsilon=0.07; M=21","call":"build_smoothed_payoff(K,S_max,epsilon,M)","gold_call":"_oracle_build_smoothed_payoff(K,S_max,epsilon,M)"},
        {"setup":"K=1.0; S_max=8.0; epsilon=1e-6; M=16","call":"build_smoothed_payoff(K,S_max,epsilon,M)","gold_call":"_oracle_build_smoothed_payoff(K,S_max,epsilon,M)"},
        {"setup":"K=1.0; S_max=2.0; epsilon=0.25; M=2","call":"build_smoothed_payoff(K,S_max,epsilon,M)","gold_call":"_oracle_build_smoothed_payoff(K,S_max,epsilon,M)"},
        {"setup":"K=1.0; S_max=2.0; epsilon=0.25; M=1\ndef run_model():\n    try: build_smoothed_payoff(K,S_max,epsilon,M); return 0\n    except ValueError: return 1\ndef run_gold():\n    try: _oracle_build_smoothed_payoff(K,S_max,epsilon,M); return 0\n    except ValueError: return 1","call":"run_model()","gold_call":"run_gold()"}
    ]
