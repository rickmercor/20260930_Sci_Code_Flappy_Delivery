"""
Execute the complete benchmark workflow for the fixed numerical instance specified by the task.



The final computation must be obtained by composing the preceding public stages in their intended dependency order: construction of the initial data, construction of the reference discretization quantities, formation and solution of the numerical evolution, construction of the analytical reference, and evaluation of the global nodal discrepancy.



Do not replace the preceding workflow with an independently hard-coded final calculation. The function represents the complete scientific experiment and must return only its requested deterministic result.

The benchmark combines the numerical and analytical components of the reference formulation into one reproducible computational experiment. Its purpose is to measure how closely the paper-specific implicit numerical construction reproduces the corresponding analytical option valuation for the prescribed finite computational domain and mesh.



The final quantity therefore depends on the complete numerical pipeline rather than on any isolated intermediate quantity.

Returns
-------
Returns the deterministic maximum nodal error for the prescribed benchmark instance as a native Python floating-point scalar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_nsfd_benchmark(r: float, sigma: float, T: float, K: float, S_max: float, epsilon: float, M: int, N: int) -> float:
    """Return the complete deterministic benchmark error.

    Parameters
    ----------
    r, sigma, T, K, S_max, epsilon : float
        Positive benchmark parameters.
    M : int
        Number of spatial intervals; must be at least 2.
    N : int
        Number of time intervals; must be at least 1.

    Returns
    -------
    float
        The maximum absolute nodal discrepancy produced by the complete composed workflow.

    Raises
    ------
    ValueError
        If the supplied benchmark parameters violate the contracts of the preceding stages.
    
    Notes
    -----
    The final scalar is obtained by composing the preceding public numerical stages.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_nsfd_benchmark(r: float, sigma: float, T: float, K: float, S_max: float, epsilon: float, M: int, N: int) -> float:
    import numpy as np
    initial_table=_oracle_build_smoothed_payoff(K,S_max,epsilon,M)
    coefficients=_oracle_compute_nsfd_coefficients(r,sigma,T,S_max,M,N)
    numerical=_oracle_solve_nsfd_surface(initial_table,coefficients,r,T,K,S_max,M,N)
    reference=_oracle_compute_black_scholes_reference(r,sigma,T,K,S_max,M,N)
    error=_oracle_compute_max_nodal_error(numerical,reference)
    if initial_table.shape!=(M+1,2): raise ValueError("upstream payoff stage returned an invalid shape")
    return float(error)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"r=0.035; sigma=0.35; T=1.1; K=1.2; S_max=8.4; epsilon=0.07; M=21; N=25","call":"run_nsfd_benchmark(r,sigma,T,K,S_max,epsilon,M,N)","gold_call":"_oracle_run_nsfd_benchmark(r,sigma,T,K,S_max,epsilon,M,N)"},
        {"setup":"r=0.04; sigma=0.4; T=1.0; K=1.0; S_max=8.0; epsilon=1e-6; M=16; N=16","call":"run_nsfd_benchmark(r,sigma,T,K,S_max,epsilon,M,N)","gold_call":"_oracle_run_nsfd_benchmark(r,sigma,T,K,S_max,epsilon,M,N)"},
        {"setup":"r=0.02; sigma=0.25; T=0.5; K=0.8; S_max=4.0; epsilon=0.1; M=7; N=9","call":"run_nsfd_benchmark(r,sigma,T,K,S_max,epsilon,M,N)","gold_call":"_oracle_run_nsfd_benchmark(r,sigma,T,K,S_max,epsilon,M,N)"},
        {"setup":"r=0.035; sigma=0.35; T=1.1; K=1.2; S_max=8.4; epsilon=0.07; M=1; N=25\ndef run_model():\n    try: run_nsfd_benchmark(r,sigma,T,K,S_max,epsilon,M,N); return 0\n    except ValueError: return 1\ndef run_gold():\n    try: _oracle_run_nsfd_benchmark(r,sigma,T,K,S_max,epsilon,M,N); return 0\n    except ValueError: return 1","call":"run_model()","gold_call":"run_gold()"}
    ]
