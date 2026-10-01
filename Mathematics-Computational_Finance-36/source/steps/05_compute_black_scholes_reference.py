"""
Evaluate the analytical reference valuation corresponding to the continuous model on the same space-time mesh used by the numerical solution.



The implementation must maintain the same parameter convention, temporal interpretation, spatial ordering, and mesh dimensions as the numerical calculation so that every analytical value corresponds to exactly one numerical node.

For constant volatility and risk-free rate, the European call model considered in the reference work admits an analytical valuation. This provides an independent continuous reference against which the numerical approximation can be assessed.



The analytical reference must be evaluated consistently with the computational time convention used by the numerical formulation. The comparison is therefore between corresponding representations of the same continuous option-pricing problem, rather than between unrelated quantities.

Returns
-------
Returns a two-dimensional NumPy array containing the analytical reference valuation at every node of the prescribed space-time mesh, with the same shape and node ordering as the numerical solution.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_black_scholes_reference(r: float, sigma: float, T: float, K: float, S_max: float, M: int, N: int) -> "np.ndarray":
    """Return the analytical reference surface on the benchmark mesh.

    Parameters
    ----------
    r, sigma, T, K, S_max : float
        Positive model and domain parameters.
    M : int
        Number of spatial intervals; must be at least 1.
    N : int
        Number of time intervals; must be at least 1.

    Returns
    -------
    numpy.ndarray
        Float64 reference surface of shape `(N+1, M+1)`.

    Raises
    ------
    ValueError
        If a scalar input violates the stated positivity or interval-count requirements.
    
    Notes
    -----
    The returned surface uses the same spatial and temporal indexing as the numerical surface.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_black_scholes_reference(r: float, sigma: float, T: float, K: float, S_max: float, M: int, N: int) -> "np.ndarray":
    import numpy as np
    from scipy.special import ndtr
    if not (r>0 and sigma>0 and T>0 and K>0 and S_max>0): raise ValueError("physical parameters must be positive")
    if not (isinstance(M,(int,np.integer)) and int(M)>=1): raise ValueError("M must be an integer >= 1")
    if not (isinstance(N,(int,np.integer)) and int(N)>=1): raise ValueError("N must be an integer >= 1")
    M=int(M); N=int(N)
    s_grid=np.arange(M+1,dtype=np.float64)*(S_max/M); times=np.arange(N+1,dtype=np.float64)*(T/N); reference=np.empty((N+1,M+1),dtype=np.float64)
    for k,tau in enumerate(times):
        for m,s in enumerate(s_grid):
            if tau<=1e-15: reference[k,m]=max(float(s-K),0.0)
            elif s<=0: reference[k,m]=0.0
            else:
                root=np.sqrt(tau); d1=(np.log(s/K)+(r+0.5*sigma**2)*tau)/(sigma*root); d2=d1-sigma*root; reference[k,m]=s*ndtr(d1)-K*np.exp(-r*tau)*ndtr(d2)
    return reference

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"r=0.035; sigma=0.35; T=1.1; K=1.2; S_max=8.4; M=21; N=25","call":"compute_black_scholes_reference(r,sigma,T,K,S_max,M,N)","gold_call":"_oracle_compute_black_scholes_reference(r,sigma,T,K,S_max,M,N)"},
        {"setup":"r=0.04; sigma=0.4; T=1.0; K=1.0; S_max=2.0; M=1; N=1","call":"compute_black_scholes_reference(r,sigma,T,K,S_max,M,N)","gold_call":"_oracle_compute_black_scholes_reference(r,sigma,T,K,S_max,M,N)"},
        {"setup":"r=0.02; sigma=0.25; T=0.5; K=0.8; S_max=4.0; M=7; N=9","call":"compute_black_scholes_reference(r,sigma,T,K,S_max,M,N)","gold_call":"_oracle_compute_black_scholes_reference(r,sigma,T,K,S_max,M,N)"},
        {"setup":"r=0.035; sigma=0.35; T=1.1; K=1.2; S_max=8.4; M=21; N=0\ndef run_model():\n    try: compute_black_scholes_reference(r,sigma,T,K,S_max,M,N); return 0\n    except ValueError: return 1\ndef run_gold():\n    try: _oracle_compute_black_scholes_reference(r,sigma,T,K,S_max,M,N); return 0\n    except ValueError: return 1","call":"run_model()","gold_call":"run_gold()"}
    ]
