"""
Construct all discretization quantities required by the reference work for the interior spatial rows and the temporal evolution of the implicit numerical formulation.






The implementation must recover the paper-specific construction from the reference method rather than replacing it with a conventional finite-difference approximation. The returned quantities must be organized so that each spatial row corresponds consistently to the associated interior grid point and can be consumed directly by the subsequent linear-system construction.

The reference method constructs a nonstandard discretization by exploiting exact information from reduced differential problems associated with the governing equation. This produces spatial and temporal discrete quantities that differ from ordinary mesh-increment substitutions.






The resulting coefficients encode the structural information used by the implicit scheme. Correct reproduction therefore requires following the construction in the reference work, including its normalization and correspondence between discrete rows and physical spatial locations.

Returns
-------
Returns a single float64 NumPy array of shape (M-1, 4) holding the complete spatial coefficient data required by the implicit update together with the temporal discretization quantity. Column 0 is the left coefficient, column 1 the center coefficient, column 2 the right coefficient, and column 3 the temporal quantity repeated on every row.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_nsfd_coefficients(r: float, sigma: float, T: float, S_max: float, M: int, N: int) -> "np.ndarray":
    """Return the coefficient data required by the reference implicit discretization.

    Parameters
    ----------
    r : float
        Positive risk-free rate.
    sigma : float
        Positive volatility.
    T : float
        Positive time horizon.
    S_max : float
        Positive spatial-domain endpoint.
    M : int
        Number of spatial intervals; must be at least 2.
    N : int
        Number of time intervals; must be at least 1.

    Returns
    -------
    "np.ndarray"
        A float64 array of shape `(M-1, 4)`. Column 0 is the left coefficient, column 1 the center coefficient, column 2 the right coefficient, and column 3 the temporal quantity repeated on every row.

    Raises
    ------
    ValueError
        If an input violates the stated positivity or interval-count requirements, or if the reference coefficient construction becomes non-finite or singular.
    
    Notes
    -----
    The coefficient rows follow the positive interior-node convention stated in the task.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_nsfd_coefficients(r: float, sigma: float, T: float, S_max: float, M: int, N: int) -> "np.ndarray":
    import numpy as np
    if not (isinstance(r,(int,float)) and float(r)>0): raise ValueError("r must be positive")
    if not (isinstance(sigma,(int,float)) and float(sigma)>0): raise ValueError("sigma must be positive")
    if not (isinstance(T,(int,float)) and float(T)>0): raise ValueError("T must be positive")
    if not (isinstance(S_max,(int,float)) and float(S_max)>0): raise ValueError("S_max must be positive")
    if not (isinstance(M,(int,np.integer)) and int(M)>=2): raise ValueError("M must be an integer >= 2")
    if not (isinstance(N,(int,np.integer)) and int(N)>=1): raise ValueError("N must be an integer >= 1")
    r=float(r); sigma=float(sigma); T=float(T); M=int(M); N=int(N)
    D=0.5*sigma**2; alpha=r/D; delta_t=T/N; psi_1=float(np.exp(r*delta_t)-1.0)
    if not np.isfinite(psi_1) or psi_1<=0: raise ValueError("invalid temporal denominator")
    B_left=np.empty(M-1,dtype=np.float64); B_center=np.empty(M-1,dtype=np.float64); B_right=np.empty(M-1,dtype=np.float64)
    for idx,m in enumerate(range(1,M)):
        A1=(m+1.0)/(m**alpha)-m/((m+1.0)**alpha)
        A2=(m+2.0)/(m**alpha)-m/((m+2.0)**alpha)
        A3=(m+2.0)/((m+1.0)**alpha)-(m+1.0)/((m+2.0)**alpha)
        q=A2-A1-A3
        if q<=0 or not np.isfinite(q): raise ValueError("reference coefficient denominator is invalid")
        B_left[idx]=(A2-A1)/q-1.0; B_center[idx]=-A2/q-1.0/psi_1; B_right[idx]=A1/q
    packed=np.empty((M-1,4),dtype=np.float64)
    packed[:,0]=B_left; packed[:,1]=B_center; packed[:,2]=B_right; packed[:,3]=psi_1
    return packed

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"r=0.035; sigma=0.35; T=1.1; S_max=8.4; M=21; N=25","call":"compute_nsfd_coefficients(r,sigma,T,S_max,M,N)","gold_call":"_oracle_compute_nsfd_coefficients(r,sigma,T,S_max,M,N)"},
        {"setup":"r=0.04; sigma=0.4; T=1.0; S_max=8.0; M=2; N=1","call":"compute_nsfd_coefficients(r,sigma,T,S_max,M,N)","gold_call":"_oracle_compute_nsfd_coefficients(r,sigma,T,S_max,M,N)"},
        {"setup":"r=0.02; sigma=0.25; T=0.5; S_max=4.0; M=7; N=9","call":"compute_nsfd_coefficients(r,sigma,T,S_max,M,N)","gold_call":"_oracle_compute_nsfd_coefficients(r,sigma,T,S_max,M,N)"},
        {"setup":"r=0.0; sigma=0.35; T=1.0; S_max=8.0; M=5; N=5\ndef run_model():\n    try: compute_nsfd_coefficients(r,sigma,T,S_max,M,N); return 0\n    except ValueError: return 1\ndef run_gold():\n    try: _oracle_compute_nsfd_coefficients(r,sigma,T,S_max,M,N); return 0\n    except ValueError: return 1","call":"run_model()","gold_call":"run_gold()"}
    ]
