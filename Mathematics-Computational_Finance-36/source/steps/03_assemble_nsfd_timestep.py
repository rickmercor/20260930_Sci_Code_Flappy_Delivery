"""
Construct the linear system associated with one implicit numerical update from the supplied previous solution and discretization data.






The construction must incorporate the spatial coefficients and the prescribed boundary information consistently with the reference formulation. This function is responsible only for forming the system representation required by the numerical solver; it must not perform the subsequent linear solve.

An implicit discretization couples neighboring spatial unknowns at a new time level. For the one-dimensional problem considered here, the reference formulation produces a structured linear system whose interior equations are affected by the spatial discretization and whose right-hand side depends on the previous state and boundary data.






Correct assembly requires preserving the distinction between interior unknowns, known boundary values, and quantities inherited from the preceding time level.

Returns
-------
A float64 NumPy array of shape (M-1, 4) containing, in columns 0 to 3, the lower band, main diagonal, upper band, and right-hand-side values of the interior tridiagonal system. The lower band holds the sub-diagonal entries of interior rows 1 to M-2 in positions 0 to M-3, followed by a trailing zero; the upper band holds the super-diagonal entries of interior rows 0 to M-3 in positions 0 to M-3, followed by a trailing zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_nsfd_timestep(previous: "np.ndarray", coefficients: "np.ndarray", S_max: float, K: float, r: float, t: float, M: int) -> "np.ndarray":
    """Assemble one interior implicit update system.

    Parameters
    ----------
    previous : numpy.ndarray
        Previous solution row with shape `(M+1,)`.
    coefficients : numpy.ndarray
        Coefficient table with shape `(M-1, 4)` produced by the coefficient stage.
    S_max, K, r, t : float
        Positive numerical/model parameters for the current update.
    M : int
        Number of spatial intervals; must be at least 2.

    Returns
    -------
    "np.ndarray"
        A float64 NumPy array of shape (M-1, 4) containing, in columns 0 to 3, the lower band, main diagonal, upper band, and right-hand-side values for the interior tridiagonal system. The lower band holds the sub-diagonal entries of interior rows 1 to M-2 in positions 0 to M-3, followed by a trailing zero; the upper band holds the super-diagonal entries of interior rows 0 to M-3 in positions 0 to M-3, followed by a trailing zero.

    Raises
    ------
    ValueError
        If shapes or scalar requirements are violated.
    
    Notes
    -----
    The returned bands are arranged for a standard tridiagonal solve of the interior unknowns.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_nsfd_timestep(previous: "np.ndarray", coefficients: "np.ndarray", S_max: float, K: float, r: float, t: float, M: int) -> "np.ndarray":
    import numpy as np
    previous=np.asarray(previous,dtype=np.float64); coefficients=np.asarray(coefficients,dtype=np.float64)
    if not (isinstance(M,(int,np.integer)) and int(M)>=2): raise ValueError("M must be an integer >= 2")
    M=int(M)
    if previous.shape!=(M+1,): raise ValueError("previous must have shape (M+1,)")
    if coefficients.shape!=(M-1,4): raise ValueError("coefficients must have shape (M-1,4)")
    B_left=coefficients[:,0]; B_center=coefficients[:,1]; B_right=coefficients[:,2]; psi_1=float(coefficients[0,3])
    if not (np.isfinite(psi_1) and psi_1>0) or not np.allclose(coefficients[:,3],psi_1,rtol=0.0,atol=0.0): raise ValueError("invalid temporal coefficient column")
    if S_max<=0 or K<=0 or r<=0 or t<=0: raise ValueError("physical inputs must be positive")
    g=float(S_max-K*np.exp(-r*t))
    lower=B_left[1:].copy(); diagonal=B_center.copy(); upper=B_right[:-1].copy(); rhs=-previous[1:M]/psi_1; rhs[-1]-=B_right[-1]*g
    packed=np.zeros((M-1,4),dtype=np.float64)
    packed[:,0]=np.r_[lower,0.0]; packed[:,1]=diagonal; packed[:,2]=np.r_[upper,0.0]; packed[:,3]=rhs
    return packed

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nM=21; previous=np.linspace(0.0,7.2,M+1)\ncoefficients=_oracle_compute_nsfd_coefficients(0.035,0.35,1.1,8.4,M,25)\nS_max=8.4; K=1.2; r=0.035; t=0.044","call":"assemble_nsfd_timestep(previous,coefficients,S_max,K,r,t,M)","gold_call":"_oracle_assemble_nsfd_timestep(previous,coefficients,S_max,K,r,t,M)"},
        {"setup":"import numpy as np\nM=2; previous=np.array([0.0,0.1,1.0]); coefficients=_oracle_compute_nsfd_coefficients(0.04,0.4,1.0,2.0,M,1); S_max=2.0; K=1.0; r=0.04; t=1.0","call":"assemble_nsfd_timestep(previous,coefficients,S_max,K,r,t,M)","gold_call":"_oracle_assemble_nsfd_timestep(previous,coefficients,S_max,K,r,t,M)"},
        {"setup":"import numpy as np\nM=5; previous=np.ones(M+1); coefficients=_oracle_compute_nsfd_coefficients(0.02,0.25,0.5,4.0,M,9); S_max=4.0; K=0.8; r=0.02; t=0.5","call":"assemble_nsfd_timestep(previous,coefficients,S_max,K,r,t,M)","gold_call":"_oracle_assemble_nsfd_timestep(previous,coefficients,S_max,K,r,t,M)"},
        {"setup":"import numpy as np\nM=4; previous=np.ones(M); coefficients=np.ones((M-1,4)); coefficients[:,3]=0.1; S_max=4.0; K=1.0; r=0.02; t=0.5\ndef run_model():\n    try: assemble_nsfd_timestep(previous,coefficients,S_max,K,r,t,M); return 0\n    except ValueError: return 1\ndef run_gold():\n    try: _oracle_assemble_nsfd_timestep(previous,coefficients,S_max,K,r,t,M); return 0\n    except ValueError: return 1","call":"run_model()","gold_call":"run_gold()"}
    ]
