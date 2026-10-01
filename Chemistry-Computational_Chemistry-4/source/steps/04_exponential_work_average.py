"""
Chains the per-time-step tilted Glauber transition matrices from Step 3 into the full tilted master-equation product (source method's Eq. 17) to compute the exact generalized work average <exp(-alpha*W)> for a complete Hamiltonian trajectory H(x, tau), tau = 0, ..., N+1. Combines the equilibrium distribution of the initial Hamiltonian (beta = 1) with the initial work increment from Step 2 and the product of N tilted transition matrices from Step 3, contracted against the all-ones vector. This is the core quantity minimized (with alpha = 2) over the free intermediate Hamiltonians in Steps 6-7 to obtain the MSE'-optimal switching protocol.

Evaluating the Jarzynski-based error proxy requires the exponential work average ⟨e^(−αW)⟩ over the full non-equilibrium trajectory, not just a single transition. The source method shows this can be computed exactly (Sec. II C, Eqs. 16–17) by drawing the initial state from the equilibrium distribution of the initial Hamiltonian, p^eq(x)∝e^(−H(x,0)), tilting it by the first work increment (Step 2), and then propagating this tilted vector forward through the chain of tilted transition matrices from Step 3 — one per time step — before finally summing over the last free state (equivalent to contracting with the all-ones vector, since the untilted transition probabilities out of the last state sum to 1). This function is the complete forward pass of the pipeline: for α=2 its output *is* the MSE′ proxy that Steps 5–7 compare against and minimize.

Returns
-------
float, the exact generalized work average ⟨e^(−αW)⟩ for the given Hamiltonian trajectory, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exponential_work_average(H: np.ndarray, alpha: float) -> float:
    '''Compute the exact generalized work average <exp(-alpha*W)>.

    Parameters
    ----------
    H : np.ndarray
        2D array of shape (m, N+2), giving the full Hamiltonian trajectory
        H(x, tau) for each state x = 1, ..., m (rows) and each time step
        tau = 0, ..., N+1 (columns, in increasing time order). Column 0 is
        the initial Hamiltonian and column N+1 is the final Hamiltonian;
        any columns strictly in between are the intermediate Hamiltonians.
        Reduced units with inverse temperature beta = 1 are used throughout,
        so the initial equilibrium distribution is p_eq(x) proportional to
        exp(-H(x, 0)).
    alpha : float
        Tilting exponent in the generalized work average <exp(-alpha*W)>.
        alpha = 1 recovers the plain Jarzynski average <exp(-W)>; alpha = 2
        gives the MSE' error proxy used for optimization.

    Returns
    -------
    result : float
        Native Python float, the exact value of <exp(-alpha*W)> for the
        given Hamiltonian trajectory, computed via the tilted master
        equation (Eq. 16-17).

    Raises
    ------
    ValueError
        If H is not convertible to a two-dimensional array of real numbers,
        if H has fewer than 2 rows (states) or fewer than 2 columns (time
        steps), if H contains any NaN or infinite value, or if alpha is not
        convertible to a finite real scalar.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_exponential_work_average(H: np.ndarray, alpha: float) -> float:
    """Reference implementation."""

    
    try:
        H_arr = np.asarray(H, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("H must be convertible to an array of real numbers.")

    if H_arr.ndim != 2:
        raise ValueError("H must be two-dimensional.")
    m, T = H_arr.shape
    if m < 2:
        raise ValueError("H must have at least 2 states (rows).")
    if T < 2:
        raise ValueError("H must have at least 2 time steps (columns).")
    if not np.all(np.isfinite(H_arr)):
        raise ValueError("H must not contain NaN or infinite values.")

    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("alpha must be convertible to a real scalar.")
    if not np.isfinite(alpha_val):
        raise ValueError("alpha must be a finite real number.")

    w0 = np.exp(-H_arr[:, 0])
    peq = w0 / np.sum(w0)
    dW = _oracle_work_increments(H_arr)              # chains to Step 2
    vec = peq * np.exp(-alpha_val * dW[:, 0])

    n_intermediate = T - 2
    for tau in range(n_intermediate):
        pi_tilde = _oracle_tilted_transition_matrix(  # chains to Step 3
            H_arr[:, tau + 1], H_arr[:, tau + 2], alpha_val
        )
        vec = vec @ pi_tilde

    return float(np.sum(vec))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: two-state trajectory with several intermediate steps, alpha=2
            "setup": (
                "import numpy as np\n"
                "H = np.array([\n"
                "    [0.0, 0.0, 0.0, 0.0, 0.0],\n"
                "    [-8.0, -5.0, -2.0, 3.0, 8.0],\n"
                "])\n"
                "alpha = 2.0"
            ),
            "call": "exponential_work_average(H, alpha)",
            "gold_call": "_oracle_exponential_work_average(H, alpha)",
        },
        {
            # Boundary case: N=0, i.e. only the initial and final Hamiltonian columns,
            # no intermediate steps at all
            "setup": (
                "import numpy as np\n"
                "H = np.array([[0.0, 0.0], [-8.0, 8.0]])\n"
                "alpha = 1.0"
            ),
            "call": "exponential_work_average(H, alpha)",
            "gold_call": "_oracle_exponential_work_average(H, alpha)",
        },
        {
            # Edge case: three-state chain, alpha=0 must give exactly 1.0
            # (<exp(0)> = 1 regardless of the trajectory)
            "setup": (
                "import numpy as np\n"
                "H = np.array([\n"
                "    [0.0, 1.0, -1.0, 0.0],\n"
                "    [2.0, 0.5, -0.5, -2.0],\n"
                "    [4.0, 2.0, 1.0, -4.0],\n"
                "])\n"
                "alpha = 0.0"
            ),
            "call": "exponential_work_average(H, alpha)",
            "gold_call": "_oracle_exponential_work_average(H, alpha)",
        },
    ]
