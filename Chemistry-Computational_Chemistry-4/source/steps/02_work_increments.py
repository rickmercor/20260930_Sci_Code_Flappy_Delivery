"""
Computes delta_W(x, tau) = H(x, tau+1) - H(x, tau) for every state x andevery time step tau = 0, ..., N (source method's Eq. 11), the elementary work performed on the system when the Hamiltonian is updated immediately before each stochastic transition. This is the quantity summed over tau to obtain the total trajectory work (Eq. 12), and its per-step values are what Step 4 uses to tilt the initial equilibrium vector.

Work is performed on the system whenever the Hamiltonian is updated between transitions, not through the stochastic jump itself. For a state occupied at time τ, the work increment δW(x,τ)=H(x,τ+1)−H(x,τ) quantifies this per-step energy injection; summing δW(x_τ,τ) over τ=0,…,N along a realized trajectory gives the total work W entering the Jarzynski relation. Computing this increment for every state at every time step (rather than only along one sampled trajectory) is what later lets the tilted-master-equation approach reweight the *entire* distribution exactly, instead of reweighting individual sampled paths.

Returns
-------
numpy.ndarray of shape (m, T-1) and dtype float64, the per-state work increments δW(x,τ)=H(x,τ+1)−H(x,τ)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def work_increments(H: np.ndarray) -> np.ndarray:
    '''Compute per-state work increments along a Hamiltonian trajectory.

    Parameters
    ----------
    H : np.ndarray
        2D array of shape (m, T), giving the Hamiltonian trajectory
        H(x, tau) for each state x = 1, ..., m (rows) and each time step
        tau = 0, ..., T-1 (columns, increasing in time).

    Returns
    -------
    dW : np.ndarray
        Array of shape (m, T-1) and dtype float64, where dW[:, tau] =
        H(:, tau+1) - H(:, tau) for tau = 0, ..., T-2.

    Raises
    ------
    ValueError
        If H is not convertible to a two-dimensional array of real numbers,
        if H has fewer than 2 rows (states) or fewer than 2 columns (time
        steps), or if H contains any NaN or infinite value.
    '''
    return dW  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_work_increments(H: np.ndarray) -> np.ndarray:
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

    return H_arr[:, 1:] - H_arr[:, :-1]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nH = np.array([[0.0,0.0,0.0,0.0],[-8.0,-5.0,-2.0,8.0]])",
            "call": "work_increments(H)",
            "gold_call": "_oracle_work_increments(H)",
        },
        {
            # Boundary case: only two time steps (T=2), a single increment
            "setup": "import numpy as np\nH = np.array([[0.0,0.0],[-8.0,8.0]])",
            "call": "work_increments(H)",
            "gold_call": "_oracle_work_increments(H)",
        },
        {
            # Edge case: constant Hamiltonian -> all increments exactly zero
            "setup": "import numpy as np\nH = np.array([[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0]])",
            "call": "work_increments(H)",
            "gold_call": "_oracle_work_increments(H)",
        },
    ]
