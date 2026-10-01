"""
Builds the exponentially "tilted" version of a single-time-step Glauber transition matrix, as used in the tilted master-equation representation of the source method's Eqs. 16-17. Given the Hamiltonian values at two consecutive time steps, this step reuses the untilted transition matrix from Step 1 (evaluated at the earlier time step) andreweights each column by exp(-alpha * delta_W(y, s)), where delta_W is the per-state work increment between the two time steps (Step 2). Chaining these tilted matrices together (one per time step) is what allows the exact, noise-free evaluation of the generalized work average <exp(-alpha*W)> that Step 4 builds the MSE' optimization objective from.

To exactly evaluate exponential work averages ⟨e^(−αW)⟩ without resorting to noisy Monte Carlo trajectory sampling, the source method reweights the Markov chain's transition matrices by the exponentiated work increment associated with the destination state at each step (Sec. II C, Eq. 16). Concretely, the tilted matrix element is π̃(x→y;τ,α) = π(x→y;τ)·e^(−αδW(y,τ)), where π(x→y;τ) is built (Step 1) from the Hamiltonian at time τ+1, and δW(y,τ)=H(y,τ+2)−H(y,τ+1) is the work increment carried by the *next* transition. Chaining these tilted matrices in sequence — one per time step — turns the full path integral defining ⟨e^(−αW)⟩ into a simple matrix-vector product (Eq. 17), which is both exact and differentiable, and is therefore the object Step 4's optimization needs to construct once per candidate set of intermediate Hamiltonians.

Returns
-------
numpy.ndarray of shape (m, m) and dtype float64, the alpha-tilted Glauber transition matrix π̃(τ,α)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tilted_transition_matrix(
    energies_here: np.ndarray, energies_ahead: np.ndarray, alpha: float
) -> np.ndarray:
    '''Build the alpha-tilted Glauber transition matrix for one time step.

    Parameters
    ----------
    energies_here : np.ndarray
        1D array of length m giving H(x, s), the Hamiltonian values at time
        step s. Used both to build the untilted Glauber transition matrix
        (as in Step 1) and as the earlier reference point of the
        destination-state work increment.
    energies_ahead : np.ndarray
        1D array of length m giving H(x, s+1), the Hamiltonian one time step
        later than energies_here. Used together with energies_here to
        compute delta_W(y, s) = H(y, s+1) - H(y, s) for each destination
        state y. Must have the same length as energies_here.
    alpha : float
        Tilting exponent used to reweight each transition by
        exp(-alpha * delta_W(y, s)).

    Returns
    -------
    pi_tilde : np.ndarray
        m x m array of dtype float64. pi_tilde[i, j] equals the untilted
        Glauber transition probability pi(i+1 -> j+1) built from
        energies_here (Step 1), multiplied by
        exp(-alpha * (energies_ahead[j] - energies_here[j])).

    Raises
    ------
    ValueError
        If energies_here or energies_ahead is not convertible to a
        one-dimensional array of real numbers, if they do not have the same
        length, if either has fewer than 2 elements, if either contains any
        NaN or infinite value, or if alpha is not convertible to a finite
        real scalar.
    '''
    return pi_tilde  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_tilted_transition_matrix(
    energies_here: np.ndarray, energies_ahead: np.ndarray, alpha: float
) -> np.ndarray:
    """Reference implementation."""
    
    try:
        e_here = np.asarray(energies_here, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("energies_here must be convertible to an array of real numbers.")
    try:
        e_ahead = np.asarray(energies_ahead, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("energies_ahead must be convertible to an array of real numbers.")

    if e_here.ndim != 1 or e_ahead.ndim != 1:
        raise ValueError("energies_here and energies_ahead must be one-dimensional.")
    if e_here.shape[0] != e_ahead.shape[0]:
        raise ValueError("energies_here and energies_ahead must have the same length.")
    if e_here.shape[0] < 2:
        raise ValueError("energies_here and energies_ahead must contain at least 2 states.")
    if not np.all(np.isfinite(e_here)) or not np.all(np.isfinite(e_ahead)):
        raise ValueError("energies_here and energies_ahead must not contain NaN or infinite values.")

    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("alpha must be convertible to a real scalar.")
    if not np.isfinite(alpha_val):
        raise ValueError("alpha must be a finite real number.")

    pi = _oracle_glauber_transition_matrix(e_here)
    delta_w = e_ahead - e_here
    weight = np.exp(-alpha_val * delta_w)
    return pi * weight[np.newaxis, :]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: two-state system, alpha=2 (the MSE' exponent used later)
            "setup": (
                "import numpy as np\n"
                "energies_here = np.array([0.0, -8.0])\n"
                "energies_ahead = np.array([0.0, -5.4])\n"
                "alpha = 2.0"
            ),
            "call": "tilted_transition_matrix(energies_here, energies_ahead, alpha)",
            "gold_call": "_oracle_tilted_transition_matrix(energies_here, energies_ahead, alpha)",
        },
        {
            # Boundary case: alpha=0 must reduce exactly to the untilted matrix from Step 1
            "setup": (
                "import numpy as np\n"
                "energies_here = np.array([0.0, 3.0, -2.0])\n"
                "energies_ahead = np.array([0.0, 1.0, -1.0])\n"
                "alpha = 0.0"
            ),
            "call": "tilted_transition_matrix(energies_here, energies_ahead, alpha)",
            "gold_call": "_oracle_tilted_transition_matrix(energies_here, energies_ahead, alpha)",
        },
        {
            # Edge case: large work increment combined with large negative alpha,
            # producing large but finite tilting weights
            "setup": (
                "import numpy as np\n"
                "energies_here = np.array([0.0, -8.0])\n"
                "energies_ahead = np.array([0.0, 8.0])\n"
                "alpha = -2.0"
            ),
            "call": "tilted_transition_matrix(energies_here, energies_ahead, alpha)",
            "gold_call": "_oracle_tilted_transition_matrix(energies_here, energies_ahead, alpha)",
        },
    ]
