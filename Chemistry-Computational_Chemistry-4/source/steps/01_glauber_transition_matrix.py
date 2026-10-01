"""
Builds the single-time-step Glauber transition matrix for a linear chain of m states, as defined by the source method's Eqs. 6-8. Given the Hamiltonian values H(x, tau+1) governing the jump out of time step tau, this step computes the row-stochastic, tridiagonal transition matrix pi(tau) with nearest-neighbor proposal probability g(x->y)=1/2 and Glauber acceptance probability f(dE)=1/(1+exp(dE)). This matrix is the core primitive reused, unmodified, by every later step in the pipeline: the equilibrium-distribution propagator, the exponentially tilted matrices used to evaluate the exact work average <exp(-alpha*W)>, and ultimately the MSE' objective that is minimized over the intermediate Hamiltonians.

The source method's discrete-time Markov models (Sec. II A, Eqs. 6–8) place m states on a line and evolve them under Glauber dynamics: a state x may only propose a jump to a nearest neighbor y=x±1, with proposal probability g(x→y)=1/2, and the jump is accepted with the Glauber probability f(ΔE)=1/(1+e^ΔE), where ΔE=H(y,τ+1)−H(x,τ+1) is evaluated using the Hamiltonian at the *upcoming* time step (work is done on the system immediately before it jumps). The remaining "stay" probability π(x→x;τ) is fixed by normalization. This tridiagonal, row-stochastic transition matrix is the single computational primitive every later step in the pipeline is built from: it is reused unmodified to build the equilibrium distribution's forward propagator, the tilted (exponentially reweighted) matrices used to evaluate ⟨e^(−αW)⟩ exactly, and ultimately the optimization objective MSE′ that is minimized over the intermediate energies. Because the acceptance function is evaluated on energy *differences* only, the gauge invariance underlying the whole method (Sec. II C) is a direct consequence of this construction: shifting every entry of `energies` by the same constant leaves the returned matrix unchanged

Returns
-------
numpy.ndarray of shape (m, m) and dtype float64, the row-stochastic Glauber transition matrix π(τ) built from `energies
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def glauber_transition_matrix(energies: np.ndarray) -> np.ndarray:
    '''Build the Glauber transition matrix for a linear chain of states.

    Parameters
    ----------
    energies : np.ndarray
        1D array of length m >= 2 giving the Hamiltonian value H(x, tau+1)
        for each state x = 1, ..., m (in order along the chain), i.e. the
        Hamiltonian that governs the transitions out of time step tau.

    Returns
    -------
    pi : np.ndarray
        m x m array of dtype float64. pi[i, j] is the Glauber transition
        probability from state (i+1) to state (j+1). Transitions are only
        nonzero between nearest neighbors on the chain (|i - j| == 1); each
        row sums to 1, with the diagonal entry pi[i, i] absorbing the
        probability of remaining in state (i+1).

    Raises
    ------
    ValueError
        If `energies` is not convertible to a one-dimensional array of real
        numbers, if it has fewer than 2 elements, or if it contains any NaN
        or infinite value.
    '''
    return pi  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_glauber_transition_matrix(energies: np.ndarray) -> np.ndarray:
    """Reference implementation."""

    def _glauber_f(delta_e: np.ndarray) -> np.ndarray:
        """Numerically stable Glauber acceptance function f(dE) = 1 / (1 + exp(dE))."""
        out = np.empty_like(delta_e, dtype=np.float64)
        pos = delta_e >= 0
        out[pos] = np.exp(-delta_e[pos]) / (1.0 + np.exp(-delta_e[pos]))
        out[~pos] = 1.0 / (1.0 + np.exp(delta_e[~pos]))
        return out

    try:
        e = np.asarray(energies, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("energies must be convertible to an array of real numbers.")

    if e.ndim != 1:
        raise ValueError("energies must be one-dimensional.")
    m = e.shape[0]
    if m < 2:
        raise ValueError("energies must contain at least 2 states.")
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must not contain NaN or infinite values.")

    pi = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        row_sum = 0.0
        if i - 1 >= 0:
            p = 0.5 * _glauber_f(np.array([e[i - 1] - e[i]]))[0]
            pi[i, i - 1] = p
            row_sum += p
        if i + 1 < m:
            p = 0.5 * _glauber_f(np.array([e[i + 1] - e[i]]))[0]
            pi[i, i + 1] = p
            row_sum += p
        pi[i, i] = 1.0 - row_sum
    return pi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: two-state system, one state far below the other in energy
            "setup": "import numpy as np\nenergies = np.array([0.0, -3.0])",
            "call": "glauber_transition_matrix(energies)",
            "gold_call": "_oracle_glauber_transition_matrix(energies)",
        },
        {
            # Boundary case: three-state chain with a very large energy gap,
            # driving one acceptance probability close to 0 and another close to 1
            "setup": "import numpy as np\nenergies = np.array([0.0, 25.0, -25.0])",
            "call": "glauber_transition_matrix(energies)",
            "gold_call": "_oracle_glauber_transition_matrix(energies)",
        },
        {
            # Edge case: all energies equal -> uniform 1/2, 1/2 hopping to neighbors,
            # and non-adjacent entries must be exactly zero
            "setup": "import numpy as np\nenergies = np.array([1.5, 1.5, 1.5, 1.5])",
            "call": "glauber_transition_matrix(energies)",
            "gold_call": "_oracle_glauber_transition_matrix(energies)",
        },
    ]
