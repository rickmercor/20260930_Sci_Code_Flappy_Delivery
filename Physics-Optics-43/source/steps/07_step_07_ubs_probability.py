"""
Compute the unified boson sampling transition probability for one input and output pattern.

The probability is the trace of the output projector against the evolved input projector. Writing the input projector through its generating function turns that trace into derivatives in x of a normally ordered Gaussian expression, while the output projection supplies the weighted Hafnian sum. Combining the previous steps,

    P(n -> m) = [ d^n / dx^n  F(x) ]_{x = 0} / n!,

with F(x) the normalized generating function and n! = prod_i n_i!. The returned probability is the real part of the differentiated expression; the imaginary part is a numerical residue and should be negligible.

The two limits of the unified sampler sit at the ends of this construction. With no input photons no derivatives are taken and the expression collapses to the Gaussian sampling result. With no squeezing the Hafnian structure degenerates and a permanent of a submatrix of the interferometer remains. Between the two, both contribute and the probability is neither a Hafnian nor a permanent.

A selection rule follows from the fact that squeezing creates and destroys photons in pairs: P vanishes unless the total detected photon number differs from the total input photon number by an even integer. This is a strong and cheap correctness check. Note also that a collision-free input pattern with equal input and output totals degenerates, because per-mode parity then forces a unique intermediate pattern and the probability reduces to a permanent expression; placing two photons in a single input mode avoids that. This task is scoped to input and output patterns with at most two photons in total, the regime in which the construction has been verified against exact Fock-space simulation.

Returns
-------
float: the transition probability P(n -> m).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def ubs_probability(T: np.ndarray, n_pattern: list, m_pattern: list, h: float = 1e-3) -> float:
    """Compute the unified boson sampling transition probability.
 
    Parameters
    ----------
    T : np.ndarray
        Complex array of shape (2M, 2M) in the block layout
        [[U, V], [conj(V), conj(U)]].
    n_pattern : list
        Sequence of M integers in {0, 1, 2}, the input photon pattern.
    m_pattern : list
        Sequence of M non-negative integers, the detected pattern.
    h : float, optional
        Base finite-difference step passed to the derivative routine.
 
    Returns
    -------
    probability : float
        The transition probability P(n -> m).
 
    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ubs_probability(T: np.ndarray, n_pattern: list, m_pattern: list, h: float = 1e-3) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
 
    T = np.asarray(T, dtype=complex)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be a square matrix")
    if T.shape[0] == 0 or T.shape[0] % 2 != 0:
        raise ValueError("T must have even dimension 2M with M >= 1")
    M = T.shape[0] // 2
    if len(n_pattern) != M:
        raise ValueError("n_pattern must have length M")
    if len(m_pattern) != M:
        raise ValueError("m_pattern must have length M")
 
    def f(x):
        return _oracle_core_generating_function(T, x, m_pattern)
 
    return float(_oracle_richardson_x_derivative(f, n_pattern, h).real)

# =============================================================================
# TEST CASES
# =============================================================================

_BENCH_T = """import numpy as np
M = 6
W = np.eye(M, dtype=complex)
for (p, q, t, f) in [(0, 1, 0.50, 0.30), (2, 3, 0.90, 1.40), (4, 5, 1.20, 0.60),
                     (1, 2, 0.70, 1.90), (3, 4, 1.10, 0.80), (0, 5, 0.40, 2.20)]:
    G0 = np.eye(M, dtype=complex)
    G0[p, p] = np.exp(1j * f) * np.cos(t); G0[p, q] = -np.sin(t)
    G0[q, p] = np.exp(1j * f) * np.sin(t); G0[q, q] = np.cos(t)
    W = G0 @ W
r = np.array([0.30, 0.45, 0.60, 0.35, 0.50, 0.40])
U = W @ np.diag(np.cosh(r)); V = W @ np.diag(np.sinh(r))
T = np.block([[U, V], [V.conj(), U.conj()]])
"""
 
 
def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark transition of the full task (normal scenario) ---
        {
            "setup": _BENCH_T + "n = [0, 0, 0, 2, 0, 0]\nm = [0, 1, 1, 0, 0, 0]\n",
            "call": "ubs_probability(T, n, m)",
            "gold_call": "_oracle_ubs_probability(T, n, m)",
        },
        # --- Valid: vacuum input reproduces the Gaussian-limit probability ---
        {
            "setup": _BENCH_T + "n = [0, 0, 0, 0, 0, 0]\nm = [0, 1, 1, 0, 0, 0]\n",
            "call": "ubs_probability(T, n, m)",
            "gold_call": "_oracle_ubs_probability(T, n, m)",
        },
        # --- Boundary: parity selection rule forbids an odd change in photon number ---
        {
            "setup": _BENCH_T + "n = [0, 0, 0, 2, 0, 0]\nm = [0, 1, 1, 1, 0, 0]\n",
            "call": "bool(abs(ubs_probability(T, n, m)) < 1e-8)",
            "gold_call": 'bool(abs(_oracle_ubs_probability(T, n, m)) < 1e-8)',
        },
        # --- Edge: a different doubly occupied input mode, checked against the oracle ---
        {
            "setup": _BENCH_T + "n = [2, 0, 0, 0, 0, 0]\nm = [1, 0, 0, 1, 0, 0]\n",
            "call": "ubs_probability(T, n, m)",
            "gold_call": "_oracle_ubs_probability(T, n, m)",
        },
    ]
