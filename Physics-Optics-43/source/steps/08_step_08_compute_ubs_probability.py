"""
Orchestrate the full unified boson sampling calculation from the raw device specification.

Given the interferometer gate list, the squeezing vector, an input photon pattern and a detected output pattern, return the unified boson sampling transition probability. The pipeline is: build the composite symplectic matrix from the gates and squeezing; for each required value of x evaluate the generating-function blocks; assemble G and take the weighted Hafnian sum for the detection pattern; normalize by |det U| and the square root of the product of the D_i; and differentiate in x to the orders set by the input pattern, dividing by n!.

This assembles the full unified boson sampler: photon-number states are injected into modes that are also squeezed, the modes are mixed by a passive interferometer, and photon-number-resolving detectors project onto an output pattern. The resulting distribution interpolates between scattershot sampling, whose probabilities are squared permanents, and Gaussian sampling, whose probabilities are squared Hafnians.

The benchmark configuration places two photons in a single input mode, which keeps several intermediate photon-number patterns in play and prevents the probability from collapsing onto a permanent, while leaving the total photon number small enough that the construction stays in the regime verified against exact Fock-space simulation.

Returns
-------
float: the transition probability P(n -> m) for the specified device.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def compute_ubs_probability(gates: list, r_list: list, n_pattern: list, m_pattern: list) -> float:
    """Compute the unified boson sampling probability for the given setup.
 
    Parameters
    ----------
    gates : list
        Ordered list of (p, q, theta, phi) interferometer gates.
    r_list : list
        List of M real squeezing parameters.
    n_pattern : list
        Sequence of M integers in {0, 1, 2}, the input pattern.
    m_pattern : list
        Sequence of M non-negative integers, the detected pattern.
 
    Returns
    -------
    probability : float
        The transition probability P(n -> m).
 
    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-07 (``build_composite_symplectic``,
    ``generating_blocks``, ``assemble_g_submatrix``,
    ``fock_projection_sum``, ``core_generating_function``,
    ``richardson_x_derivative``, ``ubs_probability``) and feed each returned
    value into the next, rather than reimplementing them. Include every
    import your implementation needs (for example ``import numpy as np``)
    inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_ubs_probability(gates: list, r_list: list, n_pattern: list, m_pattern: list) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
 
    M = len(r_list)
    if M == 0:
        raise ValueError("r_list must contain at least one mode")
    if len(n_pattern) != M:
        raise ValueError("n_pattern must have length M")
    if len(m_pattern) != M:
        raise ValueError("m_pattern must have length M")
 
    for n in n_pattern:
        if int(n) != n or int(n) not in (0, 1, 2):
            raise ValueError("n_pattern entries must be 0, 1 or 2")
 
    # -- Sub-problem 01: composite symplectic matrix of the Gaussian stage.
    T = _oracle_build_composite_symplectic(gates, r_list)
 
    def F(x):
        # -- Sub-problem 02: generating-function blocks at this x.
        D_diag, A, B = _oracle_generating_blocks(T, x)
        # -- Sub-problems 03-04: submatrix selection and weighted Hafnian sum.
        Wsum = _oracle_fock_projection_sum(A, B, m_pattern)
        # -- Sub-problem 05: normalization by |det U| and sqrt(prod D_i).
        norm = abs(np.linalg.det(T[:M, :M])) * np.sqrt(np.prod(D_diag))
        return Wsum / norm
 
    # -- Sub-problem 06: Richardson-refined derivative in the generating
    #    variables, divided by n!.  Sub-problem 07 wraps exactly this
    #    combination, so the two must agree.
    probability = float(_oracle_richardson_x_derivative(F, n_pattern, 1e-3).real)
 
    # -- Sub-problem 07: same quantity through the single-call interface.
    checked = _oracle_ubs_probability(T, n_pattern, m_pattern)
    if not np.isclose(probability, checked, rtol=1e-9, atol=1e-12):
        raise ValueError("chained pipeline disagrees with the step-07 interface")
    return probability

# =============================================================================
# TEST CASES
# =============================================================================

_BENCH = """gates = [(0, 1, 0.50, 0.30), (2, 3, 0.90, 1.40), (4, 5, 1.20, 0.60),
         (1, 2, 0.70, 1.90), (3, 4, 1.10, 0.80), (0, 5, 0.40, 2.20)]
r_list = [0.30, 0.45, 0.60, 0.35, 0.50, 0.40]
"""
 
 
def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark configuration end to end (normal scenario) ---
        # The final value itself is scored by the rubric, so it is deliberately
        # not pinned here; this checks the chain against the oracle instead.
        {
            "setup": _BENCH + "n = [0, 0, 0, 2, 0, 0]\nm = [0, 1, 1, 0, 0, 0]\n",
            "call": "compute_ubs_probability(gates, r_list, n, m)",
            "gold_call": "_oracle_compute_ubs_probability(gates, r_list, n, m)",
        },
        # --- Valid: a different input mode, checked against the oracle ---
        {
            "setup": _BENCH + "n = [0, 2, 0, 0, 0, 0]\nm = [1, 0, 0, 0, 1, 0]\n",
            "call": "compute_ubs_probability(gates, r_list, n, m)",
            "gold_call": "_oracle_compute_ubs_probability(gates, r_list, n, m)",
        },
        # --- Edge: zero squeezing with vacuum input and vacuum detection is certain ---
        {
            "setup": """gates = [(0, 1, 0.8, 1.2), (1, 2, 0.3, 0.5)]
r_list = [0.0, 0.0, 0.0]
n = [0, 0, 0]
m = [0, 0, 0]
""",
            "call": "bool(abs(compute_ubs_probability(gates, r_list, n, m) - 1.0) < 1e-10)",
            "gold_call": 'bool(abs(_oracle_compute_ubs_probability(gates, r_list, n, m) - 1.0) < 1e-10)',
        },
        # --- Boundary: forbidden by the parity selection rule ---
        {
            "setup": _BENCH + "n = [0, 0, 0, 2, 0, 0]\nm = [1, 1, 1, 0, 0, 0]\n",
            "call": "bool(abs(compute_ubs_probability(gates, r_list, n, m)) < 1e-8)",
            "gold_call": 'bool(abs(_oracle_compute_ubs_probability(gates, r_list, n, m)) < 1e-8)',
        },
        # --- Valid: smaller four-mode configuration, checked against the oracle ---
        {
            "setup": """gates = [(0, 1, 0.6, 0.4), (2, 3, 1.0, 1.3), (1, 2, 0.8, 0.9), (0, 3, 1.1, 0.5)]
r_list = [0.5, 0.35, 0.6, 0.45]
n = [2, 0, 0, 0]
m = [1, 0, 1, 0]
""",
            "call": "compute_ubs_probability(gates, r_list, n, m)",
            "gold_call": "_oracle_compute_ubs_probability(gates, r_list, n, m)",
        },
    ]
