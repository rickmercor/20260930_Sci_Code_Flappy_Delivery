"""
Step 04: Non-participation probability of a target level.

Probability that a target level has not yet taken part in the flow of probability, along a population history.

For a population history rho(t_k), k = 0 .. K - 1, the minimal-flow transition matrices T^(k) of the previous step
describe how probability moves between the levels over each sampling interval. Choose a target level M that is empty
at t_0. The probability that has never entered M evolves with the reduced matrices obtained by deleting row M and
column M from every T^(k): starting from the vector p(t_0) of the initial populations of all levels other than M, it is
propagated as p(t_(k+1)) = T~^(k) p(t_k). Probability that flows into M is removed, and probability that later flows
out of M is not added back, because it has already visited M. The non-participation probability is
P_not(t_k) = sum of the entries of p(t_k); it starts at the total initial population and never increases.

Read as a survival probability of the empty target, 1 - P_not(t_k) is the statistical confidence that the target has
participated in the flow of probability by t_k.

Returns
-------
numpy.ndarray of shape (K,), probability that the target level has not yet participated in the flow of probability at each sample
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonparticipation_probability(populations: "np.ndarray", target: int) -> "np.ndarray":
    '''Probability that the target level has not yet participated in the flow of probability, at every sample.

    Parameters
    ----------
    populations : np.ndarray
        Shape (K, n) with K >= 2 and n >= 2; non-negative populations with equal row sums, as for the minimal-flow
        transition matrices.
    target : int
        Index M of the target level, 0 <= M < n, whose population at the first sample is zero.

    Returns
    -------
    result : np.ndarray
        Shape (K,); element k is P_not(t_k), the summed reduced probability vector after k intervals.

    Raises
    ------
    ValueError
        If target is out of range or the target population at the first sample is larger than 1e-12, and for the
        same invalid population histories as the minimal-flow transition matrices.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_nonparticipation_probability(populations: "np.ndarray", target: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    rho = np.asarray(populations, dtype=float)
    if rho.ndim != 2:
        raise ValueError("populations must be a 2-D array")
    n = rho.shape[1]
    m = int(target)
    if not 0 <= m < n:
        raise ValueError("target out of range")
    if rho[0, m] > 1e-12:
        raise ValueError("the target level must be empty at the first sample")
    matrices = _oracle_least_flow_transition_matrices(rho)
    keep = np.array([k for k in range(n) if k != m])
    reduced = np.ascontiguousarray(matrices[:, keep][:, :, keep])
    states = _apply_sequence(reduced, rho[0, keep])
    return states.sum(axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: three-level history in which the target fills, empties and refills ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([[1.0, 0.0, 0.0], [0.8, 0.15, 0.05], [0.7, 0.1, 0.2], [0.75, 0.2, 0.05],\n"
                     "              [0.6, 0.1, 0.3], [0.65, 0.25, 0.1]])\n",
            "call": "nonparticipation_probability(p, 2)",
            "gold_call": "_oracle_nonparticipation_probability(p, 2)",
            "tol": 1e-12,
        },
        # --- Normal: dense history of a four-level quantum beat with fast oscillations, target level 3 ---
        {
            "setup": "import numpy as np\n"
                     "h = np.array([[0.0, 0.004, 0.0, 0.0], [0.004, 0.0181, 0.005, 0.0], [0.0, 0.005, 0.0355, 0.006],\n"
                     "              [0.0, 0.0, 0.006, 0.0521]])\n"
                     "w, v = np.linalg.eigh(h)\n"
                     "t = np.arange(0.0, 4001.0, 2.0)\n"
                     "c = (v * np.exp(-1j * np.outer(t, w))[:, None, :]) @ v[0]\n"
                     "p = np.abs(c) ** 2\n",
            "call": "nonparticipation_probability(p, 3)[::25]",
            "gold_call": "_oracle_nonparticipation_probability(p, 3)[::25]",
            "tol": 1e-9,
        },
        # --- Boundary: target in the middle of a four-level ladder with two-in/two-out intervals ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([[0.6, 0.0, 0.3, 0.1], [0.5, 0.05, 0.2, 0.25], [0.45, 0.02, 0.33, 0.2],\n"
                     "              [0.3, 0.2, 0.3, 0.2], [0.4, 0.1, 0.25, 0.25]])\n",
            "call": "nonparticipation_probability(p, 1)",
            "gold_call": "_oracle_nonparticipation_probability(p, 1)",
            "tol": 1e-12,
        },
        # --- Edge: the target never gains, so the non-participation probability stays at the total population ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([[0.7, 0.3, 0.0], [0.6, 0.4, 0.0], [0.9, 0.1, 0.0], [0.5, 0.5, 0.0]])\n",
            "call": "nonparticipation_probability(p, 2)",
            "gold_call": "_oracle_nonparticipation_probability(p, 2)",
            "tol": 1e-12,
        },
        # --- Boundary: target is level 0 and an intermediate level is emptied completely twice ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([[0.0, 0.2, 0.8], [0.1, 0.0, 0.9], [0.05, 0.25, 0.7], [0.3, 0.3, 0.4],\n"
                     "              [0.2, 0.0, 0.8], [0.4, 0.35, 0.25]])\n",
            "call": "nonparticipation_probability(p, 0)",
            "gold_call": "_oracle_nonparticipation_probability(p, 0)",
            "tol": 1e-12,
        },
        # --- Normal: six-level history with large jumps, initially spread over five levels, target level 2 ---
        {
            "setup": "import numpy as np\n"
                     "x = np.random.default_rng(7).random((40, 6))\n"
                     "p = x / x.sum(axis=1, keepdims=True)\n"
                     "p[0] = [0.3, 0.2, 0.0, 0.1, 0.25, 0.15]\n",
            "call": "nonparticipation_probability(p, 2)",
            "gold_call": "_oracle_nonparticipation_probability(p, 2)",
            "tol": 1e-12,
        },
        # --- Error: a target that already holds population must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([[0.9, 0.1], [0.8, 0.2]]), 1)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(nonparticipation_probability)",
            "gold_call": "_probe(_oracle_nonparticipation_probability)",
        },
    ]
