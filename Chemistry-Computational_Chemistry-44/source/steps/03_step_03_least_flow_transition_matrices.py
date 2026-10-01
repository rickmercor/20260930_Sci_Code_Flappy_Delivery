"""
Step 03: Minimal-flow transition matrices of a population history.

Transition matrices that move probability between levels as little as possible over each sampling interval.

A population history rho_v(t_k), k = 0 .. K - 1, of n levels (each row sums to the same total) says how much
probability each level holds at the sample times, but not where the probability went. Over the interval from t_k to
t_(k+1) the changes are d_v = rho_v(t_(k+1)) - rho_v(t_k). A transition matrix T with rho(t_(k+1)) = T rho(t_k),
whose entry T_ij is the fraction of the probability in level j at t_k that ends up in level i at t_(k+1), is fixed by
three conditions:

  1. each column sums to one (probability is conserved);
  2. probability leaves only levels whose population falls over the interval (d_j < 0) and enters only levels whose
     population rises (d_i > 0), and it flows directly from the falling to the rising levels, so that no more
     probability moves than the changes require; a level whose population rises or stays the same keeps all of its
     probability, T_jj = 1 and T_ij = 0 for i != j;
  3. when several levels rise and several fall at once, the amount -d_j leaving a falling level j is shared among the
     rising levels in proportion to their gains d_i, measured against the total gain S = sum over rising levels of
     d_i.

A falling level j therefore keeps the fraction 1 + d_j / rho_j(t_k) of its probability. If no population changes, T is
the identity.

Returns
-------
numpy.ndarray of shape (K - 1, n, n), minimal-flow transition matrices of the successive sampling intervals
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def least_flow_transition_matrices(populations: "np.ndarray") -> "np.ndarray":
    '''Minimal-flow transition matrices for every interval of a population history.

    Parameters
    ----------
    populations : np.ndarray
        Shape (K, n) with K >= 2 and n >= 2; row k holds the non-negative level populations at sample k, and all rows
        have the same sum.

    Returns
    -------
    result : np.ndarray
        Shape (K - 1, n, n); element [k, i, j] is the fraction of the probability in level j at sample k that is in
        level i at sample k + 1.

    Raises
    ------
    ValueError
        If populations is not a 2-D array with at least two rows and two columns, has a negative entry, or has row
        sums that differ by more than 1e-6.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_least_flow_transition_matrices(populations: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    rho = np.asarray(populations, dtype=float)
    if rho.ndim != 2 or rho.shape[0] < 2 or rho.shape[1] < 2:
        raise ValueError("populations must have shape (K, n) with K >= 2 and n >= 2")
    if np.any(rho < 0.0):
        raise ValueError("populations must be non-negative")
    totals = rho.sum(axis=1)
    if np.max(np.abs(totals - totals[0])) > 1e-6:
        raise ValueError("all rows must have the same total population")
    before = rho[:-1]
    change = np.diff(rho, axis=0)
    gain = np.where(change > 0.0, change, 0.0)
    loss = np.where(change < 0.0, -change, 0.0)
    total_gain = gain.sum(axis=1)
    falling = loss > 0.0
    fraction = np.where(falling, loss / np.where(falling, before, 1.0), 0.0)
    share = fraction / np.where(total_gain > 0.0, total_gain, 1.0)[:, None]
    matrices = gain[:, :, None] * share[:, None, :]
    idx = np.arange(rho.shape[1])
    matrices[:, idx, idx] = 1.0 - fraction
    return matrices

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: three levels, one falls while two rise, then two fall while one rises ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([[1.0, 0.0, 0.0], [0.9, 0.08, 0.02], [0.85, 0.05, 0.10]])\n",
            "call": "least_flow_transition_matrices(p)",
            "gold_call": "_oracle_least_flow_transition_matrices(p)",
            "tol": 1e-12,
        },
        # --- Normal: four levels with two rising and two falling at once (proportional sharing) ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([[0.4, 0.3, 0.2, 0.1], [0.3, 0.35, 0.1, 0.25], [0.32, 0.2, 0.18, 0.3]])\n",
            "call": "least_flow_transition_matrices(p)",
            "gold_call": "_oracle_least_flow_transition_matrices(p)",
            "tol": 1e-12,
        },
        # --- Boundary: an unchanged interval gives the identity, and a level emptied completely keeps nothing ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.7, 0.3]])\n",
            "call": "least_flow_transition_matrices(p)",
            "gold_call": "_oracle_least_flow_transition_matrices(p)",
            "tol": 1e-12,
        },
        # --- Edge: unnormalised total (populations sum to 2) and a zero-population level that stays empty ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([[1.2, 0.8, 0.0, 0.0], [1.0, 0.9, 0.1, 0.0], [1.1, 0.6, 0.3, 0.0]])\n",
            "call": "least_flow_transition_matrices(p)",
            "gold_call": "_oracle_least_flow_transition_matrices(p)",
            "tol": 1e-12,
        },
        # --- Normal: forty samples of a six-level history with several levels rising and falling together ---
        {
            "setup": "import numpy as np\n"
                     "x = np.random.default_rng(7).random((40, 6))\n"
                     "p = x / x.sum(axis=1, keepdims=True)\n",
            "call": "least_flow_transition_matrices(p)",
            "gold_call": "_oracle_least_flow_transition_matrices(p)",
            "tol": 1e-12,
        },
        # --- Error: rows with different totals must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([[1.0, 0.0], [0.7, 0.2]]))\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(least_flow_transition_matrices)",
            "gold_call": "_probe(_oracle_least_flow_transition_matrices)",
        },
    ]
