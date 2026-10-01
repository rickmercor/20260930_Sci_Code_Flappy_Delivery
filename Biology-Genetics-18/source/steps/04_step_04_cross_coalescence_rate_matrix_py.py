"""
Build the survival-weighted rate matrix of the exact cross-coalescence calculation for red and blue lineages in d demes.

In a cross-coalescence calculation the sampled lineages carry one of two colours, red and blue, and \(T_\times\) is the time of the first coalescence between a red and a blue lineage. Before \(T_\times\), two red lineages (or two blue lineages) in the same deme may coalesce; they are then replaced by one lineage of the same colour, so the colour counts can decrease. The state is the interleaved count vector \((\rho_1,\beta_1,\ldots,\rho_d,\beta_d)\), where \(\rho_i\) and \(\beta_i\) are the numbers of red and blue lineages in deme \(i\).

Time runs backward in generations and \(\eta_i=1/(2N_i)\). From a state, a red lineage moves from deme \(i\) to deme \(j\) at rate \(\rho_iM_{ij}\) and a blue lineage at rate \(\beta_iM_{ij}\). Two red lineages in deme \(i\) merge into one red lineage at rate \(\binom{\rho_i}{2}\eta_i\), and two blue lineages at rate \(\binom{\beta_i}{2}\eta_i\). The cross-coalescence rate

\[

\lambda_\times(\rho,\beta)=\sum_{i=1}^{d}\rho_i\beta_i\eta_i

\]

removes probability from the state space. The survival-weighted distribution \(\tilde p^{\times}_t\), with entries \(\Pr\{\text{state at } t,\ T_\times>t\}\), obeys \(d\tilde p^{\times}_t/dt=\tilde p^{\times}_t\,Q_\times\), where

\[

Q_\times=Q^{\mathrm{mig}}_{r}+Q^{\mathrm{mig}}_{b}+Q^{\mathrm{coal}}_{rr}+Q^{\mathrm{coal}}_{bb}-D_{\lambda_\times}

\]

has these transition rates off the diagonal and a diagonal equal to minus all outgoing transition rates minus the cross-coalescence rate. Every row of \(Q_\times\) therefore sums to \(-\lambda_\times\) of that state.

Returns
-------
np.ndarray with shape (S, S), the red/blue cross-coalescence rate matrix Q_x in ascending lexicographic interleaved state order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cross_coalescence_rate_matrix(n_red: int, n_blue: int, sizes: "np.ndarray", migration: "np.ndarray") -> "np.ndarray":
    """Return the cross-coalescence rate matrix Q_x on red/blue count states.
 
    Parameters
    ----------
    n_red : int
        Maximum total number of red lineages, an integer >= 1.
    n_blue : int
        Maximum total number of blue lineages, an integer >= 1.
    sizes : np.ndarray
        Finite, strictly positive diploid deme sizes with shape (d,), d >= 1.
    migration : np.ndarray
        Finite (d, d) backward-time migration rates per lineage per
        generation; entry (i, j), i != j, moves a lineage of either colour
        from deme i to deme j. Off-diagonal entries must be nonnegative;
        diagonal entries are ignored.
 
    Returns
    -------
    rate_matrix : np.ndarray
        Float array of shape (S, S). The states are all interleaved vectors
        (rho_1, beta_1, ..., rho_d, beta_d) of nonnegative integers with
        sum_i rho_i <= n_red and sum_i beta_i <= n_blue, ordered in ascending
        lexicographic order of that interleaved tuple; for d = 1,
        n_red = n_blue = 1 the order is (0, 0), (0, 1), (1, 0), (1, 1).
 
    Raises
    ------
    ValueError
        If n_red or n_blue is not an integer >= 1, if sizes is not a finite
        strictly positive one-dimensional array, or if migration does not have
        shape (d, d), is not finite, or has a negative off-diagonal entry.
    """
    return rate_matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
 
import numpy as np
 
 
def _colour_states(n_red, n_blue, d):
    """Interleaved red/blue count vectors in ascending lexicographic order."""
    reds = [s for s in itertools.product(range(n_red + 1), repeat=d) if sum(s) <= n_red]
    blues = [s for s in itertools.product(range(n_blue + 1), repeat=d) if sum(s) <= n_blue]
    return sorted(tuple(v for pair in zip(r, b) for v in pair) for r in reds for b in blues)
 
 
def _cross_coalescence_rates(states, sizes):
    """Cross-coalescence rate sum_i rho_i beta_i / (2 N_i) of each colour state."""
    eta = 1.0 / (2.0 * np.asarray(sizes, dtype=float))
    return np.array([sum(s[2 * i] * s[2 * i + 1] * eta[i] for i in range(len(eta))) for s in states], dtype=float)
 
 
def _oracle_cross_coalescence_rate_matrix(n_red: int, n_blue: int, sizes: "np.ndarray", migration: "np.ndarray") -> "np.ndarray":
    n_red = _check_count(n_red, "n_red", 1)
    n_blue = _check_count(n_blue, "n_blue", 1)
    n, mig = _check_sizes_migration(sizes, migration)
    d = n.size
    eta = 1.0 / (2.0 * n)
    states = _colour_states(n_red, n_blue, d)
    index = {s: a for a, s in enumerate(states)}
    rate = -np.diag(_cross_coalescence_rates(states, n))
    for s in states:
        a = index[s]
        for colour in range(2):
            for i in range(d):
                count = s[2 * i + colour]
                if count == 0:
                    continue
                for j in range(d):
                    if j == i or mig[i, j] == 0.0:
                        continue
                    target = list(s)
                    target[2 * i + colour] -= 1
                    target[2 * j + colour] += 1
                    r = count * mig[i, j]
                    rate[a, index[tuple(target)]] += r
                    rate[a, a] -= r
                if count >= 2:
                    target = list(s)
                    target[2 * i + colour] -= 1
                    r = 0.5 * count * (count - 1) * eta[i]
                    rate[a, index[tuple(target)]] += r
                    rate[a, a] -= r
    return rate

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
sizes = np.array([4000.0, 2500.0])
migration = np.array([[0.0, 1e-4], [2e-4, 0.0]])
""",
            "call": "cross_coalescence_rate_matrix(2, 2, sizes.copy(), migration.copy())",
            "gold_call": "_oracle_cross_coalescence_rate_matrix(2, 2, sizes.copy(), migration.copy())",
        },
        {
            "setup": """import numpy as np
sizes = np.array([700.0])
migration = np.zeros((1, 1))
""",
            "call": "cross_coalescence_rate_matrix(3, 2, sizes.copy(), migration.copy())",
            "gold_call": "_oracle_cross_coalescence_rate_matrix(3, 2, sizes.copy(), migration.copy())",
        },
        {
            "setup": """import numpy as np
sizes = np.array([600.0, 900.0, 1500.0])
migration = np.array([[0.0, 2e-3, 0.0], [0.0, 0.0, 1e-3], [3e-3, 0.0, 0.0]])
""",
            "call": "cross_coalescence_rate_matrix(1, 2, sizes.copy(), migration.copy())",
            "gold_call": "_oracle_cross_coalescence_rate_matrix(1, 2, sizes.copy(), migration.copy())",
        },
        {
            "setup": """import numpy as np
sizes = np.array([100.0, 200.0])
migration = np.array([[5.0, 0.0], [0.0, 5.0]])
""",
            "call": "cross_coalescence_rate_matrix(1, 1, sizes.copy(), migration.copy())",
            "gold_call": "_oracle_cross_coalescence_rate_matrix(1, 1, sizes.copy(), migration.copy())",
        },
        {
            "setup": """import numpy as np
sizes = np.array([1000.0, 1000.0])
migration = np.zeros((2, 2))
def run(fn):
    try:
        fn(0, 2, sizes.copy(), migration.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "run(cross_coalescence_rate_matrix)",
            "gold_call": "run(_oracle_cross_coalescence_rate_matrix)",
        },
        {
            "setup": """import numpy as np
sizes = np.array([1000.0, 1000.0])
migration = np.zeros((3, 3))
def run(fn):
    try:
        fn(2, 2, sizes.copy(), migration.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "run(cross_coalescence_rate_matrix)",
            "gold_call": "run(_oracle_cross_coalescence_rate_matrix)",
        },
    ]
