"""
Build the survival-weighted rate matrix of the exact first-coalescence calculation for m lineages in d demes.

Conditional on no coalescence yet, the \(m\) ancestral lineages of a sample are still distinct, and their configuration is the occupancy vector \(x=(x_1,\ldots,x_d)\), where \(x_i\ge 0\) counts the lineages currently in deme \(i\) and \(\sum_i x_i=m\). Time runs backward in generations. A lineage in deme \(i\) moves to deme \(j\) at the backward-time migration rate \(M_{ij}\) per lineage per generation, so the configuration jumps from \(x\) to \(x-e_i+e_j\) at rate \(x_iM_{ij}\). Every unordered pair of lineages in deme \(i\) coalesces at rate \(\eta_i=1/(2N_i)\), where \(N_i\) is the diploid size of deme \(i\), so the total first-coalescence rate in configuration \(x\) is

\[

\lambda(x)=\sum_{i=1}^{d}\binom{x_i}{2}\eta_i .

\]

The row vector \(\tilde p_t\) with entries \(\tilde p_t(x)=\Pr\{X(t)=x,\ T>t\}\), where \(T\) is the time of the first coalescence, obeys

\[

\frac{d\tilde p_t}{dt}=\tilde p_t\,\bigl(Q_{\mathrm{mig}}-D_{\lambda}\bigr),

\]

where \(Q_{\mathrm{mig}}\) holds the migration moves (off-diagonal rates, and a diagonal equal to minus the total outgoing migration rate) and \(D_{\lambda}=\operatorname{diag}(\lambda(x))\). Every row of \(Q_{\mathrm{mig}}-D_{\lambda}\) therefore sums to \(-\lambda(x)\): probability that leaves the state space is exactly the probability of a first coalescence.

Returns
-------
np.ndarray with shape (S, S), the occupancy-state rate matrix Q_mig - D_lambda in ascending lexicographic state order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def occupancy_rate_matrix(m: int, sizes: "np.ndarray", migration: "np.ndarray") -> "np.ndarray":
    """Return Q_mig - D_lambda on the occupancy states of m lineages in d demes.
 
    Parameters
    ----------
    m : int
        Number of distinct lineages, an integer m >= 0 (``bool`` is invalid).
    sizes : np.ndarray
        Finite, strictly positive diploid deme sizes N_i with shape (d,), d >= 1.
    migration : np.ndarray
        Finite (d, d) matrix of backward-time migration rates; entry (i, j),
        i != j, is the rate per lineage per generation at which a lineage in
        deme i moves to deme j. Off-diagonal entries must be nonnegative;
        diagonal entries are ignored.
 
    Returns
    -------
    rate_matrix : np.ndarray
        Float array of shape (S, S), S = C(m + d - 1, d - 1). States are all
        integer vectors x with nonnegative entries summing to m, ordered in
        ascending lexicographic order of the tuple (x_1, ..., x_d); for d = 2
        and m = 2 the order is (0, 2), (1, 1), (2, 0). Entry [x, y] for y != x
        is the migration rate from x to y, and entry [x, x] equals minus the
        total outgoing migration rate minus lambda(x).
 
    Raises
    ------
    ValueError
        If m is not an integer >= 0, if sizes is not a finite strictly positive
        one-dimensional array, or if migration does not have shape (d, d), is
        not finite, or has a negative off-diagonal entry.
    """
    return rate_matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
 
import numpy as np
 
 
def _occupancy_states(m, d):
    """Occupancy vectors of m lineages in d demes, ascending lexicographic order."""
    return [s for s in itertools.product(range(m + 1), repeat=d) if sum(s) == m]
 
 
def _occupancy_coalescence_rates(states, sizes):
    """First-coalescence rate lambda(x) = sum_i C(x_i, 2) / (2 N_i) of each state."""
    eta = 1.0 / (2.0 * np.asarray(sizes, dtype=float))
    return np.array([sum(0.5 * s[i] * (s[i] - 1) * eta[i] for i in range(len(s))) for s in states], dtype=float)
 
 
def _check_count(value, name, minimum):
    """Validate an integer count that must be at least ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)
 
 
def _check_sizes_migration(sizes, migration):
    """Validate deme sizes and a backward-time migration matrix."""
    n = np.asarray(sizes, dtype=float)
    mig = np.asarray(migration, dtype=float)
    if n.ndim != 1 or n.size < 1 or not np.all(np.isfinite(n)) or np.any(n <= 0.0):
        raise ValueError("sizes must be a finite, strictly positive 1-D array")
    d = n.size
    if mig.shape != (d, d) or not np.all(np.isfinite(mig)):
        raise ValueError("migration must be a finite (d, d) array")
    off = mig[~np.eye(d, dtype=bool)]
    if np.any(off < 0.0):
        raise ValueError("off-diagonal migration rates must be nonnegative")
    return n, mig
 
 
def _oracle_occupancy_rate_matrix(m: int, sizes: "np.ndarray", migration: "np.ndarray") -> "np.ndarray":
    m = _check_count(m, "m", 0)
    n, mig = _check_sizes_migration(sizes, migration)
    d = n.size
    states = _occupancy_states(m, d)
    index = {s: a for a, s in enumerate(states)}
    rate = -np.diag(_occupancy_coalescence_rates(states, n))
    for s in states:
        a = index[s]
        for i in range(d):
            if s[i] == 0:
                continue
            for j in range(d):
                if j == i or mig[i, j] == 0.0:
                    continue
                target = list(s)
                target[i] -= 1
                target[j] += 1
                r = s[i] * mig[i, j]
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
sizes = np.array([4000.0, 1000.0])
migration = np.array([[0.0, 1e-4], [2e-4, 0.0]])
""",
            "call": "occupancy_rate_matrix(3, sizes.copy(), migration.copy())",
            "gold_call": "_oracle_occupancy_rate_matrix(3, sizes.copy(), migration.copy())",
        },
        {
            "setup": """import numpy as np
sizes = np.array([500.0, 800.0, 1200.0])
migration = np.array([[0.0, 3e-3, 1e-3], [2e-3, 0.0, 0.0], [5e-4, 4e-3, 0.0]])
""",
            "call": "occupancy_rate_matrix(2, sizes.copy(), migration.copy())",
            "gold_call": "_oracle_occupancy_rate_matrix(2, sizes.copy(), migration.copy())",
        },
        {
            "setup": """import numpy as np
sizes = np.array([1000.0, 2000.0])
migration = np.array([[7.0, 0.0], [0.0, -3.0]])
""",
            "call": "occupancy_rate_matrix(4, sizes.copy(), migration.copy())",
            "gold_call": "_oracle_occupancy_rate_matrix(4, sizes.copy(), migration.copy())",
        },
        {
            "setup": """import numpy as np
sizes = np.array([250.0])
migration = np.zeros((1, 1))
""",
            "call": "occupancy_rate_matrix(5, sizes.copy(), migration.copy())",
            "gold_call": "_oracle_occupancy_rate_matrix(5, sizes.copy(), migration.copy())",
        },
        {
            "setup": """import numpy as np
sizes = np.array([300.0, 600.0])
migration = np.array([[0.0, 1e-3], [1e-3, 0.0]])
""",
            "call": "occupancy_rate_matrix(1, sizes.copy(), migration.copy())",
            "gold_call": "_oracle_occupancy_rate_matrix(1, sizes.copy(), migration.copy())",
        },
        {
            "setup": """import numpy as np
sizes = np.array([1000.0, 1000.0])
migration = np.array([[0.0, -1e-4], [1e-4, 0.0]])
def run(fn):
    try:
        fn(2, sizes.copy(), migration.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "run(occupancy_rate_matrix)",
            "gold_call": "run(_oracle_occupancy_rate_matrix)",
        },
        {
            "setup": """import numpy as np
sizes = np.array([1000.0, 0.0])
migration = np.zeros((2, 2))
def run(fn):
    try:
        fn(2, sizes.copy(), migration.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "run(occupancy_rate_matrix)",
            "gold_call": "run(_oracle_occupancy_rate_matrix)",
        },
    ]
