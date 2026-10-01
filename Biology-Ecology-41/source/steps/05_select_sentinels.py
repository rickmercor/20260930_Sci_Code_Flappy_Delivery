"""
Select a small sentinel node set that minimizes approximation error over a training sweep.

The result is a reproducible node-label array determined by the supplied training states, global activity curve, subset size, and random seed. The benchmark returns the lowest-error sentinel set visited during the search; exact-error ties are resolved by the lexicographically smallest sorted node-label tuple.

Returns
-------
return best_sentinel
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_sentinels(states: "np.ndarray", mean_activity: "np.ndarray", n: int, seed: int) -> "np.ndarray":
    '''Return a deterministic sentinel node set for a training sweep.

    Parameters
    ----------
    states : np.ndarray
        Finite array of shape (L, N) containing training node states.
    mean_activity : np.ndarray
        Length-L full-network activity curve corresponding to states.
    n : int
        Number of distinct sentinel nodes; 1 <= n < N.
    seed : int
        Non-negative random seed used to make the stochastic search reproducible.

    Returns
    -------
    sentinel_indices : np.ndarray
        Sorted integer node labels of length n corresponding to the lowest-error
        set visited during the search. Exact-error ties use lexicographic order.

    Raises
    ------
    ValueError
        If the state dimensions, subset size, seed, or positivity of the
        training normalization is invalid.
    '''
    return sentinel_indices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_sentinels(
    states: "np.ndarray",
    mean_activity: "np.ndarray",
    n: int,
    seed: int,
) -> "np.ndarray":
    states = np.asarray(states, dtype=float)
    mean_activity = np.asarray(mean_activity, dtype=float)

    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 2:
        raise ValueError("states must have shape (L,N) with L,N >= 2")
    if mean_activity.shape != (states.shape[0],):
        raise ValueError("mean_activity must have length L")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(mean_activity)):
        raise ValueError("states and mean_activity must be finite")
    if not isinstance(n, (int, np.integer)) or n < 1 or n >= states.shape[1]:
        raise ValueError("n must satisfy 1 <= n < N")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")

    denominator = float(np.sum(mean_activity))
    if denominator <= 0.0:
        raise ValueError("sum of mean_activity must be positive")

    rng = np.random.default_rng(int(seed))
    N = states.shape[1]
    sentinel = np.sort(rng.choice(N, size=int(n), replace=False)).astype(int)

    def _error_of(S):
        approx = np.mean(states[:, S], axis=1)
        return float(np.sum((approx - mean_activity) ** 2) / denominator)

    current_error = _error_of(sentinel)
    best_sentinel = sentinel.copy()
    best_error = current_error
    h_max = 50 * N

    for h in range(1, h_max + 1):
        replace_pos = int(rng.integers(0, n))
        outside = np.setdiff1d(
            np.arange(N, dtype=int),
            sentinel,
            assume_unique=True,
        )
        replacement = int(rng.choice(outside))
        candidate = sentinel.copy()
        candidate[replace_pos] = replacement
        candidate.sort()
        candidate_error = _error_of(candidate)

        if candidate_error < current_error:
            accept = True
        else:
            temperature = 10.0 / np.log(h + np.e - 1.0)
            probability = np.exp(
                -(candidate_error - current_error) / temperature
            )
            accept = bool(rng.random() < probability)

        if accept:
            sentinel = candidate
            current_error = candidate_error

        if (
            candidate_error < best_error
            or (
                candidate_error == best_error
                and tuple(candidate.tolist())
                < tuple(best_sentinel.tolist())
            )
        ):
            best_sentinel = candidate.copy()
            best_error = candidate_error

    return best_sentinel

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative simulated-annealing cases."""
    return [
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
states = rng.normal(size=(7,8)) + 3.0
mean_activity = states.mean(axis=1)
n = 2
seed = 123
""",
            "call": "select_sentinels(states, mean_activity, n, seed)",
            "gold_call": "_oracle_select_sentinels(states, mean_activity, n, seed)",
        },
        {
            "setup": """import numpy as np
states = np.array([[1,2,4,8],[1,3,5,7],[2,4,6,8],[3,5,7,9],[4,6,8,10]], dtype=float)
mean_activity = states.mean(axis=1)
n = 1
seed = 1
""",
            "call": "select_sentinels(states, mean_activity, n, seed)",
            "gold_call": "_oracle_select_sentinels(states, mean_activity, n, seed)",
        },
        {
            "setup": """import numpy as np
states = np.array([[1,4,2,7,3,8],[2,5,3,8,4,9],[3,6,4,9,5,10],[4,7,5,10,6,11]], dtype=float)
mean_activity = states.mean(axis=1)
n = 3
seed = 42
""",
            "call": "select_sentinels(states, mean_activity, n, seed)",
            "gold_call": "_oracle_select_sentinels(states, mean_activity, n, seed)",
        },
    ]
