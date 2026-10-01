"""
A connected pair (i, j) is treated as a Markov chain on the (K + 1)^2 joint states. Inside the pair, an infectious node infects its susceptible partner at rate beta. From outside the pair, the first node, when in stage S(x), is infected at the external rate a[x], and the second node, when in stage S(y), at the external rate b[y]. The transitions are, for every x and y from 1 to K:

- recovery, at rate 1: (I, I) to (I, S(K)) and to (S(K), I); (I, S(y)) to (S(K), S(y)); (S(x), I) to (S(x), S(K));
- ageing, at rate gamma, for x or y at least 2: (S(x), I) to (S(x - 1), I); (I, S(y)) to (I, S(y - 1)); (S(x), S(y)) to (S(x - 1), S(y)) and to (S(x), S(y - 1));
- infection of a node whose partner is infectious: (S(x), I) to (I, I) at rate beta + a[x]; (I, S(y)) to (I, I) at rate beta + b[y];
- infection of a node whose partner is susceptible: (S(x), S(y)) to (I, S(y)) at rate a[x] and to (S(x), I) at rate b[y].

The diagonal holds minus the total exit rate, so every row sums to zero.

State ordering. A single node's state is encoded by the integer 0 for I and x for S(x). The joint state (u, v), with u the state of the first node and v that of the second, has flat index u * (K + 1) + v. Entry [r, c] of the returned matrix is the rate from state r to state c.

The memory-augmented SIS model gives every node K + 1 states: the infectious state I and K susceptible stages S(1), ..., S(K). All susceptible stages are infected at exactly the same rate, beta times the number of infectious neighbours, so merging the stages into one susceptible class recovers the SIS contact process exactly and the augmentation changes nothing about the underlying process. An infectious node recovers at unit rate into the stage S(K), and each stage S(x) with x at least 2 ages into S(x - 1) at a common rate gamma, while S(1), the oldest stage, does not age further. The stage label is therefore a noisy clock of the time since the node last recovered: S(K) means recently infectious, S(1) means susceptible for a long time. With K = 1 there is a single susceptible stage and the model is the ordinary SIS model.

Returns
-------
np.ndarray of shape ((K + 1)^2, (K + 1)^2), the generator matrix of the pair chain in the ordering above, with rows summing to zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_transition_matrix(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Assemble the generator of the memory-augmented SIS chain of one connected pair.

    Parameters
    ----------
    external_first : np.ndarray
        External infection rates a[x] of the first node in stage S(x), length K.
    external_second : np.ndarray
        External infection rates b[y] of the second node in stage S(y), length K.
    beta : float
        Per-contact transmission rate, above zero.
    gamma : float
        Ageing rate between successive susceptible stages, non-negative.

    Returns
    -------
    np.ndarray
        Generator of shape ((K + 1)^2, (K + 1)^2), flat index u * (K + 1) + v with 0 for I and x for S(x).

    Raises
    ------
    ValueError
        When the external rates fail to be finite non-negative one-dimensional arrays of equal length at least 1, when beta fails to be finite and above zero, or when gamma fails to be finite and non-negative.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_pair_transition_matrix(
    external_first: np.ndarray,
    external_second: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    a = np.asarray(external_first)
    b = np.asarray(external_second)
    if a.ndim != 1 or b.ndim != 1 or a.size < 1 or a.size != b.size:
        raise ValueError("external rates must be one-dimensional arrays of equal length at least 1")
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError("external rates must be real")
    a = a.astype(float)
    b = b.astype(float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("external rates must be finite")
    if np.any(a < 0.0) or np.any(b < 0.0):
        raise ValueError("external rates must be non-negative")
    if isinstance(beta, bool) or isinstance(gamma, bool):
        raise ValueError("beta and gamma must be real numbers")
    beta = float(beta)
    gamma = float(gamma)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and above zero")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    k = a.size
    size = k + 1
    q = np.zeros((size * size, size * size))

    def _index(u, v):
        return u * size + v

    # both infectious: either node recovers into the youngest stage
    q[_index(0, 0), _index(0, k)] += 1.0
    q[_index(0, 0), _index(k, 0)] += 1.0
    for x in range(1, k + 1):
        # one node infectious, the other in stage x
        q[_index(x, 0), _index(x, k)] += 1.0
        q[_index(0, x), _index(k, x)] += 1.0
        q[_index(x, 0), _index(0, 0)] += beta + a[x - 1]
        q[_index(0, x), _index(0, 0)] += beta + b[x - 1]
        if x > 1:
            q[_index(x, 0), _index(x - 1, 0)] += gamma
            q[_index(0, x), _index(0, x - 1)] += gamma
        # both susceptible
        for y in range(1, k + 1):
            q[_index(x, y), _index(0, y)] += a[x - 1]
            q[_index(x, y), _index(x, 0)] += b[y - 1]
            if x > 1:
                q[_index(x, y), _index(x - 1, y)] += gamma
            if y > 1:
                q[_index(x, y), _index(x, y - 1)] += gamma
    q -= np.diag(q.sum(axis=1))
    return q

# =============================================================================
# TEST CASES
# =============================================================================

SETUP = """
import numpy as np

def flat(x):
    return tuple(round(float(v), 12) + 0.0 for v in np.asarray(x, dtype=float).ravel())

def verdict(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
"""


def test_cases():
    return [
        {
            # three stages with distinct external rates on the two ends: the full 16 x 16 matrix
            "setup": SETUP + """
a = np.array([0.11, 0.37, 0.52])
b = np.array([0.23, 0.05, 0.81])
""",
            "call": "flat(pair_transition_matrix(a, b, 0.4, 1.3))",
            "gold_call": "flat(_oracle_pair_transition_matrix(a, b, 0.4, 1.3))",
        },
        {
            # K = 1 is the ordinary SIS pair: a 4 x 4 generator; exchanging the two ends of the pair
            # must permute the generator by swapping u and v in the flat index
            "setup": SETUP + """
a = np.array([0.6, 0.2])
b = np.array([0.1, 0.9])
def swap_residual(fn):
    q = fn(a, b, 0.7, 0.5)
    p = fn(b, a, 0.7, 0.5)
    perm = np.array([v * 3 + u for u in range(3) for v in range(3)])
    return (round(float(np.abs(q - p[np.ix_(perm, perm)]).max()), 12),)
""",
            "call": "flat(pair_transition_matrix(np.array([0.3]), np.array([0.8]), 0.25, 0.0)) + swap_residual(pair_transition_matrix)",
            "gold_call": "flat(_oracle_pair_transition_matrix(np.array([0.3]), np.array([0.8]), 0.25, 0.0)) + swap_residual(_oracle_pair_transition_matrix)",
        },
        {
            # structural checks on eight stages with zero external pressure: rows sum to zero, the
            # oldest joint susceptible state (S(1), S(1)) is absorbing, and the total ageing and
            # recovery flow out of every state
            "setup": SETUP + """
def summary(fn):
    q = fn(np.zeros(8), np.zeros(8), 0.3, 2.1)
    off = q - np.diag(np.diag(q))
    return (round(float(np.abs(q.sum(axis=1)).max()), 12), round(float(q[10, 10]), 12),
            round(float(np.abs(q[10]).sum()), 12), round(float(off.min()), 12), round(float(np.trace(q)), 12),
            round(float(q[0, 8]), 12), round(float(q[0, 72]), 12), round(float(q[9 * 3 + 0, 0]), 12))
""",
            "call": "flat(summary(pair_transition_matrix))",
            "gold_call": "flat(summary(_oracle_pair_transition_matrix))",
        },
        {
            # malformed inputs
            "setup": SETUP + """
a = np.array([0.1, 0.2])
""",
            "call": "(verdict(pair_transition_matrix, a, a[:1], 0.3, 1.0), verdict(pair_transition_matrix, -a, a, 0.3, 1.0), verdict(pair_transition_matrix, a, a, 0.0, 1.0), verdict(pair_transition_matrix, a, a, 0.3, -1.0), verdict(pair_transition_matrix, np.array([]), np.array([]), 0.3, 1.0), verdict(pair_transition_matrix, a, np.array([0.1, np.nan]), 0.3, 1.0), verdict(pair_transition_matrix, a, a, 0.3, 1.0))",
            "gold_call": "(verdict(_oracle_pair_transition_matrix, a, a[:1], 0.3, 1.0), verdict(_oracle_pair_transition_matrix, -a, a, 0.3, 1.0), verdict(_oracle_pair_transition_matrix, a, a, 0.0, 1.0), verdict(_oracle_pair_transition_matrix, a, a, 0.3, -1.0), verdict(_oracle_pair_transition_matrix, np.array([]), np.array([]), 0.3, 1.0), verdict(_oracle_pair_transition_matrix, a, np.array([0.1, np.nan]), 0.3, 1.0), verdict(_oracle_pair_transition_matrix, a, a, 0.3, 1.0))",
        },
    ]
