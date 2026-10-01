"""
Construct the elementary transition matrix for a fixed tail in the conditional cycle ensemble.

For priors $q_e$ and reference bits $M_e$, use the conditional cycle weights

$\widetilde w_e=[q_e/(1-q_e)]^{1-2M_e}$. A state $(A,h)$ has boundary

$\{t,h\}$ for fixed tail $t$ and moving head $h$, with an empty boundary when

$h=t$. A uniformly proposed neighbor $j$ is accepted with probability



$$

a(h\to j;A)=\min\left(1,\frac{d(h)}{d(j)}\widetilde w_{hj}^{1-2A_{hj}}\right).

$$



Acceptance toggles the proposed edge and moves only the head; rejection leaves

the whole state unchanged. Closed states come first in increasing relative-mask

order. Remaining states are ordered by increasing head, then increasing mask.

The boundary vertex participates in proposals and tail selection.

Returns
-------
Return the row-stochastic elementary transition matrix with closed configurations before open head states.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_oriented_transition(
    detectors: "np.ndarray",
    probabilities: "np.ndarray",
    reference: "np.ndarray",
    tail: int,
) -> "np.ndarray":
    r"""Construct the elementary transition matrix for a fixed tail in the conditional cycle ensemble.

    Parameters
    ----------
    detectors : np.ndarray
        Binary matrix $H$ of shape $(m,E)$, $1\le E\le8$. Each column has
        support one or two. Adding a shared boundary vertex when needed must
        give a simple connected graph with $2\le n\le6$ and cycle rank
        $0\le E-n+1\le3$.
    reference : np.ndarray
        Binary reference chain $M$ of shape $(E,)$ in detector-column order.
    probabilities : np.ndarray
        Finite edge probabilities $q$ of shape $(E,)$ with $0<q_e<1$.
    tail : int
        Fixed augmented vertex index $0\le t<n$.

    Returns
    -------
    transition : np.ndarray
        Row-stochastic float matrix of shape $(nk,nk)$ with $k=2^{E-n+1}$,
        using the state ordering specified in the scientific background.

    Raises
    ------
    ValueError
        If graph/reference constraints fail, probabilities are mismatched,
        nonfinite or outside the open unit interval, or tail is invalid.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _edge_weights(probabilities, reference):
    q = np.asarray(probabilities, dtype=float)
    if (
        q.shape != reference.shape
        or not np.all(np.isfinite(q))
        or np.any((q <= 0) | (q >= 1))
    ):
        raise ValueError("probabilities must be finite edge values in (0,1)")
    odds = q / (1.0 - q)
    weights = np.where(reference == 1, 1.0 / odds, odds)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("conditional weights must be finite and positive")
    return weights


def _oriented_states(incidence, tail):
    masks, _, boundaries = _mask_data(incidence)
    states = []
    heads = [tail] + [h for h in range(incidence.shape[0]) if h != tail]
    for head in heads:
        target = np.zeros(incidence.shape[0], dtype=int)
        target[head] ^= 1
        target[tail] ^= 1
        states.extend(
            (int(a), head) for a in masks[np.all(boundaries == target, axis=1)]
        )
    return states


def _oracle_build_oriented_transition(
    detectors: "np.ndarray",
    probabilities: "np.ndarray",
    reference: "np.ndarray",
    tail: int,
) -> "np.ndarray":
    incidence = _graph(detectors)
    n, edges = incidence.shape
    reference = _reference(reference, edges)
    weights = _edge_weights(probabilities, reference)
    tail = _integer(tail, 0, n - 1, "tail")
    states = _oriented_states(incidence, tail)
    indices = {state: i for i, state in enumerate(states)}
    degrees = incidence.sum(axis=1)
    endpoints = [np.flatnonzero(incidence[:, e]) for e in range(edges)]
    transition = np.zeros((len(states), len(states)))
    for row, (mask, head) in enumerate(states):
        for edge in np.flatnonzero(incidence[head]):
            u, v = endpoints[edge]
            neighbor = int(v if head == u else u)
            ratio = (
                weights[edge] if not (mask >> int(edge)) & 1 else 1.0 / weights[edge]
            )
            acceptance = min(1.0, degrees[head] / degrees[neighbor] * ratio)
            transition[row, indices[(mask ^ (1 << int(edge)), neighbor)]] += (
                acceptance / degrees[head]
            )
            transition[row, row] += (1.0 - acceptance) / degrees[head]
    return transition

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return distinct normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
HZ = np.array([[1,1,1,0,0,0,1],[1,0,0,1,1,0,0],
               [0,1,0,1,0,1,0],[0,0,1,0,1,1,0]])
HX = np.array([[1,1,0,1,0,0,0],[1,0,1,0,1,1,0],
               [0,1,1,0,0,0,1],[0,0,0,1,1,0,0]])
MZ = np.array([0,1,0,0,0,0,1])
MX = np.array([1,0,0,0,0,1,0])
SZ = np.array([0,0,1,0])
SX = np.array([1,0,0,0])
LX = np.array([[1,0,0,0,0,0,0],[0,0,1,0,0,0,0]])
mapping = np.array([3,6,1,5,0,4,2])
samples = np.array([3,2,3,5])
spacings = np.array([2,2,1,2])
p = 0.24
q=np.full(7,2*p/3)
""",
            "call": "build_oriented_transition(HZ.copy(), q.copy(), MZ.copy(), 0)",
            "gold_call": "_oracle_build_oriented_transition(HZ.copy(), q.copy(), MZ.copy(), 0)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
H=np.array([[1]])
q=np.array([0.5])
M=np.array([0])
""",
            "call": "build_oriented_transition(H.copy(), q.copy(), M.copy(), 1)",
            "gold_call": "_oracle_build_oriented_transition(H.copy(), q.copy(), M.copy(), 1)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
HZ = np.array([[1,1,1,0,0,0,1],[1,0,0,1,1,0,0],
               [0,1,0,1,0,1,0],[0,0,1,0,1,1,0]])
HX = np.array([[1,1,0,1,0,0,0],[1,0,1,0,1,1,0],
               [0,1,1,0,0,0,1],[0,0,0,1,1,0,0]])
MZ = np.array([0,1,0,0,0,0,1])
MX = np.array([1,0,0,0,0,1,0])
SZ = np.array([0,0,1,0])
SX = np.array([1,0,0,0])
LX = np.array([[1,0,0,0,0,0,0],[0,0,1,0,0,0,0]])
mapping = np.array([3,6,1,5,0,4,2])
samples = np.array([3,2,3,5])
spacings = np.array([2,2,1,2])
p = 0.24
q=np.array([0.11,0.5,0.24,0.7,0.2,0.39,0.05])
""",
            "call": "build_oriented_transition(HX.copy(), q.copy(), MX.copy(), 4)",
            "gold_call": "_oracle_build_oriented_transition(HX.copy(), q.copy(), MX.copy(), 4)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
HZ = np.array([[1,1,1,0,0,0,1],[1,0,0,1,1,0,0],
               [0,1,0,1,0,1,0],[0,0,1,0,1,1,0]])
HX = np.array([[1,1,0,1,0,0,0],[1,0,1,0,1,1,0],
               [0,1,1,0,0,0,1],[0,0,0,1,1,0,0]])
MZ = np.array([0,1,0,0,0,0,1])
MX = np.array([1,0,0,0,0,1,0])
SZ = np.array([0,0,1,0])
SX = np.array([1,0,0,0])
LX = np.array([[1,0,0,0,0,0,0],[0,0,1,0,0,0,0]])
mapping = np.array([3,6,1,5,0,4,2])
samples = np.array([3,2,3,5])
spacings = np.array([2,2,1,2])
p = 0.24
q=np.zeros(7)

def _check_error(fn):
    try:
        fn(HZ.copy(), q.copy(), MZ.copy(), 0)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_check_error(build_oriented_transition)",
            "gold_call": "_check_error(_oracle_build_oriented_transition)",
        },
    ]
