"""
Construct the ordered physical error configurations represented by closed relative cycles.

The detector constraint defines an affine binary space. A fixed reference

$M$ gives $x=C\oplus M$, where $BC=0\pmod2$ for the augmented graph incidence

matrix $B$. Columns detected once are attached to one shared boundary vertex;

other columns retain their two detectors. Enumerate the relative masks

$\sum_e2^e C_e$ in increasing order, then convert each to its physical error

vector. The first physical row is $M$, even when it is not the smallest physical

mask. This distinction fixes every subsequent initial condition and observable.

Returns
-------
Return the binary physical-error table in ascending relative-cycle-mask order, starting with the reference chain.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_physical_cycle_ensemble(
    detectors: "np.ndarray", reference: "np.ndarray"
) -> "np.ndarray":
    r"""Construct the ordered physical error configurations represented by closed relative cycles.

    Parameters
    ----------
    detectors : np.ndarray
        Binary matrix $H$ of shape $(m,E)$, $1\le E\le8$. Each column has
        support one or two. Adding a shared boundary vertex when needed must
        give a simple connected graph with $2\le n\le6$ and cycle rank
        $0\le E-n+1\le3$.
    reference : np.ndarray
        Binary reference chain $M$ of shape $(E,)$ in detector-column order.

    Returns
    -------
    physical_errors : np.ndarray
        Binary integer array of shape $(2^{E-n+1},E)$, in ascending relative
        cycle-mask order. Row zero is the supplied reference chain.

    Raises
    ------
    ValueError
        If binary data, shapes, column supports, augmented connectivity,
        simple-graph conditions, cycle rank, or size bounds are invalid.

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


def _binary(value, ndim, name):
    array = np.asarray(value)
    if array.ndim != ndim or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be binary with {ndim} dimensions")
    return array.astype(np.int64, copy=True)


def _integer(value, low, high, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= value <= high
    ):
        raise ValueError(f"{name} is outside its integer range")
    return int(value)


def _graph(detectors):
    matrix = _binary(detectors, 2, "detectors")
    if not 1 <= matrix.shape[0] <= 6 or not 1 <= matrix.shape[1] <= 8:
        raise ValueError("invalid detector dimensions")
    support = matrix.sum(axis=0)
    if not np.all((support == 1) | (support == 2)):
        raise ValueError("each column must trigger one or two detectors")
    if np.any(support == 1):
        matrix = np.vstack((matrix, support == 1)).astype(np.int64)
    n, edges = matrix.shape
    if not 2 <= n <= 6 or not 0 <= edges - n + 1 <= 3:
        raise ValueError("invalid augmented graph size or cycle rank")
    endpoints = [tuple(np.flatnonzero(matrix[:, e])) for e in range(edges)]
    if len(set(endpoints)) != edges:
        raise ValueError("parallel edges are outside the graph contract")
    reached = {0}
    for _ in range(n):
        for u, v in endpoints:
            if u in reached or v in reached:
                reached.update((u, v))
    if len(reached) != n:
        raise ValueError("the augmented graph must be connected")
    return matrix


def _reference(reference, n_edges):
    reference = _binary(reference, 1, "reference")
    if reference.shape != (n_edges,):
        raise ValueError("reference shape must match the edge count")
    return reference


def _mask_data(incidence):
    masks = np.arange(1 << incidence.shape[1], dtype=np.int64)
    bits = (masks[:, None] >> np.arange(incidence.shape[1])) & 1
    return masks, bits, (bits @ incidence.T) % 2


def _oracle_construct_physical_cycle_ensemble(
    detectors: "np.ndarray", reference: "np.ndarray"
) -> "np.ndarray":
    incidence = _graph(detectors)
    reference = _reference(reference, incidence.shape[1])
    _, bits, boundaries = _mask_data(incidence)
    return bits[np.all(boundaries == 0, axis=1)] ^ reference

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
""",
            "call": "construct_physical_cycle_ensemble(HZ.copy(), MZ.copy())",
            "gold_call": "_oracle_construct_physical_cycle_ensemble(HZ.copy(), MZ.copy())",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
H = np.array([[1]])
M = np.array([1])
""",
            "call": "construct_physical_cycle_ensemble(H.copy(), M.copy())",
            "gold_call": "_oracle_construct_physical_cycle_ensemble(H.copy(), M.copy())",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
HZ = np.array([[1,0,1],[1,1,0]])
HX = HZ.copy()
MZ = np.array([0,0,0])
MX = np.array([1,0,0])
SZ = HZ @ MZ % 2
SX = HX @ MX % 2
LX = np.array([[1,0,0]])
mapping = np.array([2,0,1])
samples = np.array([2,1,2,3])
spacings = np.array([1,2,1,1])
p = 0.36
""",
            "call": "construct_physical_cycle_ensemble(HX.copy(), MX.copy())",
            "gold_call": "_oracle_construct_physical_cycle_ensemble(HX.copy(), MX.copy())",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
H=np.ones((3,1),dtype=int)
M=np.array([0])

def _check_error(fn):
    try:
        fn(H.copy(), M.copy())
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_check_error(construct_physical_cycle_ensemble)",
            "gold_call": "_check_error(_oracle_construct_physical_cycle_ensemble)",
        },
    ]
