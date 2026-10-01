"""
When the pair (i, j) is solved, the infection pressure on node j from outside the pair must exclude the message from i itself, since that interaction is already represented exactly inside the pair. The pressure on j in stage x is therefore the sum of the stage-x messages that j receives from all its neighbours except i. As an operator acting on messages indexed by directed edges this is the non-backtracking operator of the graph, applied separately to each stage:

C[i, j, x] = A[i, j] * sum over k != i of A[j, k] * phi[j, k, x].

The value C[i, j, x] is the external infection rate felt by node j when it sits in stage x and is paired with i, and C[j, i, x] is the external rate felt by node i when paired with j. Entries off the edge set are zero. The total pressure on node i in stage x from all its neighbours is the backtracking sum over all k, and it differs from C[j, i, x] by exactly phi[i, j, x].

The adjacency matrix describes a simple undirected graph: square, symmetric, entries 0 or 1, zero diagonal. Messages are non-negative and vanish on non-edges.

A pair approximation for the susceptible-infectious-susceptible (SIS) contact process treats every connected pair of nodes (i, j) as a small Markov system and replaces the rest of the network by the average infection pressure it exerts on each end of the pair. In the memory-augmented version of the approximation the susceptible compartment of every node is split into K stages, and the pressure a susceptible node feels depends on its stage. The quantities passed along the network are therefore stage-resolved messages: phi[i, j, x] is the rate at which node i, while in susceptible stage x, is infected by its neighbour j.

Returns
-------
np.ndarray of shape (N, N, K), the non-backtracking cavity rates C[i, j, x] defined above, zero off the edge set.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonbacktracking_cavity_rates(adjacency: np.ndarray, messages: np.ndarray) -> np.ndarray:
    """Apply the stage-resolved non-backtracking operator to edge messages.

    Parameters
    ----------
    adjacency : np.ndarray
        Symmetric 0/1 adjacency matrix of shape (N, N) with zero diagonal.
    messages : np.ndarray
        Messages phi[i, j, x] of shape (N, N, K), non-negative and zero off the edge set.

    Returns
    -------
    np.ndarray
        Cavity rates C[i, j, x] = A[i, j] * sum over k != i of A[j, k] * phi[j, k, x], of shape (N, N, K).

    Raises
    ------
    ValueError
        When the adjacency fails to be a square symmetric 0/1 matrix of size at least 2 with zero diagonal, or when the messages fail to be a finite non-negative array of shape (N, N, K) with K at least 1 that vanishes off the edge set.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_adjacency(adjacency):
    a = np.asarray(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError("adjacency must be a square matrix of size at least 2")
    if not np.all(np.isfinite(a)):
        raise ValueError("adjacency must be finite")
    a = a.astype(float)
    if not np.all((a == 0.0) | (a == 1.0)):
        raise ValueError("adjacency entries must be 0 or 1")
    if not np.array_equal(a, a.T):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.diag(a) != 0.0):
        raise ValueError("adjacency must have a zero diagonal")
    return a


def _oracle_nonbacktracking_cavity_rates(adjacency: np.ndarray, messages: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    a = _check_adjacency(adjacency)
    n = a.shape[0]
    phi = np.asarray(messages)
    if phi.ndim != 3 or phi.shape[0] != n or phi.shape[1] != n or phi.shape[2] < 1:
        raise ValueError("messages must have shape (N, N, K) with K at least 1")
    if np.iscomplexobj(phi) or not np.all(np.isfinite(phi)):
        raise ValueError("messages must be real and finite")
    phi = phi.astype(float)
    if np.any(phi < 0.0):
        raise ValueError("messages must be non-negative")
    if np.any(phi[a == 0.0] != 0.0):
        raise ValueError("messages must vanish off the edge set")

    # total[j, x] is the stage-x pressure on j from all its neighbours; removing phi[j, i, x]
    # leaves the pressure from outside the pair (i, j)
    total = phi.sum(axis=1)
    cavity = total[None, :, :] - np.transpose(phi, (1, 0, 2))
    return cavity * a[:, :, None]

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
            # a star with a pendant path: a hub, two leaves and a two-edge tail, with distinct
            # messages on every directed edge and two stages
            "setup": SETUP + """
A = np.zeros((5, 5))
for i, j in [(0, 1), (0, 2), (0, 3), (3, 4)]:
    A[i, j] = A[j, i] = 1.0
rng = np.random.default_rng(7)
phi = rng.uniform(0.05, 0.9, size=(5, 5, 2)) * A[:, :, None]
""",
            "call": "flat(nonbacktracking_cavity_rates(A, phi))",
            "gold_call": "flat(_oracle_nonbacktracking_cavity_rates(A, phi))",
        },
        {
            # the Petersen graph with uniform messages: every cavity rate on an edge is exactly
            # (degree - 1) times the message, and a leaf-free regular graph has no zero rows
            "setup": SETUP + """
outer = [(k, (k + 1) % 5) for k in range(5)]
inner = [(5 + k, 5 + (k + 2) % 5) for k in range(5)]
spokes = [(k, 5 + k) for k in range(5)]
A = np.zeros((10, 10))
for i, j in outer + inner + spokes:
    A[i, j] = A[j, i] = 1.0
phi = A[:, :, None] * np.array([0.3, 0.1, 0.7])[None, None, :]
""",
            "call": "flat(nonbacktracking_cavity_rates(A, phi)[0, 1]) + flat(nonbacktracking_cavity_rates(A, phi).sum(axis=(0, 1)))",
            "gold_call": "flat(_oracle_nonbacktracking_cavity_rates(A, phi)[0, 1]) + flat(_oracle_nonbacktracking_cavity_rates(A, phi).sum(axis=(0, 1)))",
        },
        {
            # a single edge: the cavity of each end excludes its only neighbour, so every rate is
            # zero; on a triangle with distinct messages the total pressure on i minus the cavity
            # rate C[j, i] recovers phi[i, j] on every edge
            "setup": SETUP + """
A2 = np.array([[0.0, 1.0], [1.0, 0.0]])
phi2 = np.zeros((2, 2, 1)); phi2[0, 1, 0] = 0.4; phi2[1, 0, 0] = 0.9
A3 = np.ones((3, 3)) - np.eye(3)
phi3 = A3[:, :, None] * np.arange(1.0, 10.0).reshape(3, 3, 1)
def identity(fn):
    c = fn(A3, phi3)
    residual = (phi3.sum(axis=1)[:, None, :] - np.transpose(c, (1, 0, 2)) - phi3) * A3[:, :, None]
    return (round(float(np.abs(residual).max()), 12),)
""",
            "call": "flat(nonbacktracking_cavity_rates(A2, phi2)) + identity(nonbacktracking_cavity_rates) + flat(nonbacktracking_cavity_rates(A3, phi3))",
            "gold_call": "flat(_oracle_nonbacktracking_cavity_rates(A2, phi2)) + identity(_oracle_nonbacktracking_cavity_rates) + flat(_oracle_nonbacktracking_cavity_rates(A3, phi3))",
        },
        {
            # malformed inputs: asymmetric, self-loop, weighted, messages on a non-edge, negative
            # message, wrong shape, and one valid call
            "setup": SETUP + """
A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
phi = A[:, :, None] * np.ones((1, 1, 2))
asym = A.copy(); asym[0, 2] = 1.0
loop = A.copy(); loop[1, 1] = 1.0
weighted = 0.5 * A
offedge = phi.copy(); offedge[0, 2, 0] = 0.1
negative = -phi
""",
            "call": "(verdict(nonbacktracking_cavity_rates, asym, phi), verdict(nonbacktracking_cavity_rates, loop, phi), verdict(nonbacktracking_cavity_rates, weighted, phi), verdict(nonbacktracking_cavity_rates, A, offedge), verdict(nonbacktracking_cavity_rates, A, negative), verdict(nonbacktracking_cavity_rates, A, phi[:, :, 0]), verdict(nonbacktracking_cavity_rates, A, phi))",
            "gold_call": "(verdict(_oracle_nonbacktracking_cavity_rates, asym, phi), verdict(_oracle_nonbacktracking_cavity_rates, loop, phi), verdict(_oracle_nonbacktracking_cavity_rates, weighted, phi), verdict(_oracle_nonbacktracking_cavity_rates, A, offedge), verdict(_oracle_nonbacktracking_cavity_rates, A, negative), verdict(_oracle_nonbacktracking_cavity_rates, A, phi[:, :, 0]), verdict(_oracle_nonbacktracking_cavity_rates, A, phi))",
        },
    ]
