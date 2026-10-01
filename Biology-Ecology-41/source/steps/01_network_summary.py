"""
Compute basic structural invariants of a network adjacency matrix that are used to parameterize the sentinel calculation.

The network is treated as a simple undirected graph. The returned summary records the node count and edge count needed to parameterize the downstream sentinel calculation.

Returns
-------
return np.array([float(N), M], dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def network_summary(A: "np.ndarray") -> "np.ndarray":
    '''Return structural invariants of a simple undirected network.

    Parameters
    ----------
    A : np.ndarray
        Square adjacency matrix for a simple undirected graph. Entries must be
        0 or 1, the diagonal must be zero, and the matrix must be symmetric.

    Returns
    -------
    summary : np.ndarray
        Array [N, M] as floating-point values, where N is the number of nodes
        and M is the number of undirected edges.

    Raises
    ------
    ValueError
        If A is not a finite square binary symmetric adjacency matrix with a
        zero diagonal or if the graph is disconnected.
    '''
    return summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_network_summary(A: "np.ndarray") -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 2:
        raise ValueError("A must be a square matrix with at least two nodes")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")
    if not np.all((A == 0.0) | (A == 1.0)):
        raise ValueError("A must be binary")
    if not np.allclose(A, A.T, rtol=0.0, atol=0.0):
        raise ValueError("A must be symmetric")
    if not np.all(np.diag(A) == 0.0):
        raise ValueError("A must have a zero diagonal")

    N = A.shape[0]
    seen = np.zeros(N, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(A[i]):
            j = int(j)
            if not seen[j]:
                seen[j] = True
                stack.append(j)
    if not np.all(seen):
        raise ValueError("A must describe a connected graph")

    M = float(np.sum(A) / 2.0)
    return np.array([float(N), M], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative structural test cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
""",
            "call": "network_summary(A)",
            "gold_call": "_oracle_network_summary(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,1],[1,0]], dtype=float)
""",
            "call": "network_summary(A)",
            "gold_call": "_oracle_network_summary(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
""",
            "call": "network_summary(A)",
            "gold_call": "_oracle_network_summary(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=float)
def run_model():
    try:
        network_summary(A)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_network_summary(A)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
