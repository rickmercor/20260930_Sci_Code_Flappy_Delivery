"""
Evaluate the spanning-tree constant of every vertex of a weighted generalized network.

For a weakly reversible network the kernel of the rate-weighted Laplacian restricted to each linkage class is spanned by the vector of tree constants, which is what fixes the ratios of the monomials at a complex-balanced equilibrium.

Returns
-------
np.ndarray: float tree constant of each vertex, shape (V,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_tree_constants(
    num_vertices: int, gcrn_edges: np.ndarray, edge_rates: np.ndarray
) -> np.ndarray:
    """Return the tree constant of every vertex.

    The tree constant ``K_v`` is the sum, over all spanning trees of the
    linkage class of ``v`` that are directed towards ``v`` (every other vertex
    of the class has exactly one outgoing tree edge and a directed tree path
    to ``v``), of the product of the rates of the tree edges. A vertex that
    is alone in its linkage class has ``K_v = 1``. Parallel edges are
    distinct edges.

    Parameters
    ----------
    num_vertices : int
        Number of vertices ``V``.
    gcrn_edges : np.ndarray
        Integer array with shape ``(E, 2)`` or ``(E, 3)``; the first two
        columns are the tail and head vertex of each directed edge and any
        further column is ignored.
    edge_rates : np.ndarray
        Positive rate of each edge, shape ``(E,)``.

    Returns
    -------
    np.ndarray
        Float array with shape ``(V,)``.

    Raises
    ------
    ValueError
        If ``num_vertices`` is not a positive integer, if an edge index is out
        of range or an edge is a self-loop, or if ``edge_rates`` does not
        have one finite, strictly positive entry per edge (a zero rate
        included).
    """
    return np.zeros(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_tree_constants(
    num_vertices: int, gcrn_edges: np.ndarray, edge_rates: np.ndarray
) -> np.ndarray:
    """Reference implementation (directed matrix-tree theorem)."""
    import numpy as np

    def is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    if not is_integer(num_vertices) or num_vertices < 1:
        raise ValueError("num_vertices must be a positive integer")
    size = int(num_vertices)
    edges = np.asarray(gcrn_edges)
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=int)
    if edges.ndim != 2 or edges.shape[1] < 2:
        raise ValueError("gcrn_edges must have shape (E, 2) or (E, 3)")
    if not np.all(np.isfinite(edges.astype(float))) or np.any(edges != np.round(edges)):
        raise ValueError("edge entries must be integers")
    ends = edges[:, :2].astype(int)
    if np.any(ends < 0) or np.any(ends >= size) or np.any(ends[:, 0] == ends[:, 1]):
        raise ValueError("edges must join two different existing vertices")
    rates = np.asarray(edge_rates, dtype=float)
    if rates.shape != (ends.shape[0],) or not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("edge_rates must hold one finite positive rate per edge")

    weights = np.zeros((size, size))
    linked = np.eye(size, dtype=bool)
    for (tail, head), rate in zip(ends, rates):
        weights[tail, head] += rate
        linked[tail, head] = linked[head, tail] = True
    for pivot in range(size):
        linked |= linked[:, [pivot]] & linked[[pivot], :]
    laplacian = np.diag(weights.sum(axis=1)) - weights
    constants = np.ones(size)
    for vertex in range(size):
        others = [u for u in np.flatnonzero(linked[vertex]) if u != vertex]
        if others:
            constants[vertex] = np.linalg.det(laplacian[np.ix_(others, others)])
    return constants

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    reducer = (
        "import numpy as np\n"
        "def _fsig(a, n, scale):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n,): return -1.0\n"
        "    w = np.cos(np.arange(n) + 1.0)\n"
        "    return float((np.sum(np.abs(a)) + np.sum(a * w)) / scale)\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    toy = (
        "E = np.array([[0, 1, 0], [2, 3, 1], [3, 2, 2], [1, 3, 3], [4, 0, 4], [3, 4, -1]])\n"
        "w = np.array([1.3, 0.7, 2.1, 0.9, 1.6, 2.4979991993593593])\n"
    )
    return [
        {
            "setup": reducer + "E = np.array([[0, 1], [1, 2], [2, 0]]); w = np.array([1.3, 0.7, 2.1])\n",
            "call": "_fsig(compute_tree_constants(3, E, w), 3, 1.0)",
            "gold_call": "_fsig([0.7 * 2.1, 2.1 * 1.3, 1.3 * 0.7], 3, 1.0)",
        },
        {
            "setup": reducer + toy,
            "call": "_fsig(compute_tree_constants(5, E, w), 5, 10.0)",
            "gold_call": "_fsig(_oracle_compute_tree_constants(5, E, w), 5, 10.0)",
        },
        {
            "setup": reducer + (
                "E = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [3, 4], [4, 5], [5, 3], [4, 3]])\n"
                "w = np.array([0.8, 1.9, 0.6, 2.2, 1.4, 0.5, 1.7, 0.9])\n"
            ),
            "call": "_fsig(compute_tree_constants(7, E, w), 7, 1.0)",
            "gold_call": "_fsig(_oracle_compute_tree_constants(7, E, w), 7, 1.0)",
        },
        {
            "setup": reducer + "E = np.array([[0, 1], [0, 1], [1, 0]]); w = np.array([0.4, 1.1, 0.9])\n",
            "call": "_fsig(compute_tree_constants(2, E, w), 2, 1.0)",
            "gold_call": "_fsig([0.9, 1.5], 2, 1.0)",
        },
        {
            "setup": reducer + (
                "E = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [2, 0], [1, 3]])\n"
                "w = np.array([2.5, 0.3, 1.2, 0.7, 0.45, 3.1, 1.05])\n"
            ),
            "call": "_fsig(compute_tree_constants(4, E, w), 4, 1.0)",
            "gold_call": "_fsig(_oracle_compute_tree_constants(4, E, w), 4, 1.0)",
        },
        {
            "setup": reducer + status + "E = np.array([[0, 1], [1, 0]]); w = np.array([1.0, 0.0])\n",
            "call": "_status(lambda: compute_tree_constants(2, E, w))",
            "gold_call": "_status(lambda: _oracle_compute_tree_constants(2, E, w))",
        },
        {
            "setup": reducer + status + "E = np.array([[0, 1], [1, 1]]); w = np.array([1.0, 2.0])\n",
            "call": "_status(lambda: compute_tree_constants(2, E, w))",
            "gold_call": "_status(lambda: _oracle_compute_tree_constants(2, E, w))",
        },
    ]
