"""
Fix the rate of the phantom edge so that complex-balanced equilibria of a generalized network with kinetic deficiency one exist.

When the kinetic-order network has positive kinetic deficiency, complex-balanced equilibria exist only for rate constants satisfying as many extra conditions as the kinetic deficiency, and a phantom edge rate that does not enter the dynamics is the free quantity those conditions determine.

Returns
-------
np.ndarray: float array holding the phantom rate, shape (0,) or (1,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def solve_phantom_rate(
    kinetic_complexes: np.ndarray,
    gcrn_edges: np.ndarray,
    rate_constants: np.ndarray,
    tree_constants_fn,
    rate_bracket: tuple = (1e-8, 1e8),
) -> np.ndarray:
    """Return the phantom-edge rate required for complex-balanced equilibria.

    Row ``e`` of ``gcrn_edges`` is ``[tail, head, k]``: an effective edge
    (``k >= 0``) has rate ``rate_constants[k]`` and a phantom edge
    (``k = -1``) has the unknown rate ``sigma``. Tree constants are obtained
    as ``tree_constants_fn(V, gcrn_edges, edge_rates)`` with ``V`` the number
    of vertices and ``edge_rates`` the rate of every edge. Complex-balanced
    equilibria exist when some vector ``z`` satisfies ``(y_h - y_t) @ z =
    log(K_h) - log(K_t)`` for every edge, where ``y_v`` is the kinetic
    complex of vertex ``v`` and ``K_v`` its tree constant. The kinetic
    deficiency is ``V`` minus the number of linkage classes minus the rank of
    the vectors ``y_h - y_t`` over all edges.

    If there is no phantom edge and the kinetic deficiency is 0, return an
    empty array. If there is exactly one phantom edge and the kinetic
    deficiency is 1, return the ``sigma`` inside ``rate_bracket`` that
    satisfies the existence condition, with relative accuracy ``1e-12``.

    Parameters
    ----------
    kinetic_complexes : np.ndarray
        Integer array with shape ``(V, m)``.
    gcrn_edges : np.ndarray
        Integer array with shape ``(E, 3)``.
    rate_constants : np.ndarray
        Positive rate constant of every reaction, indexed by ``k``.
    tree_constants_fn : callable
        Function ``(V, gcrn_edges, edge_rates) -> (V,)`` tree constants.
    rate_bracket : tuple
        ``(low, high)`` with ``0 < low < high``.

    Returns
    -------
    np.ndarray
        Float array with shape ``(0,)`` or ``(1,)``.

    Raises
    ------
    ValueError
        If the arrays are malformed (edge indices out of range, a reaction
        index without a finite positive rate constant), if the bracket is not
        ``0 < low < high`` with finite ends, if the numbers of phantom edges
        and the kinetic deficiency are not one of the two supported pairs
        (for example one phantom edge with kinetic deficiency 0), or if no
        ``sigma`` in the closed bracket satisfies the condition.
    """
    return np.zeros(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_phantom_rate(
    kinetic_complexes: np.ndarray,
    gcrn_edges: np.ndarray,
    rate_constants: np.ndarray,
    tree_constants_fn,
    rate_bracket: tuple = (1e-8, 1e8),
) -> np.ndarray:
    """Reference implementation (log-bisection on the consistency residual)."""
    import numpy as np

    kinetic = np.asarray(kinetic_complexes, dtype=float)
    edges = np.asarray(gcrn_edges)
    rates = np.asarray(rate_constants, dtype=float).ravel()
    if kinetic.ndim != 2 or kinetic.shape[0] == 0:
        raise ValueError("kinetic_complexes must have shape (V, m)")
    if edges.ndim != 2 or edges.shape[1] != 3 or edges.shape[0] == 0:
        raise ValueError("gcrn_edges must have shape (E, 3)")
    edges = edges.astype(int)
    size = kinetic.shape[0]
    if np.any(edges[:, :2] < 0) or np.any(edges[:, :2] >= size):
        raise ValueError("edge endpoints must be existing vertices")
    labels = edges[:, 2]
    if np.any(labels < -1) or (np.any(labels >= 0) and labels.max() >= rates.size):
        raise ValueError("every effective edge needs a rate constant")
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("rate constants must be finite and positive")
    low, high = (float(value) for value in rate_bracket)
    if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high):
        raise ValueError("rate_bracket must satisfy 0 < low < high")

    parent = list(range(size))

    def find(node):
        while parent[node] != node:
            node = parent[node]
        return node

    forest = []
    for position, (tail, head, _) in enumerate(edges):
        root_t, root_h = find(tail), find(head)
        if root_t != root_h:
            parent[root_h] = root_t
            forest.append(position)
    classes = len({find(v) for v in range(size)})
    differences = kinetic[edges[forest, 1]] - kinetic[edges[forest, 0]]
    rank = np.linalg.matrix_rank(differences, tol=1e-9) if forest else 0
    deficiency = size - classes - rank
    phantom = int(np.sum(labels == -1))
    if phantom == 0 and deficiency == 0:
        return np.zeros(0, dtype=float)
    if not (phantom == 1 and deficiency == 1):
        raise ValueError("unsupported numbers of phantom edges and kinetic deficiency")
    cokernel = np.linalg.svd(differences.T)[2][-1]

    def residual(log_sigma):
        edge_rates = np.where(labels >= 0, rates[np.maximum(labels, 0)], np.exp(log_sigma))
        tree = np.asarray(tree_constants_fn(size, edges, edge_rates), dtype=float)
        if tree.shape != (size,) or np.any(tree <= 0.0) or not np.all(np.isfinite(tree)):
            raise ValueError("tree constants must be finite and positive")
        gaps = np.log(tree[edges[forest, 1]]) - np.log(tree[edges[forest, 0]])
        return float(cokernel @ gaps)

    left, right = np.log(low), np.log(high)
    value_left, value_right = residual(left), residual(right)
    if value_left == 0.0:
        return np.array([low])
    if value_right == 0.0:
        return np.array([high])
    if value_left * value_right > 0.0:
        raise ValueError("no phantom rate in the bracket satisfies the condition")
    for _ in range(200):
        middle = 0.5 * (left + right)
        value = residual(middle)
        if value == 0.0 or right - left < 1e-15:
            left = right = middle
            break
        if (value > 0.0) == (value_left > 0.0):
            left, value_left = middle, value
        else:
            right = middle
    return np.array([float(np.exp(0.5 * (left + right)))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    tree = (
        "import numpy as np\n"
        "def _tree(n, E, w):\n"
        "    E = np.asarray(E)[:, :2].astype(int)\n"
        "    W = np.zeros((n, n))\n"
        "    L = np.eye(n, dtype=bool)\n"
        "    for (t, h), k in zip(E, w):\n"
        "        W[t, h] += k\n"
        "        L[t, h] = L[h, t] = True\n"
        "    for p in range(n): L |= L[:, [p]] & L[[p], :]\n"
        "    lap = np.diag(W.sum(axis=1)) - W\n"
        "    K = np.ones(n)\n"
        "    for v in range(n):\n"
        "        o = [u for u in np.flatnonzero(L[v]) if u != v]\n"
        "        if o: K[v] = np.linalg.det(lap[np.ix_(o, o)])\n"
        "    return K\n"
        "def _one(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    return float(a[0]) if a.shape == (1,) else -1.0\n"
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
        "Y = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]])\n"
        "E = np.array([[0, 1, 0], [2, 3, 1], [3, 2, 2], [1, 3, 3], [4, 0, 4], [3, 4, -1]])\n"
        "k = np.array([1.3, 0.7, 2.1, 0.9, 1.6])\n"
    )
    dual = (
        "Y = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0],"
        " [0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1],"
        " [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]])\n"
        "E = np.array([[0, 1, 0], [1, 2, 1], [2, 4, 2], [4, 2, 3], [4, 0, 4], [3, 5, 5],"
        " [5, 2, 6], [5, 0, 7], [6, 7, 8], [7, 6, 9], [2, 3, -1]])\n"
        "k = np.array([1.2, 2.3, 1.7, 0.6, 0.9, 2.6, 0.4, 1.1, 0.8, 1.5])\n"
    )
    return [
        {
            "setup": tree + toy,
            "call": "_one(solve_phantom_rate(Y, E, k, _tree))",
            "gold_call": "float(np.sqrt(1.3 * 2.1 * 1.6 / 0.7))",
        },
        {
            "setup": tree + toy + (
                "Y = np.pad(Y, ((0, 2), (0, 1))); Y[6, 3] = 1\n"
                "E = np.vstack([E, [[5, 6, 5], [6, 5, 6]]])\n"
                "k = np.append(k, [1.25, 0.85])\n"
            ),
            "call": "_one(solve_phantom_rate(Y, E, k, _tree))",
            "gold_call": "float(np.sqrt(1.3 * 2.1 * 1.6 / 0.7))",
        },
        {
            "setup": tree + dual,
            "call": "_one(solve_phantom_rate(Y, E, k, _tree))",
            "gold_call": "2.6 * 0.8 / 1.5",
        },
        {
            "setup": tree + dual + "k = np.array([0.35, 4.1, 2.2, 1.3, 0.25, 0.95, 3.3, 0.6, 2.7, 0.45])\n",
            "call": "_one(solve_phantom_rate(Y, E, k, _tree, (1e-3, 1e3)))",
            "gold_call": "_one(_oracle_solve_phantom_rate(Y, E, k, _tree, (1e-3, 1e3)))",
        },
        {
            "setup": tree + toy + "warp = lambda n, e, w: _tree(n, e, w) * (1.0 + 0.25 * np.arange(n))\n",
            "call": "_one(solve_phantom_rate(Y, E, k, warp))",
            "gold_call": "_one(_oracle_solve_phantom_rate(Y, E, k, warp))",
        },
        {
            "setup": tree + (
                "Y = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],"
                " [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],"
                " [0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0],"
                " [0, 0, 0, 0, 0, 0, 1]])\n"
                "E = np.array([[0, 1, 0], [2, 3, 1], [1, 0, 2], [3, 2, 3], [4, 5, 4], [5, 4, 5],"
                " [6, 7, 6], [7, 6, 7], [8, 9, 8], [9, 8, 9]])\n"
                "k = np.linspace(0.5, 2.3, 10)\n"
            ),
            "call": "float(np.asarray(solve_phantom_rate(Y, E, k, _tree)).size) + float(np.asarray(solve_phantom_rate(Y, E, k, _tree)).ndim)",
            "gold_call": "float(np.asarray(_oracle_solve_phantom_rate(Y, E, k, _tree)).size) + float(np.asarray(_oracle_solve_phantom_rate(Y, E, k, _tree)).ndim)",
        },
        {
            "setup": tree + status + (
                "Y = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])\n"
                "E = np.array([[0, 1, 0], [2, 0, 1], [3, 2, 2], [1, 2, 3], [0, 3, -1]])\n"
                "k = np.array([1.1, 0.8, 1.9, 0.6])\n"
            ),
            "call": "_status(lambda: solve_phantom_rate(Y, E, k, _tree))",
            "gold_call": "_status(lambda: _oracle_solve_phantom_rate(Y, E, k, _tree))",
        },
        {
            "setup": tree + status + toy,
            "call": "_status(lambda: solve_phantom_rate(Y, E, k, _tree, (10.0, 100.0)))",
            "gold_call": "_status(lambda: _oracle_solve_phantom_rate(Y, E, k, _tree, (10.0, 100.0)))",
        },
    ]
