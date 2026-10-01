"""
Express the complex-balanced equilibria of a generalized network as a log-linear function of designated free species.

At a complex-balanced equilibrium of a weakly reversible generalized network, the kinetic monomials of the vertices in a linkage class are proportional to their tree constants, which is a linear system for the logarithms of the concentrations.

Returns
-------
tuple: (log_offset (m,), exponents (m, d)) as float arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def parametrize_steady_states(
    kinetic_complexes: np.ndarray,
    gcrn_edges: np.ndarray,
    tree_constants: np.ndarray,
    free_species: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(log_offset, exponents)`` of the complex-balanced equilibria.

    With ``z = log(x)`` and ``y_v`` the kinetic complex of vertex ``v``, the
    complex-balanced equilibria are the solutions of ``(y_h - y_t) @ z =
    log(K_h) - log(K_t)`` over every edge ``(t, h)``. Taking ``z`` of the
    ``d`` species listed in ``free_species`` as free coordinates, every
    solution is ``z = log_offset + exponents @ z[free_species]``. The rows of
    the free species have zero offset and are the corresponding unit rows.

    Parameters
    ----------
    kinetic_complexes : np.ndarray
        Integer array with shape ``(V, m)``.
    gcrn_edges : np.ndarray
        Integer array with shape ``(E, 2)`` or ``(E, 3)`` whose first two
        columns are the tail and head vertex of each edge.
    tree_constants : np.ndarray
        Positive tree constant of each vertex, shape ``(V,)``.
    free_species : np.ndarray
        Distinct species indices, shape ``(d,)`` (possibly empty).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float arrays with shapes ``(m,)`` and ``(m, d)``.

    Raises
    ------
    ValueError
        If the arrays are malformed or an edge index is out of range, if a
        tree constant is not finite and positive, if ``free_species`` holds a
        repeated or out-of-range index, if the free species do not
        parametrize the solution set one-to-one (the solutions do not form a
        ``d``-dimensional family in which the free coordinates determine all
        others), or if the equations have no solution (relative residual
        above ``1e-9``); returning a least-squares solution in either of the
        last two situations does not satisfy this contract.
    """
    return (np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_parametrize_steady_states(
    kinetic_complexes: np.ndarray,
    gcrn_edges: np.ndarray,
    tree_constants: np.ndarray,
    free_species: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    kinetic = np.asarray(kinetic_complexes, dtype=float)
    edges = np.asarray(gcrn_edges)
    tree = np.asarray(tree_constants, dtype=float)
    if kinetic.ndim != 2 or kinetic.shape[0] == 0 or kinetic.shape[1] == 0:
        raise ValueError("kinetic_complexes must have shape (V, m)")
    if not np.all(np.isfinite(kinetic)):
        raise ValueError("kinetic complexes must be finite")
    size, species = kinetic.shape
    if edges.ndim != 2 or edges.shape[1] < 2 or edges.shape[0] == 0:
        raise ValueError("gcrn_edges must have shape (E, 2) or (E, 3)")
    ends = edges[:, :2].astype(int)
    if np.any(ends < 0) or np.any(ends >= size):
        raise ValueError("edge endpoints must be existing vertices")
    if tree.shape != (size,) or not np.all(np.isfinite(tree)) or np.any(tree <= 0.0):
        raise ValueError("tree constants must be finite and positive")
    free = [int(value) for value in np.asarray(free_species, dtype=float).ravel()]
    if len(set(free)) != len(free) or any(value < 0 or value >= species for value in free):
        raise ValueError("free_species must hold distinct in-range species indices")

    matrix = kinetic[ends[:, 1]] - kinetic[ends[:, 0]]
    rhs = np.log(tree[ends[:, 1]]) - np.log(tree[ends[:, 0]])
    others = [index for index in range(species) if index not in free]
    determined = len(others)
    if np.linalg.matrix_rank(matrix, tol=1e-9) != determined:
        raise ValueError("the free species do not match the dimension of the solution set")
    if determined and np.linalg.matrix_rank(matrix[:, others], tol=1e-9) != determined:
        raise ValueError("the free species do not determine the other log-concentrations")

    offset = np.zeros(species)
    exponents = np.zeros((species, len(free)))
    for column, index in enumerate(free):
        exponents[index, column] = 1.0
    if determined:
        pseudo = np.linalg.pinv(matrix[:, others])
        particular = pseudo @ rhs
        scale = max(1.0, float(np.linalg.norm(rhs)))
        if np.linalg.norm(matrix[:, others] @ particular - rhs) > 1e-9 * scale:
            raise ValueError("the tree constants admit no complex-balanced equilibrium")
        offset[others] = particular
        if free:
            exponents[others] = -pseudo @ matrix[:, free]
    elif np.linalg.norm(rhs) > 1e-9:
        raise ValueError("the tree constants admit no complex-balanced equilibrium")
    return offset, exponents

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
        "def _psig(res, m, d):\n"
        "    if not isinstance(res, tuple) or len(res) != 2: return -1.0\n"
        "    off = np.asarray(res[0], dtype=float); ex = np.asarray(res[1], dtype=float)\n"
        "    if off.shape != (m,) or ex.shape != (m, d): return -1.0\n"
        "    f = np.concatenate([off, ex.ravel()])\n"
        "    w = np.cos(np.arange(f.size) + 1.0)\n"
        "    return float(np.sum(np.abs(f)) + np.sum(f * w))\n"
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
        "K = _tree(5, E, np.append(k, np.sqrt(1.3 * 2.1 * 1.6 / 0.7)))\n"
    )
    dual = (
        "Y = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0],"
        " [0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1],"
        " [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]])\n"
        "E = np.array([[0, 1, 0], [1, 2, 1], [2, 4, 2], [4, 2, 3], [4, 0, 4], [3, 5, 5],"
        " [5, 2, 6], [5, 0, 7], [6, 7, 8], [7, 6, 9], [2, 3, -1]])\n"
        "k = np.array([1.2, 2.3, 1.7, 0.6, 0.9, 2.6, 0.4, 1.1, 0.8, 1.5])\n"
        "K = _tree(8, E, np.append(k, 2.6 * 0.8 / 1.5))\n"
    )
    gene = (
        "Y = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],"
        " [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],"
        " [0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0],"
        " [0, 0, 0, 0, 0, 0, 1]])\n"
        "E = np.array([[0, 1, 0], [2, 3, 1], [1, 0, 2], [3, 2, 3], [4, 5, 4], [5, 4, 5],"
        " [6, 7, 6], [7, 6, 7], [8, 9, 8], [9, 8, 9]])\n"
        "K = _tree(10, E, np.array([0.9, 1.4, 0.6, 1.1, 2.3, 0.8, 1.7, 0.5, 0.35, 1.25]))\n"
    )
    return [
        {
            "setup": tree + toy,
            "call": "_psig(parametrize_steady_states(Y, E, K, np.array([], dtype=int)), 3, 0)",
            "gold_call": "_psig((np.log([np.sqrt(2.1 * 1.6 / (1.3 * 0.7)), np.sqrt(0.7 * 1.6 / (1.3 * 2.1)), 1.6 / 0.9]), np.zeros((3, 0))), 3, 0)",
        },
        {
            "setup": tree + dual,
            "call": "_psig(parametrize_steady_states(Y, E, K, np.array([1, 3, 5])), 8, 3)",
            "gold_call": "_psig(_oracle_parametrize_steady_states(Y, E, K, np.array([1, 3, 5])), 8, 3)",
        },
        {
            "setup": tree + dual,
            "call": "_psig(parametrize_steady_states(Y, E, K, np.array([4, 0, 2])), 8, 3)",
            "gold_call": "_psig(_oracle_parametrize_steady_states(Y, E, K, np.array([4, 0, 2])), 8, 3)",
        },
        {
            "setup": tree + gene,
            "call": "_psig(parametrize_steady_states(Y, E, K, np.array([2, 5])), 7, 2)",
            "gold_call": "_psig(_oracle_parametrize_steady_states(Y, E, K, np.array([2, 5])), 7, 2)",
        },
        {
            "setup": tree + status + gene,
            "call": "_status(lambda: parametrize_steady_states(Y, E, K, np.array([0, 2])))",
            "gold_call": "_status(lambda: _oracle_parametrize_steady_states(Y, E, K, np.array([0, 2])))",
        },
        {
            "setup": tree + status + toy + "K = _tree(5, E, np.append(k, 1.0))\n",
            "call": "_status(lambda: parametrize_steady_states(Y, E, K, np.array([], dtype=int)))",
            "gold_call": "_status(lambda: _oracle_parametrize_steady_states(Y, E, K, np.array([], dtype=int)))",
        },
    ]
