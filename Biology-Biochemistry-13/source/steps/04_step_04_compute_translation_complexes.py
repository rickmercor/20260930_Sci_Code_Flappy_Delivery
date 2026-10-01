"""
Propagate translation complexes along the reaction-to-reaction graph so that every edge becomes product-to-source compatible.

Network translation adds a species combination to both sides of a reaction, which changes the complex graph but not the reaction vector, so the translated network keeps the original dynamics when each reaction retains its original source complex as kinetic complex.

Returns
-------
np.ndarray: integer translation complexes, shape (r, m).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_translation_complexes(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    reaction_edges: np.ndarray,
) -> np.ndarray:
    """Return the translation complex of every reaction.

    Reaction ``k`` is translated to ``source_complexes[k] + alpha[k] ->
    product_complexes[k] + alpha[k]``. Every edge ``(i, j)`` of the
    reaction-to-reaction graph must become product-to-source compatible, that
    is, the translated product of reaction ``i`` equals the translated source
    of reaction ``j``. On each weakly connected component of the graph (a
    reaction touched by no edge is a component by itself), ``alpha`` is
    anchored at zero on the component's lowest-index reaction and propagated
    along the edges; afterwards, for every species separately, all
    translation complexes of the component are raised by the smallest
    nonnegative integer that makes every translated source and product
    complex of that component nonnegative.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    reaction_edges : np.ndarray
        Integer array with shape ``(q, 2)`` of directed edges between
        reactions.

    Returns
    -------
    np.ndarray
        Integer array ``alpha`` with shape ``(r, m)``.

    Raises
    ------
    ValueError
        If the complex arrays are invalid (not equal ``(r, m)`` shapes with
        ``r >= 1``, or a negative, non-finite or non-integer entry), if
        ``reaction_edges`` is not a ``(q, 2)`` integer array of in-range,
        non-self-loop edges (an out-of-range index must raise ValueError,
        not IndexError), or if no choice of translation complexes makes
        every edge product-to-source compatible (the equations along the
        edges are inconsistent).
    """
    return np.zeros((0, 0), dtype=int)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_translation_complexes(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    reaction_edges: np.ndarray,
) -> np.ndarray:
    """Reference implementation (graph traversal of alpha_i - alpha_j = y_s(j) - y_p(i))."""
    import numpy as np

    source = np.asarray(source_complexes, dtype=float)
    product = np.asarray(product_complexes, dtype=float)
    if source.ndim != 2 or source.shape != product.shape or source.shape[0] == 0:
        raise ValueError("complex arrays must share a two-dimensional (r, m) shape")
    for values in (source, product):
        if not np.all(np.isfinite(values)):
            raise ValueError("complex entries must be finite")
        if np.any(values < 0.0) or np.any(values != np.round(values)):
            raise ValueError("complex entries must be nonnegative integers")
    count, species = source.shape
    edges = np.asarray(reaction_edges)
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("reaction_edges must have shape (q, 2)")
    if not np.all(np.isfinite(edges.astype(float))) or np.any(edges != np.round(edges)):
        raise ValueError("reaction indices must be integers")
    edges = edges.astype(int)
    if np.any(edges < 0) or np.any(edges >= count) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("edges must join two different existing reactions")
    source = source.astype(np.int64)
    product = product.astype(np.int64)

    neighbours = [[] for _ in range(count)]
    for tail, head in edges:
        # alpha[head] = alpha[tail] + product[tail] - source[head]
        neighbours[tail].append((head, product[tail] - source[head]))
        neighbours[head].append((tail, source[head] - product[tail]))

    alpha = np.zeros((count, species), dtype=np.int64)
    component = np.full(count, -1)
    for seed in range(count):
        if component[seed] >= 0:
            continue
        component[seed] = seed
        queue = [seed]
        while queue:
            current = queue.pop(0)
            for other, shift in neighbours[current]:
                if component[other] < 0:
                    component[other] = seed
                    alpha[other] = alpha[current] + shift
                    queue.append(other)
    for tail, head in edges:
        if not np.array_equal(product[tail] + alpha[tail], source[head] + alpha[head]):
            raise ValueError("the product-to-source equations are inconsistent")
    for seed in np.unique(component):
        members = np.flatnonzero(component == seed)
        translated = np.vstack([source[members] + alpha[members], product[members] + alpha[members]])
        alpha[members] += np.maximum(0, -translated.min(axis=0))
    return alpha.astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    builder = (
        "import numpy as np\n"
        "def _net(m, rx):\n"
        "    S = np.zeros((len(rx), m), dtype=int)\n"
        "    P = np.zeros((len(rx), m), dtype=int)\n"
        "    for r, (a, b) in enumerate(rx):\n"
        "        for i in a: S[r, i] += 1\n"
        "        for i in b: P[r, i] += 1\n"
        "    return S, P\n"
        "def _isig(a, shape):\n"
        "    a = np.asarray(a)\n"
        "    if a.shape != shape: return -1\n"
        "    f = a.ravel()\n"
        "    w = np.arange(1, f.size + 1)\n"
        "    return int(np.sum(f * w) + 3 * np.sum(f * f * (w % 5 + 1)))\n"
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
    kinase = "S, P = _net(4, [([0], [1]), ([1, 2], [0, 3]), ([0, 3], [1, 2]), ([3], [2])])\n"
    dual = (
        "S, P = _net(8, [([0], [1]), ([1, 2], [0, 3]), ([4, 3], [6]), ([6], [4, 3]),"
        " ([6], [4, 2]), ([5, 3], [7]), ([7], [5, 3]), ([7], [5, 2])])\n"
    )
    return [
        {
            "setup": builder + kinase + "E = np.array([[0, 3], [1, 0], [1, 2], [2, 1], [3, 1]])\n",
            "call": "_isig(compute_translation_complexes(S, P, E), (4, 4))",
            "gold_call": "_isig([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], (4, 4))",
        },
        {
            "setup": builder + kinase + "E = np.array([[0, 1], [1, 2], [1, 3], [2, 1], [3, 0]])\n",
            "call": "_isig(compute_translation_complexes(S, P, E), (4, 4))",
            "gold_call": "_isig([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], (4, 4))",
        },
        {
            "setup": builder + "S, P = _net(7, [([0], [0, 2]), ([2], [])])\nE = np.array([[0, 1], [1, 0]])\n",
            "call": "_isig(compute_translation_complexes(S, P, E), (2, 7))",
            "gold_call": "_isig([[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], (2, 7))",
        },
        {
            "setup": builder + dual + (
                "E = np.array([[0, 1], [1, 2], [1, 5], [2, 3], [2, 4], [3, 2],"
                " [4, 0], [5, 6], [5, 7], [6, 5], [7, 0]])\n"
            ),
            "call": "_isig(compute_translation_complexes(S, P, E), (8, 8))",
            "gold_call": "_isig(_oracle_compute_translation_complexes(S, P, E), (8, 8))",
        },
        {
            "setup": builder + dual + (
                "E = np.array([[0, 1], [1, 5], [1, 2], [5, 6], [5, 7], [6, 5],"
                " [7, 0], [2, 3], [2, 4], [3, 2], [4, 0]])\n"
            ),
            "call": "_isig(compute_translation_complexes(S, P, E), (8, 8))",
            "gold_call": "_isig(_oracle_compute_translation_complexes(S, P, E), (8, 8))",
        },
        {
            "setup": builder + (
                "S, P = _net(3, [([0, 1], [2, 1]), ([0], [1]), ([1], [0]), ([2], []), ([], [0])])\n"
                "E = np.array([[0, 3], [3, 4], [4, 0], [1, 2], [2, 1]])\n"
            ),
            "call": "_isig(compute_translation_complexes(S, P, E), (5, 3))",
            "gold_call": "_isig(_oracle_compute_translation_complexes(S, P, E), (5, 3))",
        },
        {
            "setup": builder + (
                "S, P = _net(3, [([0, 1], [2, 1]), ([0], [1]), ([1], [0]), ([2], []), ([], [0])])\n"
                "E = np.array([[0, 4], [4, 3], [3, 0], [1, 2], [2, 1]])\n"
            ),
            "call": "_isig(compute_translation_complexes(S, P, E), (5, 3))",
            "gold_call": "_isig([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 1]], (5, 3))",
        },
        {
            "setup": builder + (
                "S, P = _net(6, [([0], [1]), ([1, 2], [0, 3]), ([0, 3], [1, 2]), ([3], [2]),"
                " ([4], [5]), ([5], [4])])\n"
                "E = np.array([[0, 1], [1, 2], [1, 3], [2, 1], [3, 0], [4, 5], [5, 4]])\n"
            ),
            "call": "_isig(compute_translation_complexes(S, P, E), (6, 6))",
            "gold_call": "_isig([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], (6, 6))",
        },
        {
            "setup": builder + status + kinase + "E = np.array([[0, 1], [1, 3], [3, 0], [1, 2], [2, 1], [0, 2]])\n",
            "call": "_status(lambda: compute_translation_complexes(S, P, E))",
            "gold_call": "_status(lambda: _oracle_compute_translation_complexes(S, P, E))",
        },
        {
            "setup": builder + status + kinase + "E = np.array([[0, 4], [1, 0]])\n",
            "call": "_status(lambda: compute_translation_complexes(S, P, E))",
            "gold_call": "_status(lambda: _oracle_compute_translation_complexes(S, P, E))",
        },
    ]
