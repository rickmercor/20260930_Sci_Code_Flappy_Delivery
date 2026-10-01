"""
Count complexes, linkage classes and stoichiometric rank of a (translated) network and report its deficiency and weak reversibility.

The deficiency n - l - s measures the linear dependence among the reactions of a complex graph, and a network is weakly reversible when every reaction lies on a directed cycle of that graph.

Returns
-------
np.ndarray: integer vector [n, l, s, deficiency, weakly_reversible], shape (5,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_network_deficiency(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Return ``[n, l, s, deficiency, weakly_reversible]`` of a translated network.

    Reaction ``k`` of the translated network is
    ``source_complexes[k] + translation[k] -> product_complexes[k] +
    translation[k]``; pass a zero ``translation`` for the original network.
    ``n`` is the number of distinct complexes of the translated network,
    ``l`` its number of linkage classes, ``s`` the rank of its stoichiometric
    matrix, ``deficiency = n - l - s``, and ``weakly_reversible`` is 1 when
    every reaction lies on a directed cycle of the complex graph and 0
    otherwise.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    translation : np.ndarray
        Integer array with shape ``(r, m)``.

    Returns
    -------
    np.ndarray
        Integer array with shape ``(5,)``.

    Raises
    ------
    ValueError
        If the three arrays do not share a two-dimensional ``(r, m)`` shape
        with ``r >= 1``, if any entry is non-finite or not an integer, if the
        original or translated complexes have a negative entry, or if some
        reaction has identical source and product complexes.
    """
    return np.zeros(5, dtype=int)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_network_deficiency(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    source = np.asarray(source_complexes, dtype=float)
    product = np.asarray(product_complexes, dtype=float)
    shift = np.asarray(translation, dtype=float)
    if source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("complex arrays must be two-dimensional with r >= 1")
    if product.shape != source.shape or shift.shape != source.shape:
        raise ValueError("the three arrays must share the (r, m) shape")
    for values in (source, product, shift):
        if not np.all(np.isfinite(values)) or np.any(values != np.round(values)):
            raise ValueError("entries must be finite integers")
    tails_c = source + shift
    heads_c = product + shift
    if min(source.min(), product.min(), tails_c.min(), heads_c.min()) < 0.0:
        raise ValueError("original and translated complexes must be nonnegative")
    if np.any(np.all(source == product, axis=1)):
        raise ValueError("every reaction must change at least one species")

    index = {}
    for row in np.vstack([tails_c, heads_c]):
        index.setdefault(tuple(row), len(index))
    size = len(index)
    tails = [index[tuple(row)] for row in tails_c]
    heads = [index[tuple(row)] for row in heads_c]
    linked = np.eye(size, dtype=bool)
    reach = np.eye(size, dtype=bool)
    for tail, head in zip(tails, heads):
        linked[tail, head] = linked[head, tail] = True
        reach[tail, head] = True
    for pivot in range(size):
        linked |= linked[:, [pivot]] & linked[[pivot], :]
        reach |= reach[:, [pivot]] & reach[[pivot], :]
    linkage = len({tuple(row) for row in linked})
    rank = int(np.linalg.matrix_rank(product - source, tol=1e-9))
    reversible = all(reach[head, tail] for tail, head in zip(tails, heads))
    return np.array([size, linkage, rank, size - linkage - rank, int(reversible)], dtype=int)

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
        "def _vsig(a):\n"
        "    a = np.asarray(a)\n"
        "    if a.shape != (5,): return -1\n"
        "    return int(np.sum(a * np.array([1, 10, 100, 1000, 10000])))\n"
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
    toy = "S, P = _net(3, [([0, 1], [2, 1]), ([0], [1]), ([1], [0]), ([2], []), ([], [0])])\n"
    gene = (
        "S, P = _net(7, [([0], [0, 2]), ([1], [1, 3]), ([2], []), ([3], []),"
        " ([3, 3], [4]), ([4], [3, 3]), ([1, 2], [5]), ([5], [1, 2]),"
        " ([0, 4], [6]), ([6], [0, 4])])\n"
    )
    return [
        {
            "setup": builder + toy + "A = np.zeros_like(S)\n",
            "call": "_vsig(compute_network_deficiency(S, P, A))",
            "gold_call": "_vsig([6, 2, 3, 1, 0])",
        },
        {
            "setup": builder + toy + "A = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 1]])\n",
            "call": "_vsig(compute_network_deficiency(S, P, A))",
            "gold_call": "_vsig(_oracle_compute_network_deficiency(S, P, A))",
        },
        {
            "setup": builder + gene + "A = np.zeros_like(S)\n",
            "call": "_vsig(compute_network_deficiency(S, P, A))",
            "gold_call": "_vsig([13, 6, 5, 2, 0])",
        },
        {
            "setup": builder + gene + "A = np.zeros_like(S); A[2, 0] = 1; A[3, 1] = 1\n",
            "call": "_vsig(compute_network_deficiency(S, P, A))",
            "gold_call": "_vsig(_oracle_compute_network_deficiency(S, P, A))",
        },
        {
            "setup": builder + "S, P = _net(4, [([0], [1]), ([1, 2], [0, 3]), ([0, 3], [1, 2]), ([3], [2])])\nA = np.zeros_like(S)\n",
            "call": "_vsig(compute_network_deficiency(S, P, A))",
            "gold_call": "_vsig([6, 3, 2, 1, 0])",
        },
        {
            "setup": builder + (
                "S, P = _net(4, [([0], [1]), ([1, 2], [0, 3]), ([0, 3], [1, 2]), ([3], [2])])\n"
                "A = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])\n"
            ),
            "call": "_vsig(compute_network_deficiency(S, P, A))",
            "gold_call": "_vsig(_oracle_compute_network_deficiency(S, P, A))",
        },
        {
            "setup": builder + (
                "S, P = _net(3, [([0], [1]), ([1], [2]), ([2], [1]), ([1], [0]), ([0, 0], [2])])\n"
                "A = np.zeros_like(S)\n"
            ),
            "call": "_vsig(compute_network_deficiency(S, P, A))",
            "gold_call": "_vsig(_oracle_compute_network_deficiency(S, P, A))",
        },
        {
            "setup": builder + status + toy + "A = np.zeros_like(S); A[3, 2] = -2\n",
            "call": "_status(lambda: compute_network_deficiency(S, P, A))",
            "gold_call": "_status(lambda: _oracle_compute_network_deficiency(S, P, A))",
        },
    ]
