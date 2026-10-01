"""
Turn a translated network into a generalized network with stoichiometric and kinetic complexes, splitting shared translated complexes by phantom edges.

In a generalized network every vertex carries a stoichiometric complex that fixes the reaction vectors and a kinetic complex that fixes the mass-action monomial; phantom edges join vertices with the same stoichiometric complex and therefore never change the dynamics.

Returns
-------
tuple: (stoichiometric complexes (V, m), kinetic complexes (V, m), edges (E, 3)) as integer arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_generalized_network(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(stoichiometric_complexes, kinetic_complexes, edges)``.

    Reaction ``k`` has translated source ``c_s(k) = source_complexes[k] +
    translation[k]``, translated product ``c_p(k) = product_complexes[k] +
    translation[k]`` and kinetic complex ``source_complexes[k]``. Distinct
    translated complexes are ordered by first appearance when the reactions
    are scanned in index order, reading ``c_s(k)`` before ``c_p(k)``. A
    translated complex that is the translated source of reactions with ``d``
    different kinetic complexes becomes ``d`` vertices, one per kinetic
    complex, ordered by the smallest index of a reaction carrying that
    kinetic complex; vertices are numbered by walking the translated
    complexes in their order and, within each, its vertices in their order.
    Edges are listed as follows: first one effective edge per reaction ``k``
    in increasing ``k``, from the vertex of ``c_s(k)`` with kinetic complex
    ``source_complexes[k]`` to the first vertex of ``c_p(k)``, with third
    entry ``k``; then, walking the translated complexes in order, one phantom
    edge from each of its vertices to the next vertex of the same translated
    complex, with third entry ``-1``.

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
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Integer arrays with shapes ``(V, m)`` (stoichiometric complex of each
        vertex), ``(V, m)`` (kinetic complex of each vertex) and ``(E, 3)``
        (rows ``[tail vertex, head vertex, reaction index or -1]``).

    Raises
    ------
    ValueError
        If the three arrays do not share a two-dimensional ``(r, m)`` shape
        with ``r >= 1``, if any entry is non-finite or not an integer, if an
        original or translated complex has a negative entry, or if some
        translated complex is the translated source of no reaction, so that
        it has no kinetic complex (raise ValueError there, not a KeyError or
        IndexError).
    """
    return (np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=int), np.zeros((0, 3), dtype=int))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_generalized_network(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    import itertools
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
    source = source.astype(int)
    product = product.astype(int)
    shift = shift.astype(int)
    tails_c = source + shift
    heads_c = product + shift
    if min(source.min(), product.min(), tails_c.min(), heads_c.min()) < 0:
        raise ValueError("original and translated complexes must be nonnegative")

    order = []
    for k in range(source.shape[0]):
        for complex_ in (tuple(tails_c[k]), tuple(heads_c[k])):
            if complex_ not in order:
                order.append(complex_)
    labels = {complex_: [] for complex_ in order}
    for k in range(source.shape[0]):
        kinetic = tuple(source[k])
        if kinetic not in labels[tuple(tails_c[k])]:
            labels[tuple(tails_c[k])].append(kinetic)

    vertex = {}
    stoichiometric, kinetic_rows = [], []
    for complex_ in order:
        if not labels[complex_]:
            raise ValueError("a translated complex is the source of no reaction")
        for kinetic in labels[complex_]:
            vertex[(complex_, kinetic)] = len(stoichiometric)
            stoichiometric.append(complex_)
            kinetic_rows.append(kinetic)

    edges = []
    for k in range(source.shape[0]):
        tail = vertex[(tuple(tails_c[k]), tuple(source[k]))]
        head_complex = tuple(heads_c[k])
        head = vertex[(head_complex, labels[head_complex][0])]
        edges.append((tail, head, k))
    for complex_ in order:
        copies = labels[complex_]
        for first, second in itertools.pairwise(copies):
            edges.append((vertex[(complex_, first)], vertex[(complex_, second)], -1))
    return (
        np.array(stoichiometric, dtype=int),
        np.array(kinetic_rows, dtype=int),
        np.array(edges, dtype=int).reshape(-1, 3),
    )

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
        "def _gsig(g, v, m, e):\n"
        "    if not isinstance(g, tuple) or len(g) != 3: return -1\n"
        "    parts = [_isig(g[0], (v, m)), _isig(g[1], (v, m)), _isig(g[2], (e, 3))]\n"
        "    if min(parts) < 0: return -1\n"
        "    return parts[0] + 7 * parts[1] + 13 * parts[2]\n"
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
        "S, P = _net(3, [([0, 1], [2, 1]), ([0], [1]), ([1], [0]), ([2], []), ([], [0])])\n"
        "A = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]])\n"
    )
    dual = (
        "S, P = _net(8, [([0], [1]), ([1, 2], [0, 3]), ([4, 3], [6]), ([6], [4, 3]),"
        " ([6], [4, 2]), ([5, 3], [7]), ([7], [5, 3]), ([7], [5, 2]),"
        " ([4], [5]), ([5], [4])])\n"
        "A = np.zeros_like(S)\n"
        "A[0] = [0, 0, 1, 0, 1, 1, 0, 0]; A[1] = [0, 0, 0, 0, 1, 1, 0, 0]\n"
        "A[2:5] = [1, 0, 0, 0, 0, 1, 0, 0]; A[5:8] = [1, 0, 0, 0, 1, 0, 0, 0]\n"
    )
    return [
        {
            "setup": builder + toy,
            "call": "_gsig(build_generalized_network(S, P, A), 5, 3, 6)",
            "gold_call": "_gsig(_oracle_build_generalized_network(S, P, A), 5, 3, 6)",
        },
        {
            "setup": builder + toy,
            "call": "int(build_generalized_network(S, P, A)[2][5, 0] * 10 + build_generalized_network(S, P, A)[2][5, 1])",
            "gold_call": "34",
        },
        {
            "setup": builder + dual,
            "call": "_gsig(build_generalized_network(S, P, A), 8, 8, 11)",
            "gold_call": "_gsig(_oracle_build_generalized_network(S, P, A), 8, 8, 11)",
        },
        {
            "setup": builder + (
                "S, P = _net(7, [([0], [0, 2]), ([1], [1, 3]), ([2], []), ([3], []),"
                " ([3, 3], [4]), ([4], [3, 3]), ([1, 2], [5]), ([5], [1, 2]),"
                " ([0, 4], [6]), ([6], [0, 4])])\n"
                "A = np.zeros_like(S); A[2, 0] = 1; A[3, 1] = 1\n"
            ),
            "call": "_gsig(build_generalized_network(S, P, A), 10, 7, 10)",
            "gold_call": "_gsig(_oracle_build_generalized_network(S, P, A), 10, 7, 10)",
        },
        {
            "setup": builder + (
                "S, P = _net(4, [([0], [1]), ([1, 2], [0, 3]), ([0, 3], [1, 2]), ([3], [2])])\n"
                "A = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])\n"
            ),
            "call": "_gsig(build_generalized_network(S, P, A), 4, 4, 5)",
            "gold_call": "_gsig(_oracle_build_generalized_network(S, P, A), 4, 4, 5)",
        },
        {
            "setup": builder + (
                "S, P = _net(2, [([0], [1]), ([1], [0]), ([], [0, 1]),"
                " ([1], [0]), ([0], [1]), ([0, 1], [])])\n"
                "A = np.array([[1, 1], [2, 0], [2, 1], [1, 1], [2, 0], [2, 1]])\n"
            ),
            "call": "_gsig(build_generalized_network(S, P, A), 6, 2, 8)",
            "gold_call": "_gsig(_oracle_build_generalized_network(S, P, A), 6, 2, 8)",
        },
        {
            "setup": builder + status + toy.replace("A = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]])", "A = np.zeros((5, 3), dtype=int)"),
            "call": "_status(lambda: build_generalized_network(S, P, A))",
            "gold_call": "_status(lambda: _oracle_build_generalized_network(S, P, A))",
        },
    ]
