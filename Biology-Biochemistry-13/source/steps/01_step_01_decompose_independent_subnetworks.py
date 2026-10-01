"""
Partition the reactions of a mass-action network into its finest independent decomposition.

A partition of the reactions is an independent decomposition when the stoichiometric rank of the whole network equals the sum of the ranks of its blocks, so the positive steady states of the network are the intersection of those of its blocks.

Returns
-------
np.ndarray: integer block label of each reaction, shape (r,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def decompose_independent_subnetworks(
    source_complexes: np.ndarray, product_complexes: np.ndarray
) -> np.ndarray:
    """Return the block label of every reaction in the finest independent decomposition.

    Reaction ``k`` converts the complex ``source_complexes[k]`` into the
    complex ``product_complexes[k]``; both rows are nonnegative integer
    stoichiometric vectors over the same ordered species, and the reaction
    vector is ``product_complexes[k] - source_complexes[k]``. The finest
    independent decomposition is the partition of the reactions into the
    largest number of blocks such that the rank of the matrix of all reaction
    vectors equals the sum of the ranks of the reaction vectors of each block.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.

    Returns
    -------
    np.ndarray
        Integer array with shape ``(r,)`` holding the block of each reaction.
        Blocks are numbered ``0, 1, 2, ...`` in increasing order of the
        smallest reaction index they contain.

    Raises
    ------
    ValueError
        If the two arrays are not two-dimensional with the same shape and at
        least one reaction, if any entry is negative, non-finite or not an
        integer, or if some reaction has identical source and product
        complexes.
    """
    return np.zeros(0, dtype=int)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_decompose_independent_subnetworks(
    source_complexes: np.ndarray, product_complexes: np.ndarray
) -> np.ndarray:
    """Reference implementation."""
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
    vectors = product - source
    if np.any(np.all(vectors == 0.0, axis=1)):
        raise ValueError("every reaction must change at least one species")

    count = vectors.shape[0]
    basis = []
    for index in range(count):
        trial = basis + [index]
        if np.linalg.matrix_rank(vectors[trial], tol=1e-9) == len(trial):
            basis.append(index)

    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(first, second):
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    # Each non-basis reaction closes a fundamental circuit with the basis
    # reactions carrying a nonzero coefficient; circuits define the blocks.
    basis_matrix = vectors[basis].T
    for index in range(count):
        if index in basis:
            continue
        coefficients = np.linalg.lstsq(basis_matrix, vectors[index], rcond=None)[0]
        for member, value in zip(basis, coefficients):
            if abs(value) > 1e-9:
                join(index, member)

    labels = np.empty(count, dtype=int)
    names = {}
    for index in range(count):
        root = find(index)
        if root not in names:
            names[root] = len(names)
        labels[index] = names[root]
    return labels

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
        "def _isig(a, n):\n"
        "    a = np.asarray(a)\n"
        "    if a.shape != (n,): return -1\n"
        "    w = np.arange(1, n + 1)\n"
        "    return int(np.sum(a * w) + 3 * np.sum(a * a * (w % 5 + 1)))\n"
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
    gene = (
        "S, P = _net(7, [([0], [0, 2]), ([1], [1, 3]), ([2], []), ([3], []),"
        " ([3, 3], [4]), ([4], [3, 3]), ([1, 2], [5]), ([5], [1, 2]),"
        " ([0, 4], [6]), ([6], [0, 4])])\n"
    )
    dual = (
        "S, P = _net(8, [([0], [1]), ([1, 2], [0, 3]), ([4, 3], [6]), ([6], [4, 3]),"
        " ([6], [4, 2]), ([5, 3], [7]), ([7], [5, 3]), ([7], [5, 2]),"
        " ([4], [5]), ([5], [4])])\n"
    )
    return [
        {
            "setup": builder + gene,
            "call": "_isig(decompose_independent_subnetworks(S, P), 10)",
            "gold_call": "_isig(_oracle_decompose_independent_subnetworks(S, P), 10)",
        },
        {
            "setup": builder + gene,
            "call": "_isig(decompose_independent_subnetworks(S, P), 10)",
            "gold_call": "_isig([0, 1, 0, 1, 2, 2, 3, 3, 4, 4], 10)",
        },
        {
            "setup": builder + dual,
            "call": "_isig(decompose_independent_subnetworks(S, P), 10)",
            "gold_call": "_isig(_oracle_decompose_independent_subnetworks(S, P), 10)",
        },
        {
            "setup": builder + "S, P = _net(4, [([0], [1]), ([1, 2], [0, 3]), ([0, 3], [1, 2]), ([3], [2])])\n",
            "call": "_isig(decompose_independent_subnetworks(S, P), 4)",
            "gold_call": "_isig(_oracle_decompose_independent_subnetworks(S, P), 4)",
        },
        {
            "setup": builder + "S, P = _net(3, [([0, 1], [2, 1]), ([0], [1]), ([1], [0]), ([2], []), ([], [0])])\n",
            "call": "_isig(decompose_independent_subnetworks(S, P), 5)",
            "gold_call": "_isig(_oracle_decompose_independent_subnetworks(S, P), 5)",
        },
        {
            "setup": builder + (
                "S, P = _net(5, [([0], [1]), ([2, 0], [2, 1]), ([2], [3]), ([3], [2]),"
                " ([1, 3], [4]), ([4], [0, 2])])\n"
            ),
            "call": "_isig(decompose_independent_subnetworks(S, P), 6)",
            "gold_call": "_isig(_oracle_decompose_independent_subnetworks(S, P), 6)",
        },
        {
            "setup": builder + "S, P = _net(4, [([0], [1]), ([0, 2], [1, 2]), ([2], [3]), ([3], [2])])\n",
            "call": "_isig(decompose_independent_subnetworks(S, P), 4)",
            "gold_call": "_isig(_oracle_decompose_independent_subnetworks(S, P), 4)",
        },
        {
            "setup": builder + status + "S, P = _net(3, [([0], [1]), ([1, 2], [2, 1]), ([1], [0])])\n",
            "call": "_status(lambda: decompose_independent_subnetworks(S, P))",
            "gold_call": "_status(lambda: _oracle_decompose_independent_subnetworks(S, P))",
        },
        {
            "setup": builder + status + "S = np.array([[1, 0], [0, 1]]); P = np.array([[0, 1], [1, 0], [1, 1]])\n",
            "call": "_status(lambda: decompose_independent_subnetworks(S, P))",
            "gold_call": "_status(lambda: _oracle_decompose_independent_subnetworks(S, P))",
        },
    ]
