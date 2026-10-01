"""
Construct a directed graph on the reactions whose simple cycles are exactly the elementary flux modes and which respects common source complexes.

A reaction-to-reaction graph that is common-source and elementary-flux-mode compatible with a network whose modes are unitary and cover every reaction guarantees a weakly reversible, deficiency-zero network translation.

Returns
-------
np.ndarray: sorted integer edge list (tail reaction, head reaction), shape (q, 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_reaction_graph(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    flux_modes: np.ndarray,
) -> np.ndarray:
    """Return the edges of a compatible reaction-to-reaction graph.

    The vertices are the reactions ``0, ..., r - 1``. The returned graph must
    satisfy both conditions below, and any graph that does is acceptable.

    * Common-source compatibility: if reactions ``i`` and ``j`` have the same
      source complex and ``(k, i)`` is an edge, then ``(k, j)`` is an edge.
    * Flux-mode compatibility: the vertex sets of the simple directed cycles
      of the graph are exactly the supports of the rows of ``flux_modes``.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    flux_modes : np.ndarray
        Elementary flux modes of the network, shape ``(p, r)``.

    Returns
    -------
    np.ndarray
        Integer array with shape ``(q, 2)``; row ``(i, j)`` is the directed
        edge from reaction ``i`` to reaction ``j``. Edges are distinct, have
        ``i != j``, and rows are sorted in increasing lexicographic order.

    Raises
    ------
    ValueError
        If the complex arrays are invalid (not equal ``(r, m)`` shapes with
        ``r >= 1``, a negative, non-finite or non-integer entry, or a reaction
        with identical source and product), if ``flux_modes`` is not a
        nonempty ``(p, r)`` array, if any flux-mode entry is not 0 or 1, if
        some mode is zero or some reaction belongs to no mode, or if no graph
        satisfies both compatibility conditions; returning a graph that
        violates either condition does not satisfy this contract.
    """
    return np.zeros((0, 2), dtype=int)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_reaction_graph(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    flux_modes: np.ndarray,
) -> np.ndarray:
    """Reference implementation (exhaustive cyclic-order search)."""
    import itertools
    import numpy as np

    source = np.asarray(source_complexes, dtype=float)
    product = np.asarray(product_complexes, dtype=float)
    modes = np.asarray(flux_modes, dtype=float)
    if source.ndim != 2 or source.shape != product.shape or source.shape[0] == 0:
        raise ValueError("complex arrays must share a two-dimensional (r, m) shape")
    for values in (source, product):
        if not np.all(np.isfinite(values)):
            raise ValueError("complex entries must be finite")
        if np.any(values < 0.0) or np.any(values != np.round(values)):
            raise ValueError("complex entries must be nonnegative integers")
    if np.any(np.all(product == source, axis=1)):
        raise ValueError("every reaction must change at least one species")
    count = source.shape[0]
    if modes.ndim != 2 or modes.shape[0] == 0 or modes.shape[1] != count:
        raise ValueError("flux_modes must have shape (p, r) with p >= 1")
    if not np.all((modes == 0.0) | (modes == 1.0)):
        raise ValueError("the elementary flux modes must be unitary")
    if np.any(modes.sum(axis=1) == 0.0) or np.any(modes.sum(axis=0) == 0.0):
        raise ValueError("the flux modes must be nonzero and cover every reaction")

    supports = [tuple(np.flatnonzero(row).tolist()) for row in modes]
    targets = {frozenset(support) for support in supports}
    sources = [tuple(row) for row in source.tolist()]
    siblings = [[j for j in range(count) if sources[j] == sources[i]] for i in range(count)]

    def close(edges):
        closed = set(edges)
        for tail, head in edges:
            for other in siblings[head]:
                closed.add((tail, other))
        return closed

    def cycle_sets(edges):
        successors = [sorted(h for t, h in edges if t == node) for node in range(count)]
        found = set()
        for start in range(count):
            stack = [(start, (start,))]
            while stack:
                node, path = stack.pop()
                for nxt in successors[node]:
                    if nxt == start:
                        found.add(frozenset(path))
                    elif nxt > start and nxt not in path:
                        stack.append((nxt, path + (nxt,)))
        return found

    def cyclic_orders(support):
        for perm in itertools.permutations(support[1:]):
            ring = (support[0],) + perm
            yield {(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))}

    def search(position, edges):
        closed = close(edges)
        if any(tail == head for tail, head in closed):
            return None
        cycles = cycle_sets(closed)
        if not cycles <= targets:
            return None
        if position == len(supports):
            return closed if cycles == targets else None
        for ring in cyclic_orders(supports[position]):
            result = search(position + 1, edges | ring)
            if result is not None:
                return result
        return None

    graph = search(0, set())
    if graph is None:
        raise ValueError("no common-source and flux-mode compatible graph exists")
    return np.array(sorted(graph), dtype=int).reshape(-1, 2)

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
        "def _modes(r, sets):\n"
        "    F = np.zeros((len(sets), r), dtype=int)\n"
        "    for row, sup in enumerate(sets):\n"
        "        F[row, list(sup)] = 1\n"
        "    return F\n"
    )
    validator = (
        "def _valid(E, S, F):\n"
        "    E = np.asarray(E)\n"
        "    r = S.shape[0]\n"
        "    if E.ndim != 2 or E.shape[1] != 2 or E.shape[0] == 0: return 0\n"
        "    rows = [tuple(int(v) for v in e) for e in E.tolist()]\n"
        "    if rows != sorted(set(rows)): return 0\n"
        "    edges = set(rows)\n"
        "    if any(a == b or not (0 <= a < r and 0 <= b < r) for a, b in edges): return 0\n"
        "    src = [tuple(row) for row in S.tolist()]\n"
        "    for a, b in edges:\n"
        "        for j in range(r):\n"
        "            if src[j] == src[b] and (a, j) not in edges: return 0\n"
        "    succ = [sorted(h for t, h in edges if t == i) for i in range(r)]\n"
        "    found = set()\n"
        "    for s in range(r):\n"
        "        stack = [(s, (s,))]\n"
        "        while stack:\n"
        "            node, path = stack.pop()\n"
        "            for nxt in succ[node]:\n"
        "                if nxt == s: found.add(frozenset(path))\n"
        "                elif nxt > s and nxt not in path: stack.append((nxt, path + (nxt,)))\n"
        "    target = {frozenset(np.flatnonzero(row).tolist()) for row in F}\n"
        "    return int(found == target)\n"
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
    dual = (
        "S, P = _net(8, [([0], [1]), ([1, 2], [0, 3]), ([4, 3], [6]), ([6], [4, 3]),"
        " ([6], [4, 2]), ([5, 3], [7]), ([7], [5, 3]), ([7], [5, 2]),"
        " ([4], [5]), ([5], [4])])\n"
        "F = _modes(10, [(0, 1, 2, 4), (0, 1, 5, 7), (2, 3), (5, 6), (8, 9)])\n"
    )
    hinge = (
        "S, P = _net(6, [([0], [1]), ([1, 2], [3]), ([3], [1, 2]), ([3], [0, 4]),"
        " ([4], [2]), ([1, 5], [0, 5]), ([0, 5], [1, 5])])\n"
        "F = _modes(7, [(0, 1, 3, 4), (1, 2), (5, 6)])\n"
    )
    return [
        {
            "setup": builder + validator + dual,
            "call": "_valid(build_reaction_graph(S, P, F), S, F)",
            "gold_call": "_valid(_oracle_build_reaction_graph(S, P, F), S, F)",
        },
        {
            "setup": builder + validator + (
                "S, P = _net(4, [([0], [1]), ([1, 2], [0, 3]), ([0, 3], [1, 2]), ([3], [2])])\n"
                "F = _modes(4, [(0, 1, 3), (1, 2)])\n"
            ),
            "call": "_valid(build_reaction_graph(S, P, F), S, F)",
            "gold_call": "_valid(_oracle_build_reaction_graph(S, P, F), S, F)",
        },
        {
            "setup": builder + validator + hinge,
            "call": "_valid(build_reaction_graph(S, P, F), S, F)",
            "gold_call": "_valid(_oracle_build_reaction_graph(S, P, F), S, F)",
        },
        {
            "setup": builder + validator + dual + "F = F[[4, 2, 0, 3, 1]]\n",
            "call": "_valid(build_reaction_graph(S, P, F), S, F)",
            "gold_call": "1",
        },
        {
            "setup": builder + validator + (
                "S, P = _net(4, [([0], [1]), ([1], [0]), ([2, 2], [3]), ([3], [2, 2])])\n"
                "F = _modes(4, [(0, 1), (2, 3)])\n"
            ),
            "call": "_isig(build_reaction_graph(S, P, F), (4, 2))",
            "gold_call": "_isig([[0, 1], [1, 0], [2, 3], [3, 2]], (4, 2))",
        },
        {
            "setup": builder + validator + (
                "S, P = _net(3, [([0, 1], [2, 1]), ([2], []), ([], [0])])\n"
                "F = _modes(3, [(0, 1, 2)])\n"
            ),
            "call": "_valid(build_reaction_graph(S, P, F), S, F)",
            "gold_call": "_valid(_oracle_build_reaction_graph(S, P, F), S, F)",
        },
        {
            "setup": builder + validator + (
                "S, P = _net(5, [([0], [1]), ([1], [2]), ([2], [3]), ([3], [0]), ([2, 4], [3, 4])])\n"
                "F = _modes(5, [(0, 1, 2, 3), (0, 1, 3, 4)])\n"
            ),
            "call": "_valid(build_reaction_graph(S, P, F), S, F)",
            "gold_call": "_valid(_oracle_build_reaction_graph(S, P, F), S, F)",
        },
        {
            "setup": builder + status + "S, P = _net(3, [([0], [1]), ([1], [0]), ([2], [1]), ([1], [2])])\nF = _modes(4, [(0, 1), (2, 3)])\n",
            "call": "_status(lambda: build_reaction_graph(S, P, F))",
            "gold_call": "_status(lambda: _oracle_build_reaction_graph(S, P, F))",
        },
        {
            "setup": builder + status + "S, P = _net(3, [([0], [1]), ([1], [2]), ([2, 2], [0, 0])])\nF = np.array([[2, 2, 1]])\n",
            "call": "_status(lambda: build_reaction_graph(S, P, F))",
            "gold_call": "_status(lambda: _oracle_build_reaction_graph(S, P, F))",
        },
        {
            "setup": builder + status + dual.replace(", (8, 9)])", "])"),
            "call": "_status(lambda: build_reaction_graph(S, P, F))",
            "gold_call": "_status(lambda: _oracle_build_reaction_graph(S, P, F))",
        },
    ]
