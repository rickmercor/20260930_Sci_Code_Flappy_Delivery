"""
Turn a closed constraint relation into the canonical directed acyclic graph that realizes it, by collapsing the leaf pairs the relation forces to share one least common ancestor and keeping only the covering comparisons of the resulting partial order as arcs.

A closed constraint relation is a preorder on the constrained leaf pairs: it is reflexive on its support and transitive, but two distinct pairs may constrain each other in both directions. Two pairs related in both directions cannot be separated by any realizing DAG, because each of their LCAs would have to be a descendant of the other, and in a DAG that forces the two LCAs to be one and the same vertex; collapsing such pairs into a single class is therefore not a convenience but the only way to obtain an antisymmetric order, and the classes are precisely the vertices a realizing DAG has to provide, with the class of a singleton pair being the vertex identified with that leaf. Pairs outside the support carry no constraint at all and contribute no vertex, since the construction is not obliged to give them a well-defined LCA; that freedom is exactly what the graph-level repair of a later step exploits. Drawing every comparison as an arc would create shortcuts that carry no information and would spoil the least-common-ancestor structure, because an ancestor reachable through an intermediate class must not also be a parent; keeping only the covering comparisons, that is drawing the arc j -> i exactly when class i is strictly below class j and no third class sits strictly between them, gives the sparsest DAG whose reachability is exactly the given order, and this canonical DAG realizes the constraints whenever they are realizable at all. The vertex numbering places the n leaf classes first, in leaf order, so vertex a is leaf a for a < n, and then the remaining classes ordered by the smallest pair-table index they contain; since the singletons occupy rows 0 to n-1 of that table, numbering the classes by first appearance already puts the leaves first.

Returns
-------
np.ndarray of shape (V, V), int: the 0/1 adjacency matrix of the canonical DAG, one vertex per constraint class, leaves occupying vertices 0 to n-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def canonical_dag(n: int, rel: np.ndarray) -> np.ndarray:
    """Canonical DAG of a closed relation of pairwise LCA constraints.

    Parameters
    ----------
    n : int
        Number of leaves, labelled 0 to n-1.
    rel : np.ndarray
        Integer 0/1 array of shape (m, m) with m = n + n * (n - 1) // 2,
        holding a closed constraint relation on the canonical pair table.

    Returns
    -------
    adj : np.ndarray
        Integer 0/1 array of shape (V, V); entry [u, v] is 1 when the DAG has
        the arc u -> v. Vertices 0 to n-1 are the leaves, in leaf order, and
        the remaining vertices are the other classes in increasing order of
        their smallest pair-table index.


    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((0, 0), dtype=int)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_canonical_dag(n: int, rel: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    rel = (np.asarray(rel, dtype=np.int64) != 0)
    m = n + n * (n - 1) // 2
    if rel.shape != (m, m):
        raise ValueError("rel must be square and match the leaf count")

    # -- Mutual constraint in both directions is the equivalence; classes are
    #    numbered by the smallest pair-table index they contain.
    supported = (rel.any(axis=1) | rel.any(axis=0))
    labels = np.full(m, -1, dtype=np.int64)
    reps = []
    for i in range(m):
        if not supported[i] or labels[i] >= 0:
            continue
        labels[i] = len(reps)
        for j in range(i + 1, m):
            if supported[j] and rel[i, j] and rel[j, i]:
                labels[j] = len(reps)
        reps.append(i)

    # -- Rows 0 to n-1 are the singletons, so the leaf classes come first.
    leaf_class = []
    for a in range(n):
        if int(labels[a]) < 0:
            raise ValueError("every leaf must carry a constraint class")
        leaf_class.append(int(labels[a]))
    if len(set(leaf_class)) != n:
        raise ValueError("two leaves were merged into one class")
    seen = set(leaf_class)
    order = leaf_class + [c for c in range(len(reps)) if c not in seen]
    pos = {c: k for k, c in enumerate(order)}
    size = len(order)

    below = np.zeros((size, size), dtype=bool)
    for ci in order:
        for cj in order:
            if rel[reps[ci], reps[cj]]:
                below[pos[ci], pos[cj]] = True

    # -- Covering comparisons only: no third class strictly in between.
    adj = np.zeros((size, size), dtype=np.int64)
    for j in range(size):
        for i in range(size):
            if i == j or not below[i, j]:
                continue
            covering = True
            for c in range(size):
                if c != i and c != j and below[i, c] and below[c, j]:
                    covering = False
                    break
            if covering:
                adj[j, i] = 1
    return adj

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def closed(n, clusters):
    rows = [(a, a) for a in range(n)]
    rows += [(a, b) for a in range(n) for b in range(a + 1, n)]
    m = len(rows)
    sets = [frozenset([a]) for a in range(n)]
    sets += [frozenset(c) for c in clusters]
    top = []
    for a, b in rows:
        hit = [s for s in sets if a in s and b in s]
        top.append(min(hit, key=len) if hit else None)
    M = np.zeros((m, m), dtype=np.int64)
    for i in range(m):
        for j in range(m):
            if top[i] is not None and top[j] is not None and top[i] <= top[j]:
                M[i, j] = 1
    return M
def red(a):
    v = np.asarray(a, dtype=float).ravel()
    w = np.arange(1.0, v.size + 1.0) ** 2
    return float(v.size * 10000.0 + float(v @ w))
M4 = closed(4, [[0, 1], [0, 1, 2], [0, 1, 2, 3]])
M5 = closed(5, [[0, 1], [2, 3], [0, 1, 2, 3, 4]])
M3 = closed(3, [[0, 1], [0, 1, 2]])
M3B = closed(3, [[0, 1, 2]])
M4C = closed(4, [[0, 1], [2, 3], [0, 1, 2, 3]])
M6 = closed(6, [[0, 1], [2, 3], [0, 1, 2, 3], [4, 5], [0, 1, 2, 3, 4, 5]])
def close9():
    n = 9
    rows = [(a, a) for a in range(n)]
    rows += [(a, b) for a in range(n) for b in range(a + 1, n)]
    m = len(rows)
    idx = {p: i for i, p in enumerate(rows)}
    k = lambda u, v: idx[(min(u, v), max(u, v))]
    M = np.zeros((m, m), dtype=np.int64)
    for x, y, z in [(5, 7, 1), (4, 8, 1), (2, 7, 1), (0, 4, 2), (0, 5, 8),
                    (7, 8, 6), (1, 3, 2), (7, 8, 1), (4, 5, 2), (0, 6, 2),
                    (5, 6, 1), (0, 7, 1)]:
        M[k(x, y), k(x, z)] = 1
        M[k(x, y), k(y, z)] = 1
    sup = M.any(axis=1) | M.any(axis=0)
    for a in range(n):
        sup[k(a, a)] = True
    for i in range(m):
        if sup[i]:
            M[i, i] = 1
    rows_of = {a: [k(a, c) for c in range(n)] for a in range(n)}
    while True:
        before = int(M.sum())
        while True:
            g = ((M + ((M @ M) > 0)) > 0).astype(np.int64)
            if int(g.sum()) == int(M.sum()):
                break
            M = g
        act = M.any(axis=1) | M.any(axis=0)
        for i in range(m):
            if not act[i]:
                continue
            a, b = rows[i]
            ra = M[rows_of[a]].any(axis=0)
            rb = M[rows_of[b]].any(axis=0)
            M[i] = ((M[i] > 0) | (ra & rb)).astype(np.int64)
        if int(M.sum()) == before:
            return M
M9 = close9()
"""
    return [
        # --- Valid: a four-leaf caterpillar with nested clusters (normal scenario) ---
        {
            "setup": setup,
            "call": "red(canonical_dag(4, M4))",
            "gold_call": "red(_oracle_canonical_dag(4, M4))",
        },
        # --- Valid: two disjoint cherries above a common outgroup ---
        {
            "setup": setup,
            "call": "red(canonical_dag(5, M5))",
            "gold_call": "red(_oracle_canonical_dag(5, M5))",
        },
        # --- Boundary: one cherry above a single outgroup ---
        {
            "setup": setup,
            "call": "red(canonical_dag(3, M3))",
            "gold_call": "red(_oracle_canonical_dag(3, M3))",
        },
        # --- Boundary: an unresolved fan collapses every pair into one class ---
        {
            "setup": setup,
            "call": "red(canonical_dag(3, M3B))",
            "gold_call": "red(_oracle_canonical_dag(3, M3B))",
        },
        # --- Pinned values: the three-leaf cherry has 5 vertices and 4 arcs ---
        # Derived independently from the construction, not from the oracle: the
        # classes are {0}, {1}, {2}, {01} and {02, 12}, so V = 5, and the
        # covering arcs are root -> {01}, root -> 2, {01} -> 0, {01} -> 1,
        # giving 4 arcs and no arc from the root straight to leaf 0.
        {
            "setup": setup,
            "call": ("float(np.asarray(canonical_dag(3, M3)).shape[0] * 1000.0"
                     " + float(np.asarray(canonical_dag(3, M3)).sum()) * 10.0"
                     " + float(np.asarray(canonical_dag(3, M3))[-1, 0]))"),
            "gold_call": "5040.0",
        },
        # --- Valid: two cherries joined directly under one root ---
        {
            "setup": setup,
            "call": "red(canonical_dag(4, M4C))",
            "gold_call": "red(_oracle_canonical_dag(4, M4C))",
        },
        # --- Valid: three cherries nested under one root, on six leaves ---
        # Six leaf classes, three cherry classes, one class for the four-leaf
        # cluster and one for the root leave nine internal comparisons that
        # are not covering, so this instance separates the covering arcs from
        # the full order far more sharply than the four-leaf ones above.
        {
            "setup": setup,
            "call": "red(canonical_dag(6, M6))",
            "gold_call": "red(_oracle_canonical_dag(6, M6))",
        },
        # --- Adversarial: the closed relation of the nine-leaf required set ---
        # A full-scale constraint relation on the 45 leaf pairs of nine leaves
        # rather than a drawn cluster system: 24 classes, 40 covering arcs, and
        # a large majority of the comparisons non-covering, so nearly every arc
        # has to be defended against a third class. Small hand-drawn instances
        # leave both the class numbering and the between-ness test almost
        # unexercised.
        {
            "setup": setup,
            "call": "red(canonical_dag(9, M9))",
            "gold_call": "red(_oracle_canonical_dag(9, M9))",
        },
    ]
