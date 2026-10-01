"""
Test each triple in a list against the ancestor-based display condition in a given DAG, resolving first which leaf pairs of that DAG have a unique least common ancestor at all.

In a rooted tree any two leaves have exactly one least common ancestor, but in a DAG the minimal common ancestors of two leaves need not be comparable, so a pair can have several of them and then has no well-defined LCA at all. Every LCA-based statement about a network therefore carries a well-definedness condition that has no counterpart in tree phylogenetics, and it must be settled before any comparison of ancestral depths is attempted. That part of the computation is order-theoretic rather than metric: a vertex u is an ancestor of v when u = v or a directed path leads from u to v, the common ancestors of x and y are the vertices that are ancestors of both, the least common ancestors are those common ancestors having no other common ancestor as a proper descendant, and the pair is accepted only if exactly one vertex survives; leaves in different components have no common ancestor at all and are likewise undefined, while a leaf is its own least common ancestor with itself. Having a second parent is not by itself ambiguity, and taking the first minimal common ancestor found in place of reporting the ambiguity silently rescues pairs that a repair has deliberately broken. On top of that the ancestor-based reading of a rooted triple xy|z, which is strictly stronger than the classical reading by internally vertex-disjoint paths and comes apart from it exactly on reticulate histories, asks three things at once: that the least common ancestors of all three leaf pairs be unique, that the two ancestors involving the outgroup coincide, and that the ancestor of the inner pair be a proper descendant of that common vertex. Because the condition combines an equality with a strict descendant relation, a DAG can display at most one of the three triples on any three leaves, so counting displayed triples measures how much local ancestral structure a network actually commits to; and a pair whose least common ancestor was deliberately made ambiguous silently removes every triple that mentions it, which is precisely the effect the graph-level repair relies on.

Returns
-------
np.ndarray of shape (t,), int: one 0/1 display flag per input triple under the ancestor-based reading.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def displayed_triple_flags(adj: np.ndarray, n: int,
                           triples: np.ndarray) -> np.ndarray:
    """Flag the triples displayed by a DAG in the ancestor-based sense.

    Parameters
    ----------
    adj : np.ndarray
        Integer 0/1 array of shape (V, V); entry [u, v] is 1 when the DAG has
        the arc u -> v. Vertices 0 to n-1 are the leaves.
    n : int
        Number of leaves.
    triples : np.ndarray
        Integer array of shape (t, 3); row (x, y, z) denotes the triple xy|z.

    Returns
    -------
    flags : np.ndarray
        Integer array of shape (t,); entry i is 1 when triple i is displayed.


    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(0, dtype=int)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_displayed_triple_flags(adj: np.ndarray, n: int,
                                   triples: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    adj = (np.asarray(adj, dtype=np.int64) != 0)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square adjacency matrix")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    size = adj.shape[0]
    if n < 1 or n > size:
        raise ValueError("n must be between 1 and the number of vertices")
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)

    # -- Reachability closure; a self-loop in it would mean a directed cycle.
    reach = adj.copy()
    while True:
        grown = reach | (reach @ reach)
        if np.array_equal(grown, reach):
            break
        reach = grown
    if np.any(np.diagonal(reach)):
        raise ValueError("adj must be acyclic")
    ancestor = reach.copy()
    for v in range(size):
        ancestor[v, v] = True

    # -- Minimal common ancestors; accept only when exactly one survives.
    lca = np.full((n, n), -1, dtype=np.int64)
    for x in range(n):
        for y in range(x, n):
            common = np.nonzero(ancestor[:, x] & ancestor[:, y])[0]
            if common.size == 0:
                continue
            keep = []
            for v in common:
                if int(ancestor[v, common].sum()) == 1:
                    keep.append(int(v))
            value = keep[0] if len(keep) == 1 else -1
            lca[x, y] = value
            lca[y, x] = value

    # -- Uniqueness, then equality of the outer ancestors, then strict descent.
    flags = np.zeros(triples.shape[0], dtype=np.int64)
    for row in range(triples.shape[0]):
        x, y, z = (int(v) for v in triples[row])
        if len({x, y, z}) != 3:
            raise ValueError("a rooted triple needs three distinct leaves")
        if max(x, y, z) >= n or min(x, y, z) < 0:
            raise ValueError("triple leaf outside the leaf set")
        inner = int(lca[x, y])
        outer_x = int(lca[x, z])
        outer_y = int(lca[y, z])
        if inner < 0 or outer_x < 0 or outer_y < 0:
            continue
        if outer_x != outer_y:
            continue
        if bool(reach[outer_x, inner]):
            flags[row] = 1
    return flags

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def red(a):
    v = np.asarray(a, dtype=float).ravel()
    w = np.arange(1.0, v.size + 1.0) ** 2
    return float(v.size * 10000.0 + float(v @ w))
# caterpillar tree on four leaves
CAT = np.zeros((7, 7), dtype=np.int64)
CAT[4, 0] = 1
CAT[4, 1] = 1
CAT[5, 4] = 1
CAT[5, 2] = 1
CAT[6, 5] = 1
CAT[6, 3] = 1
# same tree with a parallel second parent over leaves 0 and 1
TWIN = np.zeros((8, 8), dtype=np.int64)
TWIN[:7, :7] = CAT
TWIN[7, 0] = 1
TWIN[7, 1] = 1
TWIN[6, 7] = 1
# two parallel parents over leaves 0,1 and two more over leaves 2,3
DBL = np.zeros((10, 10), dtype=np.int64)
DBL[:7, :7] = CAT
DBL[7, 0] = 1
DBL[7, 1] = 1
DBL[8, 2] = 1
DBL[8, 3] = 1
DBL[9, 2] = 1
DBL[9, 3] = 1
DBL[6, 7] = 1
DBL[6, 8] = 1
DBL[6, 9] = 1
# leaf 1 has two parents, yet every pair still has a unique ancestor
MULTI = np.zeros((7, 7), dtype=np.int64)
MULTI[4, 0] = 1
MULTI[4, 1] = 1
MULTI[5, 1] = 1
MULTI[5, 2] = 1
MULTI[6, 4] = 1
MULTI[6, 5] = 1
MULTI[6, 3] = 1
# two disjoint cherries with no common ancestor
SPLIT = np.zeros((6, 6), dtype=np.int64)
SPLIT[4, 0] = 1
SPLIT[4, 1] = 1
SPLIT[5, 2] = 1
SPLIT[5, 3] = 1
# an unresolved fan over three leaves
FAN = np.zeros((5, 5), dtype=np.int64)
FAN[4, 0] = 1
FAN[4, 1] = 1
FAN[4, 2] = 1
# overlapping cherries: the two outer ancestors of every triple differ
OVER = np.zeros((6, 6), dtype=np.int64)
OVER[3, 0] = 1
OVER[3, 1] = 1
OVER[4, 1] = 1
OVER[4, 2] = 1
OVER[5, 3] = 1
OVER[5, 4] = 1
ALL4 = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0], [0, 1, 3],
                 [0, 3, 1], [2, 3, 0], [0, 2, 3]], dtype=np.int64)
ALL3 = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0]], dtype=np.int64)
"""
    return [
        # --- Valid: one triple per leaf triple of a caterpillar (normal scenario) ---
        {
            "setup": setup,
            "call": "red(displayed_triple_flags(CAT, 4, ALL4))",
            "gold_call": "red(_oracle_displayed_triple_flags(CAT, 4, ALL4))",
        },
        # --- Valid: a parallel parent removes the triples over that pair ---
        {
            "setup": setup,
            "call": "red(displayed_triple_flags(TWIN, 4, ALL4))",
            "gold_call": "red(_oracle_displayed_triple_flags(TWIN, 4, ALL4))",
        },
        # --- Boundary: leaves in different components display nothing ---
        {
            "setup": setup,
            "call": "red(displayed_triple_flags(SPLIT, 4, ALL4))",
            "gold_call": "red(_oracle_displayed_triple_flags(SPLIT, 4, ALL4))",
        },
        # --- Boundary: an unresolved fan displays no triple at all ---
        {
            "setup": setup,
            "call": "red(displayed_triple_flags(FAN, 3, ALL3))",
            "gold_call": "red(_oracle_displayed_triple_flags(FAN, 3, ALL3))",
        },
        # --- Boundary: an empty triple list returns an empty flag array ---
        {
            "setup": setup,
            "call": "red(displayed_triple_flags(CAT, 4, ALL3[:0]))",
            "gold_call": "red(_oracle_displayed_triple_flags(CAT, 4, ALL3[:0]))",
        },
        # --- Pinned values: a 3-set admits exactly one resolution ---
        # Derived independently from the drawn tree, not from the oracle: in the
        # caterpillar the pair {0, 1} coalesces at 4, strictly below 5 = lca(02)
        # = lca(12), so 0 1|2 is displayed while 0 2|1 and 1 2|0 are not. The
        # flag vector over (0 1|2, 0 2|1, 1 2|0) is therefore (1, 0, 0).
        {
            "setup": setup,
            "call": ("float(displayed_triple_flags(CAT, 4, ALL3)"
                     " @ np.array([100, 10, 1]))"),
            "gold_call": "100.0",
        },
        # --- Adversarial: two leaf pairs at once without a unique ancestor ---
        # DBL gives leaves 0 and 1 a second parallel parent and leaves 2 and 3
        # two of them, so both pairs have two incomparable minimal common
        # ancestors and every triple mentioning either pair drops out. An
        # implementation that returns the first minimal common ancestor
        # instead of reporting the ambiguity keeps them all.
        {
            "setup": setup,
            "call": "red(displayed_triple_flags(DBL, 4, ALL4))",
            "gold_call": "red(_oracle_displayed_triple_flags(DBL, 4, ALL4))",
        },
        # --- Adversarial: a leaf with two parents whose ancestors stay unique ---
        # In MULTI leaf 1 is a child of two internal vertices, yet every leaf
        # pair still has exactly one minimal common ancestor, so nothing is
        # undefined. An implementation that treats a second parent as
        # ambiguity by itself, rather than testing minimality, loses triples
        # here that the network genuinely displays.
        {
            "setup": setup,
            "call": "red(displayed_triple_flags(MULTI, 4, ALL4))",
            "gold_call": "red(_oracle_displayed_triple_flags(MULTI, 4, ALL4))",
        },
        # --- Adversarial: overlapping cherries leave the outer ancestors apart ---
        # In OVER the pairs {0,1}, {1,2} and {0,2} have three different
        # ancestors, so for every one of the three triples on those leaves the
        # two ancestors involving the outgroup differ and nothing is displayed,
        # even though each ancestor on its own is perfectly unique.
        {
            "setup": setup,
            "call": "red(displayed_triple_flags(OVER, 3, ALL3))",
            "gold_call": "red(_oracle_displayed_triple_flags(OVER, 3, ALL3))",
        },
    ]
