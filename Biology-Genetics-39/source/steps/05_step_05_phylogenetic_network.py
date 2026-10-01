"""
Complete a canonical DAG into a phylogenetic network by destroying the least common ancestor of every leaf pair that only a forbidden triple mentions and then joining any remaining roots under one new root.

A forbidden triple can be blocked in two different ways, and only one of them is available once the required constraints already pin the three leaf pairs down. If a leaf pair occurs in a forbidden triple but in no required triple, nothing forces that pair to have a well-defined least common ancestor, so the cheapest repair is to give the two leaves a second, parallel parent: with two incomparable minimal common ancestors the LCA is no longer unique, and every triple that mentions the pair drops out of the displayed set without touching any required triple. The pairs needing this repair are exactly those in the pair support of the forbidden set but not in the pair support of the required set, where the pair support of a triple set collects all three leaf pairs of each of its triples. Attaching those parents adds parentless vertices, so the result may be a multi-rooted DAG; adding one new root above all roots turns it into a phylogenetic network without changing any ancestor relation among the old vertices, and hence without changing any least common ancestor that was already well-defined. When only one root remains there is nothing to join and no vertex is added.

Returns
-------
np.ndarray of shape (W, W), int: the 0/1 adjacency matrix of the phylogenetic network, two extra parents per repaired leaf pair in increasing lexicographic order of the repaired pairs, plus one new root when several roots remain.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def phylogenetic_network(adj: np.ndarray, req_triples: np.ndarray,
                         forb_triples: np.ndarray) -> np.ndarray:
    """Phylogenetic network obtained from the canonical DAG of a triple pair.

    Parameters
    ----------
    adj : np.ndarray
        Integer 0/1 array of shape (V, V), the canonical DAG; vertices 0 to
        n-1 are the leaves.
    req_triples : np.ndarray
        Integer array of shape (t, 3) listing the required triples.
    forb_triples : np.ndarray
        Integer array of shape (f, 3) listing the forbidden triples.

    Returns
    -------
    net : np.ndarray
        Integer 0/1 array of shape (W, W); entry [u, v] is 1 when the network
        has the arc u -> v. The first V vertices are those of adj, followed by
        the two extra parents of each repaired leaf pair, the repaired pairs
        taken in increasing lexicographic order, and last the new root when
        one is needed.


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

def _oracle_phylogenetic_network(adj: np.ndarray, req_triples: np.ndarray,
                                 forb_triples: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    adj = (np.asarray(adj, dtype=np.int64) != 0).astype(np.int64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] < 1:
        raise ValueError("adj must be a non-empty square adjacency matrix")
    req = np.asarray(req_triples, dtype=np.int64).reshape(-1, 3)
    forb = np.asarray(forb_triples, dtype=np.int64).reshape(-1, 3)
    size = adj.shape[0]

    def support(triples):
        out = set()
        for row in range(triples.shape[0]):
            x, y, z = (int(v) for v in triples[row])
            if len({x, y, z}) != 3:
                raise ValueError("a rooted triple needs three distinct leaves")
            if max(x, y, z) >= size or min(x, y, z) < 0:
                raise ValueError("triple leaf outside the given DAG")
            out.add((min(x, y), max(x, y)))
            out.add((min(x, z), max(x, z)))
            out.add((min(y, z), max(y, z)))
        return out

    # -- Pairs mentioned only by forbidden triples get two parallel parents.
    repair = sorted(support(forb) - support(req))
    grown = size + 2 * len(repair)
    out = np.zeros((grown, grown), dtype=np.int64)
    out[:size, :size] = adj
    cursor = size
    for (leaf_a, leaf_b) in repair:
        for _ in range(2):
            out[cursor, leaf_a] = 1
            out[cursor, leaf_b] = 1
            cursor += 1

    # -- One new root above all roots, only when more than one remains.
    roots = [v for v in range(grown) if int(out[:, v].sum()) == 0]
    if len(roots) < 2:
        return out
    final = np.zeros((grown + 1, grown + 1), dtype=np.int64)
    final[:grown, :grown] = out
    for v in roots:
        final[grown, v] = 1
    return final

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
# caterpillar DAG on four leaves: 4 = lca(01), 5 = lca(012), 6 = root
CAT = np.zeros((7, 7), dtype=np.int64)
CAT[4, 0] = 1
CAT[4, 1] = 1
CAT[5, 4] = 1
CAT[5, 2] = 1
CAT[6, 5] = 1
CAT[6, 3] = 1
REQ = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
FORB_IN = np.array([[0, 2, 3]], dtype=np.int64)
FORB_OUT = np.array([[2, 3, 0]], dtype=np.int64)
FORB_TWO = np.array([[2, 3, 0], [1, 3, 2]], dtype=np.int64)
EMPTY = np.zeros((0, 3), dtype=np.int64)
# five-leaf caterpillar: 5 = lca(01), 6, 7, 8 = root
CAT5 = np.zeros((9, 9), dtype=np.int64)
CAT5[5, 0] = 1
CAT5[5, 1] = 1
CAT5[6, 5] = 1
CAT5[6, 2] = 1
CAT5[7, 6] = 1
CAT5[7, 3] = 1
CAT5[8, 7] = 1
CAT5[8, 4] = 1
REQ5 = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
FORB5 = np.array([[2, 3, 4]], dtype=np.int64)
FORB5B = np.array([[3, 4, 2]], dtype=np.int64)
"""
    return [
        # --- Valid: one forbidden-only leaf pair triggers a repair (normal scenario) ---
        {
            "setup": setup,
            "call": "red(phylogenetic_network(CAT, REQ, FORB_OUT))",
            "gold_call": "red(_oracle_phylogenetic_network(CAT, REQ, FORB_OUT))",
        },
        # --- Valid: two forbidden triples share the same repaired pair ---
        {
            "setup": setup,
            "call": "red(phylogenetic_network(CAT, REQ, FORB_TWO))",
            "gold_call": "red(_oracle_phylogenetic_network(CAT, REQ, FORB_TWO))",
        },
        # --- Boundary: every forbidden pair is already required, nothing changes ---
        {
            "setup": setup,
            "call": "red(phylogenetic_network(CAT, REQ, FORB_IN))",
            "gold_call": "red(_oracle_phylogenetic_network(CAT, REQ, FORB_IN))",
        },
        # --- Boundary: an empty forbidden set leaves the DAG untouched ---
        {
            "setup": setup,
            "call": "red(phylogenetic_network(CAT, REQ, EMPTY))",
            "gold_call": "red(_oracle_phylogenetic_network(CAT, REQ, EMPTY))",
        },
        # --- Pinned values: one repaired pair adds two vertices, four arcs, one root ---
        # Derived independently from the construction, not from the oracle: the
        # required set 0 1|2, 0 1|3 covers the pairs 01, 02, 12, 03, 13, so the
        # forbidden triple 2 3|0 contributes only the pair 23. Repairing it adds
        # 2 vertices and 4 arcs to the 7-vertex, 6-arc caterpillar, and since the
        # two new vertices are parentless a new root is added on top: 10 vertices
        # and 6 + 4 + 3 = 13 arcs.
        {
            "setup": setup,
            "call": ("float(np.asarray(phylogenetic_network(CAT, REQ, FORB_OUT))"
                     ".shape[0] * 1000.0 + float(np.asarray("
                     "phylogenetic_network(CAT, REQ, FORB_OUT)).sum()))"),
            "gold_call": "10013.0",
        },
        # --- Adversarial: three leaf pairs need repairing at once ---
        # On the five-leaf caterpillar the required triples cover the pairs
        # among leaves 0 to 3 only, so the forbidden triple 2 3|4 leaves all
        # three of its pairs outside the required support and each has to be
        # repaired, which fixes both how many vertices are added and the order
        # in which they appear.
        {
            "setup": setup,
            "call": "red(phylogenetic_network(CAT5, REQ5, FORB5))",
            "gold_call": "red(_oracle_phylogenetic_network(CAT5, REQ5, FORB5))",
        },
        # --- Adversarial: the repaired pairs are ordered, not taken as listed ---
        # The forbidden triple 3 4|2 mentions its pairs in the order {3,4},
        # {2,3}, {2,4} and all three fall outside the required support, so the
        # extra parents are attached in the order {2,3}, {2,4}, {3,4} instead.
        # Both orders add the same number of vertices and arcs, so only the
        # attachment of individual vertices to leaves tells them apart.
        {
            "setup": setup,
            "call": "red(phylogenetic_network(CAT5, REQ5, FORB5B))",
            "gold_call": "red(_oracle_phylogenetic_network(CAT5, REQ5, FORB5B))",
        },
    ]
