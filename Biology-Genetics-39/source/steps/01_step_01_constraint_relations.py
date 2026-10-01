"""
Turn a set of required rooted triples, together with the triples a constraint-level repair has already discharged, into the relation of pairwise least-common-ancestor constraints they impose on the 1- and 2-element leaf subsets, and into the closure of that relation under the three properties every realizing DAG must respect.

Every constraint in this pipeline compares the least common ancestor (LCA) of one leaf pair with the LCA of another, so the ground set of the relation is the collection of 1- and 2-element subsets of the leaf set, of size m = n + n*(n-1)/2. A singleton {a} is admitted because lca(aa) is the leaf a itself, which is what lets the language say that a leaf lies at or below a given ancestor; the canonical order places the n singletons first, row i holding (i, i), and then the 2-element subsets (a, b) with a < b in lexicographic order, so the row and column meaning of the matrix is fixed once for the whole construction. Reading a required triple xy|z through least common ancestors says that lca(xy) is a proper descendant of lca(xz), which coincides with lca(yz); the equality need not be imposed separately, because if lca(xy) lies strictly below both outer LCAs then lca(xz) is already a common ancestor of y and z and lca(yz) one of x and z, so minimality forces them equal, and the triple therefore contributes exactly the two one-sided constraints (xy, xz) and (xy, yz). Discharging a triple runs the same reading backwards: adding the two reversed comparisons (xz, xy) and (yz, xy) makes the three leaf pairs mutually related, hence one single vertex in every realizing DAG, so no realizing DAG displays the triple while every required constraint survives. The closure is then the least relation containing all of that which is reflexive on its support, transitive, and cross-consistent. Cross-consistency is the rule specific to least common ancestors rather than to orders in general: if (ac, uv) and (bd, uv) both hold for some leaves c and d, then lca(uv) is already a common ancestor of a and b, so (ab, uv) must hold as well whenever ab is itself in the support. Transitivity and cross-consistency have to be iterated together to a fixed point, since each can expose new instances of the other, and replacing cross-consistency by a plain transitive closure leaves the leaf pairs far too weakly related and produces a badly under-resolved graph downstream.

Returns
-------
np.ndarray of shape (2, m, m), int: plane 0 the direct 0/1 constraint relation on the canonical pair table, plane 1 its closure under support-reflexivity, transitivity and cross-consistency.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def constraint_relations(n: int, req_triples: np.ndarray,
                         discharged_triples: np.ndarray) -> np.ndarray:
    """Direct and closed LCA-constraint relations of a rooted triple set.

    Parameters
    ----------
    n : int
        Number of leaves, labelled 0 to n-1.
    req_triples : np.ndarray
        Integer array of shape (t, 3); row (x, y, z) denotes the required
        rooted triple xy|z on three distinct leaves.
    discharged_triples : np.ndarray
        Integer array of shape (d, 3) listing the triples whose reversed
        comparisons have already been added by a constraint-level repair.

    Returns
    -------
    relations : np.ndarray
        Integer 0/1 array of shape (2, m, m) with m = n + n * (n - 1) // 2,
        indexed by the canonical pair table. Plane 0 is the relation the two
        triple lists impose directly, plane 1 is its closure under
        support-reflexivity, transitivity and cross-consistency.


    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((2, 0, 0), dtype=int)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_constraint_relations(n: int, req_triples: np.ndarray,
                                 discharged_triples: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    req = np.asarray(req_triples, dtype=np.int64).reshape(-1, 3)
    dis = np.asarray(discharged_triples, dtype=np.int64).reshape(-1, 3)

    # -- Canonical ground set: singletons first, then the 2-element subsets.
    rows = [(a, a) for a in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            rows.append((a, b))
    pairs = np.array(rows, dtype=np.int64)
    m = pairs.shape[0]
    index = {(int(p[0]), int(p[1])): i for i, p in enumerate(pairs)}

    def key(u, v):
        if min(u, v) < 0 or max(u, v) >= n:
            raise ValueError("triple leaf outside the leaf set")
        return index[(min(u, v), max(u, v))]

    # -- Two one-sided constraints per required triple; two reversed ones per
    #    discharged triple. The equality of the outer LCAs stays implied.
    direct = np.zeros((m, m), dtype=np.int64)
    for row in range(req.shape[0]):
        x, y, z = (int(v) for v in req[row])
        if len({x, y, z}) != 3:
            raise ValueError("a rooted triple needs three distinct leaves")
        direct[key(x, y), key(x, z)] = 1
        direct[key(x, y), key(y, z)] = 1
    for row in range(dis.shape[0]):
        x, y, z = (int(v) for v in dis[row])
        if len({x, y, z}) != 3:
            raise ValueError("a rooted triple needs three distinct leaves")
        direct[key(x, z), key(x, y)] = 1
        direct[key(y, z), key(x, y)] = 1

    # -- Support-reflexivity, over the constrained pairs and every singleton.
    out = direct.copy()
    supp = (out.any(axis=1) | out.any(axis=0))
    for a in range(n):
        supp[index[(a, a)]] = True
    for i in range(m):
        if supp[i]:
            out[i, i] = 1

    # -- Transitivity and cross-consistency, iterated to a joint fixed point.
    rows_of = {a: [key(a, c) for c in range(n)] for a in range(n)}
    while True:
        before = int(out.sum())
        while True:
            comp = ((out @ out) > 0).astype(np.int64)
            merged = ((out + comp) > 0).astype(np.int64)
            if int(merged.sum()) == int(out.sum()):
                break
            out = merged
        active = (out.any(axis=1) | out.any(axis=0))
        for k in range(m):
            if not active[k]:
                continue
            a, b = int(pairs[k, 0]), int(pairs[k, 1])
            reach_a = out[rows_of[a]].any(axis=0)
            reach_b = out[rows_of[b]].any(axis=0)
            out[k] = ((out[k] > 0) | (reach_a & reach_b)).astype(np.int64)
        if int(out.sum()) == before:
            return np.stack([direct, out]).astype(np.int64)

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
REQ = np.array([[5, 7, 1], [4, 8, 1], [2, 7, 1], [0, 4, 2],
                [0, 5, 8], [7, 8, 6], [1, 3, 2], [7, 8, 1],
                [4, 5, 2], [0, 6, 2], [5, 6, 1], [0, 7, 1]], dtype=np.int64)
NONE = np.zeros((0, 3), dtype=np.int64)
SMALL = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 0]], dtype=np.int64)
ONE = np.array([[0, 1, 2]], dtype=np.int64)
TWO4 = np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int64)
DIS9 = np.array([[0, 2, 1]], dtype=np.int64)
DIS3 = np.array([[0, 1, 2]], dtype=np.int64)
"""
    return [
        # --- Valid: the twelve required triples of the benchmark (normal scenario) ---
        {
            "setup": setup,
            "call": "red(constraint_relations(9, REQ, NONE))",
            "gold_call": "red(_oracle_constraint_relations(9, REQ, NONE))",
        },
        # --- Valid: the same set after one forbidden triple has been discharged ---
        {
            "setup": setup,
            "call": "red(constraint_relations(9, REQ, DIS9))",
            "gold_call": "red(_oracle_constraint_relations(9, REQ, DIS9))",
        },
        # --- Valid: a three-triple set on five leaves ---
        {
            "setup": setup,
            "call": "red(constraint_relations(5, SMALL, NONE))",
            "gold_call": "red(_oracle_constraint_relations(5, SMALL, NONE))",
        },
        # --- Boundary: no triples at all, so only the singleton diagonal survives ---
        {
            "setup": setup,
            "call": "red(constraint_relations(3, NONE, NONE))",
            "gold_call": "red(_oracle_constraint_relations(3, NONE, NONE))",
        },
        # --- Boundary: discharging the only triple there is on three leaves ---
        {
            "setup": setup,
            "call": "red(constraint_relations(3, NONE, DIS3))",
            "gold_call": "red(_oracle_constraint_relations(3, NONE, DIS3))",
        },
        # --- Pinned values: the ground set and the two entries one triple sets ---
        # Derived independently from the conventions, not from the oracle: with
        # n = 4 the ground set has m = 4 + 6 = 10 members, row 4 is the first
        # 2-element subset {0, 1} and row 7 is {1, 2}. The triple 0 1|2 places
        # the inner pair strictly below each outer pair, so plane 0 holds
        # (01, 02) and (01, 12) and nothing else: exactly 2 non-zero entries,
        # both in row 4. The probe reads 4, 5, 7, 2, 4, 4 in that order.
        {
            "setup": """import numpy as np
def red(a):
    v = np.asarray(a, dtype=float).ravel()
    w = np.arange(1.0, v.size + 1.0) ** 2
    return float(v.size * 10000.0 + float(v @ w))
ONE = np.array([[0, 1, 2]], dtype=np.int64)
NONE = np.zeros((0, 3), dtype=np.int64)
EXPECTED = np.array([4, 5, 7, 2, 4, 4])
def probe(R):
    R = np.asarray(R)
    hits = np.argwhere(R[0] != 0)
    return [int(R.shape[1]) - 6, int(R[0, 4, 5]) + 4, int(R[0, 4, 7]) + 6,
            int(hits.shape[0]), int(hits[:, 0].min()), int(hits[:, 0].max())]
""",
            "call": "red(probe(constraint_relations(4, ONE, NONE)))",
            "gold_call": "red(EXPECTED)",
        },
        # --- Adversarial: the equality of the outer LCAs is implied, not stored ---
        # Derived independently from the encoding: on four leaves the triple
        # 0 1|2 must leave (02, 12) and (12, 02) clear in plane 0, and must not
        # write the reverse (02, 01) either, so the probe over those three
        # entries plus the two that are set reads (1, 1, 0, 0, 0). An
        # implementation that stores the equality as a third condition, or that
        # orients the inner comparison the wrong way, disagrees here while
        # still producing a plausible closure.
        {
            "setup": """import numpy as np
def red(a):
    v = np.asarray(a, dtype=float).ravel()
    w = np.arange(1.0, v.size + 1.0) ** 2
    return float(v.size * 10000.0 + float(v @ w))
ONE = np.array([[0, 1, 2]], dtype=np.int64)
NONE = np.zeros((0, 3), dtype=np.int64)
EXPECTED = np.array([1, 1, 0, 0, 0])
def probe(R):
    R = np.asarray(R)[0]
    return [int(R[4, 5]), int(R[4, 7]), int(R[5, 7]), int(R[7, 5]),
            int(R[5, 4])]
""",
            "call": "red(probe(constraint_relations(4, ONE, NONE)))",
            "gold_call": "red(EXPECTED)",
        },
        # --- Adversarial: one sweep of the implied-constraint rule is not enough ---
        # On four leaves the triples 0 1|2 and 0 3|2 leave the closure with 37
        # entries, but a pass that applies cross-consistency once after
        # transitivity stops at 31: six further comparisons only become
        # derivable after the first sweep has added its own, so the two rules
        # have to be iterated together rather than applied in sequence.
        {
            "setup": setup,
            "call": "red(constraint_relations(4, TWO4, NONE))",
            "gold_call": "red(_oracle_constraint_relations(4, TWO4, NONE))",
        },
        # --- Adversarial: a discharged triple collapses its three pairs ---
        # Discharging 0 1|2 on three leaves adds (02, 01) and (12, 01), which
        # together with reflexivity makes the three leaf pairs mutually
        # related, so the closure relates all three in both directions. An
        # implementation that adds only one reversed comparison, or that adds
        # them in the direction a required triple would use, leaves the
        # collapse incomplete.
        {
            "setup": setup,
            "call": "red(constraint_relations(3, ONE, DIS3))",
            "gold_call": "red(_oracle_constraint_relations(3, ONE, DIS3))",
        },
    ]
