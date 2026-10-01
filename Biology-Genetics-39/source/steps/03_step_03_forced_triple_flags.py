"""
Decide, for each triple in a list, whether a constraint relation restricted to that triple's three leaf pairs is exactly the constraint pattern of the triple, so that every realizing graph is already committed to displaying it.

A rooted triple xy|z pins down the mutual order of only three leaf pairs, namely xy, xz and yz, and the pattern it pins down has four members: (xy, xz), (xy, yz), (xz, yz) and (yz, xz), the last two expressing that the two outer LCAs coincide. Restricting a relation to the six possible ordered comparisons among those three pairs and asking whether the result is exactly that four-member pattern, so in particular that (xz, xy) and (yz, xy) are both absent, is a complete test: a DAG displays the triple precisely when its own LCA relation restricts to that pattern. Applying the test to a closed constraint relation rather than to a DAG says something stronger, namely that the constraints already commit every realizing DAG to displaying the triple, which is exactly the situation that has to be repaired when the triple is forbidden. A triple that fails the test is not thereby ruled out; it merely is not yet forced, and it may still have to be blocked at the level of the graph. Both absences matter independently, and so does the presence of the two comparisons expressing the equality of the outer ancestors: a relation carrying all four present comparisons together with one of the two forbidden reverses does not force the triple, and neither does a relation carrying only the two comparisons a required triple contributes before any closure has been taken. The rows and columns are those of the canonical pair table, singletons first and then the 2-element subsets in lexicographic order, so the three indices of a triple are determined by the leaf count alone.

Returns
-------
np.ndarray of shape (t,), int: one 0/1 flag per input triple, 1 when the relation restricted to that triple's three leaf pairs is exactly the triple's four-member pattern.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def forced_triple_flags(n: int, rel: np.ndarray,
                        triples: np.ndarray) -> np.ndarray:
    """Flag the triples whose constraint pattern the relation already forces.

    Parameters
    ----------
    n : int
        Number of leaves, labelled 0 to n-1.
    rel : np.ndarray
        Integer 0/1 array of shape (m, m) with m = n + n * (n - 1) // 2,
        holding a constraint relation on the canonical pair table.
    triples : np.ndarray
        Integer array of shape (t, 3); row (x, y, z) denotes the triple xy|z.

    Returns
    -------
    flags : np.ndarray
        Integer array of shape (t,); entry i is 1 when the restriction of rel
        to the three leaf pairs of triple i equals that triple's pattern.


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

def _oracle_forced_triple_flags(n: int, rel: np.ndarray,
                                triples: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    rel = (np.asarray(rel, dtype=np.int64) != 0)
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    m = n + n * (n - 1) // 2
    if rel.shape != (m, m):
        raise ValueError("rel must be square and match the leaf count")

    rows = [(a, a) for a in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            rows.append((a, b))
    index = {p: i for i, p in enumerate(rows)}

    # -- The four present and two absent comparisons that define the pattern.
    flags = np.zeros(triples.shape[0], dtype=np.int64)
    for row in range(triples.shape[0]):
        x, y, z = (int(v) for v in triples[row])
        if len({x, y, z}) != 3:
            raise ValueError("a rooted triple needs three distinct leaves")
        if min(x, y, z) < 0 or max(x, y, z) >= n:
            raise ValueError("triple leaf outside the leaf set")
        i_xy = index[(min(x, y), max(x, y))]
        i_xz = index[(min(x, z), max(x, z))]
        i_yz = index[(min(y, z), max(y, z))]
        observed = (
            bool(rel[i_xy, i_xz]), bool(rel[i_xy, i_yz]),
            bool(rel[i_xz, i_yz]), bool(rel[i_yz, i_xz]),
            bool(rel[i_xz, i_xy]), bool(rel[i_yz, i_xy]),
        )
        target = (True, True, True, True, False, False)
        flags[row] = 1 if observed == target else 0
    return flags

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def table(n):
    rows = [(a, a) for a in range(n)]
    rows += [(a, b) for a in range(n) for b in range(a + 1, n)]
    return rows
def pattern(n, tri):
    rows = table(n)
    idx = {p: i for i, p in enumerate(rows)}
    M = np.zeros((len(rows), len(rows)), dtype=np.int64)
    k = lambda u, v: idx[(min(u, v), max(u, v))]
    for x, y, z in tri:
        M[k(x, y), k(x, z)] = 1
        M[k(x, y), k(y, z)] = 1
        M[k(x, z), k(y, z)] = 1
        M[k(y, z), k(x, z)] = 1
    return M
def red(a):
    v = np.asarray(a, dtype=float).ravel()
    w = np.arange(1.0, v.size + 1.0) ** 2
    return float(v.size * 10000.0 + float(v @ w))
def m3(ent):
    M = np.zeros((6, 6), dtype=np.int64)
    for i, j in ent:
        M[i, j] = 1
    return M
M6 = pattern(6, [(0, 1, 2), (3, 4, 5)])
PROBE6 = np.array([[0, 1, 2], [3, 4, 5], [0, 2, 1], [1, 2, 0],
                   [0, 3, 5]], dtype=np.int64)
M4 = pattern(4, [(0, 1, 2)])
PROBE4 = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
M4E = pattern(4, [])
EMPTY = np.zeros((0, 3), dtype=np.int64)
ONE012 = np.array([[0, 1, 2]], dtype=np.int64)
ALL012 = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0]], dtype=np.int64)
# closure of the relation of 0 1|2 after its two reversed comparisons
REPAIRED = m3([(0, 0), (0, 3), (0, 4), (0, 5), (1, 1), (1, 3), (1, 4),
               (1, 5), (2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4),
               (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)])
# the two comparisons a required triple contributes, before any closure
RAW = m3([(3, 4), (3, 5)])
# all four comparisons a displaying DAG induces, plus one of the two that
# the pattern requires to be absent
HALF_A = m3([(3, 4), (3, 5), (4, 5), (5, 4), (5, 3)])
HALF_B = m3([(3, 4), (3, 5), (4, 5), (5, 4), (4, 3)])
"""
    return [
        # --- Valid: two forced triples among five probes on six leaves (normal scenario) ---
        {
            "setup": setup,
            "call": "red(forced_triple_flags(6, M6, PROBE6))",
            "gold_call": "red(_oracle_forced_triple_flags(6, M6, PROBE6))",
        },
        # --- Valid: one forced and one unconstrained probe on four leaves ---
        {
            "setup": setup,
            "call": "red(forced_triple_flags(4, M4, PROBE4))",
            "gold_call": "red(_oracle_forced_triple_flags(4, M4, PROBE4))",
        },
        # --- Boundary: an empty relation flags nothing ---
        {
            "setup": setup,
            "call": "red(forced_triple_flags(4, M4E, PROBE4))",
            "gold_call": "red(_oracle_forced_triple_flags(4, M4E, PROBE4))",
        },
        # --- Boundary: an empty probe list returns an empty flag array ---
        {
            "setup": setup,
            "call": "red(forced_triple_flags(4, M4, EMPTY))",
            "gold_call": "red(_oracle_forced_triple_flags(4, M4, EMPTY))",
        },
        # --- Pinned values: only the seeded resolution of a 3-set is flagged ---
        # Derived independently from the pattern definition, not from the
        # oracle: a relation carrying exactly the pattern of 0 1|2 flags that
        # triple and neither of the two competing resolutions 0 2|1 and 1 2|0,
        # so the flag vector over (0 1|2, 0 2|1, 1 2|0) is (1, 0, 0).
        {
            "setup": setup,
            "call": ("float(forced_triple_flags(3, pattern(3, [(0, 1, 2)]),"
                     " ALL012) @ np.array([100, 10, 1]))"),
            "gold_call": "100.0",
        },
        # --- Adversarial: the state a repaired triple leaves behind ---
        # REPAIRED is the closure of the relation of 0 1|2 after its two
        # reversed comparisons have been added. All four comparisons that a
        # displaying DAG induces are present, so a test that only checks for
        # those still reports the triple as forced; the reversed comparisons
        # are present too, so the triple is in fact no longer forced.
        {
            "setup": setup,
            "call": "float(forced_triple_flags(3, REPAIRED, ONE012)[0])",
            "gold_call": "0.0",
        },
        # --- Adversarial: the required relation before it has been closed ---
        # RAW holds only the two comparisons a required triple contributes, so
        # the two comparisons expressing that the outer ancestors coincide are
        # still missing and the triple is not yet forced. A test that checks
        # only the inner comparisons and the two absences reports it as forced.
        {
            "setup": setup,
            "call": "float(forced_triple_flags(3, RAW, ONE012)[0])",
            "gold_call": "0.0",
        },
        # --- Adversarial: all three resolutions against a repaired relation ---
        # None of the three triples on the leaves 0, 1, 2 is forced once the
        # relation of 0 1|2 has been repaired, so the flag vector is all zeros.
        {
            "setup": setup,
            "call": "red(forced_triple_flags(3, REPAIRED, ALL012))",
            "gold_call": "red(_oracle_forced_triple_flags(3, REPAIRED, ALL012))",
        },
        # --- Adversarial: the two absences have to be checked separately ---
        # HALF_A and HALF_B each carry all four comparisons a displaying DAG
        # induces together with exactly one of the two that the pattern
        # requires to be absent, so neither forces 0 1|2 and both flags read 0.
        # An implementation that tests only one of the two absences accepts
        # whichever of these two relations carries the other one.
        {
            "setup": setup,
            "call": ("red([forced_triple_flags(3, HALF_A, ONE012)[0],"
                     " forced_triple_flags(3, HALF_B, ONE012)[0]])"),
            "gold_call": "red(np.zeros(2, dtype=np.int64))",
        },
    ]
