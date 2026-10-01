"""
Decide whether a relation of pairwise least-common-ancestor constraints can be realized by any directed acyclic graph, by evaluating the two conditions of the realization criterion against the direct relation and its closure.

Not every relation of least-common-ancestor constraints is realizable, and the realization theory characterizes the realizable ones by exactly two conditions. The first condition asks that (ab, xx) never belong to the direct relation whenever ab differs from the singleton xx: no directed acyclic graph can place the ancestor of a genuine two-leaf pair at or below a single leaf, because a leaf has no proper descendants, while a singleton compared with itself is of course allowed and must be exempted. The second condition asks that, whenever (ab, xy) belongs to the direct relation while (xy, ab) does not belong to its transitive closure, (xy, ab) must not belong to the full closure either; a realizing graph would otherwise have to place one ancestor both strictly below and at or above the other. The second condition is exactly where the plain transitive closure and the full closure have to be kept apart: the transitive closure decides whether the input itself already asserts the reverse comparison, while the full closure decides whether the reverse is nevertheless implied, and using one where the other belongs silently changes the verdict. The two conditions are independent, so the outcome is reported as a pair of flags rather than a single verdict, and testing them before any graph is built is what lets the construction answer that no such network exists instead of returning a graph that violates its own constraints.

Returns
-------
np.ndarray of shape (2,), int: the 0/1 outcome of the first and of the second condition of the realization criterion, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def realizability_flags(n: int, relations: np.ndarray) -> np.ndarray:
    """Evaluate the two conditions of the LCA-constraint realization criterion.

    Parameters
    ----------
    n : int
        Number of leaves, labelled 0 to n-1. Rows 0 to n-1 of each plane are
        the singletons of the canonical pair table.
    relations : np.ndarray
        Integer 0/1 array of shape (2, m, m); plane 0 is a direct constraint
        relation and plane 1 is its closure.

    Returns
    -------
    flags : np.ndarray
        Integer array of shape (2,); entry 0 is 1 when the first condition of
        the realization criterion holds for plane 0, entry 1 is 1 when the
        second holds, and each is 0 otherwise. The relation is realizable
        exactly when both entries are 1.


    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(2, dtype=int)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_realizability_flags(n: int, relations: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    stack = np.asarray(relations, dtype=np.int64)
    if stack.ndim != 3 or stack.shape[0] != 2:
        raise ValueError("relations must have shape (2, m, m)")
    m = stack.shape[1]
    if stack.shape[2] != m:
        raise ValueError("each plane must be square")
    if m != n + n * (n - 1) // 2:
        raise ValueError("relation size does not match the leaf count")
    direct = (stack[0] != 0)
    closed = (stack[1] != 0)

    # -- First condition: nothing other than a singleton itself may be
    #    constrained at or below that singleton.
    first = 1
    for j in range(n):
        for i in range(m):
            if i != j and direct[i, j]:
                first = 0
                break
        if first == 0:
            break

    # -- Transitive closure of the direct relation, kept separate from the
    #    full closure supplied in plane 1.
    trans = direct.copy()
    while True:
        grown = trans | (trans @ trans)
        if np.array_equal(grown, trans):
            break
        trans = grown

    # -- Second condition: a one-directional assertion must not have its
    #    reverse produced by the closure.
    second = 1
    for i in range(m):
        for j in range(m):
            if direct[i, j] and not trans[j, i] and closed[j, i]:
                second = 0
                break
        if second == 0:
            break

    return np.array([first, second], dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def mat(m, ent):
    M = np.zeros((m, m), dtype=np.int64)
    for i, j in ent:
        M[i, j] = 1
    return M
def stack(a, b):
    return np.stack([a, b]).astype(np.int64)
def red(a):
    v = np.asarray(a, dtype=float).ravel()
    w = np.arange(1.0, v.size + 1.0) ** 2
    return float(v.size * 10000.0 + float(v @ w))
# 3 leaves, seeded with the triple 0 1|2
REL_A = mat(6, [(3, 4), (3, 5)])
CL_A = mat(6, [(0, 0), (0, 3), (0, 4), (0, 5), (1, 1), (1, 3), (1, 4),
               (1, 5), (2, 2), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5),
               (4, 4), (4, 5), (5, 4), (5, 5)])
# the same relation written down reflexively on its whole support
REL_R = mat(6, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (3, 4),
                (3, 5)])
# 3 leaves, a one-directional assertion whose reverse the closure produces
REL_B = mat(6, [(3, 4), (5, 3)])
CL_B = mat(6, [(0, 0), (0, 3), (0, 4), (1, 1), (1, 3), (1, 4), (1, 5),
               (2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (4, 3),
               (4, 4), (5, 3), (5, 4), (5, 5)])
# 3 leaves, a two-leaf pair constrained below a singleton
REL_D = mat(6, [(3, 2)])
CL_D = mat(6, [(0, 0), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2),
               (3, 2), (3, 3)])
# 3 leaves, no constraints at all
REL_E = mat(6, [])
CL_E = mat(6, [(0, 0), (1, 1), (2, 2)])
# 4 leaves, seeded with the triples 0 1|2 and 0 1|3
REL_C = mat(10, [(4, 5), (4, 7), (4, 6), (4, 8)])
CL_C = mat(10, [(0, 0), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 1),
                (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 2), (2, 5),
                (2, 7), (3, 3), (3, 6), (3, 8), (4, 4), (4, 5), (4, 6),
                (4, 7), (4, 8), (5, 5), (5, 7), (6, 6), (6, 8), (7, 5),
                (7, 7), (8, 6), (8, 8)])
# 4 leaves, a one-directional assertion whose reverse the closure produces
REL_F = mat(10, [(4, 5), (6, 8), (7, 4)])
CL_F = mat(10, [(0, 0), (0, 4), (0, 5), (0, 6), (0, 8), (1, 1), (1, 4),
                (1, 5), (1, 7), (1, 8), (2, 2), (2, 4), (2, 5), (2, 7),
                (2, 8), (3, 3), (3, 6), (3, 8), (4, 4), (4, 5), (4, 8),
                (5, 4), (5, 5), (5, 8), (6, 6), (6, 8), (7, 4), (7, 5),
                (7, 7), (7, 8), (8, 8)])
"""
    return [
        # --- Valid: a single required triple on three leaves (normal scenario) ---
        {
            "setup": setup,
            "call": "red(realizability_flags(3, stack(REL_A, CL_A)))",
            "gold_call": "red(_oracle_realizability_flags(3, stack(REL_A, CL_A)))",
        },
        # --- Valid: two required triples sharing a cherry on four leaves ---
        {
            "setup": setup,
            "call": "red(realizability_flags(4, stack(REL_C, CL_C)))",
            "gold_call": "red(_oracle_realizability_flags(4, stack(REL_C, CL_C)))",
        },
        # --- Boundary: no constraints at all, so both conditions hold vacuously ---
        {
            "setup": setup,
            "call": "red(realizability_flags(3, stack(REL_E, CL_E)))",
            "gold_call": "red(_oracle_realizability_flags(3, stack(REL_E, CL_E)))",
        },
        # --- Edge: a two-leaf pair constrained below a single leaf ---
        {
            "setup": setup,
            "call": "red(realizability_flags(3, stack(REL_D, CL_D)))",
            "gold_call": "red(_oracle_realizability_flags(3, stack(REL_D, CL_D)))",
        },
        # --- Pinned values: the two conditions are independent ---
        # Derived independently from the criterion, not from the oracle: REL_B
        # asserts one comparison in a single direction, and its reverse is
        # absent from the transitive closure of REL_B but present in CL_B, so
        # the second condition fails while the first still holds. The flag
        # vector is therefore (1, 0), reading 10 under the probe below.
        {
            "setup": setup,
            "call": ("float(realizability_flags(3, stack(REL_B, CL_B))"
                     " @ np.array([10, 1]))"),
            "gold_call": "10.0",
        },
        # --- Adversarial: a second relation whose second condition fails ---
        # On four leaves, (01, 02) is asserted in one direction only and the
        # transitive closure of REL_F never contains (02, 01), yet the full
        # closure does. An implementation that consults the full closure where
        # the transitive closure belongs reports both conditions as holding.
        {
            "setup": setup,
            "call": ("float(realizability_flags(4, stack(REL_F, CL_F))"
                     " @ np.array([10, 1]))"),
            "gold_call": "10.0",
        },
        # --- Adversarial: a relation that is reflexive on its own support ---
        # REL_R is REL_A written down reflexively, so it holds (00, 00),
        # (11, 11) and (22, 22) and has the same closure CL_A. The first
        # condition excludes a singleton compared with itself, so it still
        # holds and the flag vector stays (1, 1), reading 11 under the probe.
        # An implementation that drops the exemption reports (0, 1) instead.
        {
            "setup": setup,
            "call": ("float(realizability_flags(3, stack(REL_R, CL_A))"
                     " @ np.array([10, 1]))"),
            "gold_call": "11.0",
        },
    ]
