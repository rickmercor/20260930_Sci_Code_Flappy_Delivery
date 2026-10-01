"""
Test a candidate matrix against the structural constraints of the encoding space.

The encoding is a bijection onto a set of integer triangular matrices cut out by linear constraints, so not every triangular matrix of the right size is the encoding of a network. The constraints require the first row and column to be one followed by zeros, the diagonal to begin one, two, three and to end at the number of leaves while moving by one at each step, the subdiagonal to follow the speciation and hybridisation rule, the second column to be a block of ones followed by zeros, and every remaining entry to lie within a band determined by its three neighbours above and to the left. The requirement that the diagonal begins one, two, three restricts the space to networks whose first two events are speciations, which is imposed because features near the root are not identifiable from data.

Returns
-------
dict with keys 'size_ok', 'P1', 'P2', 'P3', 'P4', 'P5' and 'valid', all bool
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def validate_encoding(matrix: list[list[int]], n: int, m: int) -> dict:
    '''Check a matrix against the constraints defining the encoding space.
    Parameters
    ----------
    matrix : list[list[int]]
        Candidate triangular encoding.
    n : int
        Number of leaves.
    m : int
        Number of hybridisations.
    Returns
    -------
    report : dict
        Mapping with keys 'size_ok', 'P1', 'P2', 'P3', 'P4', 'P5' and 'valid',
        all bool.
    '''
    return {}

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_validate_encoding(F, n, m):
    N = len(F)
    g = lambda i, j: F[i - 1][j - 1]
    p1 = g(1, 1) == 1 and all(g(1, k) == 0 and g(k, 1) == 0 for k in range(2, N + 1))
    p2 = (g(1, 1) == 1 and g(2, 2) == 2 and g(3, 3) == 3 and g(N, N) == n
          and all(g(i, i) > 0 for i in range(1, N + 1))
          and all(g(i, i) in (g(i - 1, i - 1) - 1, g(i - 1, i - 1) + 1)
                  for i in range(2, N + 1)))
    p3 = True
    for i in range(2, N):
        d = g(i + 1, i + 1) - g(i, i)
        if g(i + 1, i) != (g(i, i) - 1 if d == 1 else g(i, i) - 2):
            p3 = False
    col2 = [g(i, 2) for i in range(3, N + 1)]
    p4 = (g(3, 2) == 1 and set(col2) <= {0, 1}
          and all(col2[t] >= col2[t + 1] for t in range(len(col2) - 1)))
    p5 = True
    for k in range(3, N):
        for i in range(k + 1, N + 1):
            lo = max(0, g(i - 1, k) - 2, g(i, k - 1),
                     g(i, k - 1) + g(i - 1, k) - g(i - 1, k - 1) - 2)
            hi = min(g(i - 1, k), g(i, k - 1) + 2,
                     g(i, k - 1) + g(i - 1, k) - g(i - 1, k - 1))
            if not lo <= g(i, k) <= hi:
                p5 = False
    return {"size_ok": N == n + 2 * m, "P1": bool(p1), "P2": bool(p2),
            "P3": bool(p3), "P4": bool(p4), "P5": bool(p5),
            "valid": bool(N == n + 2 * m and p1 and p2 and p3 and p4 and p5)}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''
FA = [[1,0,0,0,0,0,0,0],[0,2,0,0,0,0,0,0],[0,1,3,0,0,0,0,0],[0,0,2,4,0,0,0,0],
      [0,0,1,3,5,0,0,0],[0,0,1,2,4,6,0,0],[0,0,0,0,2,4,5,0],[0,0,0,0,1,2,3,4]]
FB = [[1,0,0,0,0,0,0,0],[0,2,0,0,0,0,0,0],[0,1,3,0,0,0,0,0],[0,0,2,4,0,0,0,0],
      [0,0,2,3,5,0,0,0],[0,0,2,3,4,6,0,0],[0,0,2,3,3,4,5,0],[0,0,2,3,3,3,3,4]]
FT = [[1,0,0],[0,2,0],[0,1,3]]
FBAD = [[1,0,0],[0,2,0],[0,0,1]]
'''
    return [
        {
            "setup": setup,
            "call": (
                "(lambda r: [float(r['size_ok']), float(r['P1']), "
                "float(r['P2']), float(r['P3']), float(r['P4']), "
                "float(r['P5']), float(r['valid'])])"
                "(validate_encoding(FA, 4, 2))"
            ),
            "gold_call": "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",
        },
        {
            "setup": setup,
            "call": (
                "(lambda r: [float(r['size_ok']), float(r['P1']), "
                "float(r['P2']), float(r['P3']), float(r['P4']), "
                "float(r['P5']), float(r['valid'])])"
                "(validate_encoding(FB, 4, 2))"
            ),
            "gold_call": "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",
        },
        {
            "setup": setup,
            "call": (
                "(lambda r: [float(r['size_ok']), float(r['P1']), "
                "float(r['P2']), float(r['P3']), float(r['P4']), "
                "float(r['P5']), float(r['valid'])])"
                "(validate_encoding(FT, 3, 0))"
            ),
            "gold_call": "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",
        },
        {
            "setup": setup,
            "call": (
                "(lambda r: [float(r['size_ok']), float(r['P1']), "
                "float(r['P2']), float(r['P3']), float(r['P4']), "
                "float(r['P5']), float(r['valid'])])"
                "(validate_encoding(FBAD, 2, 0))"
            ),
            "gold_call": "[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]",
        },
    ]
