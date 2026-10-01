"""
Extract the diagonal and subdiagonal and check them against the event sequence.

The diagonal of the encoding counts the ancestral lineages present in each interval, so it starts at one above the root, rises by one at every speciation, falls by one at every hybridisation, and ends at the number of leaves. The entries immediately below the diagonal behave differently and asymmetrically: relative to the diagonal above them they drop by one after a speciation, because a single lineage is consumed by the split, but by two after a hybridisation, because two lineages are consumed by the merge. That asymmetry is a direct consequence of the spanning condition and is the cheapest available check that an encoding has been built correctly.

Returns
-------
dict with keys 'diagonal' and 'subdiagonal' (lists of int) and 'consistent' (bool)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def diagonal_profile(matrix: list[list[int]], events: list[int]) -> dict:
    '''Read the diagonal and subdiagonal of an encoding and check them against
    the event sequence.
    Parameters
    ----------
    matrix : list[list[int]]
        The triangular encoding as returned by the fourth step.
    events : list[int]
        The event codes as returned by the second step, 0 for a speciation and
        1 for a hybridisation.
    Returns
    -------
    profile : dict
        Mapping with keys 'diagonal' (list of ints), 'subdiagonal' (list of ints)
        and 'consistent' (bool).
    '''
    return {}

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_diagonal_profile(F, events):
    N = len(F)
    diag = [F[i][i] for i in range(N)]
    sub = [F[i + 1][i] for i in range(N - 1)]
    ok = True
    for i in range(N - 1):
        drop = 1 if events[i] == 0 else 2
        if sub[i] != diag[i] - drop:
            ok = False
        step = 1 if events[i] == 0 else -1
        if diag[i + 1] != diag[i] + step:
            ok = False
    return {"diagonal": diag, "subdiagonal": sub, "consistent": bool(ok)}

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
EVA = [0]*5 + [1]*2
EVT = [0, 0]
'''
    return [
        {
            "setup": setup,
            "call": (
                "(lambda p: p['diagonal'] + p['subdiagonal'] + "
                "[float(p['consistent'])])(diagonal_profile(FA, EVA))"
            ),
            "gold_call": "[1,2,3,4,5,6,5,4,0,1,2,3,4,4,3,1.0]",
        },
        {
            "setup": setup,
            "call": (
                "(lambda p: p['diagonal'] + p['subdiagonal'] + "
                "[float(p['consistent'])])(diagonal_profile(FB, EVA))"
            ),
            "gold_call": "[1,2,3,4,5,6,5,4,0,1,2,3,4,4,3,1.0]",
        },
        {
            "setup": setup,
            "call": (
                "(lambda p: p['diagonal'] + p['subdiagonal'] + "
                "[float(p['consistent'])])(diagonal_profile(FT, EVT))"
            ),
            "gold_call": "[1,2,3,0,1,1.0]",
        },
        {
            "setup": setup,
            "call": (
                "(lambda p: p['diagonal'] + p['subdiagonal'] + "
                "[float(p['consistent'])])(diagonal_profile(FBAD, EVT))"
            ),
            "gold_call": "[1,2,1,0,0,0.0]",
        },
    ]
