"""
Score each species of one web by how much cheaply-matched mass it carries to the other webs.

A species is well matched across the dataset when it carries substantial alignment mass to counterparts that resemble it. The score therefore weights each alignment entry by one minus the corresponding dissimilarity, so mass sent to a close counterpart counts for nearly its full value while mass sent to a poorly matching one counts for little, and sums over all species of every other web. Only alignments to other webs enter: a web aligned to itself would trivially match every species to itself and swamp the comparison. The species scoring highest by this measure are the ones whose structural part in the community recurs elsewhere.

Returns
-------
list of floats, one role similarity score per species of the focal web, rounded to 6 decimal places
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def role_similarity(costs_to_others: list[list[list[float]]],
                    alignments_to_others: list[list[list[float]]]) -> list[float]:
    '''Aggregate cheaply-matched alignment mass for each species of one web.
    Parameters
    ----------
    costs_to_others : list[list[list[float]]]
        One dissimilarity matrix per other web, all sharing the same row set.
    alignments_to_others : list[list[list[float]]]
        The corresponding alignments, in the same order.
    Returns
    -------
    scores : list[float]
        One score per species of the focal web, rounded to 6 decimal places.
    '''
    scores = []
    return scores  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_role_similarity(costs_to_others, alignments_to_others):
    if len(costs_to_others) != len(alignments_to_others):
        raise ValueError("one cost matrix per alignment is required")
    m = len(costs_to_others[0])
    out = [0.0] * m
    for C, T in zip(costs_to_others, alignments_to_others):
        for i in range(m):
            out[i] += sum((1.0 - C[i][j]) * T[i][j] for j in range(len(C[i])))
    return [round(x, 6) for x in out]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''
A1 = [[0,1,0],[1,0,1],[0,1,0]]
A2 = [[0,1,0],[1,0,1],[0,1,0]]
CS = [[0.2,0.9,0.5],[0.8,0.1,0.6],[0.4,0.7,0.3]]
CE = [[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0]]
MU = [1/3, 1/3, 1/3]
T0 = [[1/9]*3 for _ in range(3)]
TA = [[0.10,0.02,0.00],[0.00,0.12,0.03],[0.05,0.00,0.09]]
TB = [[0.08,0.00,0.04],[0.02,0.10,0.00],[0.00,0.06,0.07]]
TC = [[0.11,0.01,0.02],[0.03,0.09,0.00],[0.00,0.04,0.10]]
'''
    return [
        {   # normal: two other webs
            "setup": setup,
            "call": "role_similarity([CS, CE], [TA, TB])",
            "gold_call": "_oracle_role_similarity([CS, CE], [TA, TB])",
        },
        {   # boundary: a single other web
            "setup": setup,
            "call": "role_similarity([CS], [TA])",
            "gold_call": "_oracle_role_similarity([CS], [TA])",
        },
        {   # edge: no transported mass, so every score is zero
            "setup": setup,
            "call": "role_similarity([CE], [[[0.0]*3 for _ in range(3)]])",
            "gold_call": "_oracle_role_similarity([CE], [[[0.0]*3 for _ in range(3)]])",
        },
    ]
