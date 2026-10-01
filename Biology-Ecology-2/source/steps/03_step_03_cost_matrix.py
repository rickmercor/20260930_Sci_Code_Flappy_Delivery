"""
Turn two sets of species profiles into a pairwise dissimilarity matrix.

Comparing species across webs means comparing their profiles. The dissimilarity used is one minus the Pearson correlation between the two profile vectors, so two species whose profiles rise and fall together across positions are judged similar even if one participates in many more interactions than the other. This scale invariance is deliberate: a species in a large, densely connected web would otherwise be incomparable with one from a sparse web purely because of the totals involved. The correlation is undefined when a profile is constant, which is why a species that takes part in no interaction pattern cannot be compared at all. Because correlation is unchanged by any consistent reordering of the profile components, the result does not depend on the order in which positions are enumerated.

Returns
-------
list of lists of floats, the dissimilarity of each species of the first web against each of the second, rounded to 6 decimal places
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cost_matrix(profiles_a: list[list[int]],
                profiles_b: list[list[int]]) -> list[list[float]]:
    '''Pairwise dissimilarity between species of two webs.
    Parameters
    ----------
    profiles_a : list[list[int]]
        Profiles of the first web, one row per species.
    profiles_b : list[list[int]]
        Profiles of the second web, one row per species.
    Returns
    -------
    cost : list[list[float]]
        Dissimilarity of every species of the first web against every species of
        the second, each entry rounded to 6 decimal places.
    '''
    cost = []
    return cost  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
def _oracle_cost_matrix(profiles_a, profiles_b):
    def unit(P):
        out = []
        for row in P:
            m = sum(row) / len(row)
            c = [x - m for x in row]
            nrm = math.sqrt(sum(x * x for x in c))
            if nrm == 0.0:
                raise ValueError("constant role profile: correlation undefined")
            out.append([x / nrm for x in c])
        return out
    A, B = unit(profiles_a), unit(profiles_b)
    return [[round(1.0 - sum(a * b for a, b in zip(ra, rb)), 6) for rb in B]
            for ra in A]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''
def _fx(arcs, n):
    D = [[0] * n for _ in range(n)]
    for u, v in arcs:
        D[u][v] = 1
    U = [[1 if (D[i][j] or D[j][i]) else 0 for j in range(n)] for i in range(n)]
    return D, U
W1 = [(0,3),(1,5),(2,5),(3,0),(4,5),(5,0),(5,1)]
W2 = [(0,4),(1,3),(1,4),(2,0),(3,2),(3,4),(4,3),(5,2)]
W3 = [(0,5),(1,4),(2,3),(2,5),(4,1),(4,5),(5,1),(5,2)]
D1, U1 = _fx(W1, 6)
D2, U2 = _fx(W2, 6)
D3, U3 = _fx(W3, 6)
P1 = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 1]]
P2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]]
P3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1]]
'''
    return [
        {   # normal: webs 1 and 2
            "setup": setup,
            "call": "cost_matrix(P1, P2)",
            "gold_call": "_oracle_cost_matrix(P1, P2)",
        },
        {   # normal: webs 1 and 3, which contain an exactly matching pair
            "setup": setup,
            "call": "cost_matrix(P1, P3)",
            "gold_call": "_oracle_cost_matrix(P1, P3)",
        },
        {   # boundary: a web against itself, zero on the diagonal
            "setup": setup,
            "call": "cost_matrix(P1, P1)",
            "gold_call": "_oracle_cost_matrix(P1, P1)",
        },
    ]
