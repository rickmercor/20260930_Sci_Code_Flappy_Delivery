"""
Build the triangular integer encoding from the interval lineage sets.

The encoding is a lower triangular integer matrix indexed by ordered pairs of intervals. Its entry in row i and column j, for i at least j, records how much of the lineage structure present in the earlier interval is still intact by the later one. The condition is a spanning one rather than a per-step one: a lineage contributes only if it takes part in no event at all across the whole run of intervals from j down to i. Because a lineage retains its identity for exactly as long as it participates in no event, the entry is the number of lineages the two intervals have in common. Entries above the diagonal are not part of the encoding and are left at zero; the matrix is triangular, not symmetric.

Returns
-------
list of lists of ints, a lower triangular square matrix with zeros above the diagonal
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def encoding_matrix(lineage_sets: list[list[int]]) -> list[list[int]]:
    '''Build the triangular encoding from the lineage sets.

    Parameters
    ----------
    lineage_sets : list[list[int]]
        The lineage sets as returned by the third step: for each interval, the
        sorted indices of the lineages alive in it.

    Returns
    -------
    matrix : list[list[int]]
        An N by N matrix. Entry (i, j) with j at most i counts the lineages
        present in both interval j and interval i; every entry above the
        diagonal is zero.
    '''
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_encoding_matrix(interval_sets):
    N = len(interval_sets)
    sets = [set(L) for L in interval_sets]
    return [
        [
            len(sets[i] & sets[j]) if j <= i else 0
            for j in range(N)
        ]
        for i in range(N)
    ]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''
LA = [[0], [1, 2], [2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7, 8], [4, 6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [7, 9, 11, 12]]
LB = [[0], [1, 2], [2, 3, 4], [3, 4, 5, 6], [3, 4, 5, 7, 8], [3, 4, 5, 8, 9, 10], [3, 4, 5, 10, 11], [3, 4, 5, 12]]
LT = [[0], [1, 2], [1, 3, 4]]
LO = [[0], [1, 2], [2, 3, 4], [3, 4, 5, 6], [3, 5, 7]]
'''
    return [
        {   # normal: the first network
            "setup": setup,
            "call": "encoding_matrix(LA)",
            "gold_call": "_oracle_encoding_matrix(LA)",
        },
        {   # normal: the second network
            "setup": setup,
            "call": "encoding_matrix(LB)",
            "gold_call": "_oracle_encoding_matrix(LB)",
        },
        {   # boundary: a tree
            "setup": setup,
            "call": "encoding_matrix(LT)",
            "gold_call": "_oracle_encoding_matrix(LT)",
        },
        {   # edge: one hybridisation
            "setup": setup,
            "call": "encoding_matrix(LO)",
            "gold_call": "_oracle_encoding_matrix(LO)",
        },
    ]
