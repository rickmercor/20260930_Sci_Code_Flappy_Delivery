"""
Select the species that form the web's backbone.

The backbone of a web is the set of species whose roles are most consistently matched elsewhere, taken as the fixed number of highest-scoring species. Restricting attention to them is what turns a collection of pairwise comparisons into a statement about persistent structure: these are the species one expects to find playing the same part in other communities, and the interactions among them are the candidate invariant core. Ties are broken by species label so that the selection is always well defined, and the returned labels are sorted so that the backbone does not depend on the order in which scores were ranked.

Returns
-------
list of ints, the labels of the k highest-scoring species in ascending order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def backbone(scores: list[float], k: int) -> list[int]:
    '''Select the k highest-scoring species.
    Parameters
    ----------
    scores : list[float]
        Role similarity score per species.
    k : int
        Backbone size, at least 2.
    Returns
    -------
    members : list[int]
        Labels of the selected species, in ascending order.
    '''
    members = []
    return members  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_backbone(scores, k):
    if not 2 <= k <= len(scores):
        raise ValueError("k must be between 2 and the number of species")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(order[:k])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''
S1 = [0.208513, 0.147224, 0.253557, 0.003603, 0.253557, 0.0]
S2 = [0.5, 0.5, 0.5, 0.5]
S3 = [0.1, 0.9]
'''
    return [
        {   # normal: the target web, with an exact tie between two species
            "setup": setup,
            "call": "backbone(S1, 3)",
            "gold_call": "_oracle_backbone(S1, 3)",
        },
        {   # boundary: all scores equal, so labels break every tie
            "setup": setup,
            "call": "backbone(S2, 2)",
            "gold_call": "_oracle_backbone(S2, 2)",
        },
        {   # edge: k equal to the number of species
            "setup": setup,
            "call": "backbone(S3, 2)",
            "gold_call": "_oracle_backbone(S3, 2)",
        },
    ]
