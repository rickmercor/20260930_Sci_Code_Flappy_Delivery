"""
Determine which ancestral lineages are present in each time interval.

The internal events cut time into consecutive intervals: one above the root, one between each consecutive pair of events, and one below the last event down to the tips. An ancestral lineage is an edge of the network, and it is present in an interval when that interval lies between the event at which the lineage begins and the event at which it ends. The lineage above the root has no ancestor and is represented by a stem, so the first interval always contains exactly one lineage. Recording the presence of lineages interval by interval is what allows the later encoding to ask not merely how many lineages exist at a moment, but how many of them survive from one moment to a later one.

Returns
-------
list of size lists of ints; entry k lists the indices of the edges alive in interval k, indices being positions in the sorted list of network edges with the stem prepended
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lineage_sets(edges: list[tuple[str, str]], size: int,
                 root: str) -> list[list[int]]:
    '''List the lineages alive in each interval of the network.
    Parameters
    ----------
    edges : list[tuple[str, str]]
        Directed edges written ancestor to descendant.
    size : int
        The number of events, equal to n + 2m for a network with n leaves and
        m hybridisations.
    root : str
        The root vertex.
    Returns
    -------
    lineages : list[list[int]]
        One entry per interval, each a sorted list of integer indices into the
        edge list formed by prepending the stem edge above the root to the
        network edges and sorting.
    '''
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_lineage_sets(edges, size, root):
    stem = "*"

    def rank(v):
        return 0 if v == stem else (int(v[1:]) if v[0] == "v" else size)

    full = sorted([(stem, root)] + [tuple(e) for e in edges])
    index = {e: i for i, e in enumerate(full)}
    out = []
    for k in range(1, size + 1):
        alive = sorted(index[e] for e in full if rank(e[0]) < k <= rank(e[1]))
        out.append(alive)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''
EA = [("v1","v2"),("v1","v3"),("v2","v4"),("v2","v6"),("v3","v5"),("v3","v6"),
      ("v4","L1"),("v4","v7"),("v5","L2"),("v5","v7"),("v6","L3"),("v7","L4")]
EB = [("v1","v2"),("v1","v3"),("v2","L1"),("v2","L2"),("v3","L3"),("v3","v4"),
      ("v4","v5"),("v4","v6"),("v5","v6"),("v5","v7"),("v6","v7"),("v7","L4")]
ETREE = [("v1","v2"),("v1","L1"),("v2","L2"),("v2","L3")]
EONE = [("v1","v2"),("v1","v3"),("v2","L1"),("v2","v4"),("v3","L2"),("v3","v4"),("v4","L3")]
'''
    return [
        {   # normal: the first network
            "setup": setup,
            "call": "lineage_sets(EA, 8, 'v1')",
            "gold_call": "_oracle_lineage_sets(EA, 8, 'v1')",
        },
        {   # normal: the second network
            "setup": setup,
            "call": "lineage_sets(EB, 8, 'v1')",
            "gold_call": "_oracle_lineage_sets(EB, 8, 'v1')",
        },
        {   # boundary: a tree with three intervals
            "setup": setup,
            "call": "lineage_sets(ETREE, 3, 'v1')",
            "gold_call": "_oracle_lineage_sets(ETREE, 3, 'v1')",
        },
        {   # edge: one hybridisation
            "setup": setup,
            "call": "lineage_sets(EONE, 5, 'v1')",
            "gold_call": "_oracle_lineage_sets(EONE, 5, 'v1')",
        },
    ]
