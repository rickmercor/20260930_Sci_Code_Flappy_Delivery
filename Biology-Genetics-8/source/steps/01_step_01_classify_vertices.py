"""
Classify the vertices of a ranked network by their in and out degrees.

A rooted phylogenetic network is a directed acyclic graph in which the root has no incoming edge, a leaf has no outgoing edge, and every remaining vertex represents one of two biological events. A vertex with a single ancestor and two descendants is a speciation: one lineage splits in two. A vertex with two ancestors and a single descendant is a hybridisation: two lineages merge, as happens under recombination, hybridisation or horizontal gene transfer. Counting the leaves and the hybridisations fixes the size of the encoding built in later steps, because the internal events partition time into that many consecutive intervals. Any vertex whose degrees match neither pattern means the graph is not a binary network and cannot be encoded.

Returns
-------
dict with keys 'root' (str), 'speciation', 'hybridisation', 'leaves' (sorted lists of str) and 'n', 'm', 'size' (int)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def classify_vertices(edges: list[tuple[str, str]]) -> dict:
    '''Classify every vertex of a ranked network by in and out degree.
    Parameters
    ----------
    edges : list[tuple[str, str]]
        Directed edges written as (ancestor, descendant) pairs.
    Returns
    -------
    classification : dict
        Mapping with keys 'root' (str), 'speciation' and 'hybridisation' and
        'leaves' (sorted lists of str), 'n' (int, the number of leaves),
        'm' (int, the number of hybridisation vertices) and 'size' (int, the
        number of events, equal to n + 2m).
    '''
    return {}

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_classify_vertices(edges):
    if not edges:
        raise ValueError("edges must be a non-empty list of (ancestor, descendant) pairs")
    ind, outd, verts = {}, {}, set()
    for e in edges:
        if len(e) != 2:
            raise ValueError("each edge must be an (ancestor, descendant) pair")
        u, w = e
        if u == w:
            raise ValueError("self-loop at %s" % u)
        outd[u] = outd.get(u, 0) + 1
        ind[w] = ind.get(w, 0) + 1
        verts.update((u, w))
    roots = sorted(v for v in verts if ind.get(v, 0) == 0)
    if len(roots) != 1:
        raise ValueError("network must have exactly one root, found %d" % len(roots))
    leaves = sorted(v for v in verts if outd.get(v, 0) == 0)
    spec, hyb = [], []
    for v in sorted(verts):
        if v in leaves or v == roots[0]:
            continue
        if ind[v] == 1 and outd.get(v, 0) == 2:
            spec.append(v)
        elif ind[v] == 2 and outd.get(v, 0) == 1:
            hyb.append(v)
        else:
            raise ValueError("vertex %s is neither a speciation nor a hybridisation" % v)
    n, m = len(leaves), len(hyb)
    return {"root": roots[0], "speciation": spec, "hybridisation": hyb,
            "leaves": leaves, "n": n, "m": m, "size": n + 2 * m}

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
        {
            "setup": setup,
            "call": (
                "(lambda c: [int(c['root'][1:]), c['n'], c['m'], "
                "c['size'], len(c['speciation'])])(classify_vertices(EA))"
            ),
            "gold_call": "[1, 4, 2, 8, 4]",
        },
        {
            "setup": setup,
            "call": (
                "(lambda c: [int(c['root'][1:]), c['n'], c['m'], "
                "c['size'], len(c['speciation'])])(classify_vertices(EB))"
            ),
            "gold_call": "[1, 4, 2, 8, 4]",
        },
        {
            "setup": setup,
            "call": (
                "(lambda c: [int(c['root'][1:]), c['n'], c['m'], "
                "c['size'], len(c['speciation'])])(classify_vertices(ETREE))"
            ),
            "gold_call": "[1, 3, 0, 3, 1]",
        },
        {
            "setup": setup,
            "call": (
                "(lambda c: [int(c['root'][1:]), c['n'], c['m'], "
                "c['size'], len(c['speciation'])])(classify_vertices(EONE))"
            ),
            "gold_call": "[1, 3, 1, 5, 2]",
        },
    ]
