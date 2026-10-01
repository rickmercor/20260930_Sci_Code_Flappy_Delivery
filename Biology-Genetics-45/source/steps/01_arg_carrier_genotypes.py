"""
Sweep an ancestral recombination graph from left to right and return, for the

mutations it carries, the diploid genotype matrix together with the genomic

extent over which each carrying clade persists.

An ancestral recombination graph has leaf nodes for sampled haplotypes and

internal nodes for their ancestors. An edge (c, p, left, right) states that node

c inherits the interval [left, right) from node p, so the set of children of a

node changes along the sequence and every genomic position induces its own

marginal tree. Mutations are annotated on edges at a genomic position, and which

haplotypes carry a mutation is read off the graph at that position: a mutation

on the edge above node c is carried by every leaf that descends from c in the

marginal tree at its position. A diploid individual is a pair of haplotype

leaves, and its genotype at a mutation is the number of its two haplotypes that

carry it, an integer in {0, 1, 2}.




Recombination rearranges only part of the graph at a time, so the set of leaves

below a node is typically unchanged over a stretch of sequence. How far the

clade carrying each mutation persists is reported alongside the genotypes.

Returns
-------
tuple (np.ndarray of shape (n_individuals, n_mutations) of float64 diploid allele counts, np.ndarray of shape (n_mutations,) of float64 clade end positions)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def arg_carrier_genotypes(
    edges: list,
    mutations: list,
    haplotype_pairs: list,
    sequence_length: float = 100.0,
) -> tuple:
    """Build the diploid genotypes and clade extents of the mutations in an ARG.

    Args:
        edges: list of (child, parent, left, right) tuples. Each tuple states
            that node child inherits the half-open interval [left, right) from
            node parent.
        mutations: list of (child_node, position) tuples locating each mutation
            on the edge above child_node that spans that genomic position.
        haplotype_pairs: list of [hap_a, hap_b] pairs of leaf indices, one entry
            per diploid individual, in the order the individuals are reported.
        sequence_length: right end of the coordinate range, so positions lie in
            [0, sequence_length).

    Returns:
        tuple (genotypes, clade_end). genotypes holds the diploid allele counts
        with shape (n_individuals, n_mutations), columns ordered by increasing
        mutation position; mutations sharing a position keep their relative
        input order (a stable sort). clade_end has shape (n_mutations,) and is
        aligned with those columns, entry j giving the smallest position above
        the mutation at which the carrier set of column j is no longer the
        descendant set of its node, or sequence_length when no such position
        exists. Node identifiers are arbitrary non-negative integers and a leaf
        is a node without children in the marginal tree at the position.

    Raises:
        ValueError: if mutations is empty, if a mutation position lies outside
            [0, sequence_length), or if a haplotype pair does not hold exactly
            two leaf indices.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _children_by_parent(edges):
    """Group edges by parent node as (child, left, right) entries."""
    table = {}
    for child, parent, left, right in edges:
        table.setdefault(parent, []).append((int(child), float(left), float(right)))
    return table


def _next_child_edge(node, position, table, sequence_length):
    """Earliest start of a child edge of node that opens after position."""
    starts = [left for _, left, _ in table.get(node, ()) if left > position]
    return min(starts) if starts else sequence_length


def _descendants(node, position, table, cache, sequence_length):
    """Descendant leaves of node at position, memoized with an expiry bound."""
    cached = cache.get(node)
    if cached is not None and cached[1] > position:
        return cached[0]
    spanning = [
        entry for entry in table.get(node, ()) if entry[1] <= position < entry[2]
    ]
    expiry = _next_child_edge(node, position, table, sequence_length)
    if not spanning:
        cache[node] = (frozenset([node]), expiry)
        return cache[node][0]
    reached = set()
    for child, _, right in spanning:
        reached |= _descendants(child, position, table, cache, sequence_length)
        expiry = min(expiry, right, cache[child][1])
    cache[node] = (frozenset(reached), expiry)
    return cache[node][0]


def _clade_end(node, position, table, breakpoints, sequence_length):
    """Smallest position above the given one where the clade of node differs."""
    current = _descendants(node, position, table, {}, sequence_length)
    for point in breakpoints:
        if point <= position:
            continue
        if _descendants(node, point, table, {}, sequence_length) != current:
            return point
    return sequence_length


def _oracle_arg_carrier_genotypes(
    edges: list,
    mutations: list,
    haplotype_pairs: list,
    sequence_length: float = 100.0,
) -> tuple:
    """Reference implementation."""
    if len(mutations) == 0:
        raise ValueError("at least one mutation is required")
    sequence_length = float(sequence_length)
    for _, position in mutations:
        if not 0.0 <= float(position) < sequence_length:
            raise ValueError("mutation position outside [0, sequence_length)")
    for pair in haplotype_pairs:
        if len(pair) != 2:
            raise ValueError("each individual needs exactly two haplotype leaves")

    table = _children_by_parent(edges)
    breakpoints = sorted(
        {float(edge[2]) for edge in edges} | {float(edge[3]) for edge in edges}
    )
    breakpoints = [point for point in breakpoints if 0.0 < point < sequence_length]
    cache = {}
    columns = []
    extents = []
    for node, position in sorted(mutations, key=lambda item: float(item[1])):
        position = float(position)
        for stale in [key for key, value in cache.items() if value[1] <= position]:
            del cache[stale]
        carriers = _descendants(int(node), position, table, cache, sequence_length)
        columns.append(
            [sum(1 for hap in pair if hap in carriers) for pair in haplotype_pairs]
        )
        extents.append(
            _clade_end(int(node), position, table, breakpoints, sequence_length)
        )
    return np.array(columns, dtype=float).T, np.array(extents, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _PACK = """
def pack(fn, edges, mutations, pairs, span):
    genotypes, clade_end = fn(edges, mutations, pairs, span)
    return np.concatenate([
        np.asarray(genotypes.shape, dtype=float),
        np.asarray(genotypes, dtype=float).ravel(),
        np.asarray(clade_end, dtype=float).ravel(),
    ])
"""
    _EDGES = [(j, 20 + j // 2, 0, 100) for j in range(20)] + [
        (20, 30, 0, 100), (21, 30, 0, 100),
        (22, 31, 0, 40), (22, 39, 40, 70), (22, 31, 70, 100),
        (23, 31, 0, 40), (23, 35, 40, 70), (23, 31, 70, 100),
        (24, 32, 0, 40), (24, 39, 40, 70), (24, 32, 70, 100),
        (25, 32, 0, 100), (26, 33, 0, 100), (27, 33, 0, 100),
        (28, 34, 0, 70), (28, 40, 70, 100),
        (29, 34, 0, 70), (29, 38, 70, 100),
        (30, 35, 0, 100), (31, 35, 0, 40), (31, 35, 70, 100),
        (32, 36, 0, 100), (33, 36, 0, 70), (33, 40, 70, 100),
        (34, 38, 0, 70), (35, 37, 0, 100), (36, 37, 0, 100), (37, 38, 0, 100),
        (39, 32, 40, 70), (40, 36, 70, 100),
    ]
    _MUTATIONS = [
        (34, 13), (31, 16), (37, 18), (34, 22), (37, 30), (11, 35), (36, 44),
        (29, 45), (37, 52), (3, 57), (36, 64), (33, 66), (26, 72), (40, 83),
        (25, 87), (24, 92), (21, 94),
    ]
    _PAIRS = [[0, 11], [1, 14], [2, 17], [3, 8], [4, 19], [5, 12], [6, 15],
              [7, 18], [9, 16], [10, 13]]
    return [
        {
            "setup": """edges = %r
mutations = %r
pairs = %r
%s""" % (_EDGES, _MUTATIONS, _PAIRS, _PACK),
            "call": "pack(arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
            "gold_call": "pack(_oracle_arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
        },
        {
            "setup": """edges = %r
mutations = [(38, 0.0)]
pairs = %r
%s""" % (_EDGES, _PAIRS, _PACK),
            "call": "pack(arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
            "gold_call": "pack(_oracle_arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
        },
        {
            "setup": """edges = %r
mutations = [(34, 100.0)]
pairs = %r

def run_model():
    try:
        arg_carrier_genotypes(edges, mutations, pairs, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_arg_carrier_genotypes(edges, mutations, pairs, 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""" % (_EDGES, _PAIRS),
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # Mutations at and just below the breakpoints 40 and 70. The half-open reading of
        # [left, right) decides which children a node has, so a closed-interval reading gives
        # node 36 at 70.0 twelve carriers instead of ten.
        {
            "setup": """edges = %r
mutations = [(32, 39.999), (32, 40.0), (35, 69.999), (35, 70.0), (36, 69.999),
             (36, 70.0), (40, 70.0), (28, 70.0), (29, 69.999), (29, 70.0)]
pairs = %r
%s""" % (_EDGES, _PAIRS, _PACK),
            "call": "pack(arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
            "gold_call": "pack(_oracle_arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
        },
        # A second graph on [0, 50) whose leaves are nodes 20-35 (internal nodes 0-16), with a
        # unary node (9), a node whose child set changes at 25 and 40, two nodes that are roots
        # only on part of the sequence (15 on [10, 25), 16 elsewhere), mutations listed out of
        # position order, three mutations sharing position 40.0 (input order kept), and
        # positions at interval ends.
        {
            "setup": """edges = [
    (20, 0, 0, 50), (21, 0, 0, 25), (21, 9, 25, 50), (22, 1, 0, 50), (23, 1, 0, 50), (24, 2, 0, 50),
    (25, 2, 0, 50), (26, 3, 0, 50), (27, 3, 0, 50), (28, 4, 0, 50), (29, 4, 0, 50), (30, 5, 0, 50),
    (31, 5, 0, 50), (32, 6, 0, 50), (33, 6, 0, 50), (34, 7, 0, 50), (35, 7, 0, 50),
    (0, 8, 0, 50), (1, 8, 0, 40), (1, 12, 40, 50), (2, 10, 0, 40), (2, 8, 40, 50), (3, 10, 0, 50),
    (4, 11, 0, 50), (5, 11, 0, 50), (9, 11, 25, 50), (6, 12, 0, 50), (7, 12, 0, 50),
    (8, 13, 0, 50), (10, 13, 0, 50), (11, 14, 0, 10), (11, 15, 10, 25), (11, 14, 25, 50),
    (12, 14, 0, 50), (13, 16, 0, 10), (13, 15, 10, 25), (13, 16, 25, 50), (14, 16, 0, 10),
    (14, 15, 10, 25), (14, 16, 25, 50),
]
mutations = [(8, 40.0), (14, 9.999), (11, 25.0), (12, 40.0), (15, 12.0), (21, 5.0), (10, 45.0),
             (8, 39.999), (0, 40.0), (16, 25.0), (14, 10.0), (9, 30.0), (11, 24.999), (2, 49.999),
             (13, 0.0), (10, 39.999), (14, 25.0), (3, 10.0)]
pairs = [[20, 29], [21, 34], [22, 27], [23, 35], [24, 30], [25, 32], [26, 33], [28, 31]]
%s""" % (_PACK,),
            "call": "pack(arg_carrier_genotypes, edges, mutations, pairs, 50.0)",
            "gold_call": "pack(_oracle_arg_carrier_genotypes, edges, mutations, pairs, 50.0)",
        },
        # Leaf 1 leaves node 8 for node 9 at 40 and returns at 70. The clades of node 10 and of
        # the root 13 are untouched by both moves, so their extents run to the end of the range
        # while the subtree below them is rearranged twice, and the clade of node 8 ends at the
        # first move rather than the second.
        {
            "setup": """edges = [
    (0, 8, 0, 100), (1, 8, 0, 40), (1, 9, 40, 70), (1, 8, 70, 100), (2, 9, 0, 100),
    (3, 9, 0, 100), (4, 11, 0, 100), (5, 11, 0, 100), (6, 12, 0, 100), (7, 12, 0, 100),
    (8, 10, 0, 100), (9, 10, 0, 100), (10, 13, 0, 100), (11, 13, 0, 100), (12, 13, 0, 100),
]
mutations = [(10, 20.0), (8, 20.0), (9, 20.0), (13, 20.0), (8, 50.0), (10, 50.0),
             (1, 39.999), (1, 40.0), (9, 69.999), (9, 70.0), (11, 10.0)]
pairs = [[0, 4], [1, 5], [2, 6], [3, 7]]
%s""" % (_PACK,),
            "call": "pack(arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
            "gold_call": "pack(_oracle_arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
        },
        # Leaves 0 and 2 change places between nodes 10 and 11 at 30, leaf 4 leaves node 12 for
        # node 13 at 40 and returns at 70, and leaf 7 moves from node 14 to node 15 at 60. Each
        # move leaves the clade of the node above the pair it moves within untouched, so nodes 16,
        # 17 and 18 hold to the end of the range while the subtrees below them are rearranged, and
        # the swap at 30 changes which leaves node 10 reaches without changing how many.
        {
            "setup": """edges = [
    (0, 10, 0, 30), (0, 11, 30, 100), (1, 10, 0, 100), (2, 11, 0, 30), (2, 10, 30, 100),
    (3, 11, 0, 100), (4, 12, 0, 40), (4, 13, 40, 70), (4, 12, 70, 100), (5, 12, 0, 100),
    (6, 13, 0, 100), (7, 14, 0, 60), (7, 15, 60, 100), (8, 15, 0, 100), (9, 14, 0, 100),
    (10, 16, 0, 100), (11, 16, 0, 100), (12, 17, 0, 100), (13, 17, 0, 100),
    (14, 18, 0, 100), (15, 18, 0, 100), (16, 19, 0, 100), (17, 19, 0, 100), (18, 19, 0, 100),
]
mutations = [(17, 50.0), (10, 10.0), (12, 69.999), (16, 30.0), (4, 35.0), (13, 40.0), (19, 5.0),
             (11, 29.999), (14, 59.999), (12, 20.0), (18, 60.0), (10, 30.0), (15, 60.0),
             (17, 20.0), (12, 39.999), (7, 65.0), (13, 70.0), (12, 50.0), (18, 20.0), (16, 10.0)]
pairs = [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]
%s""" % (_PACK,),
            "call": "pack(arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
            "gold_call": "pack(_oracle_arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
        },
        # Leaf 0 alternates between nodes 10 and 11 at 20, 40, 60 and 80. Successive mutations on
        # the same node therefore carry different extents, while nodes 12 and 14 above the pair are
        # unaffected by every move.
        {
            "setup": """edges = [
    (0, 10, 0, 20), (0, 11, 20, 40), (0, 10, 40, 60), (0, 11, 60, 80), (0, 10, 80, 100),
    (1, 10, 0, 100), (2, 11, 0, 100), (3, 13, 0, 100), (4, 13, 0, 100), (5, 13, 0, 100),
    (10, 12, 0, 100), (11, 12, 0, 100), (12, 14, 0, 100), (13, 14, 0, 100),
]
mutations = [(10, 10.0), (11, 30.0), (12, 50.0), (10, 30.0), (11, 10.0), (10, 50.0), (14, 90.0),
             (11, 70.0), (10, 70.0), (12, 10.0), (11, 50.0), (10, 90.0), (11, 90.0), (0, 45.0),
             (13, 25.0)]
pairs = [[0, 3], [1, 4], [2, 5]]
%s""" % (_PACK,),
            "call": "pack(arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
            "gold_call": "pack(_oracle_arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
        },
        # The first of the three chromosomes the pipeline runs on. Leaf 0 moves between nodes 20
        # and 21 over [20, 40), and both are children of node 30, so the subtree below node 30 is
        # rearranged twice while the leaves it reaches never change.
        {
            "setup": """edges = [
    (0, 20, 0, 20), (0, 21, 20, 40), (0, 20, 40, 60), (1, 20, 0, 60), (2, 21, 0, 60), (3, 21,
    0, 60), (4, 22, 0, 60), (5, 22, 0, 60), (6, 23, 0, 60), (7, 23, 0, 60), (8, 24, 0, 60),
    (9, 24, 0, 60), (10, 25, 0, 60), (11, 25, 0, 60), (12, 26, 0, 60), (13, 26, 0, 60), (14,
    27, 0, 60), (15, 27, 0, 60), (16, 28, 0, 60), (17, 28, 0, 60), (18, 29, 0, 60), (19, 29,
    0, 60), (20, 30, 0, 60), (21, 30, 0, 60), (22, 31, 0, 60), (23, 31, 0, 60), (24, 32, 0,
    60), (25, 32, 0, 60), (26, 33, 0, 60), (27, 33, 0, 60), (28, 34, 0, 60), (29, 34, 0, 60),
    (30, 35, 0, 60), (31, 35, 0, 60), (32, 36, 0, 40), (32, 37, 40, 60), (33, 36, 0, 60), (34,
    37, 0, 60), (35, 37, 0, 60), (36, 38, 0, 60), (37, 38, 0, 60)
]
mutations = [
    (30, 5), (20, 5), (21, 5), (35, 19.999), (35, 20), (21, 25), (36, 10), (37, 39.999), (37,
    40), (31, 45), (32, 30), (33, 50), (3, 12), (11, 33), (17, 55), (24, 52)
]
pairs = [
    [0, 11], [1, 14], [2, 17], [3, 8], [4, 19], [5, 12], [6, 15], [7, 18], [9, 16], [10, 13]
]
%s""" % (_PACK,),
            "call": "pack(arg_carrier_genotypes, edges, mutations, pairs, 60.0)",
            "gold_call": "pack(_oracle_arg_carrier_genotypes, edges, mutations, pairs, 60.0)",
        },
        # Leaf 1 moves from node 8 to node 11 over [30, 60) and leaf 6 moves from node 12 into
        # node 9 from 50, so the clade of node 10 loses a leaf at 30, gains one at 50 and regains
        # the first at 60 while the child edges of node 10 itself never change.
        {
            "setup": """edges = [
    (0, 8, 0, 100), (1, 8, 0, 30), (1, 11, 30, 60), (1, 8, 60, 100), (2, 9, 0, 100),
    (3, 9, 0, 100), (4, 11, 0, 100), (5, 11, 0, 100), (6, 12, 0, 50), (6, 9, 50, 100),
    (7, 12, 0, 100), (8, 10, 0, 100), (9, 10, 0, 100), (10, 14, 0, 100), (11, 13, 0, 100),
    (12, 13, 0, 100), (13, 14, 0, 100),
]
mutations = [(10, 10.0), (13, 40.0), (10, 29.999), (9, 49.999), (10, 30.0), (10, 45.0),
             (11, 30.0), (10, 50.0), (9, 50.0), (13, 55.0), (8, 59.999), (10, 60.0), (8, 60.0),
             (14, 5.0), (12, 50.0)]
pairs = [[0, 4], [1, 5], [2, 6], [3, 7]]
%s""" % (_PACK,),
            "call": "pack(arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
            "gold_call": "pack(_oracle_arg_carrier_genotypes, edges, mutations, pairs, 100.0)",
        },
    ]
