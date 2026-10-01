"""
Run the whole pipeline and return the mean transitivity over the first web's backbone.

This step chains the pipeline end to end: build the three webs, profile every species by its three-node positions, convert profiles into pairwise dissimilarities, align every ordered pair of distinct webs, score the first web's species by cheaply-matched mass to the other two, take its backbone, and average the transitivity of the backbone species. The result summarises how coherently the persistent core of one community corresponds to the other two: a high value means the correspondences its best-matched species participate in agree with one another, a low value means they are individually strong but mutually inconsistent.

Returns
-------
float, the mean transitivity over the backbone of the first web, rounded to 6 decimal places
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def backbone_transitivity(webs: list[tuple[list[tuple[int, int]], int]],
                         alpha: float, eps: float, gamma: float,
                         iterations: int, k: int) -> float:
    '''Mean transitivity over the backbone of the first web.
    Parameters
    ----------
    webs : list[tuple[list[tuple[int, int]], int]]
        Three webs, each an (arc list, species count) pair.
    alpha, eps, gamma : float
        Tradeoff parameter, self-alignment parameter and step size.
    iterations : int
        Alignment iterations, with no early stopping.
    k : int
        Backbone size.
    Returns
    -------
    score : float
        Mean transitivity over the backbone, rounded to 6 decimal places.
    '''
    score = 0.0
    return score  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_backbone_transitivity(webs, alpha, eps, gamma, iterations, k):
    N = len(webs)
    if N != 3:
        raise ValueError("exactly three webs are required")
    built = [_oracle_build_web(arcs, n) for arcs, n in webs]
    profs = [_oracle_motif_profiles(b["directed"]) for b in built]
    n = built[0]["n"]
    unit = [1.0 / n] * n
    C, T = {}, {}
    for i in range(N):
        for p in range(N):
            if i == p:
                continue
            C[(i, p)] = _oracle_cost_matrix(profs[i], profs[p])
            T[(i, p)] = _oracle_align(built[i]["undirected"], built[p]["undirected"],
                                      C[(i, p)], alpha, eps, gamma, unit, unit,
                                      iterations)
    others = [p for p in range(N) if p != 0]
    scores = _oracle_role_similarity([C[(0, p)] for p in others],
                                     [T[(0, p)] for p in others])
    members = _oracle_backbone(scores, k)
    p, q = others
    vals = [_oracle_transitivity(T[(0, p)], T[(0, q)], T[(p, q)], j) for j in members]
    return round(sum(vals) / len(vals), 6)

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
WEBS = [(W1, 6), (W2, 6), (W3, 6)]
WEBS_ROT = [(W2, 6), (W3, 6), (W1, 6)]
'''
    return [
        {   # normal: the instance the task poses
            "setup": setup,
            "call": "backbone_transitivity(WEBS, 0.7, 5.0, 1.0, 300, 3)",
            "gold_call": "_oracle_backbone_transitivity(WEBS, 0.7, 5.0, 1.0, 300, 3)",
        },
        {   # normal: fewer iterations, close to the converged value
            "setup": setup,
            "call": "backbone_transitivity(WEBS, 0.7, 5.0, 1.0, 100, 3)",
            "gold_call": "_oracle_backbone_transitivity(WEBS, 0.7, 5.0, 1.0, 100, 3)",
        },
        {   # boundary: a larger step size
            "setup": setup,
            "call": "backbone_transitivity(WEBS, 0.7, 5.0, 2.0, 300, 3)",
            "gold_call": "_oracle_backbone_transitivity(WEBS, 0.7, 5.0, 2.0, 300, 3)",
        },
        {   # edge: the smallest admissible backbone
            "setup": setup,
            "call": "backbone_transitivity(WEBS, 0.7, 5.0, 1.0, 300, 2)",
            "gold_call": "_oracle_backbone_transitivity(WEBS, 0.7, 5.0, 1.0, 300, 2)",
        },
    ]
