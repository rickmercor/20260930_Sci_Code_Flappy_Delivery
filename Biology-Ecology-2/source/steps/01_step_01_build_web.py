"""
Turn an arc list into a directed food web and its underlying undirected graph.

A food web is a directed graph whose arcs run from prey to predator. Two structures are needed downstream and they are not interchangeable. The directed adjacency determines which three-node interaction patterns each species participates in and which position it occupies within them, because reversing an arc changes the pattern. The underlying undirected adjacency, obtained by ignoring arc direction, is what enters the quadratic term of the alignment objective, since that term measures whether neighbours are matched to neighbours irrespective of who eats whom. This step also records whether the web is connected, because a species isolated from the rest participates in no interaction pattern and would carry an empty profile.

Returns
-------
dict, with keys 'n', 'n_arcs' (int), 'directed', 'undirected' (n by n lists of 0/1 ints) and 'connected' (bool)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_web(arcs: list[tuple[int, int]], n: int) -> dict:
    '''Build the directed and underlying undirected adjacency of a food web.
    Parameters
    ----------
    arcs : list[tuple[int, int]]
        Trophic links written as (prey, predator) pairs of species labels.
    n : int
        Number of species, labelled 0 to n-1.
    Returns
    -------
    web : dict
        Mapping with keys 'n' (int), 'n_arcs' (int), 'directed' and 'undirected'
        (each a list of n rows of n ints), and 'connected' (bool).
    '''
    web = {}
    return web  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_web(arcs, n):
    if n < 3:
        raise ValueError("a web must have at least three species")
    D = [[0] * n for _ in range(n)]
    for a in arcs:
        if len(a) != 2:
            raise ValueError("each arc must be a (prey, predator) pair")
        u, v = a
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("species label out of range")
        if u == v:
            raise ValueError("self-loop at species %d" % u)
        D[u][v] = 1
    U = [[1 if (D[i][j] or D[j][i]) else 0 for j in range(n)] for i in range(n)]
    seen, st = set(), [0]
    while st:
        x = st.pop()
        if x in seen:
            continue
        seen.add(x)
        st.extend(j for j in range(n) if U[x][j])
    return {"n": n, "n_arcs": sum(map(sum, D)), "directed": D,
            "undirected": U, "connected": len(seen) == n}

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

def _pack_web(w):
    # [n, n_arcs, connected(0/1), directed flattened, undirected flattened]
    return (
        [float(w["n"]), float(w["n_arcs"]), 1.0 if w["connected"] else 0.0]
        + [float(x) for row in w["directed"] for x in row]
        + [float(x) for row in w["undirected"] for x in row]
    )

W1 = [(0,3),(1,5),(2,5),(3,0),(4,5),(5,0),(5,1)]
W2 = [(0,4),(1,3),(1,4),(2,0),(3,2),(3,4),(4,3),(5,2)]
W3 = [(0,5),(1,4),(2,3),(2,5),(4,1),(4,5),(5,1),(5,2)]
D1, U1 = _fx(W1, 6)
D2, U2 = _fx(W2, 6)
D3, U3 = _fx(W3, 6)
'''
    return [
        {
            "setup": setup,
            "call": "_pack_web(build_web(W1, 6))",
            "gold_call": "_pack_web(_oracle_build_web(W1, 6))",
        },
        {
            "setup": setup,
            "call": "_pack_web(build_web(W2, 6))",
            "gold_call": "_pack_web(_oracle_build_web(W2, 6))",
        },
        {
            "setup": setup,
            "call": "_pack_web(build_web([(0,1),(1,2)], 3))",
            "gold_call": "_pack_web(_oracle_build_web([(0,1),(1,2)], 3))",
        },
        {
            "setup": setup,
            "call": "_pack_web(build_web([(0,1)], 3))",
            "gold_call": "_pack_web(_oracle_build_web([(0,1)], 3))",
        },
    ]
