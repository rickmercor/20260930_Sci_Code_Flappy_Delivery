"""
Count how often each species occupies each structural position in the web's three-node subgraphs.

A species is characterised not by its identity but by the local interaction patterns it takes part in. Restricting attention to three species at a time and to the arcs induced among exactly those three, the connected patterns fall into finitely many classes up to relabelling, and within a class the three positions are not always distinguishable: symmetries of the pattern make some positions equivalent. Counting, for every species, how many times it occupies each distinguishable position yields a fixed-length profile that captures its structural part in the community. Only induced subgraphs are counted, so every arc present among the three species must be taken into account, not merely a chosen subset.



The ordering of the role-profile columns is not itself scientifically meaningful. Any deterministic global ordering of the distinguishable roles is acceptable, provided that the same ordering is used consistently for every web and every call so that corresponding role components remain aligned across webs.

Returns
-------
list of n lists of ints, one row per species giving its count for each distinguishable three-node position; role columns may use any deterministic global ordering provided that the same ordering is used consistently across all calls
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def motif_profiles(directed: list[list[int]]) -> list[list[int]]:
    '''Count each species' occupancy of every distinguishable three-node position.

    Parameters
    ----------
    directed : list[list[int]]
        Directed adjacency of one web, n rows of n entries in {0, 1}.

    Returns
    -------
    profiles : list[list[int]]
        One row per species; each row has one count per distinguishable
        three-node position. The role-column ordering may be any deterministic
        global ordering, but the same ordering must be used consistently
        across all calls.
    '''
    profiles = []
    return profiles  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import permutations, product, combinations
import math
_ARCS3 = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
def _canon3(adj):
    best = None
    for p in permutations(range(3)):
        key = tuple(sorted((p[u], p[v]) for (u, v) in adj))
        if best is None or key < best:
            best = key
    return best
def _conn3(adj):
    und = {}
    for u, v in adj:
        und.setdefault(u, set()).add(v)
        und.setdefault(v, set()).add(u)
    if len(und) < 3:
        return False
    seen, st = set(), [next(iter(und))]
    while st:
        x = st.pop()
        if x in seen:
            continue
        seen.add(x)
        st.extend(und.get(x, ()))
    return len(seen) == 3
def _role_table():
    keys = set()
    for mask in product([0, 1], repeat=6):
        adj = frozenset(a for a, m in zip(_ARCS3, mask) if m)
        if _conn3(adj):
            keys.add(_canon3(adj))
    table, rid = {}, 0
    for key in sorted(keys):
        rep = set(key)
        autos = [p for p in permutations(range(3))
                 if set((p[u], p[v]) for (u, v) in rep) == rep]
        orbits = sorted({frozenset(p[v] for p in autos) for v in range(3)}, key=sorted)
        for orb in orbits:
            table[(key, orb)] = rid
            rid += 1
    return table, rid
_ROLES, _NROLES = _role_table()
def _oracle_motif_profiles(directed):
    n = len(directed)
    prof = [[0] * _NROLES for _ in range(n)]
    for trio in combinations(range(n), 3):
        adj = frozenset((a, b) for a in range(3) for b in range(3)
                        if a != b and directed[trio[a]][trio[b]])
        if not _conn3(adj):
            continue
        key = _canon3(adj)
        rep = set(key)
        autos = [p for p in permutations(range(3))
                 if set((p[u], p[v]) for (u, v) in rep) == rep]
        iso = next(p for p in permutations(range(3))
                   if tuple(sorted((p[u], p[v]) for (u, v) in adj)) == key)
        for local in range(3):
            orb = frozenset(p[iso[local]] for p in autos)
            prof[trio[local]][_ROLES[(key, orb)]] += 1
    return prof

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

def _canon_profile_columns(P):
    rows = [list(row) for row in P]
    if not rows:
        return rows
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return rows
    cols = sorted(
        tuple(rows[i][j] for i in range(len(rows)))
        for j in range(width)
    )
    return [list(row) for row in zip(*cols)] if cols else [[] for _ in rows]

def _canon_joint_profile_columns(profile_sets):
    sets = [[list(row) for row in P] for P in profile_sets]
    widths = [len(row) for P in sets for row in P]
    if not widths or any(width != widths[0] for width in widths):
        return sets
    width = widths[0]
    cols = []
    for j in range(width):
        col = []
        for P in sets:
            col.extend(row[j] for row in P)
        cols.append(tuple(col))
    return sorted(cols)

W1 = [(0,3),(1,5),(2,5),(3,0),(4,5),(5,0),(5,1)]
W2 = [(0,4),(1,3),(1,4),(2,0),(3,2),(3,4),(4,3),(5,2)]
W3 = [(0,5),(1,4),(2,3),(2,5),(4,1),(4,5),(5,1),(5,2)]
D1, U1 = _fx(W1, 6)
D2, U2 = _fx(W2, 6)
D3, U3 = _fx(W3, 6)
'''
    return [
        {   # normal: all three benchmark webs; one global role ordering must work for every web
            "setup": setup,
            "call": "_canon_joint_profile_columns([motif_profiles(D1), motif_profiles(D2), motif_profiles(D3)])",
            "gold_call": "_canon_joint_profile_columns([_oracle_motif_profiles(D1), _oracle_motif_profiles(D2), _oracle_motif_profiles(D3)])",
        },
        {   # normal: a denser web, compared independently of the arbitrary global column labels
            "setup": setup,
            "call": "_canon_profile_columns(motif_profiles(D2))",
            "gold_call": "_canon_profile_columns(_oracle_motif_profiles(D2))",
        },
        {   # boundary: a single three-species chain
            "setup": setup,
            "call": "_canon_profile_columns(motif_profiles(_fx([(0,1),(1,2)], 3)[0]))",
            "gold_call": "_canon_profile_columns(_oracle_motif_profiles(_fx([(0,1),(1,2)], 3)[0]))",
        },
        {   # edge: no connected triple, so every count is zero
            "setup": setup,
            "call": "_canon_profile_columns(motif_profiles(_fx([(0,1)], 3)[0]))",
            "gold_call": "_canon_profile_columns(_oracle_motif_profiles(_fx([(0,1)], 3)[0]))",
        },
    ]
