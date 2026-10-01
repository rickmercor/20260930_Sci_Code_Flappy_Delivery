#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_web(arcs, n):
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
def motif_profiles(directed):
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

import math
def cost_matrix(profiles_a, profiles_b):
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

import math
def alignment_iteration(A1, A2, C, T, alpha, eps, gamma, mu, nu):
    m, n = len(C), len(C[0])
    def matmul(X, Y):
        p, q, r = len(X), len(Y), len(Y[0])
        return [[sum(X[i][k] * Y[k][j] for k in range(q)) for j in range(r)]
                for i in range(p)]
    def surrogate(M, elementwise_cost):
        if elementwise_cost:
            inner = matmul(matmul(A1, M), A2)
            core = [[C[i][j] * inner[i][j] for j in range(n)] for i in range(m)]
        else:
            core = matmul(matmul(A1, [[C[i][j] * M[i][j] for j in range(n)]
                                      for i in range(m)]), A2)
        return [[alpha * core[i][j] + 0.5 * (1 - alpha) * C[i][j]
                 - 0.5 * eps * mu[i] * nu[j] for j in range(n)] for i in range(m)]
    Q = surrogate(T, False)
    H = [[T[i][j] * math.exp(-Q[i][j] / gamma) for j in range(n)] for i in range(m)]
    rf = [min(mu[i] / s, 1.0) if (s := sum(H[i])) > 0 else 1.0 for i in range(m)]
    H = [[H[i][j] * rf[i] for j in range(n)] for i in range(m)]
    Qp = surrogate(H, True)
    H = [[H[i][j] * math.exp(-Qp[i][j] / gamma) for j in range(n)] for i in range(m)]
    cf = [min(nu[j] / s, 1.0) if (s := sum(H[i][j] for i in range(m))) > 0 else 1.0
          for j in range(n)]
    return [[round(H[i][j] * cf[j], 9) for j in range(n)] for i in range(m)]

def align(A1, A2, C, alpha, eps, gamma, mu, nu, iterations):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    m, n = len(C), len(C[0])
    T = [[1.0 / (m * n)] * n for _ in range(m)]
    for _ in range(iterations):
        T = alignment_iteration(A1, A2, C, T, alpha, eps, gamma, mu, nu)
    return [[round(x, 6) for x in row] for row in T]

def role_similarity(costs_to_others, alignments_to_others):
    if len(costs_to_others) != len(alignments_to_others):
        raise ValueError("one cost matrix per alignment is required")
    m = len(costs_to_others[0])
    out = [0.0] * m
    for C, T in zip(costs_to_others, alignments_to_others):
        for i in range(m):
            out[i] += sum((1.0 - C[i][j]) * T[i][j] for j in range(len(C[i])))
    return [round(x, 6) for x in out]

def backbone(scores, k):
    if not 2 <= k <= len(scores):
        raise ValueError("k must be between 2 and the number of species")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return sorted(order[:k])

def transitivity(T_ip, T_iq, T_pq, j):
    num = den = 0.0
    for a, x in enumerate(T_ip[j]):
        if x == 0.0:
            continue
        for b, y in enumerate(T_iq[j]):
            if y == 0.0:
                continue
            den += x * y
            num += x * y * T_pq[a][b]
    if den <= 0.0:
        raise ValueError("no alignment mass at species %d: ratio undefined" % j)
    return round(num / den, 6)

def backbone_transitivity(webs, alpha, eps, gamma, iterations, k):
    N = len(webs)
    if N != 3:
        raise ValueError("exactly three webs are required")
    built = [build_web(arcs, n) for arcs, n in webs]
    profs = [motif_profiles(b["directed"]) for b in built]
    n = built[0]["n"]
    unit = [1.0 / n] * n
    C, T = {}, {}
    for i in range(N):
        for p in range(N):
            if i == p:
                continue
            C[(i, p)] = cost_matrix(profs[i], profs[p])
            T[(i, p)] = align(built[i]["undirected"], built[p]["undirected"],
                                      C[(i, p)], alpha, eps, gamma, unit, unit,
                                      iterations)
    others = [p for p in range(N) if p != 0]
    scores = role_similarity([C[(0, p)] for p in others],
                                     [T[(0, p)] for p in others])
    members = backbone(scores, k)
    p, q = others
    vals = [transitivity(T[(0, p)], T[(0, q)], T[(p, q)], j) for j in members]
    return round(sum(vals) / len(vals), 6)
SCICODE_GOLD_EOF
