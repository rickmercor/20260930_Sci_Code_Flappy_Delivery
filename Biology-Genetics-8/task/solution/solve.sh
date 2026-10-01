#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def classify_vertices(edges):
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

def event_sequence(classification):
    hyb = set(classification["hybridisation"])
    internal = ([classification["root"]] + classification["speciation"]
                + classification["hybridisation"])
    ordered = sorted(internal, key=lambda v: int(v[1:]))
    return [1 if v in hyb else 0 for v in ordered]

def lineage_sets(edges, size, root):
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

def encoding_matrix(interval_sets):
    N = len(interval_sets)
    sets = [set(L) for L in interval_sets]
    return [
        [
            len(sets[i] & sets[j]) if j <= i else 0
            for j in range(N)
        ]
        for i in range(N)
    ]

def diagonal_profile(F, events):
    N = len(F)
    diag = [F[i][i] for i in range(N)]
    sub = [F[i + 1][i] for i in range(N - 1)]
    ok = True
    for i in range(N - 1):
        drop = 1 if events[i] == 0 else 2
        if sub[i] != diag[i] - drop:
            ok = False
        step = 1 if events[i] == 0 else -1
        if diag[i + 1] != diag[i] + step:
            ok = False
    return {"diagonal": diag, "subdiagonal": sub, "consistent": bool(ok)}

def validate_encoding(F, n, m):
    N = len(F)
    g = lambda i, j: F[i - 1][j - 1]
    p1 = g(1, 1) == 1 and all(g(1, k) == 0 and g(k, 1) == 0 for k in range(2, N + 1))
    p2 = (g(1, 1) == 1 and g(2, 2) == 2 and g(3, 3) == 3 and g(N, N) == n
          and all(g(i, i) > 0 for i in range(1, N + 1))
          and all(g(i, i) in (g(i - 1, i - 1) - 1, g(i - 1, i - 1) + 1)
                  for i in range(2, N + 1)))
    p3 = True
    for i in range(2, N):
        d = g(i + 1, i + 1) - g(i, i)
        if g(i + 1, i) != (g(i, i) - 1 if d == 1 else g(i, i) - 2):
            p3 = False
    col2 = [g(i, 2) for i in range(3, N + 1)]
    p4 = (g(3, 2) == 1 and set(col2) <= {0, 1}
          and all(col2[t] >= col2[t + 1] for t in range(len(col2) - 1)))
    p5 = True
    for k in range(3, N):
        for i in range(k + 1, N + 1):
            lo = max(0, g(i - 1, k) - 2, g(i, k - 1),
                     g(i, k - 1) + g(i - 1, k) - g(i - 1, k - 1) - 2)
            hi = min(g(i - 1, k), g(i, k - 1) + 2,
                     g(i, k - 1) + g(i - 1, k) - g(i - 1, k - 1))
            if not lo <= g(i, k) <= hi:
                p5 = False
    return {"size_ok": N == n + 2 * m, "P1": bool(p1), "P2": bool(p2),
            "P3": bool(p3), "P4": bool(p4), "P5": bool(p5),
            "valid": bool(N == n + 2 * m and p1 and p2 and p3 and p4 and p5)}

def weight_matrix(times):
    if len(times) < 2:
        raise ValueError("times must list u_0 through u_N")
    N = len(times) - 1
    for k in range(1, len(times)):
        if times[k] <= times[k - 1]:
            raise ValueError("event times must be strictly increasing")
    return [[(times[i] - times[j - 1]) if j < i else 0
             for j in range(1, N + 1)] for i in range(1, N + 1)]

def weighted_distance(matrix_a, weights_a, matrix_b, weights_b):
    import math
    N = len(matrix_a)
    if not (len(weights_a) == len(matrix_b) == len(weights_b) == N):
        raise ValueError("all four matrices must have the same size")
    total = 0.0
    for i in range(N):
        for j in range(N):
            total += (matrix_a[i][j] * weights_a[i][j]
                      - matrix_b[i][j] * weights_b[i][j]) ** 2
    return round(math.sqrt(total), 6)

def timed_network_distance(edges_a, times_a, edges_b, times_b):
    ca = classify_vertices(edges_a)
    cb = classify_vertices(edges_b)
    if ca["size"] != cb["size"]:
        raise ValueError("encodings differ in size; event alignment would be required")
    out = []
    for edges, c in ((edges_a, ca), (edges_b, cb)):
        F = encoding_matrix(lineage_sets(edges, c["size"], c["root"]))
        if not validate_encoding(F, c["n"], c["m"])["valid"]:
            raise ValueError("network is not admitted to the encoding space")
        if not diagonal_profile(F, event_sequence(c))["consistent"]:
            raise ValueError("encoding diagonal disagrees with the event sequence")
        out.append(F)
    return weighted_distance(out[0], weight_matrix(times_a),
                                     out[1], weight_matrix(times_b))
SCICODE_GOLD_EOF
