#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def constraint_relations(n: int, req_triples: np.ndarray,
                                 discharged_triples: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    req = np.asarray(req_triples, dtype=np.int64).reshape(-1, 3)
    dis = np.asarray(discharged_triples, dtype=np.int64).reshape(-1, 3)

    # -- Canonical ground set: singletons first, then the 2-element subsets.
    rows = [(a, a) for a in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            rows.append((a, b))
    pairs = np.array(rows, dtype=np.int64)
    m = pairs.shape[0]
    index = {(int(p[0]), int(p[1])): i for i, p in enumerate(pairs)}

    def key(u, v):
        if min(u, v) < 0 or max(u, v) >= n:
            raise ValueError("triple leaf outside the leaf set")
        return index[(min(u, v), max(u, v))]

    # -- Two one-sided constraints per required triple; two reversed ones per
    #    discharged triple. The equality of the outer LCAs stays implied.
    direct = np.zeros((m, m), dtype=np.int64)
    for row in range(req.shape[0]):
        x, y, z = (int(v) for v in req[row])
        if len({x, y, z}) != 3:
            raise ValueError("a rooted triple needs three distinct leaves")
        direct[key(x, y), key(x, z)] = 1
        direct[key(x, y), key(y, z)] = 1
    for row in range(dis.shape[0]):
        x, y, z = (int(v) for v in dis[row])
        if len({x, y, z}) != 3:
            raise ValueError("a rooted triple needs three distinct leaves")
        direct[key(x, z), key(x, y)] = 1
        direct[key(y, z), key(x, y)] = 1

    # -- Support-reflexivity, over the constrained pairs and every singleton.
    out = direct.copy()
    supp = (out.any(axis=1) | out.any(axis=0))
    for a in range(n):
        supp[index[(a, a)]] = True
    for i in range(m):
        if supp[i]:
            out[i, i] = 1

    # -- Transitivity and cross-consistency, iterated to a joint fixed point.
    rows_of = {a: [key(a, c) for c in range(n)] for a in range(n)}
    while True:
        before = int(out.sum())
        while True:
            comp = ((out @ out) > 0).astype(np.int64)
            merged = ((out + comp) > 0).astype(np.int64)
            if int(merged.sum()) == int(out.sum()):
                break
            out = merged
        active = (out.any(axis=1) | out.any(axis=0))
        for k in range(m):
            if not active[k]:
                continue
            a, b = int(pairs[k, 0]), int(pairs[k, 1])
            reach_a = out[rows_of[a]].any(axis=0)
            reach_b = out[rows_of[b]].any(axis=0)
            out[k] = ((out[k] > 0) | (reach_a & reach_b)).astype(np.int64)
        if int(out.sum()) == before:
            return np.stack([direct, out]).astype(np.int64)

def realizability_flags(n: int, relations: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    stack = np.asarray(relations, dtype=np.int64)
    if stack.ndim != 3 or stack.shape[0] != 2:
        raise ValueError("relations must have shape (2, m, m)")
    m = stack.shape[1]
    if stack.shape[2] != m:
        raise ValueError("each plane must be square")
    if m != n + n * (n - 1) // 2:
        raise ValueError("relation size does not match the leaf count")
    direct = (stack[0] != 0)
    closed = (stack[1] != 0)

    # -- First condition: nothing other than a singleton itself may be
    #    constrained at or below that singleton.
    first = 1
    for j in range(n):
        for i in range(m):
            if i != j and direct[i, j]:
                first = 0
                break
        if first == 0:
            break

    # -- Transitive closure of the direct relation, kept separate from the
    #    full closure supplied in plane 1.
    trans = direct.copy()
    while True:
        grown = trans | (trans @ trans)
        if np.array_equal(grown, trans):
            break
        trans = grown

    # -- Second condition: a one-directional assertion must not have its
    #    reverse produced by the closure.
    second = 1
    for i in range(m):
        for j in range(m):
            if direct[i, j] and not trans[j, i] and closed[j, i]:
                second = 0
                break
        if second == 0:
            break

    return np.array([first, second], dtype=np.int64)

def forced_triple_flags(n: int, rel: np.ndarray,
                                triples: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    rel = (np.asarray(rel, dtype=np.int64) != 0)
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    m = n + n * (n - 1) // 2
    if rel.shape != (m, m):
        raise ValueError("rel must be square and match the leaf count")

    rows = [(a, a) for a in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            rows.append((a, b))
    index = {p: i for i, p in enumerate(rows)}

    # -- The four present and two absent comparisons that define the pattern.
    flags = np.zeros(triples.shape[0], dtype=np.int64)
    for row in range(triples.shape[0]):
        x, y, z = (int(v) for v in triples[row])
        if len({x, y, z}) != 3:
            raise ValueError("a rooted triple needs three distinct leaves")
        if min(x, y, z) < 0 or max(x, y, z) >= n:
            raise ValueError("triple leaf outside the leaf set")
        i_xy = index[(min(x, y), max(x, y))]
        i_xz = index[(min(x, z), max(x, z))]
        i_yz = index[(min(y, z), max(y, z))]
        observed = (
            bool(rel[i_xy, i_xz]), bool(rel[i_xy, i_yz]),
            bool(rel[i_xz, i_yz]), bool(rel[i_yz, i_xz]),
            bool(rel[i_xz, i_xy]), bool(rel[i_yz, i_xy]),
        )
        target = (True, True, True, True, False, False)
        flags[row] = 1 if observed == target else 0
    return flags

def canonical_dag(n: int, rel: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    rel = (np.asarray(rel, dtype=np.int64) != 0)
    m = n + n * (n - 1) // 2
    if rel.shape != (m, m):
        raise ValueError("rel must be square and match the leaf count")

    # -- Mutual constraint in both directions is the equivalence; classes are
    #    numbered by the smallest pair-table index they contain.
    supported = (rel.any(axis=1) | rel.any(axis=0))
    labels = np.full(m, -1, dtype=np.int64)
    reps = []
    for i in range(m):
        if not supported[i] or labels[i] >= 0:
            continue
        labels[i] = len(reps)
        for j in range(i + 1, m):
            if supported[j] and rel[i, j] and rel[j, i]:
                labels[j] = len(reps)
        reps.append(i)

    # -- Rows 0 to n-1 are the singletons, so the leaf classes come first.
    leaf_class = []
    for a in range(n):
        if int(labels[a]) < 0:
            raise ValueError("every leaf must carry a constraint class")
        leaf_class.append(int(labels[a]))
    if len(set(leaf_class)) != n:
        raise ValueError("two leaves were merged into one class")
    seen = set(leaf_class)
    order = leaf_class + [c for c in range(len(reps)) if c not in seen]
    pos = {c: k for k, c in enumerate(order)}
    size = len(order)

    below = np.zeros((size, size), dtype=bool)
    for ci in order:
        for cj in order:
            if rel[reps[ci], reps[cj]]:
                below[pos[ci], pos[cj]] = True

    # -- Covering comparisons only: no third class strictly in between.
    adj = np.zeros((size, size), dtype=np.int64)
    for j in range(size):
        for i in range(size):
            if i == j or not below[i, j]:
                continue
            covering = True
            for c in range(size):
                if c != i and c != j and below[i, c] and below[c, j]:
                    covering = False
                    break
            if covering:
                adj[j, i] = 1
    return adj

def phylogenetic_network(adj: np.ndarray, req_triples: np.ndarray,
                                 forb_triples: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    adj = (np.asarray(adj, dtype=np.int64) != 0).astype(np.int64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] < 1:
        raise ValueError("adj must be a non-empty square adjacency matrix")
    req = np.asarray(req_triples, dtype=np.int64).reshape(-1, 3)
    forb = np.asarray(forb_triples, dtype=np.int64).reshape(-1, 3)
    size = adj.shape[0]

    def support(triples):
        out = set()
        for row in range(triples.shape[0]):
            x, y, z = (int(v) for v in triples[row])
            if len({x, y, z}) != 3:
                raise ValueError("a rooted triple needs three distinct leaves")
            if max(x, y, z) >= size or min(x, y, z) < 0:
                raise ValueError("triple leaf outside the given DAG")
            out.add((min(x, y), max(x, y)))
            out.add((min(x, z), max(x, z)))
            out.add((min(y, z), max(y, z)))
        return out

    # -- Pairs mentioned only by forbidden triples get two parallel parents.
    repair = sorted(support(forb) - support(req))
    grown = size + 2 * len(repair)
    out = np.zeros((grown, grown), dtype=np.int64)
    out[:size, :size] = adj
    cursor = size
    for (leaf_a, leaf_b) in repair:
        for _ in range(2):
            out[cursor, leaf_a] = 1
            out[cursor, leaf_b] = 1
            cursor += 1

    # -- One new root above all roots, only when more than one remains.
    roots = [v for v in range(grown) if int(out[:, v].sum()) == 0]
    if len(roots) < 2:
        return out
    final = np.zeros((grown + 1, grown + 1), dtype=np.int64)
    final[:grown, :grown] = out
    for v in roots:
        final[grown, v] = 1
    return final

def displayed_triple_flags(adj: np.ndarray, n: int,
                                   triples: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    adj = (np.asarray(adj, dtype=np.int64) != 0)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square adjacency matrix")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    size = adj.shape[0]
    if n < 1 or n > size:
        raise ValueError("n must be between 1 and the number of vertices")
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)

    # -- Reachability closure; a self-loop in it would mean a directed cycle.
    reach = adj.copy()
    while True:
        grown = reach | (reach @ reach)
        if np.array_equal(grown, reach):
            break
        reach = grown
    if np.any(np.diagonal(reach)):
        raise ValueError("adj must be acyclic")
    ancestor = reach.copy()
    for v in range(size):
        ancestor[v, v] = True

    # -- Minimal common ancestors; accept only when exactly one survives.
    lca = np.full((n, n), -1, dtype=np.int64)
    for x in range(n):
        for y in range(x, n):
            common = np.nonzero(ancestor[:, x] & ancestor[:, y])[0]
            if common.size == 0:
                continue
            keep = []
            for v in common:
                if int(ancestor[v, common].sum()) == 1:
                    keep.append(int(v))
            value = keep[0] if len(keep) == 1 else -1
            lca[x, y] = value
            lca[y, x] = value

    # -- Uniqueness, then equality of the outer ancestors, then strict descent.
    flags = np.zeros(triples.shape[0], dtype=np.int64)
    for row in range(triples.shape[0]):
        x, y, z = (int(v) for v in triples[row])
        if len({x, y, z}) != 3:
            raise ValueError("a rooted triple needs three distinct leaves")
        if max(x, y, z) >= n or min(x, y, z) < 0:
            raise ValueError("triple leaf outside the leaf set")
        inner = int(lca[x, y])
        outer_x = int(lca[x, z])
        outer_y = int(lca[y, z])
        if inner < 0 or outer_x < 0 or outer_y < 0:
            continue
        if outer_x != outer_y:
            continue
        if bool(reach[outer_x, inner]):
            flags[row] = 1
    return flags

def triple_resolution_rate(n: int, req_triples: np.ndarray,
                                   forb_triples: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    # -- Validate the orchestrator inputs.
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 3:
        raise ValueError("at least three leaves are needed")
    req = np.asarray(req_triples, dtype=np.int64).reshape(-1, 3)
    forb = np.asarray(forb_triples, dtype=np.int64).reshape(-1, 3)

    # -- Sub-problems 01 and 02: constraints, closure and realizability.
    discharged = np.zeros((0, 3), dtype=np.int64)
    rels = np.asarray(constraint_relations(n, req, discharged),
                      dtype=np.int64)
    if np.any(np.asarray(realizability_flags(n, rels)) == 0):
        raise ValueError("the required constraints are not realizable")

    # -- Sub-problem 03 in a loop: discharge every forbidden triple forced.
    for _ in range(forb.shape[0] + 1):
        flags = np.asarray(forced_triple_flags(n, rels[1], forb))
        forced = np.nonzero(flags.reshape(-1) != 0)[0]
        if forced.size == 0:
            break
        discharged = np.vstack(
            [discharged, forb[int(forced[0])].reshape(1, 3)])
        rels = np.asarray(constraint_relations(n, req, discharged),
                          dtype=np.int64)
    else:
        raise ValueError("the repair loop did not settle")

    # -- Sub-problems 04 and 05: canonical DAG and phylogenetic network.
    dag = canonical_dag(n, rels[1])
    net = np.asarray(phylogenetic_network(dag, req, forb),
                     dtype=np.int64)

    # -- Sub-problem 06: the ancestor-based display test.
    if req.shape[0] and not np.all(
            np.asarray(displayed_triple_flags(net, n, req)) != 0):
        raise ValueError("no phylogenetic network agrees with the triple sets")
    if forb.shape[0] and np.any(
            np.asarray(displayed_triple_flags(net, n, forb)) != 0):
        raise ValueError("no phylogenetic network agrees with the triple sets")

    # -- Sweep the whole triple space and normalise by the 3-element subsets.
    space = []
    for x in range(n):
        for y in range(x + 1, n):
            for z in range(n):
                if z != x and z != y:
                    space.append((x, y, z))
    space = np.array(space, dtype=np.int64).reshape(-1, 3)
    shown = np.asarray(displayed_triple_flags(net, n, space))
    subsets = n * (n - 1) * (n - 2) // 6
    return float(int(np.count_nonzero(shown)) / subsets)
SCICODE_GOLD_EOF
