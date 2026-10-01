"""
Insert the expansion p = p^(0) + eps p^(1) + eps^2 p^(2) + ... into the master equation dp/dt = (F / eps + S) p and collect equal powers of eps. The construction is fixed by three conventions and nothing else:

1. the projection of the k-th full-space term is carried by the k-th reduced variable, Q p^(k) = p_tilde^(k);
2. every full-space term is linear in the reduced variables introduced so far, with matrix coefficients that do not depend on time;
3. the coefficient of p_tilde^(k) in p^(k) is the leading-order map K^(0), and every coefficient of a lower-order reduced variable is annihilated by Q.

Carried to order K this determines, in particular, the response matrix that multiplies p_tilde^(0) in each of p^(1), ..., p^(K), and it closes the K + 1 reduced equations into a single linear autonomous system

    dy/dt = L y,   y = (p_tilde^(0), p_tilde^(1), ..., p_tilde^(K)).

Both are needed downstream: L propagates the reduced coefficients, and the response matrices rebuild the full-space probability from them, which is what an observable that varies between the configurations of one class requires. The hierarchy itself contains no eps; the separation parameter enters only when the terms are recombined.

Nothing in the construction may assume that Q is a 0/1 matrix, and nothing asks the fast process to be reversible.

F and S may be scipy.sparse matrices as large as the lattice's 32768 configurations, so the closure may not form any dense array with m x m entries. The responses and L are returned as dense arrays.

Insert the expansion p = p^(0) + eps p^(1) + eps^2 p^(2) + ... into the master equation dp/dt = (F / eps + S) p and collect equal powers of eps. The construction is fixed by three conventions and nothing else:

Returns
-------
tuple (resp, L): np.ndarray of shape (m, K n) holding the response matrices of p_tilde^(0) in p^(1), ..., p^(K) side by side in that order, each annihilated by Q, and np.ndarray of shape ((K+1) n, (K+1) n) with the generator of the closed hierarchy for (p_tilde^(0), ..., p_tilde^(K)).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hierarchy_closure(F: "np.ndarray | scipy.sparse.spmatrix", S: "np.ndarray | scipy.sparse.spmatrix", Q: "np.ndarray", K0: "np.ndarray", order: int) -> tuple: """Parameters: fast and slow generators F and S, projector Q, leading map K0, and integer order >= 1. Returns: response matrices resp and reduced hierarchy generator L. Raises: ValueError if F or S is not a non-empty square 2D array, holds a non-finite entry, has a negative off-diagonal rate or a column that does not sum to zero; if F and S differ in shape; if Q does not have shape (n, m) with n >= 1, holds a non-finite entry, is negative anywhere, has a column that does not sum to one, or does not annihilate F; if K0 is not a finite array of shape (m, n), Q @ K0 is not the identity, or F @ K0 does not vanish; if order is not an integer >= 1; or if the constrained system has no solution."""; return resp, L

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse


def _as_generator(M, name):
    """CSR copy of a square generator after the column-convention checks."""
    import scipy.sparse as sp
    if sp.issparse(M):
        M = sp.csr_matrix(M, dtype=float)
    else:
        M = np.asarray(M, dtype=float)
        if M.ndim != 2:
            raise ValueError(name + " must be a non-empty square 2D array")
        M = sp.csr_matrix(M)
    if M.shape[0] != M.shape[1] or M.shape[0] < 1:
        raise ValueError(name + " must be a non-empty square 2D array")
    M.sum_duplicates()
    if not np.all(np.isfinite(M.data)):
        raise ValueError(name + " must be finite")
    scale = max(1.0, float(abs(M).max()))
    coo = M.tocoo()
    off = coo.row != coo.col
    if off.any() and coo.data[off].min() < -1e-12 * scale:
        raise ValueError(name + " has a negative off-diagonal rate")
    if np.abs(np.asarray(M.sum(axis=0)).ravel()).max() > 1e-9 * scale:
        raise ValueError("every column of " + name + " must sum to zero")
    return M


def _check_projector(F, Q):
    Q = np.asarray(Q, dtype=float)
    m = F.shape[0]
    if Q.ndim != 2 or Q.shape[1] != m or Q.shape[0] < 1:
        raise ValueError("Q must have shape (n, m) with n >= 1")
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q must be finite")
    if Q.min() < -1e-9 or np.abs(Q.sum(axis=0) - 1.0).max() > 1e-9:
        raise ValueError("Q must be non-negative with every column summing to one")
    if np.abs(F.T @ Q.T).max() > 1e-8 * max(1.0, float(abs(F).max())):
        raise ValueError("Q must annihilate F")
    return Q


def _oracle_hierarchy_closure(F: "np.ndarray | scipy.sparse.spmatrix",
                              S: "np.ndarray | scipy.sparse.spmatrix",
                              Q: "np.ndarray", K0: "np.ndarray", order: int) -> tuple:
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    F = _as_generator(F, "F")
    S = _as_generator(S, "S")
    if S.shape != F.shape:
        raise ValueError("F and S must have the same shape")
    Q = _check_projector(F, Q)
    K0 = np.asarray(K0, dtype=float)
    n, m = Q.shape
    if K0.shape != (m, n) or not np.all(np.isfinite(K0)):
        raise ValueError("K0 must be a finite array of shape (m, n)")
    if np.abs(Q @ K0 - np.eye(n)).max() > 1e-8:
        raise ValueError("Q K0 must be the identity")
    fscale = max(1.0, float(abs(F).max()))
    if np.abs(F @ K0).max() > 1e-6 * fscale:
        raise ValueError("F K0 must vanish")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be an integer >= 1")
    order = int(order)
    # F X = R with Q X = 0 is the bordered system [[F, K0], [Q, 0]] [X; lam] = [R; 0]:
    # Q R = 0 forces lam = 0, and the border removes the null space of F.
    bordered = sp.bmat([[F, sp.csr_matrix(K0)], [sp.csr_matrix(Q), None]], format="csc")
    try:
        lu = splu(bordered)
    except RuntimeError:
        raise ValueError("the constrained system has no solution")
    resp = np.empty((m, order * n))
    G = [K0] + [resp[:, k * n:(k + 1) * n] for k in range(order)]
    C = [Q @ (S @ K0)]
    width = 24
    for j in range(order):
        for lo in range(0, n, width):
            cols = slice(lo, min(n, lo + width))
            rhs = -(S @ G[j][:, cols])
            for i in range(j + 1):
                rhs += G[j - i] @ C[i][:, cols]
            X = lu.solve(np.vstack([rhs, np.zeros((n, rhs.shape[1]))]))[:m]
            res = F @ X - rhs
            if max(res.max(), -res.min()) > 1e-6 * max(1.0, float(np.abs(rhs).max())):
                raise ValueError("the constrained system has no solution")
            G[j + 1][:, cols] = X
        C.append(Q @ (S @ G[j + 1]))
    L = np.zeros(((order + 1) * n, (order + 1) * n))
    for r in range(order + 1):
        for c in range(r + 1):
            L[r * n:(r + 1) * n, c * n:(c + 1) * n] = C[r - c]
    return resp, L

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup_common = """import numpy as np
import scipy.sparse
def _circ(sizes, seed, one_way=False, balanced=False):
    g = np.random.default_rng(seed)
    m = int(sum(sizes))
    perm = g.permutation(m)
    J = np.zeros((m, m))
    pi = np.zeros(m)
    groups = []
    off = 0
    for sz in sizes:
        idx = perm[off:off + sz]
        off += sz
        groups.append(sorted(int(i) for i in idx))
        w = g.uniform(0.5, 2.0, size=sz)
        pi[idx] = w / w.sum()
        if sz == 1:
            continue
        if not balanced:
            c = g.uniform(0.3, 1.5)
            for a in range(sz):
                J[idx[(a + 1) % sz], idx[a]] += c
            if sz >= 4:
                sub = g.choice(idx, size=3, replace=False)
                c = g.uniform(0.2, 1.0)
                for a in range(3):
                    J[sub[(a + 1) % 3], sub[a]] += c
        if not one_way:
            for a in range(1 if sz == 2 else sz):
                i, j = idx[a], idx[(a + 1) % sz]
                c = g.uniform(0.1, 0.8)
                J[j, i] += c
                J[i, j] += c
    F = J / pi[None, :]
    F[np.diag_indices(m)] = 0.0
    F[np.diag_indices(m)] = -F.sum(axis=0)
    groups.sort(key=min)
    Q = np.zeros((len(groups), m))
    K0 = np.zeros((m, len(groups)))
    for r, grp in enumerate(groups):
        Q[r, grp] = 1.0
        K0[grp, r] = pi[grp]
    return F, Q, K0
def _rates(m, seed, scale=1.0):
    g = np.random.default_rng(seed)
    S = g.uniform(0.05, 2.0, size=(m, m)) * scale
    np.fill_diagonal(S, 0.0)
    S[np.diag_indices(m)] = -S.sum(axis=0)
    return S
def _named(fast, slow, m):
    F = np.zeros((m, m)); S = np.zeros((m, m))
    for (i, j), k in fast.items():
        F[j, i] += k
    for (i, j), k in slow.items():
        S[j, i] += k
    F[np.diag_indices(m)] = -F.sum(axis=0); S[np.diag_indices(m)] = -S.sum(axis=0)
    return F, S
def _group(groups, m):
    Q = np.zeros((len(groups), m))
    for b, grp in enumerate(groups):
        for s in grp:
            Q[b, s] = 1.0
    return Q
def _cols(cols, m):
    K0 = np.zeros((m, len(cols)))
    for r, col in enumerate(cols):
        for s, v in col.items():
            K0[s, r] = v
    return K0
def _bigc(n_classes, max_size, n_left, seed):
    g = np.random.default_rng(seed)
    sizes = g.integers(1, max_size + 1, size=n_classes)
    held_n = int(sizes.sum())
    m = held_n + n_left
    perm = g.permutation(m)
    label = np.full(m, -1)
    pi = np.zeros(m)
    rows, cols, flux = [], [], []
    off = 0
    for k, sz in enumerate(sizes):
        idx = perm[off:off + sz]
        off += sz
        label[idx] = k
        w = g.uniform(0.5, 2.0, size=sz)
        pi[idx] = w / w.sum()
        if sz == 1:
            continue
        nxt = np.roll(idx, -1)
        sym = g.uniform(0.1, 0.8, size=sz)
        rows += [nxt, idx]
        cols += [idx, nxt]
        flux += [sym + g.uniform(0.3, 1.5), sym]
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(flux) / pi[c]
    left = perm[held_n:]
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    into = g.integers(0, n_classes, size=(2, n_left))
    out = g.uniform(0.2, 2.0, size=(2, n_left))
    for e in range(2):
        r = np.concatenate([r, perm[start[into[e]] + g.integers(0, sizes[into[e]])]])
        c = np.concatenate([c, left])
        v = np.concatenate([v, out[e]])
    F = scipy.sparse.csr_matrix((v, (r, c)), shape=(m, m))
    F = (F - scipy.sparse.diags(np.asarray(F.sum(axis=0)).ravel())).tocsr()
    first = np.full(n_classes, m)
    np.minimum.at(first, label[perm[:held_n]], perm[:held_n])
    rank = np.argsort(np.argsort(first))
    held = perm[:held_n]
    Q = np.zeros((n_classes, m))
    Q[rank[label[held]], held] = 1.0
    for e in range(2):
        np.add.at(Q, (rank[into[e]], left), out[e] / out.sum(axis=0))
    K0 = np.zeros((m, n_classes))
    K0[held, rank[label[held]]] = pi[held]
    return F, Q, K0
def _slow(m, per, seed):
    g = np.random.default_rng(seed)
    src = np.repeat(np.arange(m), per)
    dst = g.integers(0, m, size=m * per)
    keep = src != dst
    S = scipy.sparse.csr_matrix((g.uniform(0.05, 2.0, size=int(keep.sum())), (dst[keep], src[keep])),
                                shape=(m, m))
    return (S - scipy.sparse.diags(np.asarray(S.sum(axis=0)).ravel())).tocsr()
def _fp(res):
    G, L = res
    out = 0.0
    for w, M in ((1.0, G), (2.0, L)):
        M = np.asarray(M, dtype=float)
        if M.ndim != 2:
            return float("nan")
        a = np.cos(0.37 * np.arange(M.shape[0]) + 0.2)
        b = np.sin(0.71 * np.arange(M.shape[1]) + 0.3)
        c = np.cos(1.29 * np.arange(M.shape[1]) + 0.7)
        out += w * (a @ M @ b + 0.5 * (a * a) @ M @ c + 1e-3 * M.shape[0] + 1e-2 * M.shape[1])
    return float(out)
"""
    three = """
F, S = _named({(0, 1): 2.3, (1, 0): 1.1}, {(1, 2): 1.4, (2, 1): 4.2}, 3)
Q = _group([[0, 1], [2]], 3)
K0 = _cols([{0: 0.3235294117647059, 1: 0.6764705882352942}, {2: 1.0}], 3)
"""
    def _raises(args):
        return """
def run_model():
    try:
        hierarchy_closure(%s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_hierarchy_closure(%s)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""" % (args, args)
    return [
        # --- Normal: about 33000 configurations held as sparse matrices, 200
        # driven classes and three thousand configurations that the fast events
        # leave for good in one step, each towards two classes, so that the
        # projector carries fractional weights, with sparse slow events, at
        # third order ---
        {"setup": setup_common + """
F, Q, K0 = _bigc(200, 300, 3000, 41)
S = _slow(F.shape[0], 4, 42)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 3))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 3))"},
        # --- Normal: sixty configurations on shuffled indices in twelve driven
        # classes, slow events joining every pair, at third order ---
        {"setup": setup_common + """
F, Q, K0 = _circ([1, 1, 2, 3, 3, 4, 5, 5, 6, 8, 9, 13], 71)
S = _rates(60, 72)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 3))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 3))"},
        # --- Normal: five classes of unequal size at third order ---
        {"setup": setup_common + """
F, Q, K0 = _circ([1, 2, 3, 4, 2], 101)
S = _rates(12, 202)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 3))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 3))"},
        # --- Normal: the eight-state benchmark network of the source paper at
        # second order ---
        {"setup": setup_common + """
F, S = _named({(0, 1): 1.2, (1, 0): 2.1, (1, 2): 2.3, (2, 1): 3.2, (3, 4): 4.5,
               (4, 3): 5.4, (5, 6): 6.7, (6, 5): 7.6},
              {(2, 3): 3.4, (3, 2): 4.3, (4, 5): 5.6, (5, 4): 6.5,
               (6, 7): 1.9, (7, 6): 3.7}, 8)
Q = _group([[0, 1, 2], [3, 4], [5, 6], [7]], 8)
K0 = _cols([{0: 0.5045045045045045, 1: 0.2882882882882883, 2: 0.2072072072072072},
            {3: 0.5454545454545454, 4: 0.45454545454545453},
            {5: 0.5314685314685315, 6: 0.46853146853146854}, {7: 1.0}], 8)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 2))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 2))"},
        # --- Normal: configurations the fast events leave for good, so that the
        # projector carries fractional weights, at third order ---
        {"setup": setup_common + """
F, S = _named({(0, 1): 2.0, (0, 4): 0.5, (3, 2): 1.0, (3, 4): 3.0, (3, 5): 1.0,
               (1, 2): 4.0, (2, 1): 1.5, (5, 6): 2.2, (6, 5): 0.9},
              {(4, 5): 0.8, (5, 4): 1.2, (2, 0): 0.6, (6, 3): 0.9, (1, 4): 0.3,
               (4, 0): 0.4}, 7)
Q = _group([[1, 2], [4], [5, 6]], 7)
Q[:, 0] = [0.8, 0.2, 0.0]
Q[:, 3] = [0.2, 0.6, 0.2]
K0 = _cols([{1: 0.2727272727272727, 2: 0.7272727272727273}, {4: 1.0},
            {5: 0.2903225806451613, 6: 0.7096774193548387}], 7)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 3))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 3))"},
        # --- Normal: the same fractional projector with different slow events, at
        # second order ---
        {"setup": setup_common + """
F, S = _named({(0, 1): 2.0, (0, 4): 0.5, (3, 2): 1.0, (3, 4): 3.0, (3, 5): 1.0,
               (1, 2): 4.0, (2, 1): 1.5, (5, 6): 2.2, (6, 5): 0.9},
              {(2, 5): 0.5, (6, 1): 0.7, (4, 3): 0.6, (5, 0): 0.2, (1, 6): 0.4}, 7)
Q = _group([[1, 2], [4], [5, 6]], 7)
Q[:, 0] = [0.8, 0.2, 0.0]
Q[:, 3] = [0.2, 0.6, 0.2]
K0 = _cols([{1: 0.2727272727272727, 2: 0.7272727272727273}, {4: 1.0},
            {5: 0.2903225806451613, 6: 0.7096774193548387}], 7)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 2))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 2))"},
        # --- Normal: irreducible fast cycles at fourth order, where the
        # leading-order vector carries a current ---
        {"setup": setup_common + """
F, S = _named({(0, 1): 2.0, (1, 2): 5.0, (2, 0): 1.0, (3, 4): 3.0, (4, 5): 1.5, (5, 3): 4.0},
              {(2, 3): 0.8, (3, 2): 1.3, (1, 4): 0.6, (4, 1): 0.4, (0, 5): 0.9}, 6)
Q = _group([[0, 1, 2], [3, 4, 5]], 6)
K0 = _cols([{0: 0.29411764705882354, 1: 0.11764705882352941, 2: 0.5882352941176471},
            {3: 0.26666666666666666, 4: 0.5333333333333333, 5: 0.2}], 6)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 4))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 4))"},
        # --- Normal: the eight-state network again, both generators passed as
        # sparse matrices, at third order ---
        {"setup": setup_common + """
F, S = _named({(0, 1): 1.2, (1, 0): 2.1, (1, 2): 2.3, (2, 1): 3.2, (3, 4): 4.5,
               (4, 3): 5.4, (5, 6): 6.7, (6, 5): 7.6},
              {(2, 3): 3.4, (3, 2): 4.3, (4, 5): 5.6, (5, 4): 6.5,
               (6, 7): 1.9, (7, 6): 3.7}, 8)
F, S = scipy.sparse.csr_matrix(F), scipy.sparse.csc_matrix(S)
Q = _group([[0, 1, 2], [3, 4], [5, 6], [7]], 8)
K0 = _cols([{0: 0.5045045045045045, 1: 0.2882882882882883, 2: 0.2072072072072072},
            {3: 0.5454545454545454, 4: 0.45454545454545453},
            {5: 0.5314685314685315, 6: 0.46853146853146854}, {7: 1.0}], 8)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 3))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 3))"},
        # --- Boundary: the three-state network of the source paper closed at
        # first order only ---
        {"setup": setup_common + three,
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 1))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 1))"},
        # --- Boundary: fast links that all obey detailed balance, at third order ---
        {"setup": setup_common + """
F, Q, K0 = _circ([2, 3, 4, 6, 1], 303, balanced=True)
S = _rates(16, 304)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 3))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 3))"},
        # --- Boundary: a slow generator far weaker than the fast one, carried to
        # fifth order ---
        {"setup": setup_common + """
F, Q, K0 = _circ([3, 3, 2], 305)
S = _rates(8, 404, scale=1e-3)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 5))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 5))"},
        # --- Edge: a sixth-order closure ---
        {"setup": setup_common + """
F, Q, K0 = _circ([2, 3, 1, 3], 309)
S = _rates(9, 310, scale=0.3)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 6))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 6))"},
        # --- Edge: no fast events ---
        {"setup": setup_common + """
F = np.zeros((4, 4))
_, S = _named({}, {(0, 1): 0.3, (1, 0): 0.9, (1, 2): 1.1, (2, 3): 0.2, (3, 0): 0.7}, 4)
Q = np.eye(4); K0 = np.eye(4)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 2))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 2))"},
        # --- Edge: every fast link runs one way only, on shuffled indices ---
        {"setup": setup_common + """
F, Q, K0 = _circ([3, 5, 2, 4], 307, one_way=True)
S = _rates(14, 308)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 3))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 3))"},
        # --- Edge: one class holding the whole state space ---
        {"setup": setup_common + """
F, Q, K0 = _circ([6], 505)
S = _rates(6, 606)
""",
         "call": "_fp(hierarchy_closure(F, S, Q, K0, 3))",
         "gold_call": "_fp(_oracle_hierarchy_closure(F, S, Q, K0, 3))"},
        # --- Invalid: a projector that does not annihilate the fast generator ---
        {"setup": setup_common + three + "Qbad = _group([[0], [1, 2]], 3)\n"
         + _raises("F, S, Qbad, K0, 3"),
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a leading-order map that the fast generator does not
        # annihilate ---
        {"setup": setup_common + three + "K0bad = np.array([[0.5, 0.0], [0.5, 0.0], [0.0, 1.0]])\n"
         + _raises("F, S, Q, K0bad, 3"),
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: an order below one ---
        {"setup": setup_common + three + _raises("F, S, Q, K0, 0"),
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
