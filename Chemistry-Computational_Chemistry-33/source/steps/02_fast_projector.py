"""
When the generator of a master equation splits as W = F / eps + S with a small eps, the fast part F acts on a far shorter timescale than S, and the reduction replaces the full probability vector p by a smaller vector of slow variables, p_tilde = Q p. The projector Q must be chosen so that the fast events cannot change p_tilde, Q F = 0: its rows span the left null space of F. That identity is what makes the singular-perturbation hierarchy of the later steps closable, because it removes the unknown higher-order term from every projected equation.

The left null space has one dimension for every recurrent class of the fast process, that is, every set of configurations that the fast events keep circulating among and never leave. When every configuration belongs to such a class, the fast events split the configurations into disjoint superbasins and Q is the 0/1 matrix that assigns each configuration to its own. A fast network can also contain configurations that the fast events leave for good and never re-enter. Such a configuration belongs to no single class, and its weight in Q is shared among the classes the fast process can carry it to.

A basis of the left null space is not unique, so it is returned in one canonical form. Every entry is non-negative, every column sums to one, and row r equals one on every configuration of the r-th recurrent class and zero on every configuration of every other recurrent class. Rows are ordered by the smallest configuration index belonging to their recurrent class, so that the reduced coordinates are canonical and the later steps can be chained without a relabelling. An off-diagonal entry of F counts as a fast transition when it exceeds 1e-12 times the larger of 1 and the largest absolute entry of F.

The fast generator can be large. On the task's lattice it has 32768 configurations, and a dense array with that many rows and columns would take 8.6 GB, so F may be given either as a dense np.ndarray or as a scipy.sparse matrix, and the construction has to work from its nonzero entries: it may not form any dense array with m x m entries. The projector itself is returned as a dense array, since it has only one row per recurrent class.

When the generator of a master equation splits as W = F / eps + S with a small eps, the fast part F acts on a far shorter timescale than S, and the reduction replaces the full probability vector p by a smaller vector of slow variables, p_tilde = Q p. The projector Q must be chosen so that the fast events cannot change p_tilde, Q F = 0: its rows span the left null space of F. That identity is what makes the singular-perturbation hierarchy of the later steps closable, because it removes the unknown higher-order term from every projected equation.

Returns
-------
np.ndarray of float with shape (n, m): the canonical basis of the left null space of F, one row per recurrent class of the fast process, non-negative with every column summing to one, ordered by the smallest configuration index of each class.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fast_projector(F: "np.ndarray | scipy.sparse.spmatrix") -> "np.ndarray": """Parameters: F is a square fast generator in the column convention. Returns: Q, the canonical recurrent-class projector satisfying Q @ F = 0. Raises: ValueError if F is not a non-empty square 2D array, holds a non-finite entry, has a negative off-diagonal rate, or has a column that does not sum to zero."""; return Q

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


def _recurrent_classes(F):
    """Recurrent classes of the fast process, ordered by smallest member."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    m = F.shape[0]
    scale = max(1.0, float(abs(F).max()))
    coo = F.tocoo()
    keep = (coo.row != coo.col) & (coo.data > 1e-12 * scale)
    src, dst = coo.col[keep], coo.row[keep]
    graph = sp.csr_matrix((np.ones(src.size), (src, dst)), shape=(m, m))
    ncomp, lab = connected_components(graph, directed=True, connection="strong")
    leaves = np.zeros(ncomp, dtype=bool)
    leaves[lab[src[lab[src] != lab[dst]]]] = True
    first = np.full(ncomp, m, dtype=np.int64)
    np.minimum.at(first, lab, np.arange(m))
    closed = [c for c in np.argsort(first) if not leaves[c]]
    order = np.full(ncomp, -1, dtype=np.int64)
    order[closed] = np.arange(len(closed))
    return order[lab]


def _oracle_fast_projector(F: "np.ndarray | scipy.sparse.spmatrix") -> "np.ndarray":
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    F = _as_generator(F, "F")
    m = F.shape[0]
    cls = _recurrent_classes(F)
    n = int(cls.max()) + 1
    R = np.flatnonzero(cls >= 0)
    T = np.flatnonzero(cls < 0)
    Q = np.zeros((n, m))
    Q[cls[R], R] = 1.0
    if T.size:
        member = sp.csr_matrix((np.ones(R.size), (cls[R], R)), shape=(n, m))
        into = np.asarray((member @ F[:, T]).todense())
        FTT = F[T][:, T].T.tocsc()
        Q[:, T] = splu(FTT).solve(np.ascontiguousarray(-into.T)).T
    return Q

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup_common = """import numpy as np
import scipy.sparse
def _gen(rates, m):
    F = np.zeros((m, m))
    for (i, j), k in rates.items():
        F[j, i] += k
    F[np.diag_indices(m)] = -F.sum(axis=0)
    return F
def _network(sizes, n_left, seed, one_way=False):
    g = np.random.default_rng(seed)
    m = int(sum(sizes)) + n_left
    perm = g.permutation(m)
    F = np.zeros((m, m))
    off = 0
    for sz in sizes:
        idx = perm[off:off + sz]
        off += sz
        if sz == 1:
            continue
        for a in range(sz):
            F[idx[(a + 1) % sz], idx[a]] += g.uniform(0.5, 3.0)
            if not one_way:
                F[idx[a], idx[(a + 1) % sz]] += g.uniform(0.1, 1.0)
    held, left = perm[:off], perm[off:]
    for t, s in enumerate(left):
        for j in g.choice(held, size=2, replace=False):
            F[j, s] += g.uniform(0.2, 2.0)
        if t + 1 < n_left:
            F[left[t + 1], s] += g.uniform(0.2, 2.0)
            if t % 2 == 0:
                F[s, left[t + 1]] += g.uniform(0.2, 2.0)
    F[np.diag_indices(m)] = 0.0
    F[np.diag_indices(m)] = -F.sum(axis=0)
    return F
def _big(n_classes, max_size, n_left, seed):
    g = np.random.default_rng(seed)
    sizes = g.integers(1, max_size + 1, size=n_classes)
    m = int(sizes.sum()) + n_left
    perm = g.permutation(m)
    rows, cols, vals = [], [], []
    def link(src, dst, lo, hi):
        cols.append(src)
        rows.append(dst)
        vals.append(g.uniform(lo, hi, size=len(src)))
    off = 0
    for sz in sizes:
        idx = perm[off:off + sz]
        off += sz
        if sz == 1:
            continue
        nxt = np.roll(idx, -1)
        link(idx, nxt, 0.5, 3.0)
        link(nxt, idx, 0.1, 1.0)
        a = g.choice(idx, size=sz // 3)
        b = g.choice(idx, size=sz // 3)
        link(a[a != b], b[a != b], 0.2, 1.5)
    held, left = perm[:off], perm[off:]
    link(left, g.choice(held, size=n_left), 0.2, 2.0)
    link(left, g.choice(held, size=n_left), 0.2, 2.0)
    link(left[:-1], left[1:], 0.2, 2.0)
    even = np.arange(0, n_left - 1, 2)
    link(left[even + 1], left[even], 0.2, 2.0)
    F = scipy.sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(m, m))
    return (F - scipy.sparse.diags(np.asarray(F.sum(axis=0)).ravel())).tocsr()
def _fp(M):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        return float("nan")
    a = np.cos(0.37 * np.arange(M.shape[0]) + 0.2)
    b = np.sin(0.71 * np.arange(M.shape[1]) + 0.3)
    c = np.cos(1.29 * np.arange(M.shape[1]) + 0.7)
    return float(a @ M @ b + 0.5 * (a * a) @ M @ c + 1e-3 * M.shape[0] + 1e-2 * M.shape[1])
"""
    raises = """
def run_model():
    try:
        fast_projector(F)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_fast_projector(F)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- Normal: about 34000 configurations held as a sparse matrix, two
        # hundred recurrent classes of up to three hundred configurations each,
        # and four thousand configurations the fast events leave for good, in
        # chains that feed each other and share their weight among classes ---
        {"setup": setup_common + "F = _big(200, 300, 4000, 11)\n",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Normal: sixty configurations on shuffled indices, ten recurrent
        # classes of mixed size carrying driven currents, and six configurations
        # the fast events leave for good, some of them feeding each other ---
        {"setup": setup_common + "F = _network([1, 2, 3, 3, 4, 5, 6, 8, 9, 13], 6, 7)\n",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Normal: the eight-state benchmark network of the source paper, fast
        # transitions only ---
        {"setup": setup_common + """
F = _gen({(0, 1): 1.2, (1, 0): 2.1, (1, 2): 2.3, (2, 1): 3.2, (3, 4): 4.5,
          (4, 3): 5.4, (5, 6): 6.7, (6, 5): 7.6}, 8)
""",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Normal: the eight-state network again, passed as a sparse matrix ---
        {"setup": setup_common + """
F = scipy.sparse.csr_matrix(_gen({(0, 1): 1.2, (1, 0): 2.1, (1, 2): 2.3, (2, 1): 3.2,
                                  (3, 4): 4.5, (4, 3): 5.4, (5, 6): 6.7, (6, 5): 7.6}, 8))
""",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Normal: one configuration the fast events leave for good, which can
        # end up in either of two configurations that the fast events never
        # leave, beside a driven three-cycle ---
        {"setup": setup_common + """
F = _gen({(0, 1): 2.0, (0, 2): 0.5, (3, 4): 3.0, (4, 5): 1.5, (5, 3): 4.0}, 6)
""",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Normal: two configurations left for good that feed each other before
        # reaching a two-state class or a one-configuration class ---
        {"setup": setup_common + """
F = _gen({(0, 1): 2.0, (0, 3): 0.5, (0, 4): 1.0, (4, 0): 3.0, (4, 2): 0.7,
          (1, 2): 4.0, (2, 1): 1.5, (5, 6): 2.2, (6, 5): 0.9}, 8)
""",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Normal: a chain of three configurations left for good, each of which
        # can also leave the chain for a different class ---
        {"setup": setup_common + """
F = _gen({(0, 1): 1.5, (0, 5): 0.4, (1, 2): 0.9, (1, 7): 1.1, (2, 3): 2.2,
          (2, 8): 0.3, (3, 4): 1.0, (4, 3): 2.0, (5, 6): 0.7, (6, 5): 1.3}, 9)
""",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Normal: three configurations left for good that circulate among
        # themselves before escaping to one of two classes ---
        {"setup": setup_common + """
F = _gen({(4, 5): 2.0, (5, 6): 1.5, (6, 4): 1.0, (5, 4): 0.5, (4, 0): 0.3,
          (6, 2): 0.8, (0, 1): 1.2, (1, 0): 0.6, (2, 3): 3.0, (3, 2): 2.5}, 7)
""",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Boundary: no fast transitions at all ---
        {"setup": setup_common + "F = np.zeros((5, 5))\n",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Boundary: configurations left for good that can reach only one class
        # between them ---
        {"setup": setup_common + """
F = _gen({(0, 1): 1.3, (1, 2): 0.8, (2, 3): 2.5, (3, 2): 1.1, (4, 5): 0.6,
          (5, 4): 0.6}, 6)
""",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Edge: the lowest-indexed configurations are ones the fast events
        # leave for good ---
        {"setup": setup_common + """
F = _gen({(0, 5): 1.0, (0, 3): 3.0, (1, 0): 0.4, (1, 6): 2.0, (3, 4): 1.7,
          (4, 3): 0.9, (5, 6): 5.0, (6, 5): 0.5, (2, 7): 1.0, (7, 2): 1.0}, 8)
""",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Edge: every fast link runs one way only, on shuffled indices ---
        {"setup": setup_common + "F = _network([3, 3, 4, 2, 5, 1], 0, 8, one_way=True)\n",
         "call": "_fp(fast_projector(F))",
         "gold_call": "_fp(_oracle_fast_projector(F))"},
        # --- Invalid: a negative off-diagonal rate must raise ValueError ---
        {"setup": setup_common + """
F = _gen({(0, 1): 1.0, (1, 0): 1.0}, 2)
F[0, 1] = -0.5
F[np.diag_indices(2)] = 0.0
F[np.diag_indices(2)] = -F.sum(axis=0)
""" + raises,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a column that does not sum to zero must raise ValueError ---
        {"setup": setup_common + """
F = _gen({(0, 1): 1.0, (1, 2): 2.0}, 3)
F[2, 2] = -0.5
""" + raises,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a non-square array must raise ValueError ---
        {"setup": setup_common + "F = np.zeros((2, 3))\n" + raises,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
