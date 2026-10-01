"""
Collecting the terms of order 1/eps in the expansion p = p^(0) + eps p^(1) + eps^2 p^(2) + ... of the master equation dp/dt = (F / eps + S) p leaves an algebraic condition on the leading term alone: the fast generator annihilates it, F p^(0) = 0.

The leading term is carried by a conditional probability matrix, p^(0) = K^(0) p_tilde^(0), with one column for every row of the projector Q built in the previous step. Column r is supported on the r-th recurrent class, where it is non-negative and sums to one, and it is zero on every other configuration. The two properties

    F K^(0) = 0,   Q K^(0) = I

fix it uniquely, whatever fractional entries Q carries on the configurations that belong to no recurrent class.

Nothing about the shape of that distribution may be assumed. It is whatever the supplied fast generator makes it, and the fast generator is the only place it can come from: the same set of configurations, given a different fast process on it, carries a different leading-order distribution even though the interaction energies of its configurations are unchanged.

F may be a scipy.sparse matrix as large as the lattice's 32768 configurations, and the leading-order map is needed for every class at once, so the construction may not form any dense array with m x m entries. It is returned as a dense array of shape (m, n).

Collecting the terms of order 1/eps in the expansion p = p^(0) + eps p^(1) + eps^2 p^(2) + ... of the master equation dp/dt = (F / eps + S) p leaves an algebraic condition on the leading term alone: the fast generator annihilates it, F p^(0) = 0.

Returns
-------
np.ndarray of float with shape (m, n): the leading-order conditional probabilities, column r the normalised stationary distribution of the fast process on the r-th recurrent class and zero on every other configuration, satisfying F @ K0 = 0 and Q @ K0 = I.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def driven_steady_state(F: "np.ndarray | scipy.sparse.spmatrix", Q: "np.ndarray") -> "np.ndarray": """Parameters: F is a fast generator and Q its projector. Returns: K0, the stationary within-class map satisfying F @ K0 = 0 and Q @ K0 = I. Raises: ValueError if F is not a non-empty square 2D array, holds a non-finite entry, has a negative off-diagonal rate or a column that does not sum to zero; if Q does not have shape (n, m) with n >= 1, holds a non-finite entry, is negative anywhere, or has a column that does not sum to one; if Q does not annihilate F; or if a row of Q does not carry exactly one stationary state of F."""; return K0

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


def _stationary(block):
    """Normalised null vector of an irreducible generator block."""
    from scipy.sparse.linalg import splu
    size = block.shape[0]
    if size == 1:
        return np.ones(1)
    rhs = np.zeros(size)
    rhs[-1] = 1.0
    if size <= 2000:
        A = block.toarray()
        A[-1, :] = 1.0
        return np.linalg.solve(A, rhs)
    A = block.tolil()
    A[size - 1, :] = np.ones(size)
    return splu(A.tocsc()).solve(rhs)


def _oracle_driven_steady_state(F: "np.ndarray | scipy.sparse.spmatrix",
                                Q: "np.ndarray") -> "np.ndarray":
    F = _as_generator(F, "F")
    Q = _check_projector(F, Q)
    n, m = Q.shape
    cls = _recurrent_classes(F)
    K0 = np.zeros((m, n))
    for r in range(n):
        inside = np.unique(cls[(Q[r] > 1e-12) & (cls >= 0)])
        if inside.size != 1:
            raise ValueError("each row of Q must carry exactly one stationary state of F")
        idx = np.flatnonzero(cls == inside[0])
        K0[idx, r] = _stationary(F[idx][:, idx])
    return K0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup_common = """import numpy as np
import scipy.sparse
def _classes(sizes, seed, one_way=False, symmetric=False):
    g = np.random.default_rng(seed)
    m = int(sum(sizes))
    perm = g.permutation(m)
    F = np.zeros((m, m))
    groups = []
    off = 0
    for sz in sizes:
        idx = perm[off:off + sz]
        off += sz
        groups.append(sorted(int(i) for i in idx))
        if sz == 1:
            continue
        for a in range(sz):
            i, j = idx[a], idx[(a + 1) % sz]
            if symmetric:
                k = g.uniform(0.3, 3.0)
                F[j, i] += k
                F[i, j] += k
            else:
                F[j, i] += g.uniform(0.5, 3.0)
                if not one_way:
                    F[i, j] += g.uniform(0.1, 1.0)
    F[np.diag_indices(m)] = 0.0
    F[np.diag_indices(m)] = -F.sum(axis=0)
    groups.sort(key=min)
    Q = np.zeros((len(groups), m))
    for r, grp in enumerate(groups):
        Q[r, grp] = 1.0
    return F, Q
def _gen(rates, m):
    F = np.zeros((m, m))
    for (i, j), k in rates.items():
        F[j, i] += k
    F[np.diag_indices(m)] = -F.sum(axis=0)
    return F
def _group(groups, m):
    Q = np.zeros((len(groups), m))
    for b, grp in enumerate(groups):
        for s in grp:
            Q[b, s] = 1.0
    return Q
def _big(sizes, seed):
    g = np.random.default_rng(seed)
    sizes = np.asarray(sizes)
    m = int(sizes.sum())
    perm = g.permutation(m)
    label = np.empty(m, dtype=int)
    rows, cols, vals = [], [], []
    def link(src, dst, lo, hi):
        cols.append(src)
        rows.append(dst)
        vals.append(g.uniform(lo, hi, size=len(src)))
    off = 0
    for k, sz in enumerate(sizes):
        idx = perm[off:off + sz]
        off += sz
        label[idx] = k
        if sz == 1:
            continue
        nxt = np.roll(idx, -1)
        link(idx, nxt, 0.5, 3.0)
        link(nxt, idx, 0.1, 1.0)
        a = g.choice(idx, size=sz // 3)
        b = g.choice(idx, size=sz // 3)
        link(a[a != b], b[a != b], 0.2, 1.5)
    F = scipy.sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(m, m))
    F = (F - scipy.sparse.diags(np.asarray(F.sum(axis=0)).ravel())).tocsr()
    first = np.full(len(sizes), m)
    np.minimum.at(first, label, np.arange(m))
    rank = np.argsort(np.argsort(first))
    Q = np.zeros((len(sizes), m))
    Q[rank[label], np.arange(m)] = 1.0
    return F, Q
def _fp(M):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        return float("nan")
    a = np.cos(0.37 * np.arange(M.shape[0]) + 0.2)
    b = np.sin(0.71 * np.arange(M.shape[1]) + 0.3)
    c = np.cos(1.29 * np.arange(M.shape[1]) + 0.7)
    return float(a @ M @ b + 0.5 * (a * a) @ M @ c + 1e-3 * M.shape[0])
"""
    raises = """
def run_model():
    try:
        driven_steady_state(F, Qbad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_driven_steady_state(F, Qbad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- Normal: about 38000 configurations held as a sparse matrix, in 140
        # driven classes of up to five hundred configurations and one class of
        # four thousand ---
        {"setup": setup_common + """
sizes = np.random.default_rng(3).integers(1, 501, size=140).tolist() + [4000]
F, Q = _big(sizes, 4)
""",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Normal: sixty configurations on shuffled indices in ten classes of
        # mixed size, each carrying a driven current ---
        {"setup": setup_common + "F, Q = _classes([1, 2, 3, 3, 4, 5, 6, 8, 9, 19], 5)\n",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Normal: the same classes with every link symmetric, so detailed
        # balance holds inside each ---
        {"setup": setup_common + "F, Q = _classes([1, 2, 3, 3, 4, 5, 6, 8, 9, 19], 5, symmetric=True)\n",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Normal: two configurations the fast events leave for good, sharing
        # their weight among a two-state class, a one-configuration class and a
        # second two-state class ---
        {"setup": setup_common + """
F = _gen({(0, 1): 2.0, (0, 4): 0.5, (3, 2): 1.0, (3, 4): 3.0, (3, 5): 1.0,
          (1, 2): 4.0, (2, 1): 1.5, (5, 6): 2.2, (6, 5): 0.9}, 7)
Q = _group([[1, 2], [4], [5, 6]], 7)
Q[:, 0] = [0.8, 0.2, 0.0]
Q[:, 3] = [0.2, 0.6, 0.2]
""",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Normal: an irreducible three-cycle, which sustains a current with no
        # energy landscape behind it ---
        {"setup": setup_common + """
F = _gen({(0, 1): 2.0, (1, 2): 5.0, (2, 0): 1.0, (3, 4): 3.0, (4, 5): 1.5, (5, 3): 4.0}, 6)
Q = _group([[0, 1, 2], [3, 4, 5]], 6)
""",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Normal: the two driven three-cycles again, passed as a sparse matrix ---
        {"setup": setup_common + """
F = scipy.sparse.csr_matrix(_gen({(0, 1): 2.0, (1, 2): 5.0, (2, 0): 1.0, (3, 4): 3.0,
                                  (4, 5): 1.5, (5, 3): 4.0}, 6))
Q = _group([[0, 1, 2], [3, 4, 5]], 6)
""",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Boundary: a class that two configurations drain into ---
        {"setup": setup_common + """
F = _gen({(0, 1): 3.0, (1, 2): 1.7, (0, 2): 0.4}, 4)
Q = _group([[0, 1, 2], [3]], 4)
""",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Boundary: a configuration left for good whose only exit is one
        # class, beside a class it cannot reach ---
        {"setup": setup_common + """
F = _gen({(0, 2): 1.4, (2, 3): 2.0, (3, 4): 0.5, (4, 2): 1.7, (3, 2): 0.8,
          (1, 5): 0.9, (5, 1): 0.9}, 6)
Q = _group([[1, 5], [0, 2, 3, 4]], 6)
""",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Edge: no fast events ---
        {"setup": setup_common + "F = np.zeros((4, 4))\nQ = np.eye(4)\n",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Edge: every fast link runs one way only, on shuffled indices ---
        {"setup": setup_common + "F, Q = _classes([3, 3, 4, 2, 5, 1], 8, one_way=True)\n",
         "call": "_fp(driven_steady_state(F, Q))",
         "gold_call": "_fp(_oracle_driven_steady_state(F, Q))"},
        # --- Invalid: a projector that does not annihilate the fast generator ---
        {"setup": setup_common + """
F = _gen({(0, 1): 2.3, (1, 0): 1.1}, 3)
Qbad = _group([[0], [1, 2]], 3)
""" + raises,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a projector whose columns do not each sum to one ---
        {"setup": setup_common + """
F = _gen({(0, 1): 2.3, (1, 0): 1.1}, 3)
Qbad = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
""" + raises,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
