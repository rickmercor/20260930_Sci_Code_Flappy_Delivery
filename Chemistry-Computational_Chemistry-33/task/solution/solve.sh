#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
import scipy.sparse


def lattice_generators(nrow: int, ncol: int, k_ads: float, k_des: float,
                               k_rxn: float, k_rxn_cross: float, kappa: float,
                               drive_bias: float, kappa_cross: float,
                               interaction: float) -> tuple:
    import scipy.sparse as sp
    for name, v in (("nrow", nrow), ("ncol", ncol)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if v < 1:
            raise ValueError(name + " must be >= 1")
    nrow, ncol = int(nrow), int(ncol)
    nsites = nrow * ncol
    if nsites > 15:
        raise ValueError("nrow * ncol must not exceed 15")
    rates = [float(k_ads), float(k_des), float(k_rxn), float(k_rxn_cross),
             float(kappa), float(kappa_cross)]
    if any((not np.isfinite(k)) or k < 0.0 for k in rates):
        raise ValueError("rate constants must be finite and non-negative")
    k_ads, k_des, k_rxn, k_rxn_cross, kappa, kappa_cross = rates
    drive_bias = float(drive_bias)
    if not np.isfinite(drive_bias) or abs(drive_bias) > 1.0:
        raise ValueError("drive_bias must be finite and lie in [-1, 1]")
    interaction = float(interaction)
    if not np.isfinite(interaction):
        raise ValueError("interaction must be finite")
    along, across = set(), set()
    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            n = r * ncol + (c + 1) % ncol
            if n != s:
                along.add((min(s, n), max(s, n)))
            n = ((r + 1) % nrow) * ncol + c
            if n != s:
                across.add((min(s, n), max(s, n)))
    along, across = sorted(along), sorted(across)
    m = 1 << nsites
    conf = np.arange(m, dtype=np.int64)
    occ = ((conf[:, None] >> np.arange(nsites, dtype=np.int64)) & 1).astype(bool)
    pairs_along = np.zeros(m)
    for a, b in along:
        pairs_along += occ[:, a] & occ[:, b]
    pairs_across = np.zeros(m)
    for a, b in across:
        pairs_across += occ[:, a] & occ[:, b]
    pair_total = pairs_along + pairs_across
    fast, slow = ([], [], []), ([], [], [])

    def _add(store, mask, target, rate):
        src = np.flatnonzero(mask)
        store[0].append(target[src])
        store[1].append(src)
        store[2].append(np.broadcast_to(rate, (m,))[src])

    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            targets = {}
            down = r * ncol + (c + 1) % ncol
            up = r * ncol + (c - 1) % ncol
            if down != s:
                targets[down] = targets.get(down, 0.0) + kappa * (1.0 + drive_bias)
            if up != s:
                targets[up] = targets.get(up, 0.0) + kappa * (1.0 - drive_bias)
            for t, prefactor in targets.items():
                j = conf ^ (1 << s) ^ (1 << t)
                _add(fast, occ[:, s] & ~occ[:, t], j,
                    prefactor * np.exp(-0.5 * interaction * (pair_total[j] - pair_total)))
    for s in range(nsites):
        _add(slow, ~occ[:, s], conf | (1 << s), k_ads)
        _add(slow, occ[:, s], conf & ~(1 << s), k_des)
    for group, k_react in ((along, k_rxn), (across, k_rxn_cross)):
        if k_react > 0.0:
            for a, b in group:
                _add(slow, occ[:, a] & occ[:, b], conf & ~(1 << a) & ~(1 << b), k_react)
    for a, b in across:
        j = conf ^ (1 << a) ^ (1 << b)
        _add(slow, occ[:, a] != occ[:, b], j,
            kappa_cross * np.exp(-0.5 * interaction * (pair_total[j] - pair_total)))

    def _assemble(store):
        if store[0]:
            rows, cols, vals = (np.concatenate(x) for x in store)
        else:
            rows = cols = np.zeros(0, dtype=np.int64)
            vals = np.zeros(0)
        M = sp.csr_matrix((vals, (rows, cols)), shape=(m, m))
        out = np.asarray(M.sum(axis=0)).ravel()
        return (M - sp.diags(out, format="csr")).tocsr()

    F = _assemble(fast)
    S = _assemble(slow)
    event_rate = k_rxn * pairs_along + k_rxn_cross * pairs_across
    return F, S, event_rate

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


def fast_projector(F: "np.ndarray | scipy.sparse.spmatrix") -> "np.ndarray":
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


def driven_steady_state(F: "np.ndarray | scipy.sparse.spmatrix",
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


def hierarchy_closure(F: "np.ndarray | scipy.sparse.spmatrix",
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

import numpy as np


def window_averaged_state(L: "np.ndarray", ptilde0: "np.ndarray",
                                  t_start: float, t_end: float) -> "np.ndarray":
    from scipy.linalg import expm
    L = np.asarray(L, dtype=float)
    ptilde0 = np.asarray(ptilde0, dtype=float).reshape(-1)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square 2D array")
    n = ptilde0.size
    if n < 1 or L.shape[0] % n != 0 or L.shape[0] < 2 * n:
        raise ValueError("L must have shape ((K+1) n, (K+1) n) with K >= 1 and n = len(ptilde0)")
    if not (np.all(np.isfinite(L)) and np.all(np.isfinite(ptilde0))):
        raise ValueError("inputs must be finite")
    if np.any(ptilde0 < -1e-12) or abs(ptilde0.sum() - 1.0) > 1e-8:
        raise ValueError("ptilde0 must be a probability distribution")
    t_start, t_end = float(t_start), float(t_end)
    if not np.isfinite(t_start) or t_start < 0.0:
        raise ValueError("t_start must be a finite non-negative number")
    if not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("t_end must be finite and greater than t_start")
    size = L.shape[0]
    blocks = size // n
    # the last column of exp([[L, y0], [0, 0]] t) holds the integral of exp(L s) y0 over [0, t]
    aug = np.zeros((size + 1, size + 1))
    aug[:size, :size] = L
    aug[:n, size] = ptilde0
    ybar = (expm(aug * t_end)[:size, size] - expm(aug * t_start)[:size, size]) / (t_end - t_start)
    return ybar.reshape(blocks, n)

import numpy as np


def turnover_orders(event_rate: "np.ndarray", K0: "np.ndarray", resp: "np.ndarray",
                            coeffs: "np.ndarray") -> "np.ndarray":
    event_rate = np.asarray(event_rate, dtype=float).reshape(-1)
    K0 = np.asarray(K0, dtype=float)
    resp = np.asarray(resp, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    if K0.ndim != 2 or resp.ndim != 2 or coeffs.ndim != 2:
        raise ValueError("K0, resp and coeffs must be 2D arrays")
    m, n = K0.shape
    if event_rate.size != m or resp.shape[0] != m:
        raise ValueError("event_rate, K0 and resp must share the configuration count")
    if n < 1 or resp.shape[1] % n != 0 or resp.shape[1] < n:
        raise ValueError("resp must have shape (m, K n) with K >= 1")
    order = resp.shape[1] // n
    if coeffs.shape != (order + 1, n):
        raise ValueError("coeffs must have shape (K + 1, n)")
    for M in (event_rate, K0, resp, coeffs):
        if not np.all(np.isfinite(M)):
            raise ValueError("inputs must be finite")
    G = [K0] + [resp[:, k * n:(k + 1) * n] for k in range(order)]
    out = np.empty(order + 1)
    for k in range(order + 1):
        term = sum(G[k - l] @ coeffs[l] for l in range(k + 1))
        out[k] = float(event_rate @ term)
    return out

import numpy as np


def rate_rescaling_bias(nrow: int = 3, ncol: int = 5, k_ads: float = 8.5,
                                k_des: float = 4.0, k_rxn: float = 10.0,
                                k_rxn_cross: float = 4.0, kappa: float = 1.0,
                                drive_bias: float = 0.5, kappa_cross: float = 0.4,
                                interaction: float = 1.5, eps: float = 0.02,
                                t_start: float = 0.02, t_end: float = 0.08,
                                initial_config: int = 0, order: int = 3) -> float:
    for name, v in (("nrow", nrow), ("ncol", ncol), ("initial_config", initial_config),
                    ("order", order)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    nrow, ncol, initial_config, order = int(nrow), int(ncol), int(initial_config), int(order)
    if nrow < 1 or ncol < 1 or nrow * ncol > 15:
        raise ValueError("lattice must have between 1 and 15 sites")
    if initial_config < 0 or initial_config >= (1 << (nrow * ncol)):
        raise ValueError("initial_config must index a configuration of the lattice")
    if order < 1:
        raise ValueError("order must be >= 1")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite number")
    t_start, t_end = float(t_start), float(t_end)
    if not np.isfinite(t_start) or t_start < 0.0:
        raise ValueError("t_start must be a finite non-negative number")
    if not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("t_end must be finite and greater than t_start")
    F, S, event_rate = lattice_generators(
        nrow, ncol, k_ads, k_des, k_rxn, k_rxn_cross, kappa, drive_bias,
        kappa_cross, interaction)
    event_rate = np.asarray(event_rate, dtype=float)
    Q = np.asarray(fast_projector(F), dtype=float)
    K0 = driven_steady_state(F, Q)
    resp, L = hierarchy_closure(F, S, Q, K0, order)
    ptilde0 = Q[:, initial_config].copy()
    coeffs = window_averaged_state(L, ptilde0, t_start, t_end)
    orders = np.asarray(turnover_orders(event_rate, K0, resp, coeffs), dtype=float)
    if orders[0] == 0.0:
        raise ValueError("the quasi-equilibrium rate vanishes, so the bias is undefined")
    correction = sum(eps ** k * orders[k] for k in range(1, order + 1))
    return float(correction / orders[0])
SCICODE_GOLD_EOF
