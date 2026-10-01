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
from numpy.linalg import det, slogdet


def contact_laplacian(coords: "np.ndarray", r_c: float, kT: float) -> "np.ndarray":
    """Eq. (1): w_ij = exp(-d_ij/kT) for every C-alpha pair with d_ij <= r_c (backbone on the same footing)."""
    X = np.asarray(coords, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 3 or not np.all(np.isfinite(X)):
        raise ValueError("coords must be a finite (N, 3) array with N >= 3")
    r_c, kT = float(r_c), float(kT)
    if not (np.isfinite(r_c) and r_c > 0.0 and np.isfinite(kT) and kT > 0.0):
        raise ValueError("r_c and kT must be finite positive numbers")
    n = X.shape[0]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    W = np.where((D <= r_c) & ~np.eye(n, dtype=bool), np.exp(-D / kT), 0.0)
    L = np.diag(W.sum(axis=1)) - W
    return L

import numpy as np
from numpy.linalg import det, slogdet


def _check_positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + " must be a finite positive number")
    return x


def _check_laplacian(L):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite square matrix of size >= 3")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):
        raise ValueError("L must be a symmetric Laplacian with zero row sums")
    return L


def _edges_from_laplacian(L):
    """Lexicographic (i < j) edge list, lengths and weights recovered from the weighted Laplacian."""
    n = L.shape[0]
    ii, jj = np.where(np.triu(L, 1) < 0.0)
    edges = list(zip(ii.tolist(), jj.tolist()))
    if not edges:
        raise ValueError("L has no edges")
    w = -L[ii, jj]
    return edges, w


def _transfer_current(L, edges, w):
    """Symmetric transfer-current matrix K_ab = sqrt(w_a w_b) Y(e_a, e_b), eqs. (8)-(9). The edge vectors chi_e are
    orthogonal to the constant null vector of L, so any generalised inverse gives the same Y as the Moore-Penrose
    pseudoinverse; the grounded inverse of the reduced Laplacian is used because it is well conditioned and exact."""
    n = L.shape[0]
    Kp = np.zeros((n, n))
    Kp[1:, 1:] = np.linalg.inv(L[1:, 1:])
    chi = np.zeros((len(edges), n))
    for a, (i, j) in enumerate(edges):
        chi[a, i] = 1.0
        chi[a, j] = -1.0
    Y = chi @ Kp @ chi.T
    return Y * np.sqrt(np.outer(w, w))


def global_tree_thermodynamics(L: "np.ndarray", kT: float) -> "np.ndarray":
    """Eqs. (3)-(6) and 2.1.7: ln Z from the reduced Laplacian; <E> and Var(E) over the spanning-tree ensemble
    through the marginal inclusion P(e in T) = w_e R_e and the transfer-current covariances Cov(1_e, 1_f) = -K_ef^2."""
    L = _check_laplacian(L)
    kT = _check_positive(kT, "kT")
    edges, w = _edges_from_laplacian(L)
    sign, lnZ = slogdet(L[1:, 1:])
    if sign <= 0:
        raise ValueError("the contact graph is disconnected")
    d = -kT * np.log(w)                                   # edge lengths recovered from eq. (1)
    K = _transfer_current(L, edges, w)
    p = np.diag(K)
    cov = -K ** 2
    np.fill_diagonal(cov, p * (1.0 - p))
    U = float(d @ p)
    varE = float(d @ cov @ d)
    F = -kT * lnZ
    S = (U - F) / kT
    C = varE / kT ** 2
    return np.array([lnZ, F, U, S, C], dtype=np.float64)

import numpy as np
from numpy.linalg import det, slogdet


def _edges_from_laplacian(L):
    """Lexicographic (i < j) edge list, lengths and weights recovered from the weighted Laplacian."""
    n = L.shape[0]
    ii, jj = np.where(np.triu(L, 1) < 0.0)
    edges = list(zip(ii.tolist(), jj.tolist()))
    if not edges:
        raise ValueError("L has no edges")
    w = -L[ii, jj]
    return edges, w


def _transfer_current(L, edges, w):
    """Symmetric transfer-current matrix K_ab = sqrt(w_a w_b) Y(e_a, e_b), eqs. (8)-(9). The edge vectors chi_e are
    orthogonal to the constant null vector of L, so any generalised inverse gives the same Y as the Moore-Penrose
    pseudoinverse; the grounded inverse of the reduced Laplacian is used because it is well conditioned and exact."""
    n = L.shape[0]
    Kp = np.zeros((n, n))
    Kp[1:, 1:] = np.linalg.inv(L[1:, 1:])
    chi = np.zeros((len(edges), n))
    for a, (i, j) in enumerate(edges):
        chi[a, i] = 1.0
        chi[a, j] = -1.0
    Y = chi @ Kp @ chi.T
    return Y * np.sqrt(np.outer(w, w))


def edge_transfer_current(L: "np.ndarray", kT: float) -> "np.ndarray":
    """Eqs. (7)-(9): E x E matrix K_ab = Y(e_a, e_b) sqrt(w_a w_b), edges in lexicographic (i < j) order."""
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite square matrix of size >= 3")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):
        raise ValueError("L must be a symmetric Laplacian with zero row sums")
    if not (np.isfinite(float(kT)) and float(kT) > 0.0):
        raise ValueError("kT must be a finite positive number")
    edges, w = _edges_from_laplacian(L)
    return _transfer_current(L, edges, w)

import numpy as np
from numpy.linalg import det, slogdet


def _check_coords(coords):
    X = np.asarray(coords, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 3 or not np.all(np.isfinite(X)):
        raise ValueError("coords must be a finite (N, 3) array with N >= 3")
    return X


def _check_positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + " must be a finite positive number")
    return x


def _check_laplacian(L):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite square matrix of size >= 3")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):
        raise ValueError("L must be a symmetric Laplacian with zero row sums")
    return L


def _edges_from_laplacian(L):
    """Lexicographic (i < j) edge list, lengths and weights recovered from the weighted Laplacian."""
    n = L.shape[0]
    ii, jj = np.where(np.triu(L, 1) < 0.0)
    edges = list(zip(ii.tolist(), jj.tolist()))
    if not edges:
        raise ValueError("L has no edges")
    w = -L[ii, jj]
    return edges, w


def _transfer_current(L, edges, w):
    """Symmetric transfer-current matrix K_ab = sqrt(w_a w_b) Y(e_a, e_b), eqs. (8)-(9). The edge vectors chi_e are
    orthogonal to the constant null vector of L, so any generalised inverse gives the same Y as the Moore-Penrose
    pseudoinverse; the grounded inverse of the reduced Laplacian is used because it is well conditioned and exact."""
    n = L.shape[0]
    Kp = np.zeros((n, n))
    Kp[1:, 1:] = np.linalg.inv(L[1:, 1:])
    chi = np.zeros((len(edges), n))
    for a, (i, j) in enumerate(edges):
        chi[a, i] = 1.0
        chi[a, j] = -1.0
    Y = chi @ Kp @ chi.T
    return Y * np.sqrt(np.outer(w, w))


def _tree_energy_cumulants(X, r_c, kT):
    """Exact (Var(E), kappa_3) of the tree energy at temperature kT from the determinantal edge-indicator structure:
    Var = d^T Cov d with Cov = -K*K off the diagonal and p(1-p) on it; kappa_3 = sum d^3 p - 3 sum d_e^2 d_f K_ef^2
    + 2 sum d_e d_f d_g K_ef K_fg K_ge (= -dVar/dbeta)."""
    L = contact_laplacian(X, r_c, kT)
    edges, w = _edges_from_laplacian(L)
    d = -kT * np.log(w)
    K = _transfer_current(L, edges, w)
    p = np.diag(K)
    K2 = K ** 2
    cov = -K2
    np.fill_diagonal(cov, p * (1.0 - p))
    var = float(d @ cov @ d)
    k3 = float(np.sum(d ** 3 * p) - 3.0 * np.sum((d ** 2)[:, None] * d[None, :] * K2)
               + 2.0 * (d @ (K * (K @ np.diag(d) @ K)) @ d))
    return var, k3


def _peak_residual(X, r_c, kT):
    """g(kT) = kappa_3 - 2 kT Var(E): positive below the heat-capacity peak, negative above it (dC/dbeta = beta^2 g / kT)."""
    var, k3 = _tree_energy_cumulants(X, r_c, kT)
    return k3 - 2.0 * kT * var


def heat_capacity_peak_temperature(coords: "np.ndarray", r_c: float, kT_lo: float, kT_hi: float) -> float:
    """The effective temperature kT* in (kT_lo, kT_hi) at which the global spanning-tree heat capacity C(kT) = Var(E)/(kT)^2
    is maximal, to 1e-12. Stationarity of C in beta = 1/kT gives 2 beta Var(E) - beta^2 kappa_3 = 0, i.e. kappa_3 = 2 kT Var(E),
    with kappa_3 = -dVar/dbeta the third cumulant of the tree energy, evaluated exactly (see _tree_energy_cumulants). The root
    of g(kT) = kappa_3 - 2 kT Var(E) is bracketed by its sign change and polished by safeguarded Newton steps."""
    X = _check_coords(coords)
    r_c = _check_positive(r_c, "r_c")
    lo, hi = float(kT_lo), float(kT_hi)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("kT_lo and kT_hi must be finite with 0 < kT_lo < kT_hi")
    g_lo, g_hi = _peak_residual(X, r_c, lo), _peak_residual(X, r_c, hi)
    if not (g_lo > 0.0 > g_hi):
        raise ValueError("the heat capacity has no interior maximum bracketed by [kT_lo, kT_hi]")
    a, b = lo, hi
    x = 0.5 * (a + b)
    for _ in range(200):
        gx = _peak_residual(X, r_c, x)
        if gx > 0.0:
            a = x
        else:
            b = x
        h = 1e-6 * x
        dg = (_peak_residual(X, r_c, x + h) - _peak_residual(X, r_c, x - h)) / (2.0 * h)
        x_new = x - gx / dg if dg != 0.0 else 0.5 * (a + b)
        if not (a < x_new < b):
            x_new = 0.5 * (a + b)
        if abs(x_new - x) < 1e-13 or (b - a) < 1e-13:
            x = x_new
            break
        x = x_new
    return float(x)

import numpy as np
from numpy.linalg import det, slogdet


def _check_laplacian(L):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite square matrix of size >= 3")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):
        raise ValueError("L must be a symmetric Laplacian with zero row sums")
    return L


def channel_paths(L: "np.ndarray", s: int, t: int, max_nodes: int) -> "np.ndarray":
    """All simple paths from s to t with 2..max_nodes nodes, as a lexicographically sorted int array padded with -1."""
    L = _check_laplacian(L)
    n = L.shape[0]
    for name, v in (("s", s), ("t", t), ("max_nodes", max_nodes)):
        if isinstance(v, bool) or int(v) != v:
            raise ValueError(name + " must be an integer")
    s, t, max_nodes = int(s), int(t), int(max_nodes)
    if not (0 <= s < n and 0 <= t < n) or s == t or max_nodes < 2:
        raise ValueError("s and t must be distinct residue indices and max_nodes >= 2")
    adj = [np.where(L[v] < 0.0)[0].tolist() for v in range(n)]
    out = []
    stack = [(s, (s,))]
    while stack:
        v, path = stack.pop()
        if v == t:
            out.append(path)
            continue
        if len(path) >= max_nodes:
            continue
        for u in adj[v]:
            if u not in path:
                stack.append((u, path + (u,)))
    out.sort()
    P = -np.ones((len(out), max_nodes), dtype=np.int64)
    for r, path in enumerate(out):
        P[r, :len(path)] = path
    return P

import numpy as np
from numpy.linalg import det, slogdet


def _check_positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + " must be a finite positive number")
    return x


def _check_laplacian(L):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite square matrix of size >= 3")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):
        raise ValueError("L must be a symmetric Laplacian with zero row sums")
    return L


def _edges_from_laplacian(L):
    """Lexicographic (i < j) edge list, lengths and weights recovered from the weighted Laplacian."""
    n = L.shape[0]
    ii, jj = np.where(np.triu(L, 1) < 0.0)
    edges = list(zip(ii.tolist(), jj.tolist()))
    if not edges:
        raise ValueError("L has no edges")
    w = -L[ii, jj]
    return edges, w


def path_energy_table(L: "np.ndarray", kT: float, K: "np.ndarray", paths: "np.ndarray") -> "np.ndarray":
    """Eqs. (10), (25)-(26): per path the Burton-Pemantle probability det(K_pi), the physical length E_E = sum d_e
    and the topological contribution E_T = -kT ln det(Y_pi) with Y_pi the UNWEIGHTED edge-to-edge distance block."""
    L = _check_laplacian(L)
    kT = _check_positive(kT, "kT")
    edges, w = _edges_from_laplacian(L)
    K = np.asarray(K, dtype=np.float64)
    P = np.asarray(paths)
    if K.shape != (len(edges), len(edges)) or P.ndim != 2 or P.shape[0] == 0:
        raise ValueError("K must be E x E for the edges of L and paths must be a non-empty (m, max_nodes) array")
    d = -kT * np.log(w)
    eidx = {}
    for a, (i, j) in enumerate(edges):
        eidx[(i, j)] = a
        eidx[(j, i)] = a
    out = np.zeros((P.shape[0], 3), dtype=np.float64)
    for r in range(P.shape[0]):
        nodes = [int(v) for v in P[r] if v >= 0]
        if len(nodes) < 2:
            raise ValueError("every path needs at least two nodes")
        try:
            ea = [eidx[(nodes[k], nodes[k + 1])] for k in range(len(nodes) - 1)]
        except KeyError:
            raise ValueError("path uses a pair that is not an edge of L")
        Kb = K[np.ix_(ea, ea)]
        Yb = Kb / np.sqrt(np.outer(w[ea], w[ea]))
        out[r, 0] = det(Kb)
        out[r, 1] = d[ea].sum()
        out[r, 2] = -kT * np.log(det(Yb))
    return out

import numpy as np
from numpy.linalg import det, slogdet


def active_channel_weights(P: "np.ndarray", eta: float) -> "np.ndarray":
    """Eqs. (11), (28): corridor-normalised probabilities sorted by ascending surprisal; keep the top paths until their
    cumulative share first reaches eta (that path included); renormalise on the active set, zero elsewhere."""
    P = np.asarray(P, dtype=np.float64).ravel()
    eta = float(eta)
    if P.size == 0 or not np.all(np.isfinite(P)) or np.any(P <= 0.0):
        raise ValueError("P must be a non-empty array of finite positive path probabilities")
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must lie in (0, 1]")
    q = P / P.sum()
    order = np.argsort(-q, kind="stable")
    cum = np.cumsum(q[order])
    m = int(np.searchsorted(cum, eta, side="left")) + 1
    m = min(m, P.size)
    act = order[:m]
    p = np.zeros_like(P)
    p[act] = P[act] / P[act].sum()
    return p

import numpy as np
from numpy.linalg import det, slogdet


def _check_positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + " must be a finite positive number")
    return x


def channel_thermodynamics(p: "np.ndarray", table: "np.ndarray", kT: float) -> "np.ndarray":
    """Eqs. (13), (16)-(18) and 2.3.2: [S, PR, C, C_E, C_T, C_X] over the active channel distribution."""
    p = np.asarray(p, dtype=np.float64).ravel()
    T = np.asarray(table, dtype=np.float64)
    kT = _check_positive(kT, "kT")
    if T.ndim != 2 or T.shape[1] != 3 or T.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("p must be a probability vector aligned with the rows of table (m, 3)")
    act = p > 0.0
    pa = p[act]
    EE = T[act, 1]
    ET = T[act, 2]
    mean = lambda x: float(pa @ x)
    var = lambda x: float(pa @ (x - mean(x)) ** 2)
    cov = float(pa @ ((EE - mean(EE)) * (ET - mean(ET))))
    S = -float(pa @ np.log(pa))
    PR = 1.0 / float(pa @ pa)
    E = EE + ET
    return np.array([S, PR, var(E) / kT ** 2, var(EE) / kT ** 2, var(ET) / kT ** 2, 2.0 * cov / kT ** 2])

import numpy as np
from numpy.linalg import det, slogdet


def allosteric_importance(paths: "np.ndarray", p: "np.ndarray", n_res: int) -> "np.ndarray":
    """2.3.2: I_k = share of the active channel ensemble whose path passes through interior residue k (endpoints 0)."""
    P = np.asarray(paths)
    p = np.asarray(p, dtype=np.float64).ravel()
    if isinstance(n_res, bool) or int(n_res) != n_res or int(n_res) < 3:
        raise ValueError("n_res must be an integer >= 3")
    n_res = int(n_res)
    if P.ndim != 2 or P.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0) or P.max() >= n_res:
        raise ValueError("paths (m, max_nodes) must align with the probability vector p and index residues < n_res")
    I = np.zeros(n_res, dtype=np.float64)
    for r in range(P.shape[0]):
        if p[r] > 0.0:
            nodes = [int(v) for v in P[r] if v >= 0]
            for v in nodes[1:-1]:
                I[v] += p[r]
    return I

import numpy as np
from numpy.linalg import det, slogdet


def _path_distribution(P, p):
    """Map each active path (node tuple) to its occupancy weight; validates alignment and normalisation."""
    P = np.asarray(P)
    p = np.asarray(p, dtype=np.float64).ravel()
    if P.ndim != 2 or P.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("each path table must align with its probability vector")
    return {tuple(int(v) for v in P[r] if v >= 0): float(p[r]) for r in range(P.shape[0]) if p[r] > 0.0}


def channel_divergence(paths_a: "np.ndarray", p_a: "np.ndarray", paths_b: "np.ndarray", p_b: "np.ndarray") -> float:
    """Eqs. (19)-(20): Jensen-Shannon divergence (natural log) between two channel distributions over the union of paths."""
    for P, p in ((paths_a, p_a), (paths_b, p_b)):
        P = np.asarray(P)
        p = np.asarray(p, dtype=np.float64).ravel()
        if P.ndim != 2 or P.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("each path table must align with its probability vector summing to one")
    A = _path_distribution(paths_a, p_a)
    B = _path_distribution(paths_b, p_b)
    keys = sorted(set(A) | set(B))
    pa = np.array([A.get(k, 0.0) for k in keys])
    pb = np.array([B.get(k, 0.0) for k in keys])
    m = 0.5 * (pa + pb)
    kl = lambda x, y: float(np.sum(x[x > 0] * np.log(x[x > 0] / y[x > 0])))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)

import numpy as np
from numpy.linalg import det, slogdet


def _check_laplacian(L):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite square matrix of size >= 3")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):
        raise ValueError("L must be a symmetric Laplacian with zero row sums")
    return L


def _effective_resistance(L, a, b):
    """Eq. (7) from the grounded generalised inverse of a connected Laplacian."""
    n = L.shape[0]
    G = np.zeros((n, n))
    G[1:, 1:] = np.linalg.inv(L[1:, 1:])
    return G[a, a] + G[b, b] - 2.0 * G[a, b]


def channel_convergence_ratio(L: "np.ndarray", s: int, t: int, max_nodes: int) -> float:
    """Sec. 3.2.1: r(L) = R_st(G(L)) / R_st(G), where G(L) is the INDUCED subgraph on the union of nodes visited by the
    simple paths of node length <= max_nodes (every edge of G between two visited nodes is retained, not only path edges)."""
    L = _check_laplacian(L)
    for name, v in (("s", s), ("t", t), ("max_nodes", max_nodes)):
        if isinstance(v, bool) or int(v) != v:
            raise ValueError(name + " must be an integer")
    if not (0 <= int(s) < L.shape[0] and 0 <= int(t) < L.shape[0]) or int(s) == int(t) or int(max_nodes) < 2:
        raise ValueError("s and t must be distinct residue indices and max_nodes >= 2")
    P = channel_paths(L, s, t, max_nodes)
    nodes = sorted(set(int(v) for v in P.ravel() if v >= 0))
    sub = L[np.ix_(nodes, nodes)].copy()
    W = -sub
    np.fill_diagonal(W, 0.0)
    Ls = np.diag(W.sum(axis=1)) - W
    idx = {v: i for i, v in enumerate(nodes)}
    return float(_effective_resistance(Ls, idx[int(s)], idx[int(t)]) / _effective_resistance(L, int(s), int(t)))

import numpy as np
from numpy.linalg import det, slogdet


def importance_shift_table(importance_wt: "np.ndarray", importance_mut: "np.ndarray") -> "np.ndarray":
    """Table 3 of the source: relative shift Delta I_k = 100 (I_k^mut - I_k^wt) / I_k^wt in percent, reported only for
    residues whose wild-type occupancy reaches the source's reporting threshold I_k^wt >= 0.03; zero elsewhere."""
    Iw = np.asarray(importance_wt, dtype=np.float64).ravel()
    Im = np.asarray(importance_mut, dtype=np.float64).ravel()
    if Iw.size != Im.size or Iw.size < 3 or not np.all(np.isfinite(Iw)) or not np.all(np.isfinite(Im)) or np.any(Iw < 0) or np.any(Im < 0):
        raise ValueError("importance vectors must be finite, nonnegative and of equal length >= 3")
    out = np.zeros_like(Iw)
    keep = Iw >= 0.03
    out[keep] = 100.0 * (Im[keep] - Iw[keep]) / Iw[keep]
    return out

import numpy as np
from numpy.linalg import det, slogdet


def _check_coords(coords):
    X = np.asarray(coords, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 3 or not np.all(np.isfinite(X)):
        raise ValueError("coords must be a finite (N, 3) array with N >= 3")
    return X


def _check_positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + " must be a finite positive number")
    return x


def _check_laplacian(L):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite square matrix of size >= 3")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):
        raise ValueError("L must be a symmetric Laplacian with zero row sums")
    return L


def _edges_from_laplacian(L):
    """Lexicographic (i < j) edge list, lengths and weights recovered from the weighted Laplacian."""
    n = L.shape[0]
    ii, jj = np.where(np.triu(L, 1) < 0.0)
    edges = list(zip(ii.tolist(), jj.tolist()))
    if not edges:
        raise ValueError("L has no edges")
    w = -L[ii, jj]
    return edges, w


def _transfer_current(L, edges, w):
    """Symmetric transfer-current matrix K_ab = sqrt(w_a w_b) Y(e_a, e_b), eqs. (8)-(9). The edge vectors chi_e are
    orthogonal to the constant null vector of L, so any generalised inverse gives the same Y as the Moore-Penrose
    pseudoinverse; the grounded inverse of the reduced Laplacian is used because it is well conditioned and exact."""
    n = L.shape[0]
    Kp = np.zeros((n, n))
    Kp[1:, 1:] = np.linalg.inv(L[1:, 1:])
    chi = np.zeros((len(edges), n))
    for a, (i, j) in enumerate(edges):
        chi[a, i] = 1.0
        chi[a, j] = -1.0
    Y = chi @ Kp @ chi.T
    return Y * np.sqrt(np.outer(w, w))


def _path_distribution(P, p):
    """Map each active path (node tuple) to its occupancy weight; validates alignment and normalisation."""
    P = np.asarray(P)
    p = np.asarray(p, dtype=np.float64).ravel()
    if P.ndim != 2 or P.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("each path table must align with its probability vector")
    return {tuple(int(v) for v in P[r] if v >= 0): float(p[r]) for r in range(P.shape[0]) if p[r] > 0.0}


def _effective_resistance(L, a, b):
    """Eq. (7) from the grounded generalised inverse of a connected Laplacian."""
    n = L.shape[0]
    G = np.zeros((n, n))
    G[1:, 1:] = np.linalg.inv(L[1:, 1:])
    return G[a, a] + G[b, b] - 2.0 * G[a, b]


def _tree_energy_cumulants(X, r_c, kT):
    """Exact (Var(E), kappa_3) of the tree energy at temperature kT from the determinantal edge-indicator structure:
    Var = d^T Cov d with Cov = -K*K off the diagonal and p(1-p) on it; kappa_3 = sum d^3 p - 3 sum d_e^2 d_f K_ef^2
    + 2 sum d_e d_f d_g K_ef K_fg K_ge (= -dVar/dbeta)."""
    L = contact_laplacian(X, r_c, kT)
    edges, w = _edges_from_laplacian(L)
    d = -kT * np.log(w)
    K = _transfer_current(L, edges, w)
    p = np.diag(K)
    K2 = K ** 2
    cov = -K2
    np.fill_diagonal(cov, p * (1.0 - p))
    var = float(d @ cov @ d)
    k3 = float(np.sum(d ** 3 * p) - 3.0 * np.sum((d ** 2)[:, None] * d[None, :] * K2)
               + 2.0 * (d @ (K * (K @ np.diag(d) @ K)) @ d))
    return var, k3


def _peak_residual(X, r_c, kT):
    """g(kT) = kappa_3 - 2 kT Var(E): positive below the heat-capacity peak, negative above it (dC/dbeta = beta^2 g / kT)."""
    var, k3 = _tree_energy_cumulants(X, r_c, kT)
    return k3 - 2.0 * kT * var


def mutation_cross_coupling_shift(coords_wt: "np.ndarray", coords_mut: "np.ndarray", r_c: float, kT_lo: float, kT_hi: float,
                                          s: int, t: int, max_nodes: int, eta: float) -> float:
    """Delta C_X = C_X(mutant) - C_X(wild type) for the channel s -> t (Table 2 of the source), both structures evaluated
    at the wild type's heat-capacity peak temperature kT* found in (kT_lo, kT_hi)."""
    Xw = _check_coords(coords_wt)
    Xm = _check_coords(coords_mut)
    if Xw.shape != Xm.shape:
        raise ValueError("wild-type and mutant coordinates must have the same shape")
    kT = heat_capacity_peak_temperature(Xw, r_c, kT_lo, kT_hi)
    ensembles = []
    for X in (Xw, Xm):
        L = contact_laplacian(X, r_c, kT)
        thermo = global_tree_thermodynamics(L, kT)          # also certifies a connected graph
        if not np.all(np.isfinite(thermo)):
            raise ValueError("global tree thermodynamics are not finite")
        K = edge_transfer_current(L, kT)
        paths = channel_paths(L, s, t, max_nodes)
        table = path_energy_table(L, kT, K, paths)
        p = active_channel_weights(table[:, 0], eta)
        channel = channel_thermodynamics(p, table, kT)
        importance = allosteric_importance(paths, p, X.shape[0])
        if importance[int(s)] != 0.0 or importance[int(t)] != 0.0 or np.any(importance > 1.0 + 1e-12):
            raise ValueError("allosteric importance must vanish at the endpoints and never exceed one")
        ratio = channel_convergence_ratio(L, s, t, max_nodes)
        if ratio < 1.0 - 1e-9:                                      # Rayleigh monotonicity: R_st(G(L)) >= R_st(G)
            raise ValueError("the truncated channel cannot have a lower resistance than the full graph")
        ensembles.append((paths, p, channel, importance))
    d_js = channel_divergence(ensembles[0][0], ensembles[0][1], ensembles[1][0], ensembles[1][1])
    if not (-1e-12 <= d_js <= np.log(2.0) + 1e-12):
        raise ValueError("the channel divergence must lie in [0, ln 2]")
    shifts = importance_shift_table(ensembles[0][3], ensembles[1][3])
    if not np.all(np.isfinite(shifts)):
        raise ValueError("importance shifts must be finite")
    return float(ensembles[1][2][5] - ensembles[0][2][5])
SCICODE_GOLD_EOF
