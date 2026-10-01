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
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def _positions(N, r_double, r_single, angle_deg):
    """All-trans zigzag geometry: bond i (atom i to atom i + 1) has length r_double for even i
    and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain
    axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg."""
    phi = (180.0 - angle_deg) * pi / 360.0
    pos = np.zeros((N, 2))
    for i in range(N - 1):
        ell = r_double if i % 2 == 0 else r_single
        ang = phi if i % 2 == 0 else -phi
        pos[i + 1] = pos[i] + ell * np.array([cos(ang), sin(ang)])
    return pos

def _ohno(U, eps, r):
    return U / np.sqrt(1.0 + (U * eps * r / 14.397) ** 2)

def coulomb_matrix(N: int, U: float, eps: float, r_double: float, r_single: float, angle_deg: float) -> np.ndarray:
    _check_model(N, U, eps, 1.0, 0.0, r_double, r_single, angle_deg)
    if N < 2:
        raise ValueError("N must be at least 2")
    pos = _positions(N, r_double, r_single, angle_deg)
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    V = _ohno(U, eps, r)
    np.fill_diagonal(V, U)
    return V

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def _positions(N, r_double, r_single, angle_deg):
    """All-trans zigzag geometry: bond i (atom i to atom i + 1) has length r_double for even i
    and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain
    axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg."""
    phi = (180.0 - angle_deg) * pi / 360.0
    pos = np.zeros((N, 2))
    for i in range(N - 1):
        ell = r_double if i % 2 == 0 else r_single
        ang = phi if i % 2 == 0 else -phi
        pos[i + 1] = pos[i] + ell * np.array([cos(ang), sin(ang)])
    return pos

def _ohno(U, eps, r):
    return U / np.sqrt(1.0 + (U * eps * r / 14.397) ** 2)

def _popcount_table(N):
    t = np.zeros(1 << N, dtype=np.int64)
    for b in range(N):
        t[(np.arange(1 << N) >> b) & 1 == 1] += 1
    return t

def _configs(N, k):
    """All bit masks of N bits with exactly k bits set, ascending as integers."""
    allm = np.arange(1 << N, dtype=np.int64)
    return allm[_popcount_table(N)[allm] == k]

def _sector(N, n_up, n_down):
    """Determinant basis of the (n_up, n_down) sector: index = i_up * D_down + i_down, up and
    down configurations ascending as integers; creation operators ordered all-up (ascending
    site) then all-down (ascending site)."""
    cu = _configs(N, n_up)
    cd = _configs(N, n_down)
    u = np.repeat(cu, len(cd))
    d = np.tile(cd, len(cu))
    return cu, cd, u, d

def _index(cu, cd, u, d):
    iu = np.searchsorted(cu, u)
    idn = np.searchsorted(cd, d)
    return iu * len(cd) + idn

def _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down):
    cu, cd, u, d = _sector(N, n_up, n_down)
    D = len(u)
    pos = _positions(N, r_double, r_single, angle_deg)
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    V = _ohno(U, eps, r)
    np.fill_diagonal(V, 0.0)
    occ_u = ((u[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    occ_d = ((d[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    q = occ_u + occ_d - 1.0
    diag = U * ((occ_u - 0.5) * (occ_d - 0.5)).sum(axis=1) + 0.5 * np.einsum("ki,ij,kj->k", q, V, q)
    H = np.zeros((D, D))
    H[np.arange(D), np.arange(D)] = diag
    cols = np.arange(D)
    for i in range(N - 1):
        t = t0 * (1.0 + delta) if i % 2 == 0 else t0 * (1.0 - delta)
        j = i + 1
        for a, b in ((i, j), (j, i)):
            ma, mb = 1 << a, 1 << b
            for spin in (0, 1):
                occ = u if spin == 0 else d
                sel = ((occ & ma) != 0) & ((occ & mb) == 0)
                new = occ[sel] ^ ma ^ mb
                if spin == 0:
                    rows = _index(cu, cd, new, d[sel])
                else:
                    rows = _index(cu, cd, u[sel], new)
                H[rows, cols[sel]] += -t
    return H

def ppp_hamiltonian(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_up: int, n_down: int) -> np.ndarray:
    _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg)
    for x in (n_up, n_down):
        if not (isinstance(x, (int, np.integer)) and not isinstance(x, bool)) or x < 0 or x > N:
            raise ValueError("electron numbers must be integers between 0 and N")
    return _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def _positions(N, r_double, r_single, angle_deg):
    """All-trans zigzag geometry: bond i (atom i to atom i + 1) has length r_double for even i
    and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain
    axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg."""
    phi = (180.0 - angle_deg) * pi / 360.0
    pos = np.zeros((N, 2))
    for i in range(N - 1):
        ell = r_double if i % 2 == 0 else r_single
        ang = phi if i % 2 == 0 else -phi
        pos[i + 1] = pos[i] + ell * np.array([cos(ang), sin(ang)])
    return pos

def _ohno(U, eps, r):
    return U / np.sqrt(1.0 + (U * eps * r / 14.397) ** 2)

def _popcount_table(N):
    t = np.zeros(1 << N, dtype=np.int64)
    for b in range(N):
        t[(np.arange(1 << N) >> b) & 1 == 1] += 1
    return t

def _configs(N, k):
    """All bit masks of N bits with exactly k bits set, ascending as integers."""
    allm = np.arange(1 << N, dtype=np.int64)
    return allm[_popcount_table(N)[allm] == k]

def _sector(N, n_up, n_down):
    """Determinant basis of the (n_up, n_down) sector: index = i_up * D_down + i_down, up and
    down configurations ascending as integers; creation operators ordered all-up (ascending
    site) then all-down (ascending site)."""
    cu = _configs(N, n_up)
    cd = _configs(N, n_down)
    u = np.repeat(cu, len(cd))
    d = np.tile(cd, len(cu))
    return cu, cd, u, d

def _index(cu, cd, u, d):
    iu = np.searchsorted(cu, u)
    idn = np.searchsorted(cd, d)
    return iu * len(cd) + idn

def _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down):
    cu, cd, u, d = _sector(N, n_up, n_down)
    D = len(u)
    pos = _positions(N, r_double, r_single, angle_deg)
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    V = _ohno(U, eps, r)
    np.fill_diagonal(V, 0.0)
    occ_u = ((u[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    occ_d = ((d[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    q = occ_u + occ_d - 1.0
    diag = U * ((occ_u - 0.5) * (occ_d - 0.5)).sum(axis=1) + 0.5 * np.einsum("ki,ij,kj->k", q, V, q)
    H = np.zeros((D, D))
    H[np.arange(D), np.arange(D)] = diag
    cols = np.arange(D)
    for i in range(N - 1):
        t = t0 * (1.0 + delta) if i % 2 == 0 else t0 * (1.0 - delta)
        j = i + 1
        for a, b in ((i, j), (j, i)):
            ma, mb = 1 << a, 1 << b
            for spin in (0, 1):
                occ = u if spin == 0 else d
                sel = ((occ & ma) != 0) & ((occ & mb) == 0)
                new = occ[sel] ^ ma ^ mb
                if spin == 0:
                    rows = _index(cu, cd, new, d[sel])
                else:
                    rows = _index(cu, cd, u[sel], new)
                H[rows, cols[sel]] += -t
    return H

def _splus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S+ = sum_i c+_{i up} c_{i down} from the sector of (u, d) to the sector
    (cu2, cd2) with one more up electron, as (rows, cols, vals)."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((d & m) != 0) & ((u & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up + pc[dd & below] + pc[uu & below])
        rows.append(_index(cu2, cd2, uu | m, dd ^ m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _sminus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S- = sum_i c+_{i down} c_{i up}."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((u & m) != 0) & ((d & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up - 1 + pc[uu & below] + pc[dd & below])
        rows.append(_index(cu2, cd2, uu ^ m, dd | m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _apply(op, x, D_out):
    rows, cols, vals = op
    y = np.zeros((D_out,) + x.shape[1:])
    np.add.at(y, rows, vals[:, None] * x[cols] if x.ndim == 2 else vals * x[cols])
    return y

def _reverse_bits(x, N):
    y = np.zeros_like(x)
    for i in range(N):
        y |= ((x >> i) & 1) << (N - 1 - i)
    return y

def _resolve(N, n_up, n_down, w, v, tol=1e-7):
    """Rotate eigenvectors inside every (near-)degenerate energy cluster so that S^2, the site
    reversal P and the occupation complement J are simultaneously diagonal, order the members
    of a cluster by (S(S+1), p, j) ascending and return (vectors, [S(S+1), p, j] per state)."""
    cu, cd, u, d = _sector(N, n_up, n_down)
    full = (1 << N) - 1
    ip = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ij = _index(cu, cd, full ^ u, full ^ d)
    sz = 0.5 * (n_up - n_down)
    if n_up < N:
        cu2, cd2, u2, d2 = _sector(N, n_up + 1, n_down - 1)
        sp = _splus(N, cu, cd, u, d, cu2, cd2)
        sm = _sminus(N, cu2, cd2, u2, d2, cu, cd)
        D2 = len(u2)

    def _s2_apply(X):
        if n_up == N:
            return sz * (sz + 1.0) * X
        return _apply(sm, _apply(sp, X, D2), len(u)) + sz * (sz + 1.0) * X

    ops = (_s2_apply, lambda X: X[ip], lambda X: X[ij])
    v = v.copy()
    lab = np.zeros((len(w), 3))
    scale = max(1.0, float(np.max(np.abs(w))))
    start = 0
    while start < len(w):
        stop = start + 1
        while stop < len(w) and w[stop] - w[stop - 1] < tol * scale:
            stop += 1
        groups = [list(range(start, stop))]
        for op in ops:
            newg = []
            for g in groups:
                X = v[:, g]
                M = X.T @ op(X)
                M = 0.5 * (M + M.T)
                ev, Q = np.linalg.eigh(M)
                X = X @ Q
                v[:, g] = X
                cut = [0] + [t for t in range(1, len(g)) if ev[t] - ev[t - 1] > 1e-6] + [len(g)]
                newg += [[g[t] for t in range(cut[a], cut[a + 1])] for a in range(len(cut) - 1)]
            groups = newg
        X = v[:, start:stop]
        vals = np.column_stack([np.einsum("ik,ik->k", X, op(X)) for op in ops])
        keys = np.round(vals, 6)  # order degenerate members by their integer labels, not by rounding noise
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        v[:, start:stop] = X[:, order]
        lab[start:stop] = vals[order]
        start = stop
    return v, lab

def _lowest_states(H, N, n, n_states):
    """Lowest n_states eigenpairs of H in the (n, n) sector, obtained block by block in the
    four symmetry-adapted subspaces of the group generated by the spin flip (u, d) -> (d, u)
    and the site reversal; the eigenvectors are expanded back to the full sector basis."""
    cu, cd, u, d = _sector(N, n, n)
    D = len(u)
    pf = _index(cu, cd, d, u)
    pp = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ar = np.arange(D)
    orb = np.column_stack([ar, pf, pp, pf[pp]])
    rep = orb.min(axis=1)
    own = np.nonzero(rep == ar)[0]
    ws, vs = [], []
    for chi_f in (1.0, -1.0):
        for chi_p in (1.0, -1.0):
            ch = np.array([1.0, chi_f, chi_p, chi_f * chi_p])
            cols_all = orb[own]
            coef = np.zeros((len(own), 4))
            for a in range(4):
                for b in range(4):
                    coef[:, a] += ch[b] * (cols_all[:, b] == cols_all[:, a])
            for a in range(1, 4):
                for b in range(a):
                    coef[:, a] *= (cols_all[:, a] != cols_all[:, b])
            coef = coef / 4.0
            norm = np.sqrt((coef * coef).sum(axis=1) * 1.0)
            keep = norm > 1e-12
            if not keep.any():
                continue
            cols_all = cols_all[keep]
            coef = coef[keep] / norm[keep][:, None]
            nb = cols_all.shape[0]
            HB = np.zeros((D, nb))
            for a in range(4):
                HB += H[:, cols_all[:, a]] * coef[:, a][None, :]
            Hb = np.zeros((nb, nb))
            for a in range(4):
                Hb += coef[:, a][:, None] * HB[cols_all[:, a], :]
            Hb = 0.5 * (Hb + Hb.T)
            k = min(n_states, nb)
            wb, vb = eigh(Hb, driver="evr", subset_by_index=[0, k - 1])
            V = np.zeros((D, k))
            for a in range(4):
                np.add.at(V, cols_all[:, a], coef[:, a][:, None] * vb)
            ws.append(wb)
            vs.append(V)
    w = np.concatenate(ws)
    v = np.hstack(vs)
    order = np.argsort(w, kind="stable")[:n_states]
    return w[order], v[:, order]

def _labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states):
    """Lowest n_states eigenpairs of the half-filled Sz = 0 sector with labels
    [E_k - E_0, S(S+1), p_k p_0, j_k j_0]; p = site reversal, j = occupation complement."""
    n = N // 2
    H = _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n, n)
    if n_states > H.shape[0]:
        raise ValueError("n_states exceeds the dimension of the half-filled Sz = 0 sector")
    w, v = _lowest_states(H, N, n, n_states)
    v, sl = _resolve(N, n, n, w, v)
    lab = np.column_stack([w - w[0], sl[:, 0], sl[:, 1] * sl[0, 1], sl[:, 2] * sl[0, 2]])
    return w, v, lab, H

def symmetry_labels(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int) -> np.ndarray:
    _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg)
    if not (isinstance(n_states, (int, np.integer)) and not isinstance(n_states, bool)) or n_states < 1:
        raise ValueError("n_states must be a positive integer")
    w, v, lab, H = _labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
    return lab

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def _positions(N, r_double, r_single, angle_deg):
    """All-trans zigzag geometry: bond i (atom i to atom i + 1) has length r_double for even i
    and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain
    axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg."""
    phi = (180.0 - angle_deg) * pi / 360.0
    pos = np.zeros((N, 2))
    for i in range(N - 1):
        ell = r_double if i % 2 == 0 else r_single
        ang = phi if i % 2 == 0 else -phi
        pos[i + 1] = pos[i] + ell * np.array([cos(ang), sin(ang)])
    return pos

def _ohno(U, eps, r):
    return U / np.sqrt(1.0 + (U * eps * r / 14.397) ** 2)

def _popcount_table(N):
    t = np.zeros(1 << N, dtype=np.int64)
    for b in range(N):
        t[(np.arange(1 << N) >> b) & 1 == 1] += 1
    return t

def _configs(N, k):
    """All bit masks of N bits with exactly k bits set, ascending as integers."""
    allm = np.arange(1 << N, dtype=np.int64)
    return allm[_popcount_table(N)[allm] == k]

def _sector(N, n_up, n_down):
    """Determinant basis of the (n_up, n_down) sector: index = i_up * D_down + i_down, up and
    down configurations ascending as integers; creation operators ordered all-up (ascending
    site) then all-down (ascending site)."""
    cu = _configs(N, n_up)
    cd = _configs(N, n_down)
    u = np.repeat(cu, len(cd))
    d = np.tile(cd, len(cu))
    return cu, cd, u, d

def _index(cu, cd, u, d):
    iu = np.searchsorted(cu, u)
    idn = np.searchsorted(cd, d)
    return iu * len(cd) + idn

def _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down):
    cu, cd, u, d = _sector(N, n_up, n_down)
    D = len(u)
    pos = _positions(N, r_double, r_single, angle_deg)
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    V = _ohno(U, eps, r)
    np.fill_diagonal(V, 0.0)
    occ_u = ((u[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    occ_d = ((d[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    q = occ_u + occ_d - 1.0
    diag = U * ((occ_u - 0.5) * (occ_d - 0.5)).sum(axis=1) + 0.5 * np.einsum("ki,ij,kj->k", q, V, q)
    H = np.zeros((D, D))
    H[np.arange(D), np.arange(D)] = diag
    cols = np.arange(D)
    for i in range(N - 1):
        t = t0 * (1.0 + delta) if i % 2 == 0 else t0 * (1.0 - delta)
        j = i + 1
        for a, b in ((i, j), (j, i)):
            ma, mb = 1 << a, 1 << b
            for spin in (0, 1):
                occ = u if spin == 0 else d
                sel = ((occ & ma) != 0) & ((occ & mb) == 0)
                new = occ[sel] ^ ma ^ mb
                if spin == 0:
                    rows = _index(cu, cd, new, d[sel])
                else:
                    rows = _index(cu, cd, u[sel], new)
                H[rows, cols[sel]] += -t
    return H

def _splus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S+ = sum_i c+_{i up} c_{i down} from the sector of (u, d) to the sector
    (cu2, cd2) with one more up electron, as (rows, cols, vals)."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((d & m) != 0) & ((u & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up + pc[dd & below] + pc[uu & below])
        rows.append(_index(cu2, cd2, uu | m, dd ^ m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _sminus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S- = sum_i c+_{i down} c_{i up}."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((u & m) != 0) & ((d & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up - 1 + pc[uu & below] + pc[dd & below])
        rows.append(_index(cu2, cd2, uu ^ m, dd | m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _apply(op, x, D_out):
    rows, cols, vals = op
    y = np.zeros((D_out,) + x.shape[1:])
    np.add.at(y, rows, vals[:, None] * x[cols] if x.ndim == 2 else vals * x[cols])
    return y

def _reverse_bits(x, N):
    y = np.zeros_like(x)
    for i in range(N):
        y |= ((x >> i) & 1) << (N - 1 - i)
    return y

def _resolve(N, n_up, n_down, w, v, tol=1e-7):
    """Rotate eigenvectors inside every (near-)degenerate energy cluster so that S^2, the site
    reversal P and the occupation complement J are simultaneously diagonal, order the members
    of a cluster by (S(S+1), p, j) ascending and return (vectors, [S(S+1), p, j] per state)."""
    cu, cd, u, d = _sector(N, n_up, n_down)
    full = (1 << N) - 1
    ip = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ij = _index(cu, cd, full ^ u, full ^ d)
    sz = 0.5 * (n_up - n_down)
    if n_up < N:
        cu2, cd2, u2, d2 = _sector(N, n_up + 1, n_down - 1)
        sp = _splus(N, cu, cd, u, d, cu2, cd2)
        sm = _sminus(N, cu2, cd2, u2, d2, cu, cd)
        D2 = len(u2)

    def _s2_apply(X):
        if n_up == N:
            return sz * (sz + 1.0) * X
        return _apply(sm, _apply(sp, X, D2), len(u)) + sz * (sz + 1.0) * X

    ops = (_s2_apply, lambda X: X[ip], lambda X: X[ij])
    v = v.copy()
    lab = np.zeros((len(w), 3))
    scale = max(1.0, float(np.max(np.abs(w))))
    start = 0
    while start < len(w):
        stop = start + 1
        while stop < len(w) and w[stop] - w[stop - 1] < tol * scale:
            stop += 1
        groups = [list(range(start, stop))]
        for op in ops:
            newg = []
            for g in groups:
                X = v[:, g]
                M = X.T @ op(X)
                M = 0.5 * (M + M.T)
                ev, Q = np.linalg.eigh(M)
                X = X @ Q
                v[:, g] = X
                cut = [0] + [t for t in range(1, len(g)) if ev[t] - ev[t - 1] > 1e-6] + [len(g)]
                newg += [[g[t] for t in range(cut[a], cut[a + 1])] for a in range(len(cut) - 1)]
            groups = newg
        X = v[:, start:stop]
        vals = np.column_stack([np.einsum("ik,ik->k", X, op(X)) for op in ops])
        keys = np.round(vals, 6)  # order degenerate members by their integer labels, not by rounding noise
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        v[:, start:stop] = X[:, order]
        lab[start:stop] = vals[order]
        start = stop
    return v, lab

def _lowest_states(H, N, n, n_states):
    """Lowest n_states eigenpairs of H in the (n, n) sector, obtained block by block in the
    four symmetry-adapted subspaces of the group generated by the spin flip (u, d) -> (d, u)
    and the site reversal; the eigenvectors are expanded back to the full sector basis."""
    cu, cd, u, d = _sector(N, n, n)
    D = len(u)
    pf = _index(cu, cd, d, u)
    pp = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ar = np.arange(D)
    orb = np.column_stack([ar, pf, pp, pf[pp]])
    rep = orb.min(axis=1)
    own = np.nonzero(rep == ar)[0]
    ws, vs = [], []
    for chi_f in (1.0, -1.0):
        for chi_p in (1.0, -1.0):
            ch = np.array([1.0, chi_f, chi_p, chi_f * chi_p])
            cols_all = orb[own]
            coef = np.zeros((len(own), 4))
            for a in range(4):
                for b in range(4):
                    coef[:, a] += ch[b] * (cols_all[:, b] == cols_all[:, a])
            for a in range(1, 4):
                for b in range(a):
                    coef[:, a] *= (cols_all[:, a] != cols_all[:, b])
            coef = coef / 4.0
            norm = np.sqrt((coef * coef).sum(axis=1) * 1.0)
            keep = norm > 1e-12
            if not keep.any():
                continue
            cols_all = cols_all[keep]
            coef = coef[keep] / norm[keep][:, None]
            nb = cols_all.shape[0]
            HB = np.zeros((D, nb))
            for a in range(4):
                HB += H[:, cols_all[:, a]] * coef[:, a][None, :]
            Hb = np.zeros((nb, nb))
            for a in range(4):
                Hb += coef[:, a][:, None] * HB[cols_all[:, a], :]
            Hb = 0.5 * (Hb + Hb.T)
            k = min(n_states, nb)
            wb, vb = eigh(Hb, driver="evr", subset_by_index=[0, k - 1])
            V = np.zeros((D, k))
            for a in range(4):
                np.add.at(V, cols_all[:, a], coef[:, a][:, None] * vb)
            ws.append(wb)
            vs.append(V)
    w = np.concatenate(ws)
    v = np.hstack(vs)
    order = np.argsort(w, kind="stable")[:n_states]
    return w[order], v[:, order]

def _labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states):
    """Lowest n_states eigenpairs of the half-filled Sz = 0 sector with labels
    [E_k - E_0, S(S+1), p_k p_0, j_k j_0]; p = site reversal, j = occupation complement."""
    n = N // 2
    H = _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n, n)
    if n_states > H.shape[0]:
        raise ValueError("n_states exceeds the dimension of the half-filled Sz = 0 sector")
    w, v = _lowest_states(H, N, n, n_states)
    v, sl = _resolve(N, n, n, w, v)
    lab = np.column_stack([w - w[0], sl[:, 0], sl[:, 1] * sl[0, 1], sl[:, 2] * sl[0, 2]])
    return w, v, lab, H

def _pick(lab, s2_target, p_target, j_target, order):
    sel = [k for k in range(lab.shape[0]) if abs(lab[k, 1] - s2_target) < 1e-6
           and abs(lab[k, 2] - p_target) < 1e-6 and abs(lab[k, 3] - j_target) < 1e-6]
    if len(sel) <= order:
        raise ValueError("requested state not found among the computed eigenstates; raise n_states")
    return sel[order]

def _dark_states(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states):
    """Indices and vectors of 1^1Ag+, 2^1Ag+, 1^1Bu-, 1^1Bu+, 1^3Bu, 1^5Ag+ (covalent label:
    singlets and quintets share the ground-state complement eigenvalue, covalent triplets
    carry the opposite one)."""
    w, v, lab, H = _labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
    k_gs = _pick(lab, 0.0, 1.0, 1.0, 0)
    if k_gs != 0:
        raise ValueError("ground state is not a totally symmetric covalent singlet")
    k_2ag = _pick(lab, 0.0, 1.0, 1.0, 1)
    k_bum = _pick(lab, 0.0, -1.0, -1.0, 0)
    k_bup = _pick(lab, 0.0, -1.0, 1.0, 0)
    k_t = _pick(lab, 2.0, -1.0, -1.0, 0)
    k_q = _pick(lab, 6.0, 1.0, 1.0, 0)
    return w, v, lab, H, dict(gs=k_gs, ag2=k_2ag, bum=k_bum, bup=k_bup, t1=k_t, q1=k_q)

def dark_state_energies(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int) -> np.ndarray:
    _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg)
    if not (isinstance(n_states, (int, np.integer)) and not isinstance(n_states, bool)) or n_states < 2:
        raise ValueError("n_states must be an integer >= 2")
    w, v, lab, H, ks = _dark_states(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
    e0 = w[ks["gs"]]
    return np.array([w[ks["ag2"]] - e0, w[ks["bum"]] - e0, w[ks["bup"]] - e0, w[ks["t1"]] - e0,
                     w[ks["q1"]] - e0, w[ks["q1"]] - w[ks["ag2"]]])

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def _positions(N, r_double, r_single, angle_deg):
    """All-trans zigzag geometry: bond i (atom i to atom i + 1) has length r_double for even i
    and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain
    axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg."""
    phi = (180.0 - angle_deg) * pi / 360.0
    pos = np.zeros((N, 2))
    for i in range(N - 1):
        ell = r_double if i % 2 == 0 else r_single
        ang = phi if i % 2 == 0 else -phi
        pos[i + 1] = pos[i] + ell * np.array([cos(ang), sin(ang)])
    return pos

def _ohno(U, eps, r):
    return U / np.sqrt(1.0 + (U * eps * r / 14.397) ** 2)

def _popcount_table(N):
    t = np.zeros(1 << N, dtype=np.int64)
    for b in range(N):
        t[(np.arange(1 << N) >> b) & 1 == 1] += 1
    return t

def _configs(N, k):
    """All bit masks of N bits with exactly k bits set, ascending as integers."""
    allm = np.arange(1 << N, dtype=np.int64)
    return allm[_popcount_table(N)[allm] == k]

def _sector(N, n_up, n_down):
    """Determinant basis of the (n_up, n_down) sector: index = i_up * D_down + i_down, up and
    down configurations ascending as integers; creation operators ordered all-up (ascending
    site) then all-down (ascending site)."""
    cu = _configs(N, n_up)
    cd = _configs(N, n_down)
    u = np.repeat(cu, len(cd))
    d = np.tile(cd, len(cu))
    return cu, cd, u, d

def _index(cu, cd, u, d):
    iu = np.searchsorted(cu, u)
    idn = np.searchsorted(cd, d)
    return iu * len(cd) + idn

def _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down):
    cu, cd, u, d = _sector(N, n_up, n_down)
    D = len(u)
    pos = _positions(N, r_double, r_single, angle_deg)
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    V = _ohno(U, eps, r)
    np.fill_diagonal(V, 0.0)
    occ_u = ((u[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    occ_d = ((d[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    q = occ_u + occ_d - 1.0
    diag = U * ((occ_u - 0.5) * (occ_d - 0.5)).sum(axis=1) + 0.5 * np.einsum("ki,ij,kj->k", q, V, q)
    H = np.zeros((D, D))
    H[np.arange(D), np.arange(D)] = diag
    cols = np.arange(D)
    for i in range(N - 1):
        t = t0 * (1.0 + delta) if i % 2 == 0 else t0 * (1.0 - delta)
        j = i + 1
        for a, b in ((i, j), (j, i)):
            ma, mb = 1 << a, 1 << b
            for spin in (0, 1):
                occ = u if spin == 0 else d
                sel = ((occ & ma) != 0) & ((occ & mb) == 0)
                new = occ[sel] ^ ma ^ mb
                if spin == 0:
                    rows = _index(cu, cd, new, d[sel])
                else:
                    rows = _index(cu, cd, u[sel], new)
                H[rows, cols[sel]] += -t
    return H

def _splus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S+ = sum_i c+_{i up} c_{i down} from the sector of (u, d) to the sector
    (cu2, cd2) with one more up electron, as (rows, cols, vals)."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((d & m) != 0) & ((u & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up + pc[dd & below] + pc[uu & below])
        rows.append(_index(cu2, cd2, uu | m, dd ^ m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _sminus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S- = sum_i c+_{i down} c_{i up}."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((u & m) != 0) & ((d & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up - 1 + pc[uu & below] + pc[dd & below])
        rows.append(_index(cu2, cd2, uu ^ m, dd | m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _apply(op, x, D_out):
    rows, cols, vals = op
    y = np.zeros((D_out,) + x.shape[1:])
    np.add.at(y, rows, vals[:, None] * x[cols] if x.ndim == 2 else vals * x[cols])
    return y

def _reverse_bits(x, N):
    y = np.zeros_like(x)
    for i in range(N):
        y |= ((x >> i) & 1) << (N - 1 - i)
    return y

def _resolve(N, n_up, n_down, w, v, tol=1e-7):
    """Rotate eigenvectors inside every (near-)degenerate energy cluster so that S^2, the site
    reversal P and the occupation complement J are simultaneously diagonal, order the members
    of a cluster by (S(S+1), p, j) ascending and return (vectors, [S(S+1), p, j] per state)."""
    cu, cd, u, d = _sector(N, n_up, n_down)
    full = (1 << N) - 1
    ip = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ij = _index(cu, cd, full ^ u, full ^ d)
    sz = 0.5 * (n_up - n_down)
    if n_up < N:
        cu2, cd2, u2, d2 = _sector(N, n_up + 1, n_down - 1)
        sp = _splus(N, cu, cd, u, d, cu2, cd2)
        sm = _sminus(N, cu2, cd2, u2, d2, cu, cd)
        D2 = len(u2)

    def _s2_apply(X):
        if n_up == N:
            return sz * (sz + 1.0) * X
        return _apply(sm, _apply(sp, X, D2), len(u)) + sz * (sz + 1.0) * X

    ops = (_s2_apply, lambda X: X[ip], lambda X: X[ij])
    v = v.copy()
    lab = np.zeros((len(w), 3))
    scale = max(1.0, float(np.max(np.abs(w))))
    start = 0
    while start < len(w):
        stop = start + 1
        while stop < len(w) and w[stop] - w[stop - 1] < tol * scale:
            stop += 1
        groups = [list(range(start, stop))]
        for op in ops:
            newg = []
            for g in groups:
                X = v[:, g]
                M = X.T @ op(X)
                M = 0.5 * (M + M.T)
                ev, Q = np.linalg.eigh(M)
                X = X @ Q
                v[:, g] = X
                cut = [0] + [t for t in range(1, len(g)) if ev[t] - ev[t - 1] > 1e-6] + [len(g)]
                newg += [[g[t] for t in range(cut[a], cut[a + 1])] for a in range(len(cut) - 1)]
            groups = newg
        X = v[:, start:stop]
        vals = np.column_stack([np.einsum("ik,ik->k", X, op(X)) for op in ops])
        keys = np.round(vals, 6)  # order degenerate members by their integer labels, not by rounding noise
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        v[:, start:stop] = X[:, order]
        lab[start:stop] = vals[order]
        start = stop
    return v, lab

def _subchain_family(m, U, eps, t0, delta, r_double, r_single, angle_deg, all_covalent=False):
    """Covalent triplets of an isolated open subchain of 2m sites: the triplet eigenstates
    (S(S+1) = 2) whose complement eigenvalue is opposite to that of the subchain ground state,
    the m lowest of them (the family) unless all_covalent. Returns (energies, T0 vectors,
    T+1 vectors, T-1 vectors, three sectors, reversal parities)."""
    Ns = 2 * m
    cu, cd, u, d = _sector(Ns, m, m)
    H = _hamiltonian(Ns, U, eps, t0, delta, r_double, r_single, angle_deg, m, m)
    w, v = np.linalg.eigh(H)
    v, sl = _resolve(Ns, m, m, w, v)
    sel = [k for k in range(len(w)) if abs(sl[k, 0] - 2.0) < 1e-6 and abs(sl[k, 2] * sl[0, 2] + 1.0) < 1e-6]
    if len(sel) < m:
        raise ValueError("fewer covalent triplets than family members")
    if not all_covalent:
        sel = sel[:m]
    cu_p, cd_p, u_p, d_p = _sector(Ns, m + 1, m - 1)
    cu_m, cd_m, u_m, d_m = _sector(Ns, m - 1, m + 1)
    sp = _splus(Ns, cu, cd, u, d, cu_p, cd_p)
    sm = _sminus(Ns, cu, cd, u, d, cu_m, cd_m)
    t0v = v[:, sel]
    tp = _apply(sp, t0v, len(u_p)) / sqrt(2.0)
    tm = _apply(sm, t0v, len(u_m)) / sqrt(2.0)
    return w[sel] - w[0], t0v, tp, tm, (cu, cd, u, d), (cu_p, cd_p, u_p, d_p), (cu_m, cd_m, u_m, d_m), sl[sel, 1] * sl[0, 1]

def triplet_family(m: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float) -> np.ndarray:
    if not (isinstance(m, (int, np.integer)) and not isinstance(m, bool)) or m < 1:
        raise ValueError("m must be a positive integer")
    _check_model(2 * m, U, eps, t0, delta, r_double, r_single, angle_deg)
    e, t0v, tp, tm, s0, sp, sm, p = _subchain_family(m, U, eps, t0, delta, r_double, r_single, angle_deg)
    Ns = 2 * m
    cu_p, cd_p, u_p, d_p = sp
    occ = (((u_p[:, None] >> np.arange(Ns)[None, :]) & 1) - ((d_p[:, None] >> np.arange(Ns)[None, :]) & 1)) * 0.5
    dens = (tp ** 2).T @ occ
    return np.column_stack([e, p, dens])

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def _positions(N, r_double, r_single, angle_deg):
    """All-trans zigzag geometry: bond i (atom i to atom i + 1) has length r_double for even i
    and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain
    axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg."""
    phi = (180.0 - angle_deg) * pi / 360.0
    pos = np.zeros((N, 2))
    for i in range(N - 1):
        ell = r_double if i % 2 == 0 else r_single
        ang = phi if i % 2 == 0 else -phi
        pos[i + 1] = pos[i] + ell * np.array([cos(ang), sin(ang)])
    return pos

def _ohno(U, eps, r):
    return U / np.sqrt(1.0 + (U * eps * r / 14.397) ** 2)

def _popcount_table(N):
    t = np.zeros(1 << N, dtype=np.int64)
    for b in range(N):
        t[(np.arange(1 << N) >> b) & 1 == 1] += 1
    return t

def _configs(N, k):
    """All bit masks of N bits with exactly k bits set, ascending as integers."""
    allm = np.arange(1 << N, dtype=np.int64)
    return allm[_popcount_table(N)[allm] == k]

def _sector(N, n_up, n_down):
    """Determinant basis of the (n_up, n_down) sector: index = i_up * D_down + i_down, up and
    down configurations ascending as integers; creation operators ordered all-up (ascending
    site) then all-down (ascending site)."""
    cu = _configs(N, n_up)
    cd = _configs(N, n_down)
    u = np.repeat(cu, len(cd))
    d = np.tile(cd, len(cu))
    return cu, cd, u, d

def _index(cu, cd, u, d):
    iu = np.searchsorted(cu, u)
    idn = np.searchsorted(cd, d)
    return iu * len(cd) + idn

def _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down):
    cu, cd, u, d = _sector(N, n_up, n_down)
    D = len(u)
    pos = _positions(N, r_double, r_single, angle_deg)
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    V = _ohno(U, eps, r)
    np.fill_diagonal(V, 0.0)
    occ_u = ((u[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    occ_d = ((d[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    q = occ_u + occ_d - 1.0
    diag = U * ((occ_u - 0.5) * (occ_d - 0.5)).sum(axis=1) + 0.5 * np.einsum("ki,ij,kj->k", q, V, q)
    H = np.zeros((D, D))
    H[np.arange(D), np.arange(D)] = diag
    cols = np.arange(D)
    for i in range(N - 1):
        t = t0 * (1.0 + delta) if i % 2 == 0 else t0 * (1.0 - delta)
        j = i + 1
        for a, b in ((i, j), (j, i)):
            ma, mb = 1 << a, 1 << b
            for spin in (0, 1):
                occ = u if spin == 0 else d
                sel = ((occ & ma) != 0) & ((occ & mb) == 0)
                new = occ[sel] ^ ma ^ mb
                if spin == 0:
                    rows = _index(cu, cd, new, d[sel])
                else:
                    rows = _index(cu, cd, u[sel], new)
                H[rows, cols[sel]] += -t
    return H

def _splus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S+ = sum_i c+_{i up} c_{i down} from the sector of (u, d) to the sector
    (cu2, cd2) with one more up electron, as (rows, cols, vals)."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((d & m) != 0) & ((u & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up + pc[dd & below] + pc[uu & below])
        rows.append(_index(cu2, cd2, uu | m, dd ^ m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _sminus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S- = sum_i c+_{i down} c_{i up}."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((u & m) != 0) & ((d & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up - 1 + pc[uu & below] + pc[dd & below])
        rows.append(_index(cu2, cd2, uu ^ m, dd | m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _apply(op, x, D_out):
    rows, cols, vals = op
    y = np.zeros((D_out,) + x.shape[1:])
    np.add.at(y, rows, vals[:, None] * x[cols] if x.ndim == 2 else vals * x[cols])
    return y

def _reverse_bits(x, N):
    y = np.zeros_like(x)
    for i in range(N):
        y |= ((x >> i) & 1) << (N - 1 - i)
    return y

def _resolve(N, n_up, n_down, w, v, tol=1e-7):
    """Rotate eigenvectors inside every (near-)degenerate energy cluster so that S^2, the site
    reversal P and the occupation complement J are simultaneously diagonal, order the members
    of a cluster by (S(S+1), p, j) ascending and return (vectors, [S(S+1), p, j] per state)."""
    cu, cd, u, d = _sector(N, n_up, n_down)
    full = (1 << N) - 1
    ip = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ij = _index(cu, cd, full ^ u, full ^ d)
    sz = 0.5 * (n_up - n_down)
    if n_up < N:
        cu2, cd2, u2, d2 = _sector(N, n_up + 1, n_down - 1)
        sp = _splus(N, cu, cd, u, d, cu2, cd2)
        sm = _sminus(N, cu2, cd2, u2, d2, cu, cd)
        D2 = len(u2)

    def _s2_apply(X):
        if n_up == N:
            return sz * (sz + 1.0) * X
        return _apply(sm, _apply(sp, X, D2), len(u)) + sz * (sz + 1.0) * X

    ops = (_s2_apply, lambda X: X[ip], lambda X: X[ij])
    v = v.copy()
    lab = np.zeros((len(w), 3))
    scale = max(1.0, float(np.max(np.abs(w))))
    start = 0
    while start < len(w):
        stop = start + 1
        while stop < len(w) and w[stop] - w[stop - 1] < tol * scale:
            stop += 1
        groups = [list(range(start, stop))]
        for op in ops:
            newg = []
            for g in groups:
                X = v[:, g]
                M = X.T @ op(X)
                M = 0.5 * (M + M.T)
                ev, Q = np.linalg.eigh(M)
                X = X @ Q
                v[:, g] = X
                cut = [0] + [t for t in range(1, len(g)) if ev[t] - ev[t - 1] > 1e-6] + [len(g)]
                newg += [[g[t] for t in range(cut[a], cut[a + 1])] for a in range(len(cut) - 1)]
            groups = newg
        X = v[:, start:stop]
        vals = np.column_stack([np.einsum("ik,ik->k", X, op(X)) for op in ops])
        keys = np.round(vals, 6)  # order degenerate members by their integer labels, not by rounding noise
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        v[:, start:stop] = X[:, order]
        lab[start:stop] = vals[order]
        start = stop
    return v, lab

def _lowest_states(H, N, n, n_states):
    """Lowest n_states eigenpairs of H in the (n, n) sector, obtained block by block in the
    four symmetry-adapted subspaces of the group generated by the spin flip (u, d) -> (d, u)
    and the site reversal; the eigenvectors are expanded back to the full sector basis."""
    cu, cd, u, d = _sector(N, n, n)
    D = len(u)
    pf = _index(cu, cd, d, u)
    pp = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ar = np.arange(D)
    orb = np.column_stack([ar, pf, pp, pf[pp]])
    rep = orb.min(axis=1)
    own = np.nonzero(rep == ar)[0]
    ws, vs = [], []
    for chi_f in (1.0, -1.0):
        for chi_p in (1.0, -1.0):
            ch = np.array([1.0, chi_f, chi_p, chi_f * chi_p])
            cols_all = orb[own]
            coef = np.zeros((len(own), 4))
            for a in range(4):
                for b in range(4):
                    coef[:, a] += ch[b] * (cols_all[:, b] == cols_all[:, a])
            for a in range(1, 4):
                for b in range(a):
                    coef[:, a] *= (cols_all[:, a] != cols_all[:, b])
            coef = coef / 4.0
            norm = np.sqrt((coef * coef).sum(axis=1) * 1.0)
            keep = norm > 1e-12
            if not keep.any():
                continue
            cols_all = cols_all[keep]
            coef = coef[keep] / norm[keep][:, None]
            nb = cols_all.shape[0]
            HB = np.zeros((D, nb))
            for a in range(4):
                HB += H[:, cols_all[:, a]] * coef[:, a][None, :]
            Hb = np.zeros((nb, nb))
            for a in range(4):
                Hb += coef[:, a][:, None] * HB[cols_all[:, a], :]
            Hb = 0.5 * (Hb + Hb.T)
            k = min(n_states, nb)
            wb, vb = eigh(Hb, driver="evr", subset_by_index=[0, k - 1])
            V = np.zeros((D, k))
            for a in range(4):
                np.add.at(V, cols_all[:, a], coef[:, a][:, None] * vb)
            ws.append(wb)
            vs.append(V)
    w = np.concatenate(ws)
    v = np.hstack(vs)
    order = np.argsort(w, kind="stable")[:n_states]
    return w[order], v[:, order]

def _labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states):
    """Lowest n_states eigenpairs of the half-filled Sz = 0 sector with labels
    [E_k - E_0, S(S+1), p_k p_0, j_k j_0]; p = site reversal, j = occupation complement."""
    n = N // 2
    H = _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n, n)
    if n_states > H.shape[0]:
        raise ValueError("n_states exceeds the dimension of the half-filled Sz = 0 sector")
    w, v = _lowest_states(H, N, n, n_states)
    v, sl = _resolve(N, n, n, w, v)
    lab = np.column_stack([w - w[0], sl[:, 0], sl[:, 1] * sl[0, 1], sl[:, 2] * sl[0, 2]])
    return w, v, lab, H

def _pick(lab, s2_target, p_target, j_target, order):
    sel = [k for k in range(lab.shape[0]) if abs(lab[k, 1] - s2_target) < 1e-6
           and abs(lab[k, 2] - p_target) < 1e-6 and abs(lab[k, 3] - j_target) < 1e-6]
    if len(sel) <= order:
        raise ValueError("requested state not found among the computed eigenstates; raise n_states")
    return sel[order]

def _dark_states(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states):
    """Indices and vectors of 1^1Ag+, 2^1Ag+, 1^1Bu-, 1^1Bu+, 1^3Bu, 1^5Ag+ (covalent label:
    singlets and quintets share the ground-state complement eigenvalue, covalent triplets
    carry the opposite one)."""
    w, v, lab, H = _labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
    k_gs = _pick(lab, 0.0, 1.0, 1.0, 0)
    if k_gs != 0:
        raise ValueError("ground state is not a totally symmetric covalent singlet")
    k_2ag = _pick(lab, 0.0, 1.0, 1.0, 1)
    k_bum = _pick(lab, 0.0, -1.0, -1.0, 0)
    k_bup = _pick(lab, 0.0, -1.0, 1.0, 0)
    k_t = _pick(lab, 2.0, -1.0, -1.0, 0)
    k_q = _pick(lab, 6.0, 1.0, 1.0, 0)
    return w, v, lab, H, dict(gs=k_gs, ag2=k_2ag, bum=k_bum, bup=k_bup, t1=k_t, q1=k_q)

def _subchain_family(m, U, eps, t0, delta, r_double, r_single, angle_deg, all_covalent=False):
    """Covalent triplets of an isolated open subchain of 2m sites: the triplet eigenstates
    (S(S+1) = 2) whose complement eigenvalue is opposite to that of the subchain ground state,
    the m lowest of them (the family) unless all_covalent. Returns (energies, T0 vectors,
    T+1 vectors, T-1 vectors, three sectors, reversal parities)."""
    Ns = 2 * m
    cu, cd, u, d = _sector(Ns, m, m)
    H = _hamiltonian(Ns, U, eps, t0, delta, r_double, r_single, angle_deg, m, m)
    w, v = np.linalg.eigh(H)
    v, sl = _resolve(Ns, m, m, w, v)
    sel = [k for k in range(len(w)) if abs(sl[k, 0] - 2.0) < 1e-6 and abs(sl[k, 2] * sl[0, 2] + 1.0) < 1e-6]
    if len(sel) < m:
        raise ValueError("fewer covalent triplets than family members")
    if not all_covalent:
        sel = sel[:m]
    cu_p, cd_p, u_p, d_p = _sector(Ns, m + 1, m - 1)
    cu_m, cd_m, u_m, d_m = _sector(Ns, m - 1, m + 1)
    sp = _splus(Ns, cu, cd, u, d, cu_p, cd_p)
    sm = _sminus(Ns, cu, cd, u, d, cu_m, cd_m)
    t0v = v[:, sel]
    tp = _apply(sp, t0v, len(u_p)) / sqrt(2.0)
    tm = _apply(sm, t0v, len(u_m)) / sqrt(2.0)
    return w[sel] - w[0], t0v, tp, tm, (cu, cd, u, d), (cu_p, cd_p, u_p, d_p), (cu_m, cd_m, u_m, d_m), sl[sel, 1] * sl[0, 1]

def _site_sign(N, u, d):
    """Parity converting a site-ordered operator string (site 0 up, site 0 down, site 1 up, ...)
    into the all-up-then-all-down ordering: (-1)^(sum over occupied down sites i of the number
    of occupied up sites above i)."""
    pc = _popcount_table(N)
    s = np.zeros_like(u)
    for i in range(N):
        s += ((d >> i) & 1) * pc[u >> (i + 1)]
    return (-1.0) ** (s % 2)

def _jw_product(N, NL, secL, vecL, secR, vecR, cu, cd):
    """Site-ordered (Jordan-Wigner) tensor product of a left-subchain state (sites 0..NL-1) and
    a right-subchain state (sites NL..N-1), returned in the full-chain sector basis (cu, cd)."""
    cuL, cdL, uL, dL = secL
    cuR, cdR, uR, dR = secR
    NR = N - NL
    sL = _site_sign(NL, uL, dL)
    sR = _site_sign(NR, uR, dR)
    u = (uL[:, None] | (uR[None, :] << NL)).ravel()
    d = (dL[:, None] | (dR[None, :] << NL)).ravel()
    coef = ((vecL * sL)[:, None] * (vecR * sR)[None, :]).ravel() * _site_sign(N, u, d)
    out = np.zeros(len(cu) * len(cd))
    np.add.at(out, _index(cu, cd, u, d), coef)
    return out

def triplet_pair_product(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int, m: int, j: int, k: int) -> np.ndarray:
    _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg)
    Nd = N // 2
    if not (isinstance(m, (int, np.integer)) and not isinstance(m, bool)) or m < 1 or m > Nd - 1:
        raise ValueError("m must be an integer between 1 and N/2 - 1")
    if not (0 <= j < m) or not (0 <= k < Nd - m):
        raise ValueError("family indices out of range: 0 <= j < m and 0 <= k < N/2 - m")
    w, v, lab, H, ks = _dark_states(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
    n = N // 2
    cu, cd, u, d = _sector(N, n, n)
    eL, t0L, tpL, tmL, s0L, spL, smL, pL = _subchain_family(m, U, eps, t0, delta, r_double, r_single, angle_deg)
    eR, t0R, tpR, tmR, s0R, spR, smR, pR = _subchain_family(Nd - m, U, eps, t0, delta, r_double, r_single, angle_deg)
    NL = 2 * m
    v00 = _jw_product(N, NL, s0L, t0L[:, j], s0R, t0R[:, k], cu, cd)
    vpm = _jw_product(N, NL, spL, tpL[:, j], smR, tmR[:, k], cu, cd)
    vmp = _jw_product(N, NL, smL, tmL[:, j], spR, tpR[:, k], cu, cd)
    vS = (vpm - v00 + vmp) / sqrt(3.0)
    vQ = (vpm + 2.0 * v00 + vmp) / sqrt(6.0)
    psi = v[:, ks["ag2"]]
    psq = v[:, ks["q1"]]
    psb = v[:, ks["bup"]]
    psg = v[:, ks["gs"]]
    e0 = w[ks["gs"]]
    return np.array([(v00 @ psi) ** 2, (vS @ psi) ** 2, (v00 @ psq) ** 2, (vQ @ psq) ** 2,
                     (vS @ psb) ** 2, (vS @ psg) ** 2, vS @ (H @ vS) - e0])

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def _positions(N, r_double, r_single, angle_deg):
    """All-trans zigzag geometry: bond i (atom i to atom i + 1) has length r_double for even i
    and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain
    axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg."""
    phi = (180.0 - angle_deg) * pi / 360.0
    pos = np.zeros((N, 2))
    for i in range(N - 1):
        ell = r_double if i % 2 == 0 else r_single
        ang = phi if i % 2 == 0 else -phi
        pos[i + 1] = pos[i] + ell * np.array([cos(ang), sin(ang)])
    return pos

def _ohno(U, eps, r):
    return U / np.sqrt(1.0 + (U * eps * r / 14.397) ** 2)

def _popcount_table(N):
    t = np.zeros(1 << N, dtype=np.int64)
    for b in range(N):
        t[(np.arange(1 << N) >> b) & 1 == 1] += 1
    return t

def _configs(N, k):
    """All bit masks of N bits with exactly k bits set, ascending as integers."""
    allm = np.arange(1 << N, dtype=np.int64)
    return allm[_popcount_table(N)[allm] == k]

def _sector(N, n_up, n_down):
    """Determinant basis of the (n_up, n_down) sector: index = i_up * D_down + i_down, up and
    down configurations ascending as integers; creation operators ordered all-up (ascending
    site) then all-down (ascending site)."""
    cu = _configs(N, n_up)
    cd = _configs(N, n_down)
    u = np.repeat(cu, len(cd))
    d = np.tile(cd, len(cu))
    return cu, cd, u, d

def _index(cu, cd, u, d):
    iu = np.searchsorted(cu, u)
    idn = np.searchsorted(cd, d)
    return iu * len(cd) + idn

def _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down):
    cu, cd, u, d = _sector(N, n_up, n_down)
    D = len(u)
    pos = _positions(N, r_double, r_single, angle_deg)
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    V = _ohno(U, eps, r)
    np.fill_diagonal(V, 0.0)
    occ_u = ((u[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    occ_d = ((d[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    q = occ_u + occ_d - 1.0
    diag = U * ((occ_u - 0.5) * (occ_d - 0.5)).sum(axis=1) + 0.5 * np.einsum("ki,ij,kj->k", q, V, q)
    H = np.zeros((D, D))
    H[np.arange(D), np.arange(D)] = diag
    cols = np.arange(D)
    for i in range(N - 1):
        t = t0 * (1.0 + delta) if i % 2 == 0 else t0 * (1.0 - delta)
        j = i + 1
        for a, b in ((i, j), (j, i)):
            ma, mb = 1 << a, 1 << b
            for spin in (0, 1):
                occ = u if spin == 0 else d
                sel = ((occ & ma) != 0) & ((occ & mb) == 0)
                new = occ[sel] ^ ma ^ mb
                if spin == 0:
                    rows = _index(cu, cd, new, d[sel])
                else:
                    rows = _index(cu, cd, u[sel], new)
                H[rows, cols[sel]] += -t
    return H

def _splus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S+ = sum_i c+_{i up} c_{i down} from the sector of (u, d) to the sector
    (cu2, cd2) with one more up electron, as (rows, cols, vals)."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((d & m) != 0) & ((u & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up + pc[dd & below] + pc[uu & below])
        rows.append(_index(cu2, cd2, uu | m, dd ^ m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _sminus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S- = sum_i c+_{i down} c_{i up}."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((u & m) != 0) & ((d & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up - 1 + pc[uu & below] + pc[dd & below])
        rows.append(_index(cu2, cd2, uu ^ m, dd | m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _apply(op, x, D_out):
    rows, cols, vals = op
    y = np.zeros((D_out,) + x.shape[1:])
    np.add.at(y, rows, vals[:, None] * x[cols] if x.ndim == 2 else vals * x[cols])
    return y

def _reverse_bits(x, N):
    y = np.zeros_like(x)
    for i in range(N):
        y |= ((x >> i) & 1) << (N - 1 - i)
    return y

def _resolve(N, n_up, n_down, w, v, tol=1e-7):
    """Rotate eigenvectors inside every (near-)degenerate energy cluster so that S^2, the site
    reversal P and the occupation complement J are simultaneously diagonal, order the members
    of a cluster by (S(S+1), p, j) ascending and return (vectors, [S(S+1), p, j] per state)."""
    cu, cd, u, d = _sector(N, n_up, n_down)
    full = (1 << N) - 1
    ip = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ij = _index(cu, cd, full ^ u, full ^ d)
    sz = 0.5 * (n_up - n_down)
    if n_up < N:
        cu2, cd2, u2, d2 = _sector(N, n_up + 1, n_down - 1)
        sp = _splus(N, cu, cd, u, d, cu2, cd2)
        sm = _sminus(N, cu2, cd2, u2, d2, cu, cd)
        D2 = len(u2)

    def _s2_apply(X):
        if n_up == N:
            return sz * (sz + 1.0) * X
        return _apply(sm, _apply(sp, X, D2), len(u)) + sz * (sz + 1.0) * X

    ops = (_s2_apply, lambda X: X[ip], lambda X: X[ij])
    v = v.copy()
    lab = np.zeros((len(w), 3))
    scale = max(1.0, float(np.max(np.abs(w))))
    start = 0
    while start < len(w):
        stop = start + 1
        while stop < len(w) and w[stop] - w[stop - 1] < tol * scale:
            stop += 1
        groups = [list(range(start, stop))]
        for op in ops:
            newg = []
            for g in groups:
                X = v[:, g]
                M = X.T @ op(X)
                M = 0.5 * (M + M.T)
                ev, Q = np.linalg.eigh(M)
                X = X @ Q
                v[:, g] = X
                cut = [0] + [t for t in range(1, len(g)) if ev[t] - ev[t - 1] > 1e-6] + [len(g)]
                newg += [[g[t] for t in range(cut[a], cut[a + 1])] for a in range(len(cut) - 1)]
            groups = newg
        X = v[:, start:stop]
        vals = np.column_stack([np.einsum("ik,ik->k", X, op(X)) for op in ops])
        keys = np.round(vals, 6)  # order degenerate members by their integer labels, not by rounding noise
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        v[:, start:stop] = X[:, order]
        lab[start:stop] = vals[order]
        start = stop
    return v, lab

def _subchain_family(m, U, eps, t0, delta, r_double, r_single, angle_deg, all_covalent=False):
    """Covalent triplets of an isolated open subchain of 2m sites: the triplet eigenstates
    (S(S+1) = 2) whose complement eigenvalue is opposite to that of the subchain ground state,
    the m lowest of them (the family) unless all_covalent. Returns (energies, T0 vectors,
    T+1 vectors, T-1 vectors, three sectors, reversal parities)."""
    Ns = 2 * m
    cu, cd, u, d = _sector(Ns, m, m)
    H = _hamiltonian(Ns, U, eps, t0, delta, r_double, r_single, angle_deg, m, m)
    w, v = np.linalg.eigh(H)
    v, sl = _resolve(Ns, m, m, w, v)
    sel = [k for k in range(len(w)) if abs(sl[k, 0] - 2.0) < 1e-6 and abs(sl[k, 2] * sl[0, 2] + 1.0) < 1e-6]
    if len(sel) < m:
        raise ValueError("fewer covalent triplets than family members")
    if not all_covalent:
        sel = sel[:m]
    cu_p, cd_p, u_p, d_p = _sector(Ns, m + 1, m - 1)
    cu_m, cd_m, u_m, d_m = _sector(Ns, m - 1, m + 1)
    sp = _splus(Ns, cu, cd, u, d, cu_p, cd_p)
    sm = _sminus(Ns, cu, cd, u, d, cu_m, cd_m)
    t0v = v[:, sel]
    tp = _apply(sp, t0v, len(u_p)) / sqrt(2.0)
    tm = _apply(sm, t0v, len(u_m)) / sqrt(2.0)
    return w[sel] - w[0], t0v, tp, tm, (cu, cd, u, d), (cu_p, cd_p, u_p, d_p), (cu_m, cd_m, u_m, d_m), sl[sel, 1] * sl[0, 1]

def _site_sign(N, u, d):
    """Parity converting a site-ordered operator string (site 0 up, site 0 down, site 1 up, ...)
    into the all-up-then-all-down ordering: (-1)^(sum over occupied down sites i of the number
    of occupied up sites above i)."""
    pc = _popcount_table(N)
    s = np.zeros_like(u)
    for i in range(N):
        s += ((d >> i) & 1) * pc[u >> (i + 1)]
    return (-1.0) ** (s % 2)

def _jw_product(N, NL, secL, vecL, secR, vecR, cu, cd):
    """Site-ordered (Jordan-Wigner) tensor product of a left-subchain state (sites 0..NL-1) and
    a right-subchain state (sites NL..N-1), returned in the full-chain sector basis (cu, cd)."""
    cuL, cdL, uL, dL = secL
    cuR, cdR, uR, dR = secR
    NR = N - NL
    sL = _site_sign(NL, uL, dL)
    sR = _site_sign(NR, uR, dR)
    u = (uL[:, None] | (uR[None, :] << NL)).ravel()
    d = (dL[:, None] | (dR[None, :] << NL)).ravel()
    coef = ((vecL * sL)[:, None] * (vecR * sR)[None, :]).ravel() * _site_sign(N, u, d)
    out = np.zeros(len(cu) * len(cd))
    np.add.at(out, _index(cu, cd, u, d), coef)
    return out

def _pair_products(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind):
    """Triplet-pair product states of the N-site chain for the requested basis ('paper':
    T_j(m) x T_1(Nd - m); 'family': T_j(m) x T_k(Nd - m) over the two families; 'covalent':
    every pair of covalent triplets of the two subchains): the (m, j, k) list and the three
    coupled products per partition as columns (T0 T0, singlet-coupled, quintet-coupled)."""
    if basis_kind not in ("paper", "family", "covalent"):
        raise ValueError("basis_kind must be 'paper', 'family' or 'covalent'")
    Nd = N // 2
    n = N // 2
    cu, cd, u, d = _sector(N, n, n)
    fam = {}
    for m in range(1, Nd):
        for mm in (m, Nd - m):
            if mm not in fam:
                fam[mm] = _subchain_family(mm, U, eps, t0, delta, r_double, r_single, angle_deg,
                                           basis_kind == "covalent")
    parts = []
    for m in range(1, Nd):
        nL, nR = fam[m][1].shape[1], fam[Nd - m][1].shape[1]
        if basis_kind == "paper":
            parts += [(m, j, 0) for j in range(nL)]
        else:
            parts += [(m, j, k) for j in range(nL) for k in range(nR)]
    c00, cS, cQ = [], [], []
    for (m, j, k) in parts:
        eL, t0L, tpL, tmL, s0L, spL, smL, pL = fam[m]
        eR, t0R, tpR, tmR, s0R, spR, smR, pR = fam[Nd - m]
        NL = 2 * m
        v00 = _jw_product(N, NL, s0L, t0L[:, j], s0R, t0R[:, k], cu, cd)
        vpm = _jw_product(N, NL, spL, tpL[:, j], smR, tmR[:, k], cu, cd)
        vmp = _jw_product(N, NL, smL, tmL[:, j], spR, tpR[:, k], cu, cd)
        c00.append(v00)
        cS.append((vpm - v00 + vmp) / sqrt(3.0))
        cQ.append((vpm + 2.0 * v00 + vmp) / sqrt(6.0))
    return parts, np.array(c00).T, np.array(cS).T, np.array(cQ).T

def pair_basis_gram(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, basis_kind: str) -> np.ndarray:
    _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg)
    if basis_kind not in ("paper", "family", "covalent"):
        raise ValueError("basis_kind must be 'paper', 'family' or 'covalent'")
    if N < 4:
        raise ValueError("a triplet pair needs at least two dimers")
    parts, B00, BS, BQ = _pair_products(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)
    g00 = np.sort(np.linalg.eigvalsh(B00.T @ B00))[::-1]
    gS = np.sort(np.linalg.eigvalsh(BS.T @ BS))[::-1]
    return np.vstack([g00, gS])

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _EPS_RANK():
    return 1e-8

def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def _positions(N, r_double, r_single, angle_deg):
    """All-trans zigzag geometry: bond i (atom i to atom i + 1) has length r_double for even i
    and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain
    axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg."""
    phi = (180.0 - angle_deg) * pi / 360.0
    pos = np.zeros((N, 2))
    for i in range(N - 1):
        ell = r_double if i % 2 == 0 else r_single
        ang = phi if i % 2 == 0 else -phi
        pos[i + 1] = pos[i] + ell * np.array([cos(ang), sin(ang)])
    return pos

def _ohno(U, eps, r):
    return U / np.sqrt(1.0 + (U * eps * r / 14.397) ** 2)

def _popcount_table(N):
    t = np.zeros(1 << N, dtype=np.int64)
    for b in range(N):
        t[(np.arange(1 << N) >> b) & 1 == 1] += 1
    return t

def _configs(N, k):
    """All bit masks of N bits with exactly k bits set, ascending as integers."""
    allm = np.arange(1 << N, dtype=np.int64)
    return allm[_popcount_table(N)[allm] == k]

def _sector(N, n_up, n_down):
    """Determinant basis of the (n_up, n_down) sector: index = i_up * D_down + i_down, up and
    down configurations ascending as integers; creation operators ordered all-up (ascending
    site) then all-down (ascending site)."""
    cu = _configs(N, n_up)
    cd = _configs(N, n_down)
    u = np.repeat(cu, len(cd))
    d = np.tile(cd, len(cu))
    return cu, cd, u, d

def _index(cu, cd, u, d):
    iu = np.searchsorted(cu, u)
    idn = np.searchsorted(cd, d)
    return iu * len(cd) + idn

def _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down):
    cu, cd, u, d = _sector(N, n_up, n_down)
    D = len(u)
    pos = _positions(N, r_double, r_single, angle_deg)
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    V = _ohno(U, eps, r)
    np.fill_diagonal(V, 0.0)
    occ_u = ((u[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    occ_d = ((d[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    q = occ_u + occ_d - 1.0
    diag = U * ((occ_u - 0.5) * (occ_d - 0.5)).sum(axis=1) + 0.5 * np.einsum("ki,ij,kj->k", q, V, q)
    H = np.zeros((D, D))
    H[np.arange(D), np.arange(D)] = diag
    cols = np.arange(D)
    for i in range(N - 1):
        t = t0 * (1.0 + delta) if i % 2 == 0 else t0 * (1.0 - delta)
        j = i + 1
        for a, b in ((i, j), (j, i)):
            ma, mb = 1 << a, 1 << b
            for spin in (0, 1):
                occ = u if spin == 0 else d
                sel = ((occ & ma) != 0) & ((occ & mb) == 0)
                new = occ[sel] ^ ma ^ mb
                if spin == 0:
                    rows = _index(cu, cd, new, d[sel])
                else:
                    rows = _index(cu, cd, u[sel], new)
                H[rows, cols[sel]] += -t
    return H

def _splus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S+ = sum_i c+_{i up} c_{i down} from the sector of (u, d) to the sector
    (cu2, cd2) with one more up electron, as (rows, cols, vals)."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((d & m) != 0) & ((u & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up + pc[dd & below] + pc[uu & below])
        rows.append(_index(cu2, cd2, uu | m, dd ^ m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _sminus(N, cu, cd, u, d, cu2, cd2):
    """Sparse data of S- = sum_i c+_{i down} c_{i up}."""
    pc = _popcount_table(N)
    rows, cols, vals = [], [], []
    n_up = pc[cu[0]] if len(cu) else 0
    for i in range(N):
        m = 1 << i
        below = m - 1
        sel = ((u & m) != 0) & ((d & m) == 0)
        if not sel.any():
            continue
        uu, dd = u[sel], d[sel]
        sign = (-1.0) ** (n_up - 1 + pc[uu & below] + pc[dd & below])
        rows.append(_index(cu2, cd2, uu ^ m, dd | m))
        cols.append(np.nonzero(sel)[0])
        vals.append(sign)
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def _apply(op, x, D_out):
    rows, cols, vals = op
    y = np.zeros((D_out,) + x.shape[1:])
    np.add.at(y, rows, vals[:, None] * x[cols] if x.ndim == 2 else vals * x[cols])
    return y

def _reverse_bits(x, N):
    y = np.zeros_like(x)
    for i in range(N):
        y |= ((x >> i) & 1) << (N - 1 - i)
    return y

def _resolve(N, n_up, n_down, w, v, tol=1e-7):
    """Rotate eigenvectors inside every (near-)degenerate energy cluster so that S^2, the site
    reversal P and the occupation complement J are simultaneously diagonal, order the members
    of a cluster by (S(S+1), p, j) ascending and return (vectors, [S(S+1), p, j] per state)."""
    cu, cd, u, d = _sector(N, n_up, n_down)
    full = (1 << N) - 1
    ip = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ij = _index(cu, cd, full ^ u, full ^ d)
    sz = 0.5 * (n_up - n_down)
    if n_up < N:
        cu2, cd2, u2, d2 = _sector(N, n_up + 1, n_down - 1)
        sp = _splus(N, cu, cd, u, d, cu2, cd2)
        sm = _sminus(N, cu2, cd2, u2, d2, cu, cd)
        D2 = len(u2)

    def _s2_apply(X):
        if n_up == N:
            return sz * (sz + 1.0) * X
        return _apply(sm, _apply(sp, X, D2), len(u)) + sz * (sz + 1.0) * X

    ops = (_s2_apply, lambda X: X[ip], lambda X: X[ij])
    v = v.copy()
    lab = np.zeros((len(w), 3))
    scale = max(1.0, float(np.max(np.abs(w))))
    start = 0
    while start < len(w):
        stop = start + 1
        while stop < len(w) and w[stop] - w[stop - 1] < tol * scale:
            stop += 1
        groups = [list(range(start, stop))]
        for op in ops:
            newg = []
            for g in groups:
                X = v[:, g]
                M = X.T @ op(X)
                M = 0.5 * (M + M.T)
                ev, Q = np.linalg.eigh(M)
                X = X @ Q
                v[:, g] = X
                cut = [0] + [t for t in range(1, len(g)) if ev[t] - ev[t - 1] > 1e-6] + [len(g)]
                newg += [[g[t] for t in range(cut[a], cut[a + 1])] for a in range(len(cut) - 1)]
            groups = newg
        X = v[:, start:stop]
        vals = np.column_stack([np.einsum("ik,ik->k", X, op(X)) for op in ops])
        keys = np.round(vals, 6)  # order degenerate members by their integer labels, not by rounding noise
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        v[:, start:stop] = X[:, order]
        lab[start:stop] = vals[order]
        start = stop
    return v, lab

def _lowest_states(H, N, n, n_states):
    """Lowest n_states eigenpairs of H in the (n, n) sector, obtained block by block in the
    four symmetry-adapted subspaces of the group generated by the spin flip (u, d) -> (d, u)
    and the site reversal; the eigenvectors are expanded back to the full sector basis."""
    cu, cd, u, d = _sector(N, n, n)
    D = len(u)
    pf = _index(cu, cd, d, u)
    pp = _index(cu, cd, _reverse_bits(u, N), _reverse_bits(d, N))
    ar = np.arange(D)
    orb = np.column_stack([ar, pf, pp, pf[pp]])
    rep = orb.min(axis=1)
    own = np.nonzero(rep == ar)[0]
    ws, vs = [], []
    for chi_f in (1.0, -1.0):
        for chi_p in (1.0, -1.0):
            ch = np.array([1.0, chi_f, chi_p, chi_f * chi_p])
            cols_all = orb[own]
            coef = np.zeros((len(own), 4))
            for a in range(4):
                for b in range(4):
                    coef[:, a] += ch[b] * (cols_all[:, b] == cols_all[:, a])
            for a in range(1, 4):
                for b in range(a):
                    coef[:, a] *= (cols_all[:, a] != cols_all[:, b])
            coef = coef / 4.0
            norm = np.sqrt((coef * coef).sum(axis=1) * 1.0)
            keep = norm > 1e-12
            if not keep.any():
                continue
            cols_all = cols_all[keep]
            coef = coef[keep] / norm[keep][:, None]
            nb = cols_all.shape[0]
            HB = np.zeros((D, nb))
            for a in range(4):
                HB += H[:, cols_all[:, a]] * coef[:, a][None, :]
            Hb = np.zeros((nb, nb))
            for a in range(4):
                Hb += coef[:, a][:, None] * HB[cols_all[:, a], :]
            Hb = 0.5 * (Hb + Hb.T)
            k = min(n_states, nb)
            wb, vb = eigh(Hb, driver="evr", subset_by_index=[0, k - 1])
            V = np.zeros((D, k))
            for a in range(4):
                np.add.at(V, cols_all[:, a], coef[:, a][:, None] * vb)
            ws.append(wb)
            vs.append(V)
    w = np.concatenate(ws)
    v = np.hstack(vs)
    order = np.argsort(w, kind="stable")[:n_states]
    return w[order], v[:, order]

def _labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states):
    """Lowest n_states eigenpairs of the half-filled Sz = 0 sector with labels
    [E_k - E_0, S(S+1), p_k p_0, j_k j_0]; p = site reversal, j = occupation complement."""
    n = N // 2
    H = _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n, n)
    if n_states > H.shape[0]:
        raise ValueError("n_states exceeds the dimension of the half-filled Sz = 0 sector")
    w, v = _lowest_states(H, N, n, n_states)
    v, sl = _resolve(N, n, n, w, v)
    lab = np.column_stack([w - w[0], sl[:, 0], sl[:, 1] * sl[0, 1], sl[:, 2] * sl[0, 2]])
    return w, v, lab, H

def _pick(lab, s2_target, p_target, j_target, order):
    sel = [k for k in range(lab.shape[0]) if abs(lab[k, 1] - s2_target) < 1e-6
           and abs(lab[k, 2] - p_target) < 1e-6 and abs(lab[k, 3] - j_target) < 1e-6]
    if len(sel) <= order:
        raise ValueError("requested state not found among the computed eigenstates; raise n_states")
    return sel[order]

def _dark_states(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states):
    """Indices and vectors of 1^1Ag+, 2^1Ag+, 1^1Bu-, 1^1Bu+, 1^3Bu, 1^5Ag+ (covalent label:
    singlets and quintets share the ground-state complement eigenvalue, covalent triplets
    carry the opposite one)."""
    w, v, lab, H = _labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
    k_gs = _pick(lab, 0.0, 1.0, 1.0, 0)
    if k_gs != 0:
        raise ValueError("ground state is not a totally symmetric covalent singlet")
    k_2ag = _pick(lab, 0.0, 1.0, 1.0, 1)
    k_bum = _pick(lab, 0.0, -1.0, -1.0, 0)
    k_bup = _pick(lab, 0.0, -1.0, 1.0, 0)
    k_t = _pick(lab, 2.0, -1.0, -1.0, 0)
    k_q = _pick(lab, 6.0, 1.0, 1.0, 0)
    return w, v, lab, H, dict(gs=k_gs, ag2=k_2ag, bum=k_bum, bup=k_bup, t1=k_t, q1=k_q)

def _subchain_family(m, U, eps, t0, delta, r_double, r_single, angle_deg, all_covalent=False):
    """Covalent triplets of an isolated open subchain of 2m sites: the triplet eigenstates
    (S(S+1) = 2) whose complement eigenvalue is opposite to that of the subchain ground state,
    the m lowest of them (the family) unless all_covalent. Returns (energies, T0 vectors,
    T+1 vectors, T-1 vectors, three sectors, reversal parities)."""
    Ns = 2 * m
    cu, cd, u, d = _sector(Ns, m, m)
    H = _hamiltonian(Ns, U, eps, t0, delta, r_double, r_single, angle_deg, m, m)
    w, v = np.linalg.eigh(H)
    v, sl = _resolve(Ns, m, m, w, v)
    sel = [k for k in range(len(w)) if abs(sl[k, 0] - 2.0) < 1e-6 and abs(sl[k, 2] * sl[0, 2] + 1.0) < 1e-6]
    if len(sel) < m:
        raise ValueError("fewer covalent triplets than family members")
    if not all_covalent:
        sel = sel[:m]
    cu_p, cd_p, u_p, d_p = _sector(Ns, m + 1, m - 1)
    cu_m, cd_m, u_m, d_m = _sector(Ns, m - 1, m + 1)
    sp = _splus(Ns, cu, cd, u, d, cu_p, cd_p)
    sm = _sminus(Ns, cu, cd, u, d, cu_m, cd_m)
    t0v = v[:, sel]
    tp = _apply(sp, t0v, len(u_p)) / sqrt(2.0)
    tm = _apply(sm, t0v, len(u_m)) / sqrt(2.0)
    return w[sel] - w[0], t0v, tp, tm, (cu, cd, u, d), (cu_p, cd_p, u_p, d_p), (cu_m, cd_m, u_m, d_m), sl[sel, 1] * sl[0, 1]

def _site_sign(N, u, d):
    """Parity converting a site-ordered operator string (site 0 up, site 0 down, site 1 up, ...)
    into the all-up-then-all-down ordering: (-1)^(sum over occupied down sites i of the number
    of occupied up sites above i)."""
    pc = _popcount_table(N)
    s = np.zeros_like(u)
    for i in range(N):
        s += ((d >> i) & 1) * pc[u >> (i + 1)]
    return (-1.0) ** (s % 2)

def _jw_product(N, NL, secL, vecL, secR, vecR, cu, cd):
    """Site-ordered (Jordan-Wigner) tensor product of a left-subchain state (sites 0..NL-1) and
    a right-subchain state (sites NL..N-1), returned in the full-chain sector basis (cu, cd)."""
    cuL, cdL, uL, dL = secL
    cuR, cdR, uR, dR = secR
    NR = N - NL
    sL = _site_sign(NL, uL, dL)
    sR = _site_sign(NR, uR, dR)
    u = (uL[:, None] | (uR[None, :] << NL)).ravel()
    d = (dL[:, None] | (dR[None, :] << NL)).ravel()
    coef = ((vecL * sL)[:, None] * (vecR * sR)[None, :]).ravel() * _site_sign(N, u, d)
    out = np.zeros(len(cu) * len(cd))
    np.add.at(out, _index(cu, cd, u, d), coef)
    return out

def _pair_products(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind):
    """Triplet-pair product states of the N-site chain for the requested basis ('paper':
    T_j(m) x T_1(Nd - m); 'family': T_j(m) x T_k(Nd - m) over the two families; 'covalent':
    every pair of covalent triplets of the two subchains): the (m, j, k) list and the three
    coupled products per partition as columns (T0 T0, singlet-coupled, quintet-coupled)."""
    if basis_kind not in ("paper", "family", "covalent"):
        raise ValueError("basis_kind must be 'paper', 'family' or 'covalent'")
    Nd = N // 2
    n = N // 2
    cu, cd, u, d = _sector(N, n, n)
    fam = {}
    for m in range(1, Nd):
        for mm in (m, Nd - m):
            if mm not in fam:
                fam[mm] = _subchain_family(mm, U, eps, t0, delta, r_double, r_single, angle_deg,
                                           basis_kind == "covalent")
    parts = []
    for m in range(1, Nd):
        nL, nR = fam[m][1].shape[1], fam[Nd - m][1].shape[1]
        if basis_kind == "paper":
            parts += [(m, j, 0) for j in range(nL)]
        else:
            parts += [(m, j, k) for j in range(nL) for k in range(nR)]
    c00, cS, cQ = [], [], []
    for (m, j, k) in parts:
        eL, t0L, tpL, tmL, s0L, spL, smL, pL = fam[m]
        eR, t0R, tpR, tmR, s0R, spR, smR, pR = fam[Nd - m]
        NL = 2 * m
        v00 = _jw_product(N, NL, s0L, t0L[:, j], s0R, t0R[:, k], cu, cd)
        vpm = _jw_product(N, NL, spL, tpL[:, j], smR, tmR[:, k], cu, cd)
        vmp = _jw_product(N, NL, smL, tmL[:, j], spR, tpR[:, k], cu, cd)
        c00.append(v00)
        cS.append((vpm - v00 + vmp) / sqrt(3.0))
        cQ.append((vpm + 2.0 * v00 + vmp) / sqrt(6.0))
    return parts, np.array(c00).T, np.array(cS).T, np.array(cQ).T

def _span_population(B, psi):
    """Squared norm of the projection of psi on the span of the columns of B (rank-revealing)."""
    Uo, s, Vt = np.linalg.svd(B, full_matrices=False)
    keep = s > _EPS_RANK() * s[0]
    c = Uo[:, keep].T @ psi
    return float(c @ c)

def triplet_pair_populations(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int, basis_kind: str) -> np.ndarray:
    _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg)
    if basis_kind not in ("paper", "family", "covalent"):
        raise ValueError("basis_kind must be 'paper', 'family' or 'covalent'")
    if N < 4:
        raise ValueError("a triplet pair needs at least two dimers")
    w, v, lab, H, ks = _dark_states(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
    parts, B00, BS, BQ = _pair_products(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)
    out = np.zeros((3, 2))
    for row, key in enumerate(("ag2", "bup")):
        psi = v[:, ks[key]]
        out[row, 0] = 3.0 * _span_population(B00, psi)
        out[row, 1] = _span_population(BS, psi)
    psq = v[:, ks["q1"]]
    out[2, 0] = 1.5 * _span_population(B00, psq)
    out[2, 1] = _span_population(BQ, psq)
    return out

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _EPS_RANK():
    return 1e-8

def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def triplet_pair_audit(N: int, U_list: list, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int) -> np.ndarray:
    _check_model(N, 0.0, eps, t0, delta, r_double, r_single, angle_deg)
    if not isinstance(U_list, (list, tuple, np.ndarray)) or len(U_list) < 1:
        raise ValueError("U_list must contain at least one Coulomb parameter")
    if N < 4:
        raise ValueError("a triplet pair needs at least two dimers")
    Nd = N // 2
    rows = []
    for U in U_list:
        U = float(U)
        V = coulomb_matrix(N, U, eps, r_double, r_single, angle_deg)
        Hs = ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, Nd, Nd)
        lab = symmetry_labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
        en = dark_state_energies(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
        fam = triplet_family(Nd - 1, U, eps, t0, delta, r_double, r_single, angle_deg)
        prod = triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, 1, 0, 0)
        pops = []
        grams = []
        for kind in ("covalent", "family", "paper"):
            pk = triplet_pair_populations(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, kind)
            gk = pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, kind)
            pops.append(pk)
            grams.append([gk.shape[1], gk[1, 0], gk[1, -1], float(np.sum(gk[1] > _EPS_RANK() * gk[1, 0]))])
        pops = np.array(pops)
        ncov = float(np.sum((np.abs(lab[:, 1] - 2.0) < 1e-6) & (np.abs(lab[:, 3] + 1.0) < 1e-6) & (lab[:, 0] < en[0])))
        row = [pops[0, 0, 1], pops[0, 0, 0], pops[1, 0, 1], pops[1, 0, 0], pops[2, 0, 1], pops[2, 0, 0],
               pops[0, 1, 1], pops[0, 1, 0], pops[1, 1, 1], pops[1, 1, 0], pops[2, 1, 1], pops[2, 1, 0],
               pops[0, 2, 1], pops[0, 2, 0], pops[1, 2, 1], pops[1, 2, 0], pops[2, 2, 1], pops[2, 2, 0]]
        row += list(en)
        row += [V[0, 1], V[0, N - 1], float(np.trace(Hs)) / Hs.shape[0], ncov, fam[0, 0], fam[-1, 0], fam[0, 2]]
        row += [prod[0], prod[1], prod[5], prod[6]]
        for g in grams:
            row += g
        rows.append(row)
    return np.array(rows, dtype=float)
SCICODE_GOLD_EOF
