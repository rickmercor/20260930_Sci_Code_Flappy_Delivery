"""
For the single triplet-pair product built from the family member T_{j+1}(m) of the left subchain of m dimers and T_{k+1}(N_d - m) of the right subchain (0-based j, k), return the (7,) array [|<T0T0|2^1Ag+>|^2, |<1TT|2^1Ag+>|^2, |<T0T0|1^5Ag+>|^2, |<2TT|1^5Ag+>|^2, |<1TT|1^1Bu+>|^2, |<1TT|1^1Ag+>|^2, <1TT|H|1TT> - E_0], the target states being identified among the n_states lowest eigenstates of the full chain. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid parameters, if m is not an integer between 1 and N_d - 1, or if j or k is outside its family (0 <= j < m, 0 <= k < N_d - m).

For one product state the spin-coupled overlap of a singlet is exactly three times the T0T0 overlap and that of a quintet exactly three halves of it; the same product's energy expectation is the diabatic energy of an unrelaxed triplet pair, from which the coupling to the charge-transfer configurations pulls the dark state down.

Returns
-------
A (7,) float64 array.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def triplet_pair_product(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int, m: int, j: int, k: int) -> np.ndarray:
    """For the single triplet-pair product built from the family member T_{j+1}(m) of the left
    subchain of m dimers and T_{k+1}(N_d - m) of the right subchain (0-based j, k), return
    the (7,) array [|<T0T0|2^1Ag+>|^2, |<1TT|2^1Ag+>|^2, |<T0T0|1^5Ag+>|^2,
    |<2TT|1^5Ag+>|^2, |<1TT|1^1Bu+>|^2, |<1TT|1^1Ag+>|^2, <1TT|H|1TT> - E_0], the target
    states being identified among the n_states lowest eigenstates of the full chain.

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U : float
        On-site Coulomb parameter in eV (>= 0).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).
    n_states : int
        Number of lowest eigenstates of the half-filled S_z = 0 sector to compute (1 <= n_states <= sector dimension) and to search for the named states.
    m : int
        Number of dimers of the (left) segment (1 <= m; for products 1 <= m <= N/2 - 1).
    j : int
        Zero-based family index on the left segment (0 <= j < m).
    k : int
        Zero-based family index on the right segment (0 <= k < N/2 - m).

    Returns
    -------
    A (7,) float64 array.

    Raises
    ------
    ValueError
        For invalid parameters, if m is not an integer between 1 and N_d - 1, or if j or
        k is outside its family (0 <= j < m, 0 <= k < N_d - m).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

def _oracle_triplet_pair_product(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int, m: int, j: int, k: int) -> np.ndarray:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nN = 8\nU = 8.0\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.35\nr_single = 1.45\nangle_deg = 120.0\nn_states = 100\nm = 2\nj = 1\nk = 0\n',
         'call': 'triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, m, j, k)',
         'gold_call': '_oracle_triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, m, j, k)'},
        {'setup': 'import numpy as np\nN = 6\nU = 4.0\neps = 1.0\nt0 = 2.5\ndelta = 0.1\nr_double = 1.4\nr_single = 1.4\nangle_deg = 180.0\nn_states = 40\nm = 1\nj = 0\nk = 1\n',
         'call': 'triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, m, j, k)',
         'gold_call': '_oracle_triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, m, j, k)'},
        {'setup': 'import numpy as np\nN = 4\nU = 14.0\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.34\nr_single = 1.46\nangle_deg = 125.0\nn_states = 36\nm = 1\nj = 0\nk = 0\n',
         'call': 'triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, m, j, k)',
         'gold_call': '_oracle_triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, m, j, k)'},
        {'setup': 'import numpy as np\nN = 8\nU = 4.0\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.35\nr_single = 1.45\nangle_deg = 120.0\nn_states = 100\nm = 3\nj = 2\nk = 0\n',
         'call': 'triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, m, j, k)',
         'gold_call': '_oracle_triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, m, j, k)'},
        {"setup": "import numpy as np\n# invalid input: a zero hopping integral must raise ValueError\ndef _probe(fn):\n    try:\n        fn(8, 8.0, 2.0, 0.0, 1.0 / 12.0, 1.35, 1.45, 120.0, 100, 1, 1, 1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n", "call": "_probe(triplet_pair_product)", "gold_call": "_probe(_oracle_triplet_pair_product)"}
    ]
