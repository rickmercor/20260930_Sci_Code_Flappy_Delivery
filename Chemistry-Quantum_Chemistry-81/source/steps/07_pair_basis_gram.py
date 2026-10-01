"""
Return the (2, n_b) array of the eigenvalues, in descending order, of the Gram (overlap) matrices of the n_b product states of the requested pair basis ('paper', 'family' or 'covalent'): row 0 for the T0T0 products, row 1 for the singlet-coupled products 1TT. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid parameters, if N < 4 or if basis_kind is not one of the three names.

Products on different partitions of the chain overlap, and the overlap of two M = 0 products mixes the singlet-coupled and the quintet-coupled channels with weights 1/3 and 2/3, so the metric of the T0T0 set is not the metric of the spin-adapted set; the Gram spectra make that difference, and the linear dependencies of the complete covalent basis, explicit.

Returns
-------
A (2, n_b) float64 array of descending Gram eigenvalues.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_basis_gram(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, basis_kind: str) -> np.ndarray:
    """Return the (2, n_b) array of the eigenvalues, in descending order, of the Gram (overlap)
    matrices of the n_b product states of the requested pair basis ('paper', 'family' or
    'covalent'): row 0 for the T0T0 products, row 1 for the singlet-coupled products 1TT.
    Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0,
    ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi
    electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma}
    t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) +
    sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 +
    delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i;
    Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in
    angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity.

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
    basis_kind : str
        Pair basis: 'paper', 'family' or 'covalent'.

    Returns
    -------
    A (2, n_b) float64 array of descending Gram eigenvalues.

    Raises
    ------
    ValueError
        For invalid parameters, if N < 4 or if basis_kind is not one of the three names.
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

def _oracle_pair_basis_gram(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, basis_kind: str) -> np.ndarray:
    _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg)
    if basis_kind not in ("paper", "family", "covalent"):
        raise ValueError("basis_kind must be 'paper', 'family' or 'covalent'")
    if N < 4:
        raise ValueError("a triplet pair needs at least two dimers")
    parts, B00, BS, BQ = _pair_products(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)
    g00 = np.sort(np.linalg.eigvalsh(B00.T @ B00))[::-1]
    gS = np.sort(np.linalg.eigvalsh(BS.T @ BS))[::-1]
    return np.vstack([g00, gS])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': "import numpy as np\nN = 8\nU = 8.0\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.35\nr_single = 1.45\nangle_deg = 120.0\nn_states = 100\nbasis_kind = 'paper'\n",
         'call': 'pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)',
         'gold_call': '_oracle_pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)'},
        {'setup': "import numpy as np\nN = 8\nU = 8.0\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.35\nr_single = 1.45\nangle_deg = 120.0\nn_states = 100\nbasis_kind = 'family'\n",
         'call': 'pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)',
         'gold_call': '_oracle_pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)'},
        {'setup': "import numpy as np\nN = 6\nU = 4.0\neps = 1.0\nt0 = 2.5\ndelta = 0.1\nr_double = 1.4\nr_single = 1.4\nangle_deg = 180.0\nn_states = 40\nbasis_kind = 'covalent'\n",
         'call': 'pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)',
         'gold_call': '_oracle_pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)'},
        {'setup': "import numpy as np\nN = 4\nU = 14.0\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.34\nr_single = 1.46\nangle_deg = 125.0\nn_states = 36\nbasis_kind = 'family'\n",
         'call': 'pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)',
         'gold_call': '_oracle_pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, basis_kind)'},
        {"setup": "import numpy as np\n# invalid input: a negative bond length must raise ValueError\ndef _probe(fn):\n    try:\n        fn(8, 8.0, 2.0, 2.4, 1.0 / 12.0, -1.35, 1.45, 120.0, 'paper')\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n", "call": "_probe(pair_basis_gram)", "gold_call": "_probe(_oracle_pair_basis_gram)"}
    ]
