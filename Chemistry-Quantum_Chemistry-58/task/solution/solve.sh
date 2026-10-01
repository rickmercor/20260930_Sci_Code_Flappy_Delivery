#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _check_even_chain(n_sites):
    if not isinstance(n_sites, (int, np.integer)) or isinstance(n_sites, bool):
        raise ValueError("n_sites must be an integer")
    if n_sites < 2 or n_sites % 2 != 0:
        raise ValueError("n_sites must be an even integer >= 2")


def ohno_interaction_matrix(n_sites: int, bond_double: float, bond_single: float, bond_angle_deg: float, U: float, eps_r: float) -> np.ndarray:
    _check_even_chain(n_sites)
    for name, x in (("bond_double", bond_double), ("bond_single", bond_single)):
        if not (np.isfinite(x) and x > 0.0):
            raise ValueError(name + " must be positive")
    if not (np.isfinite(bond_angle_deg) and 0.0 < bond_angle_deg <= 180.0):
        raise ValueError("bond_angle_deg must lie in (0, 180]")
    if not (np.isfinite(U) and U > 0.0):
        raise ValueError("U must be positive")
    if not (np.isfinite(eps_r) and eps_r > 0.0):
        raise ValueError("eps_r must be positive")
    beta = math.radians(180.0 - bond_angle_deg) / 2.0
    pos = np.zeros((n_sites, 2))
    for k in range(n_sites - 1):
        L = bond_double if k % 2 == 0 else bond_single
        ang = beta if k % 2 == 0 else -beta
        pos[k + 1] = pos[k] + L * np.array([math.cos(ang), math.sin(ang)])
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    V = U / np.sqrt(1.0 + (U * eps_r * r / 14.397) ** 2)
    np.fill_diagonal(V, 0.0)
    return V

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _strings(n_sites, n_occ):
    """All occupation bit-strings of n_sites sites holding n_occ electrons of one spin, ascending."""
    return [sum(1 << i for i in c) for c in itertools.combinations(range(n_sites), n_occ)]


def _hop_matrix(strings, n_sites, t):
    """One-spin hopping matrix -sum_k t_k (c+_k c_{k+1} + h.c.) on the given bit-strings (creation
    operators ordered by ascending site index)."""
    idx = {s: k for k, s in enumerate(strings)}
    rows, cols, vals = [], [], []
    for k, s in enumerate(strings):
        for bond in range(n_sites - 1):
            for p, q in ((bond, bond + 1), (bond + 1, bond)):
                if (s >> q) & 1 and not (s >> p) & 1:
                    s2 = s ^ (1 << q) ^ (1 << p)
                    lo, hi = min(p, q), max(p, q)
                    between = bin(s & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))).count("1")
                    rows.append(idx[s2]); cols.append(k); vals.append(-t[bond] * (-1) ** between)
    return sp.coo_matrix((vals, (rows, cols)), shape=(len(strings), len(strings))).tocsr()


def _sector(n_sites, n_up, n_down):
    """Determinant basis of one (n_up, n_down) sector: spin strings, index maps, occupations, as a namespace."""
    su = _strings(n_sites, n_up)
    sd = su if n_down == n_up else _strings(n_sites, n_down)
    occu = np.array([[(s >> i) & 1 for i in range(n_sites)] for s in su], dtype=float)
    occd = occu if sd is su else np.array([[(s >> i) & 1 for i in range(n_sites)] for s in sd], dtype=float)
    return types.SimpleNamespace(n=n_sites, n_up=n_up, n_down=n_down, su=su, sd=sd,
                                 iu={s: k for k, s in enumerate(su)}, id={s: k for k, s in enumerate(sd)},
                                 nu=len(su), nd=len(sd), dim=len(su) * len(sd), occu=occu, occd=occd)


def _hoppings(n_sites, t0, delta):
    return np.array([t0 * (1.0 + delta) if k % 2 == 0 else t0 * (1.0 - delta) for k in range(n_sites - 1)])


def _hamiltonian(sec, t0, delta, V, U):
    n = sec.n
    t = _hoppings(n, t0, delta)
    Hu = _hop_matrix(sec.su, n, t); Hd = _hop_matrix(sec.sd, n, t)
    H = sp.kron(Hu, sp.identity(sec.nd, format="csr"), format="csr") + sp.kron(sp.identity(sec.nu, format="csr"), Hd, format="csr")
    X = sec.occu - 0.5; Y = sec.occd - 0.5
    Vm = np.asarray(V, dtype=float)
    onsite = U * (X @ Y.T)
    xx = np.einsum("ai,ij,aj->a", X, Vm, X); yy = np.einsum("bi,ij,bj->b", Y, Vm, Y); xy = X @ Vm @ Y.T
    diag = onsite + 0.5 * (xx[:, None] + yy[None, :]) + xy
    H = H + sp.diags(diag.ravel(), format="csr")
    return H


def _lowest(H, k, dim):
    """Lowest k eigenpairs, ascending; dense below 1500, ARPACK above (fixed start vector)."""
    k = int(min(k, dim))
    if dim <= 1500:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    kk = min(k, dim - 2)
    w, v = spla.eigsh(H, k=kk, which="SA", v0=np.ones(dim), tol=1e-13, ncv=min(dim, max(3 * kk + 20, 60)))
    o = np.argsort(w)
    return w[o], v[:, o]


def _validate_model(n_sites, t0, delta, V, U):
    _check_even_chain(n_sites)
    Vm = np.asarray(V, dtype=float)
    if Vm.ndim != 2 or Vm.shape != (n_sites, n_sites):
        raise ValueError("V must be an (n_sites, n_sites) array")
    if not np.allclose(Vm, Vm.T, rtol=0.0, atol=1e-12):
        raise ValueError("V must be symmetric")
    if not (np.isfinite(t0) and t0 > 0.0):
        raise ValueError("t0 must be positive")
    if not (np.isfinite(delta) and -1.0 < delta < 1.0):
        raise ValueError("delta must lie in (-1, 1)")
    if not (np.isfinite(U) and U > 0.0):
        raise ValueError("U must be positive")
    return Vm


def sector_eigenvalues(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float, n_up: int, n_down: int, n_states: int) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    for name, x in (("n_up", n_up), ("n_down", n_down), ("n_states", n_states)):
        if not isinstance(x, (int, np.integer)) or isinstance(x, bool):
            raise ValueError(name + " must be an integer")
    if not (0 <= n_up <= n_sites and 0 <= n_down <= n_sites):
        raise ValueError("n_up and n_down must lie between 0 and n_sites")
    sec = _sector(n_sites, n_up, n_down)
    if not (1 <= n_states <= sec.dim):
        raise ValueError("n_states must lie between 1 and the sector dimension")
    H = _hamiltonian(sec, t0, delta, Vm, U)
    w, _ = _lowest(H, n_states, sec.dim)
    return np.array(w[:n_states], dtype=float)

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _site_maps(sec_from, sec_to, spin, site, create):
    """Index maps for c+_site (create=True) or c_site (create=False) acting on one spin string set.
    Returns (idx_from, idx_to, sign) for every string of sec_from's spin set that the operator does not kill."""
    strs = sec_from.su if spin == 0 else sec_from.sd
    lookup = sec_to.iu if spin == 0 else sec_to.id
    f, t_, s = [], [], []
    for k, st in enumerate(strs):
        occ = (st >> site) & 1
        if create and occ or (not create and not occ):
            continue
        sgn = (-1) ** bin(st & ((1 << site) - 1)).count("1")
        st2 = st | (1 << site) if create else st ^ (1 << site)
        f.append(k); t_.append(lookup[st2]); s.append(sgn)
    return np.array(f, dtype=int), np.array(t_, dtype=int), np.array(s, dtype=float)


def _s_plus(psi, sec):
    """S+ psi, mapping the (n_up, n_down) sector to (n_up+1, n_down-1). Returns (vector, target sector)."""
    tgt = _sector(sec.n, sec.n_up + 1, sec.n_down - 1)
    P = psi.reshape(sec.nu, sec.nd)
    out = np.zeros((tgt.nu, tgt.nd))
    cross = (-1) ** sec.n_up  # c+_{i up} written left of all up operators passes none; c_{i dn} passes n_up up-operators
    for i in range(sec.n):
        fu, tu, su = _site_maps(sec, tgt, 0, i, True)
        fd, td, sd = _site_maps(sec, tgt, 1, i, False)
        if fu.size == 0 or fd.size == 0:
            continue
        out[np.ix_(tu, td)] += cross * (su[:, None] * sd[None, :]) * P[np.ix_(fu, fd)]
    return out.ravel(), tgt


def _s2_expectation(psi, sec):
    """<S^2> for a normalised vector of a sector with S_z = (n_up - n_down)/2: S^2 = S- S+ + Sz^2 + Sz."""
    sz = 0.5 * (sec.n_up - sec.n_down)
    if sec.n_down == 0:
        return sz * sz + sz
    v, _ = _s_plus(psi, sec)
    return float(v @ v) + sz * sz + sz


def _reflect_strings(strings, n_sites):
    """Spatial inversion i -> n-1-i on one spin string set: (permutation, sign)."""
    lookup = {s: k for k, s in enumerate(strings)}
    perm = np.zeros(len(strings), dtype=int); sgn = np.zeros(len(strings))
    for k, s in enumerate(strings):
        occ = [i for i in range(n_sites) if (s >> i) & 1]
        m = len(occ)
        perm[k] = lookup[sum(1 << (n_sites - 1 - i) for i in occ)]
        sgn[k] = (-1) ** (m * (m - 1) // 2)
    return perm, sgn


def _inversion_expectation(psi, sec):
    pu, su = _reflect_strings(sec.su, sec.n)
    pd, sd = _reflect_strings(sec.sd, sec.n) if sec.sd is not sec.su else (pu, su)
    P = psi.reshape(sec.nu, sec.nd)
    Q = np.zeros_like(P)
    Q[np.ix_(pu, pd)] = (su[:, None] * sd[None, :]) * P
    return float(P.ravel() @ Q.ravel())


def _conjugate_strings(strings, n_sites):
    """Alternancy map on one spin string: apply prod_{i in s, ascending} (-1)^i c_i to the completely filled
    string (creation operators ascending). Returns (permutation to the complementary string, sign)."""
    lookup = {s: k for k, s in enumerate(strings)}
    full = (1 << n_sites) - 1
    perm = np.zeros(len(strings), dtype=int); sgn = np.zeros(len(strings))
    for k, s in enumerate(strings):
        occ = [i for i in range(n_sites) if (s >> i) & 1]
        cur = full; sign = 1
        for i in reversed(occ):  # rightmost operator acts first
            sign *= (-1) ** bin(cur & ((1 << i) - 1)).count("1") * (-1) ** i
            cur ^= (1 << i)
        perm[k] = lookup[cur]; sgn[k] = sign
    return perm, sgn


def _alternancy_expectation(psi, sec):
    pu, su = _conjugate_strings(sec.su, sec.n)
    pd, sd = _conjugate_strings(sec.sd, sec.n) if sec.sd is not sec.su else (pu, su)
    P = psi.reshape(sec.nu, sec.nd)
    Q = np.zeros_like(P)
    Q[np.ix_(pu, pd)] = (su[:, None] * sd[None, :]) * P
    return float(P.ravel() @ Q.ravel())


def _labelled_spectrum(n_sites, t0, delta, V, U, k):
    """Lowest k Sz=0 eigenpairs with labels (E, S(S+1), inversion, alternancy)."""
    sec = _sector(n_sites, n_sites // 2, n_sites // 2)
    H = _hamiltonian(sec, t0, delta, V, U)
    w, v = _lowest(H, k, sec.dim)
    lab = np.array([[w[i], _s2_expectation(v[:, i], sec), _inversion_expectation(v[:, i], sec), _alternancy_expectation(v[:, i], sec)] for i in range(len(w))])
    return sec, w, v, lab


def _covalent_triplets(n_sites, t0, delta, V, U, n_members):
    """The n_members lowest Sz=0 triplet eigenstates sharing the alternancy eigenvalue of the lowest triplet."""
    sec = _sector(n_sites, n_sites // 2, n_sites // 2)
    k = min(sec.dim, max(4 * n_members + 8, 16))
    while True:
        sec, w, v, lab = _labelled_spectrum(n_sites, t0, delta, V, U, k)
        trip = [i for i in range(len(w)) if abs(lab[i, 1] - 2.0) < 1e-6]
        if trip:
            ref = np.sign(lab[trip[0], 3])
            fam = [i for i in trip if np.sign(lab[i, 3]) == ref]
            if len(fam) >= n_members:
                return sec, w, v, lab, fam[:n_members]
        if k >= sec.dim:
            raise ValueError("the sector does not contain the requested number of covalent triplets")
        k = min(sec.dim, 2 * k)


def covalent_triplet_family(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float, n_members: int) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if not isinstance(n_members, (int, np.integer)) or isinstance(n_members, bool) or n_members < 1:
        raise ValueError("n_members must be a positive integer")
    sec, w, v, lab, fam = _covalent_triplets(n_sites, t0, delta, Vm, U, n_members)
    cov = (np.abs(sec.occu[:, None, :] - sec.occd[None, :, :]) == 1).all(-1)  # every site singly occupied
    out = np.zeros((n_members, 2))
    for r, i in enumerate(fam):
        P = v[:, i].reshape(sec.nu, sec.nd)
        out[r, 0] = w[i]
        out[r, 1] = float((P[cov] ** 2).sum())
    return out

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _s_minus(psi, sec):
    """S- psi, mapping (n_up, n_down) to (n_up-1, n_down+1)."""
    tgt = _sector(sec.n, sec.n_up - 1, sec.n_down + 1)
    P = psi.reshape(sec.nu, sec.nd)
    out = np.zeros((tgt.nu, tgt.nd))
    cross = (-1) ** tgt.n_up  # c_{i up} acts on the up block directly; c+_{i dn} passes the remaining up operators
    for i in range(sec.n):
        fu, tu, su = _site_maps(sec, tgt, 0, i, False)
        fd, td, sd = _site_maps(sec, tgt, 1, i, True)
        if fu.size == 0 or fd.size == 0:
            continue
        out[np.ix_(tu, td)] += cross * (su[:, None] * sd[None, :]) * P[np.ix_(fu, fd)]
    return out.ravel(), tgt


def _dark_states(n_sites, t0, delta, V, U):
    """Ground state, 2^1Ag- (lowest inversion-even excited singlet sharing the ground state alternancy eigenvalue) in the Sz=0 sector, and the lowest quintet via the Sz=2 sector."""
    sec = _sector(n_sites, n_sites // 2, n_sites // 2)
    k = min(sec.dim, 12)
    while True:
        sec, w, v, lab = _labelled_spectrum(n_sites, t0, delta, V, U, k)
        gs_sign = np.sign(lab[0, 3])
        sing_ag = [i for i in range(len(w)) if abs(lab[i, 1]) < 1e-6 and lab[i, 2] > 0.999 and np.sign(lab[i, 3]) == gs_sign]
        trip = [i for i in range(len(w)) if abs(lab[i, 1] - 2.0) < 1e-6]
        if len(sing_ag) >= 2 and trip:
            break
        if k >= sec.dim:
            raise ValueError("the Sz=0 sector does not contain a covalent inversion-even excited singlet")
        k = min(sec.dim, 2 * k)
    q_sec = _sector(n_sites, n_sites // 2 + 2, n_sites // 2 - 2)
    Hq = _hamiltonian(q_sec, t0, delta, V, U)
    wq, vq = _lowest(Hq, 1, q_sec.dim)
    return dict(sec=sec, w=w, v=v, lab=lab, gs=sing_ag[0], ag2=sing_ag[1], t1=trip[0], q_sec=q_sec, eq=float(wq[0]), vq=vq[:, 0])


def dark_state_energies(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if n_sites < 4:
        raise ValueError("n_sites must be at least 4 for a quintet state")
    d = _dark_states(n_sites, t0, delta, Vm, U)
    return np.array([d["w"][d["gs"]], d["w"][d["t1"]], d["w"][d["ag2"]], d["eq"]], dtype=float)

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _embed(psi_l, sec_l, psi_r, sec_r, sec_f):
    """Tensor product of a left-subchain state (sites 0..nl-1) and a right-subchain state (sites nl..n-1)
    as a vector of the full-chain sector (global sign immaterial for populations)."""
    nl = sec_l.n
    PL = psi_l.reshape(sec_l.nu, sec_l.nd); PR = psi_r.reshape(sec_r.nu, sec_r.nd)
    iu = np.array([[sec_f.iu[a | (b << nl)] for b in sec_r.su] for a in sec_l.su])
    idn = np.array([[sec_f.id[a | (b << nl)] for b in sec_r.sd] for a in sec_l.sd])
    out = np.zeros(sec_f.dim)
    idx = iu[:, :, None, None] * sec_f.nd + idn[None, None, :, :]
    out[idx.ravel()] = (PL[:, None, :, None] * PR[None, :, None, :]).ravel()
    return out


def _subchain_family(Vm, t0, delta, U, start, sites, members, cache):
    """Covalent triplet family of the subchain of `sites` atoms starting at atom index `start`, memoised in `cache`."""
    key = (start, sites, members)
    if key not in cache:
        cache[key] = _covalent_triplets(sites, t0, delta, Vm[start:start + sites, start:start + sites], U, members)
    return cache[key]


def _triplet_pair_basis(n_sites, t0, delta, V, U):
    """Raw (non-orthogonal) T0 x T0 products |T_j(m)> x |T_1(Nd-m)>, ordered by m = 1..Nd-1 and j = 1..m."""
    nd = n_sites // 2
    Vm = np.asarray(V, dtype=float)
    sec_f = _sector(n_sites, nd, nd)
    cache = {}
    cols = []
    for m in range(1, nd):
        sec_l, wl, vl, labl, faml = _subchain_family(Vm, t0, delta, U, 0, 2 * m, m, cache)
        sec_r, wr, vr, labr, famr = _subchain_family(Vm, t0, delta, U, 2 * m, 2 * (nd - m), 1, cache)
        for j in range(m):
            cols.append(_embed(vl[:, faml[j]], sec_l, vr[:, famr[0]], sec_r, sec_f))
    return sec_f, np.array(cols).T


def triplet_pair_gram_spectrum(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if n_sites < 4:
        raise ValueError("n_sites must be at least 4 to form a triplet pair")
    sec_f, B = _triplet_pair_basis(n_sites, t0, delta, Vm, U)
    S = B.T @ B
    return np.sort(np.linalg.eigvalsh(0.5 * (S + S.T)))

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _lowdin_populations(B, psi, factor):
    S = B.T @ B
    ev, Uv = np.linalg.eigh(0.5 * (S + S.T))
    if ev.min() <= 1e-10:
        raise ValueError("the triplet-pair products are linearly dependent")
    Bo = B @ (Uv @ np.diag(ev ** -0.5) @ Uv.T)
    return factor * (Bo.T @ psi) ** 2


def triplet_pair_populations(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if n_sites < 4:
        raise ValueError("n_sites must be at least 4 to form a triplet pair")
    d = _dark_states(n_sites, t0, delta, Vm, U)
    sec_f, B = _triplet_pair_basis(n_sites, t0, delta, Vm, U)
    return _lowdin_populations(B, d["v"][:, d["ag2"]], 3.0)

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def quintet_population_and_binding(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if n_sites < 4:
        raise ValueError("n_sites must be at least 4 for a quintet state")
    d = _dark_states(n_sites, t0, delta, Vm, U)
    # Sz=0 component of the lowest quintet: apply S- twice to the Sz=2 eigenvector and normalise
    v1, s1 = _s_minus(d["vq"], d["q_sec"])
    v0, s0 = _s_minus(v1, s1)
    v0 = v0 / np.linalg.norm(v0)
    sec_f, B = _triplet_pair_basis(n_sites, t0, delta, Vm, U)
    pq = float(_lowdin_populations(B, v0, 1.5).sum())
    return np.array([pq, d["eq"] - d["w"][d["ag2"]]], dtype=float)

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _fit_three(nvals, pvals):
    n6, n8, n10 = nvals; p6, p8, p10 = pvals
    g = lambda a: (p6 - p8) * (n8 ** -a - n10 ** -a) - (p8 - p10) * (n6 ** -a - n8 ** -a)
    lo, hi = 1e-6, 60.0
    if g(lo) * g(hi) > 0:
        raise ValueError("no positive decay exponent fits the three populations")
    alpha = brentq(g, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    a = (p8 - p10) / (n8 ** -alpha - n10 ** -alpha)
    c = p10 - a * n10 ** -alpha
    return alpha, a, c


def extrapolated_triplet_pair_population(bond_double: float, bond_single: float, bond_angle_deg: float, t0: float, delta: float, U: float, eps_r: float, chain_lengths: np.ndarray) -> float:
    lens = np.asarray(chain_lengths)
    if lens.ndim != 1 or lens.size != 3:
        raise ValueError("chain_lengths must hold exactly three chain lengths")
    lens = [int(x) for x in lens]
    if any(x != lens_i for x, lens_i in zip(np.asarray(chain_lengths), lens)) or not (4 <= lens[0] < lens[1] < lens[2]):
        raise ValueError("chain_lengths must be three increasing even integers >= 4")
    pops = []
    for n in lens:
        V = ohno_interaction_matrix(n, bond_double, bond_single, bond_angle_deg, U, eps_r)
        pops.append(float(triplet_pair_populations(n, t0, delta, V, U).sum()))
    alpha, a, c = _fit_three(lens, pops)
    return float(c)
SCICODE_GOLD_EOF
