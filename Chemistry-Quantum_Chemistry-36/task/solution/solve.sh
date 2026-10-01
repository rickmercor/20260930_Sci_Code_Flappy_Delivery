#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def _chain_coordinates(bonds):
    import numpy as np
    b = np.asarray(bonds, dtype=float)
    xs = np.concatenate([[0.0], np.cumsum(b)])
    return np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)


def _ohno_kernel(coords_a, u_a, coords_b, u_b):
    import numpy as np
    d = np.linalg.norm(np.asarray(coords_a)[:, None, :] - np.asarray(coords_b)[None, :, :], axis=-1)
    a = 2.0 * 14.397 / (np.asarray(u_a, dtype=float)[:, None] + np.asarray(u_b, dtype=float)[None, :])
    return 14.397 / np.sqrt(d * d + a * a)


def ppp_monomer_matrices(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                 hubbard_u: "np.ndarray") -> "np.ndarray":
    import numpy as np
    b = np.atleast_1d(np.asarray(bonds, dtype=float))
    be = np.atleast_1d(np.asarray(betas, dtype=float))
    al = np.atleast_1d(np.asarray(alphas, dtype=float))
    u = np.atleast_1d(np.asarray(hubbard_u, dtype=float))
    if b.ndim != 1 or b.size == 0 or not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError("bonds must be a non-empty array of positive lengths")
    n = b.size + 1
    if be.shape != b.shape:
        raise ValueError("betas must have one entry per bond")
    if al.shape != (n,) or u.shape != (n,):
        raise ValueError("alphas and hubbard_u must have one entry per site")
    if not np.all(np.isfinite(u)) or np.any(u <= 0.0):
        raise ValueError("hubbard_u must be positive")
    h = np.diag(al)
    for k in range(b.size):
        h[k, k + 1] = be[k]
        h[k + 1, k] = be[k]
    X = _chain_coordinates(b)
    gamma = _ohno_kernel(X, u, X, u)
    return np.stack([h, gamma])

def _antiparallel_partner(bonds, separation):
    import numpy as np
    XA = _chain_coordinates(bonds)
    xc = XA[:, 0].mean()
    XB = np.column_stack([2.0 * xc - XA[:, 0], XA[:, 1], XA[:, 2] + float(separation)])
    return XA, XB


def antiparallel_stack_interactions(bonds: "np.ndarray", hubbard_u: "np.ndarray",
                                            separation: float) -> "np.ndarray":
    import numpy as np
    b = np.atleast_1d(np.asarray(bonds, dtype=float))
    u = np.atleast_1d(np.asarray(hubbard_u, dtype=float))
    if b.ndim != 1 or b.size == 0 or not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError("bonds must be a non-empty array of positive lengths")
    if u.shape != (b.size + 1,) or not np.all(np.isfinite(u)) or np.any(u <= 0.0):
        raise ValueError("hubbard_u must hold one positive value per site")
    if not np.isfinite(separation) or separation <= 0.0:
        raise ValueError("separation must be a positive finite number")
    XA, XB = _antiparallel_partner(b, separation)
    return _ohno_kernel(XA, u, XB, u)

def _check_ppp_system(h, gamma, core_charges, n_occ):
    import numpy as np
    H = np.asarray(h, dtype=float)
    G = np.asarray(gamma, dtype=float)
    Z = np.atleast_1d(np.asarray(core_charges, dtype=float))
    if H.ndim != 2 or H.shape[0] != H.shape[1] or not np.allclose(H, H.T, atol=1e-12):
        raise ValueError("h must be a square symmetric matrix")
    if G.shape != H.shape or not np.allclose(G, G.T, atol=1e-12):
        raise ValueError("gamma must be a symmetric matrix of the same size as h")
    n = H.shape[0]
    if Z.shape != (n,):
        raise ValueError("core_charges must have one entry per site")
    if int(n_occ) != n_occ or not (1 <= int(n_occ) < n):
        raise ValueError("n_occ must be an integer with 1 <= n_occ < n_sites")
    return H, G, Z, int(n_occ)


def _rhf_iterations(H, G, Z, nocc, tol=1e-12, maxit=20000):
    import numpy as np
    off = G - np.diag(np.diag(G))
    e, C = np.linalg.eigh(H)
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    for it in range(maxit):
        F = H + np.diag(0.5 * np.diag(P) * np.diag(G) + off @ (np.diag(P) - Z)) - 0.5 * P * off
        e, C = np.linalg.eigh(F)
        Pn = 2.0 * C[:, :nocc] @ C[:, :nocc].T
        if np.abs(Pn - P).max() <= tol:
            P = Pn
            break
        P = Pn if it < 200 else 0.5 * (P + Pn)
    F = H + np.diag(0.5 * np.diag(P) * np.diag(G) + off @ (np.diag(P) - Z)) - 0.5 * P * off
    e, C = np.linalg.eigh(F)
    for k in range(C.shape[1]):
        mags = np.abs(C[:, k])
        j = int(np.flatnonzero(mags >= mags.max() - 1e-8)[0])
        if C[j, k] < 0.0:
            C[:, k] = -C[:, k]
    return e, C


def ppp_rhf_orbitals(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                             n_occ: int) -> "np.ndarray":
    import numpy as np
    H, G, Z, nocc = _check_ppp_system(h, gamma, core_charges, n_occ)
    e, C = _rhf_iterations(H, G, Z, nocc)
    return np.vstack([e[None, :], C])

def _fock_space(n_sites, n_per_spin):
    import itertools
    strings = [sum(1 << i for i in c) for c in itertools.combinations(range(n_sites), n_per_spin)]
    dets = [(a, b) for a in strings for b in strings]
    return dict(n=n_sites, dets=dets, idx={d: k for k, d in enumerate(dets)}, dim=len(dets), cache={})


def _site_hop(space, p, q, s):
    import numpy as np
    import scipy.sparse as sp
    key = ("hop", p, q, s)
    if key in space["cache"]:
        return space["cache"][key]
    n = space["n"]
    P, Q = p + s * n, q + s * n
    rows, cols, vals = [], [], []
    for j, (a, b) in enumerate(space["dets"]):
        f = a | (b << n)
        if not (f >> Q) & 1:
            continue
        sg = (-1) ** bin(f & ((1 << Q) - 1)).count("1")
        f ^= 1 << Q
        if (f >> P) & 1:
            continue
        sg *= (-1) ** bin(f & ((1 << P) - 1)).count("1")
        f |= 1 << P
        rows.append(space["idx"][(f & ((1 << n) - 1), f >> n)])
        cols.append(j)
        vals.append(sg)
    M = sp.csr_matrix((np.array(vals, dtype=float), (rows, cols)), shape=(space["dim"], space["dim"]))
    space["cache"][key] = M
    return M


def _site_occupation(space, p):
    import numpy as np
    import scipy.sparse as sp
    key = ("occ", p)
    if key not in space["cache"]:
        v = np.array([((a >> p) & 1) + ((b >> p) & 1) for a, b in space["dets"]], dtype=float)
        space["cache"][key] = sp.diags(v).tocsr()
    return space["cache"][key]


def _ppp_many_body(space, h, g, Z):
    import numpy as np
    import scipy.sparse as sp
    n = space["n"]
    I = sp.identity(space["dim"], format="csr")
    H = sp.csr_matrix((space["dim"], space["dim"]))
    for p in range(n):
        for q in range(n):
            if h[p, q] != 0.0:
                H = H + h[p, q] * (_site_hop(space, p, q, 0) + _site_hop(space, p, q, 1))
    dn = [_site_occupation(space, p) - Z[p] * I for p in range(n)]
    for p in range(n):
        H = H + g[p, p] * (_site_hop(space, p, p, 0) @ _site_hop(space, p, p, 1))
        for q in range(p + 1, n):
            if g[p, q] != 0.0:
                H = H + g[p, q] * (dn[p] @ dn[q])
    return H.tocsr()


def _cc_build(space, H, C, occ):
    import itertools
    import numpy as np
    import scipy.sparse as sp
    n = space["n"]
    occ = [int(i) for i in occ]
    vir = [p for p in range(n) if p not in occ]
    emo = {}

    def _E(a, i, s):
        if (a, i, s) not in emo:
            M = sp.csr_matrix((space["dim"], space["dim"]))
            for mu in range(n):
                for nu in range(n):
                    c = C[mu, a] * C[nu, i]
                    if abs(c) > 1e-15:
                        M = M + c * _site_hop(space, mu, nu, s)
            emo[(a, i, s)] = M.tocsr()
        return emo[(a, i, s)]

    so_o = [(i, s) for s in (0, 1) for i in occ]
    so_v = [(a, s) for s in (0, 1) for a in vir]
    man = [(((a, s),), ((i, s),)) for s in (0, 1) for i in occ for a in vir]
    for i, j in itertools.combinations(so_o, 2):
        for a, b in itertools.combinations(so_v, 2):
            if i[1] + j[1] == a[1] + b[1]:
                man.append(((a, b), (i, j)))
    tau = []
    for cre, ann in man:
        if len(cre) == 1:
            tau.append(_E(cre[0][0], ann[0][0], cre[0][1]))
        else:
            (a, sa), (b, sb) = cre
            (i, si), (j, sj) = ann
            if sa == si and sb == sj:
                tau.append((_E(a, i, sa) @ _E(b, j, sb)).tocsr())
            else:
                tau.append((-(_E(a, j, sa) @ _E(b, i, sb))).tocsr())
    ref = np.zeros(space["dim"])
    for k, (a, b) in enumerate(space["dets"]):
        sa = [p for p in range(n) if (a >> p) & 1]
        sb = [p for p in range(n) if (b >> p) & 1]
        ref[k] = np.linalg.det(C[np.ix_(sa, occ)]) * np.linalg.det(C[np.ix_(sb, occ)])
    B = np.column_stack([ref] + [t @ ref for t in tau])
    return dict(space=space, H=H, C=C, occ=occ, vir=vir, man=man, tau=tau, ref=ref, B=B)


def _cc_expv(T, v, sign=1.0):
    import numpy as np
    out = v.copy()
    term = v.copy()
    for k in range(1, 80):
        term = sign * (T @ term) / k
        if not np.any(term):
            break
        out = out + term
    return out


def _cc_tmat(cc, t):
    import scipy.sparse as sp
    T = sp.csr_matrix((cc["space"]["dim"], cc["space"]["dim"]))
    for x, op in zip(t, cc["tau"]):
        if x != 0.0:
            T = T + x * op
    return T.tocsr()


def _cc_sim(Op, v, T):
    return _cc_expv(T, Op @ _cc_expv(T, v), -1.0)


def _cc_jacobian(cc, T):
    import numpy as np
    cols = [(cc["B"].T @ _cc_sim(cc["H"] @ tau - tau @ cc["H"], cc["ref"], T))[1:] for tau in cc["tau"]]
    return np.array(cols).T


def _cc_solve(cc, tol=1e-10, maxit=200):
    import numpy as np
    t = np.zeros(len(cc["man"]))
    for it in range(maxit):
        T = _cc_tmat(cc, t)
        r = (cc["B"].T @ _cc_sim(cc["H"], cc["ref"], T))[1:]
        if np.abs(r).max() < 1e-13:
            break
        t = t - np.linalg.solve(_cc_jacobian(cc, T), r)
    T = _cc_tmat(cc, t)
    r = (cc["B"].T @ _cc_sim(cc["H"], cc["ref"], T))[1:]
    if np.abs(r).max() > tol:
        raise ValueError("CCSD amplitude equations did not converge")
    cc["t"] = t
    cc["T"] = T
    cc["E0"] = float(cc["ref"] @ _cc_sim(cc["H"], cc["ref"], T))
    return cc


def _cc_block(cc, Op):
    import numpy as np
    return np.column_stack([cc["B"].T @ _cc_sim(Op, cc["B"][:, k], cc["T"]) for k in range(cc["B"].shape[1])])


def _ccsd_state(H_ppp, G, Z, C, occ):
    import numpy as np
    n = H_ppp.shape[0]
    space = _fock_space(n, len(occ))
    Hmb = _ppp_many_body(space, H_ppp, G, Z)
    return _cc_solve(_cc_build(space, Hmb, np.asarray(C, dtype=float), occ))


def ccsd_energy(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray", n_occ: int) -> float:
    import numpy as np
    H, G, Z, nocc = _check_ppp_system(h, gamma, core_charges, n_occ)
    orb = ppp_rhf_orbitals(H, G, Z, nocc)
    cc = _ccsd_state(H, G, Z, orb[1:], list(range(nocc)))
    return float(cc["E0"])

def _spin_swap_ops(ops):
    f = [(p, 1 - s) for p, s in ops]
    order = sorted(range(len(f)), key=lambda z: (f[z][1], f[z][0]))
    sign = -1.0 if (len(f) == 2 and order == [1, 0]) else 1.0
    return tuple(f[z] for z in order), sign


def _spin_exchange_matrix(cc):
    import numpy as np
    key = {m: k for k, m in enumerate(cc["man"])}
    dim = len(cc["man"]) + 1
    P = np.zeros((dim, dim))
    P[0, 0] = 1.0
    for k, (cre, ann) in enumerate(cc["man"]):
        c2, s1 = _spin_swap_ops(cre)
        a2, s2 = _spin_swap_ops(ann)
        P[1 + key[(c2, a2)], 1 + k] = s1 * s2
    return P


def _s_squared(cc, r):
    import numpy as np
    space = cc["space"]; n = space["n"]
    v = _cc_expv(cc["T"], cc["B"] @ r)
    out = {}
    for k, (a, b) in enumerate(space["dets"]):
        if v[k] == 0.0:
            continue
        f = a | (b << n)
        for p in range(n):
            if not (f >> (n + p)) & 1 or (f >> p) & 1:
                continue
            sg = (-1) ** bin(f & ((1 << (n + p)) - 1)).count("1")
            g = f ^ (1 << (n + p))
            sg *= (-1) ** bin(g & ((1 << p) - 1)).count("1")
            g |= 1 << p
            out[g] = out.get(g, 0.0) + sg * v[k]
    return float(sum(x * x for x in out.values()) / (v @ v))


def _eom_singlets(cc, E0):
    """Singlet right/left eigenpairs of the projected transformed Hamiltonian, sorted by energy."""
    import numpy as np
    M = _cc_block(cc, cc["H"])
    P = _spin_exchange_matrix(cc)
    w, VR = np.linalg.eig(M)
    wl, VL = np.linalg.eig(M.T)
    out = []
    for k in np.argsort(w.real):
        if abs(w[k] - E0) < 1e-8:
            continue
        r = VR[:, k].real
        if (r @ P @ r) / (r @ r) > 0.5 and _s_squared(cc, r) < 0.5:
            r = r / np.linalg.norm(r)
            j = int(np.argmax(np.abs(r)))
            if r[j] < 0.0:
                r = -r
            kl = int(np.argmin(np.abs(wl - w[k])))
            l = VL[:, kl].real
            l = l / (l @ r)
            out.append((float(w[k].real - E0), r, l))
    kg = int(np.argmin(np.abs(wl - E0)))
    lg = VL[:, kg].real
    lg = lg / lg[0]
    return out, M, lg


def eom_ccsd_singlet_energies(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                                      n_occ: int, n_roots: int) -> "np.ndarray":
    import numpy as np
    H, G, Z, nocc = _check_ppp_system(h, gamma, core_charges, n_occ)
    if int(n_roots) != n_roots or int(n_roots) < 1:
        raise ValueError("n_roots must be a positive integer")
    E0 = ccsd_energy(H, G, Z, nocc)
    orb = ppp_rhf_orbitals(H, G, Z, nocc)
    cc = _ccsd_state(H, G, Z, orb[1:], list(range(nocc)))
    roots, _, _ = _eom_singlets(cc, E0)
    if int(n_roots) > len(roots):
        raise ValueError("fewer singlet excited states are available than requested")
    return np.array([roots[k][0] for k in range(int(n_roots))])

def _site_transition_density(cc, bra, ket_coeffs):
    import numpy as np
    v = _cc_expv(cc["T"], cc["B"] @ ket_coeffs)
    n = cc["space"]["n"]
    return np.array([bra @ (cc["B"].T @ _cc_expv(cc["T"], _site_occupation(cc["space"], p) @ v, -1.0))
                     for p in range(n)])


def _monomer_transition(H, G, Z, nocc, root):
    """EOM-CCSD state of a monomer: returns (omega, r, l, lambda-vector, rho10, rho01, cc, rhf C)."""
    import numpy as np
    if int(root) != root or int(root) < 0:
        raise ValueError("root must be a non-negative integer")
    omegas = eom_ccsd_singlet_energies(H, G, Z, nocc, int(root) + 1)
    E0 = ccsd_energy(H, G, Z, nocc)
    orb = ppp_rhf_orbitals(H, G, Z, nocc)
    cc = _ccsd_state(H, G, Z, orb[1:], list(range(nocc)))
    roots, M, lg = _eom_singlets(cc, E0)
    k = int(np.argmin([abs(x[0] - omegas[int(root)]) for x in roots]))
    omega, r, l = roots[k]
    e0 = np.zeros(M.shape[0])
    e0[0] = 1.0
    rho10 = _site_transition_density(cc, l, e0)
    rho01 = _site_transition_density(cc, lg, r)
    return dict(omega=omega, r=r, l=l, lg=lg, rho10=rho10, rho01=rho01, cc=cc, C=orb[1:], E0=E0)


def eom_transition_density_product(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                                           n_occ: int, root: int) -> "np.ndarray":
    import numpy as np
    H, G, Z, nocc = _check_ppp_system(h, gamma, core_charges, n_occ)
    st = _monomer_transition(H, G, Z, nocc, root)
    return np.outer(st["rho10"], st["rho01"])

def fragment_excitonic_coupling(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                        hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                        root: int, separation: float) -> float:
    import numpy as np
    M = ppp_monomer_matrices(bonds, betas, alphas, hubbard_u)
    G = antiparallel_stack_interactions(bonds, hubbard_u, separation)
    P = eom_transition_density_product(M[0], M[1], core_charges, n_occ, root)
    return float(abs(np.sum(G * P)))

def _embed_manifold(man_mono, man_dimer, shift):
    import numpy as np
    key = {m: k for k, m in enumerate(man_dimer)}
    return np.array([key[(tuple((p + shift, s) for p, s in c), tuple((p + shift, s) for p, s in a))]
                     for c, a in man_mono], dtype=int)


def _intermolecular_operator(space, G, ZA, ZB, nA):
    import scipy.sparse as sp
    I = sp.identity(space["dim"], format="csr")
    W = sp.csr_matrix((space["dim"], space["dim"]))
    for mu in range(nA):
        for nu in range(G.shape[1]):
            W = W + G[mu, nu] * ((_site_occupation(space, mu) - ZA[mu] * I) @ (_site_occupation(space, nA + nu) - ZB[nu] * I))
    return W.tocsr()


def _supersystem_pieces(bonds, betas, alphas, hubbard_u, core_charges, n_occ, root, separation):
    import numpy as np
    M = ppp_monomer_matrices(bonds, betas, alphas, hubbard_u)
    G = antiparallel_stack_interactions(bonds, hubbard_u, separation)
    H1, G1, Z, nocc = _check_ppp_system(M[0], M[1], core_charges, n_occ)
    n = H1.shape[0]
    st = _monomer_transition(H1, G1, Z, nocc, root)
    # non-interacting dimer on the product of the monomer RHF determinants
    Hd = np.zeros((2 * n, 2 * n)); Hd[:n, :n] = H1; Hd[n:, n:] = H1
    Gd = np.zeros((2 * n, 2 * n)); Gd[:n, :n] = G1; Gd[n:, n:] = G1
    Cd = np.zeros((2 * n, 2 * n)); Cd[:n, :n] = st["C"]; Cd[n:, n:] = st["C"]
    Zd = np.concatenate([Z, Z])
    E0_dimer = ccsd_energy(Hd, Gd, Zd, 2 * nocc)
    occ = list(range(nocc)) + list(range(n, n + nocc))
    cc = _ccsd_state(Hd, Gd, Zd, Cd, occ)
    if abs(cc["E0"] - E0_dimer) > 1e-8:
        raise ValueError("dimer reference is not size-consistent")
    Mb = _cc_block(cc, cc["H"])
    W = _intermolecular_operator(cc["space"], G, Z, Z, n)
    mapA = _embed_manifold(st["cc"]["man"], cc["man"], 0)
    mapB = _embed_manifold(st["cc"]["man"], cc["man"], n)
    dim = Mb.shape[0]
    isA = np.zeros(dim, bool); isA[1 + mapA] = True
    isB = np.zeros(dim, bool); isB[1 + mapB] = True
    isX = ~(isA | isB); isX[0] = False
    E = E0_dimer + st["omega"]
    RB = np.zeros(dim); RB[0] = st["r"][0]; RB[1 + mapB] = st["r"][1:]
    LA = np.zeros(dim); LA[1 + mapA] = st["l"][1:]
    LX = np.linalg.solve((E * np.eye(int(isX.sum())) - Mb[np.ix_(isX, isX)]).T, LA[isA] @ Mb[np.ix_(isA, isX)])
    LAX = LA.copy(); LAX[isX] = LX
    Wb = _cc_block(cc, W)
    t1 = np.linalg.solve(_cc_jacobian(cc, cc["T"]), -Wb[1:, 0])
    T1 = _cc_tmat(cc, t1)
    Wrel = _cc_block(cc, W + cc["H"] @ T1 - T1 @ cc["H"])
    return dict(Vs=float(abs(LAX @ Wrel @ RB)), LX_weight=float(np.linalg.norm(LX) / np.linalg.norm(st["l"][1:])),
                Vs_frozen=float(abs(LAX @ Wb @ RB)), Vs_local=float(abs(LA @ Wrel @ RB)), omega=st["omega"],
                E0_dimer=E0_dimer)


def supersystem_excitonic_coupling(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                           hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                           root: int, separation: float) -> float:
    return _supersystem_pieces(bonds, betas, alphas, hubbard_u, core_charges, n_occ, root, separation)["Vs"]

def separability_error_percent(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                       hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                       root: int, separation: float) -> float:
    vc = fragment_excitonic_coupling(bonds, betas, alphas, hubbard_u, core_charges, n_occ, root, separation)
    vs = supersystem_excitonic_coupling(bonds, betas, alphas, hubbard_u, core_charges, n_occ, root, separation)
    if vs < 1e-12:
        raise ValueError("the supersystem coupling vanishes")
    return float(100.0 * (vc - vs) / vs)
SCICODE_GOLD_EOF
