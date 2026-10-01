"""
Coupled-cluster singles and doubles ground-state energy of a closed-shell PPP pi system.

Coupled-cluster theory writes the ground state as |Psi_0> = exp(T)|Phi_0>, where |Phi_0> is the closed-shell RHF determinant of the previous step and T is a linear combination of excitation operators out of it. In the singles and doubles approximation (CCSD) T contains every single and every double excitation of the spin-orbital RHF determinant, with no orbital frozen and no restriction by spatial symmetry. The amplitudes are fixed by requiring that the similarity-transformed Hamiltonian exp(-T) H exp(T) has no component connecting |Phi_0> to any singly or doubly excited determinant, and the CCSD energy is then the expectation value of that transformed Hamiltonian in |Phi_0>.

The Hamiltonian is the PPP operator with core charges of the previous step, including the constant core-core repulsion, so the energy returned is the total pi energy in eV rather than a correlation energy. The amplitude equations are solved until every projected residual is below 1e-10 eV in magnitude.

CCSD is exact for any system in which no determinant beyond double excitation can be reached from the reference, which includes every two-electron system and every system with only one doubly occupied or only one virtual spatial orbital. It is not exact once four or more electrons are distributed over a virtual space that allows triple and quadruple excitations, as for butadiene or for a dimer of two chromophores.

The energy is invariant to any rotation among the occupied orbitals and to any rotation among the virtual orbitals, so degenerate RHF orbitals, as they occur for two identical non-interacting molecules, do not make it ambiguous.

Returns
-------
float: the CCSD total pi energy in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ccsd_energy(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray", n_occ: int) -> float:
    '''CCSD ground-state total pi energy on the RHF reference.

    Parameters
    ----------
    h : numpy.ndarray
        Symmetric one-electron matrix of shape (n_sites, n_sites), in eV.
    gamma : numpy.ndarray
        Symmetric electron-repulsion matrix of shape (n_sites, n_sites), in eV.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals; must satisfy 1 <= n_occ < n_sites.

    Returns
    -------
    energy : float
        The CCSD total energy <Phi_0| exp(-T) H exp(T) |Phi_0> in eV, with H the PPP
        operator including the core-core repulsion.

    Raises
    ------
    ValueError
        If h or gamma is not a square symmetric matrix of the same size, if
        core_charges does not have length n_sites, or if n_occ is not an integer
        with 1 <= n_occ < n_sites.
    '''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_ccsd_energy(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray", n_occ: int) -> float:
    import numpy as np
    H, G, Z, nocc = _check_ppp_system(h, gamma, core_charges, n_occ)
    orb = _oracle_ppp_rhf_orbitals(H, G, Z, nocc)
    cc = _ccsd_state(H, G, Z, orb[1:], list(range(nocc)))
    return float(cc["E0"])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark C=C-N monomer, where CCSD coincides with full CI ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]))
""",
            "call": "ccsd_energy(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2)",
            "gold_call": "_oracle_ccsd_energy(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2)",
            "tol": 1e-9,
        },
        # --- Normal: butadiene, where CCSD is not exact ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.35, 1.46, 1.35]), np.array([-2.6, -2.2, -2.6]), np.zeros(4), np.full(4, 11.13))
""",
            "call": "ccsd_energy(M[0].copy(), M[1].copy(), np.ones(4), 2)",
            "gold_call": "_oracle_ccsd_energy(M[0].copy(), M[1].copy(), np.ones(4), 2)",
            "tol": 1e-9,
        },
        # --- Boundary: a two-electron polar unit ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]))
""",
            "call": "ccsd_energy(M[0].copy(), M[1].copy(), np.ones(2), 1)",
            "gold_call": "_oracle_ccsd_energy(M[0].copy(), M[1].copy(), np.ones(2), 1)",
            "tol": 1e-9,
        },
        # --- Edge: two non-interacting C=C-N molecules, degenerate orbitals, energy twice the monomer ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]))
Z = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0])
h = np.zeros((6, 6)); h[:3, :3] = M[0]; h[3:, 3:] = M[0]
g = np.zeros((6, 6)); g[:3, :3] = M[1]; g[3:, 3:] = M[1]
""",
            "call": "ccsd_energy(h.copy(), g.copy(), Z.copy(), 4)",
            "gold_call": "_oracle_ccsd_energy(h.copy(), g.copy(), Z.copy(), 4)",
            "tol": 1e-9,
        },
        # --- Edge: the interacting antiparallel dimer at 7 angstrom, eight electrons, CCSD differs from full CI ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]))
G = _oracle_antiparallel_stack_interactions(np.array([1.34, 1.40]), np.array([11.13, 11.13, 16.76]), 7.0)
Z = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0])
h = np.zeros((6, 6)); h[:3, :3] = M[0]; h[3:, 3:] = M[0]
g = np.zeros((6, 6)); g[:3, :3] = M[1]; g[3:, 3:] = M[1]; g[:3, 3:] = G; g[3:, :3] = G.T
""",
            "call": "ccsd_energy(h.copy(), g.copy(), Z.copy(), 4)",
            "gold_call": "_oracle_ccsd_energy(h.copy(), g.copy(), Z.copy(), 4)",
            "tol": 1e-9,
        },
        # --- Edge: an allyl cation with a charged core ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13))
""",
            "call": "ccsd_energy(M[0].copy(), M[1].copy(), np.ones(3), 1)",
            "gold_call": "_oracle_ccsd_energy(M[0].copy(), M[1].copy(), np.ones(3), 1)",
            "tol": 1e-9,
        },
        # --- Invalid: core charges of the wrong length ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, 0.0]), np.array([11.13, 11.13]))
def _exception_code(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "_exception_code(lambda: ccsd_energy(M[0].copy(), M[1].copy(), np.ones(3), 1))",
            "gold_call": "_exception_code(lambda: _oracle_ccsd_energy(M[0].copy(), M[1].copy(), np.ones(3), 1))",
        },
    ]
