"""
Equation-of-motion CCSD excitation energies of the lowest singlet states of a closed-shell PPP pi system.

In the equation-of-motion formulation excited states are eigenstates of the same similarity-transformed Hamiltonian that defines the CCSD ground state, exp(-T) H exp(T), represented in the space spanned by the reference determinant and all singly and doubly excited determinants. That projected matrix is not symmetric, so it has distinct left and right eigenvectors, but a single set of eigenvalues. Its lowest eigenvalue is the CCSD ground-state energy, and every other eigenvalue minus that energy is an EOM-CCSD excitation energy.

The reference is the closed-shell RHF determinant, so the projected problem contains the M_S = 0 components of singlet, triplet and, once four electrons can be unpaired, quintet states together. A state is a singlet when the total spin S^2 of its right eigenvector, taken as the wavefunction exp(T) R |Phi_0>, is zero. Exchanging the alpha and beta spin labels does not separate singlets from quintets, because both are unchanged by that exchange. Only singlet states are wanted.

The excitation energies are returned in increasing order for the requested number of lowest singlet excited states, in eV. The ground state itself is not counted.

Returns
-------
numpy.ndarray of shape (n_roots,): the lowest n_roots EOM-CCSD singlet excitation energies in eV, in increasing order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def eom_ccsd_singlet_energies(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                              n_occ: int, n_roots: int) -> "np.ndarray":
    '''Lowest singlet EOM-CCSD excitation energies.

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
    n_roots : int
        Number of singlet excited states wanted; must be at least 1 and no larger
        than the number of singlet excited states in the singles and doubles space.

    Returns
    -------
    energies : numpy.ndarray
        Real array of shape (n_roots,) with the EOM-CCSD singlet excitation
        energies in eV in increasing order.

    Raises
    ------
    ValueError
        If h or gamma is not a square symmetric matrix of the same size, if
        core_charges does not have length n_sites, if n_occ is not an integer with
        1 <= n_occ < n_sites, or if n_roots is not a positive integer or exceeds the
        number of singlet excited states available.
    '''
    return energies

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_eom_ccsd_singlet_energies(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                                      n_occ: int, n_roots: int) -> "np.ndarray":
    import numpy as np
    H, G, Z, nocc = _check_ppp_system(h, gamma, core_charges, n_occ)
    if int(n_roots) != n_roots or int(n_roots) < 1:
        raise ValueError("n_roots must be a positive integer")
    E0 = _oracle_ccsd_energy(H, G, Z, nocc)
    orb = _oracle_ppp_rhf_orbitals(H, G, Z, nocc)
    cc = _ccsd_state(H, G, Z, orb[1:], list(range(nocc)))
    roots, _, _ = _eom_singlets(cc, E0)
    if int(n_roots) > len(roots):
        raise ValueError("fewer singlet excited states are available than requested")
    return np.array([roots[k][0] for k in range(int(n_roots))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark C=C-N monomer ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]))
""",
            "call": "eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2, 3)",
            "gold_call": "_oracle_eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2, 3)",
            "tol": 1e-9,
        },
        # --- Normal: butadiene, where EOM-CCSD is not exact and the doubly excited 2Ag state is misplaced ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.35, 1.46, 1.35]), np.array([-2.6, -2.2, -2.6]), np.zeros(4), np.full(4, 11.13))
""",
            "call": "eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.ones(4), 2, 4)",
            "gold_call": "_oracle_eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.ones(4), 2, 4)",
            "tol": 1e-9,
        },
        # --- Boundary: a two-electron polar unit asked for every singlet it has ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]))
""",
            "call": "eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.ones(2), 1, 2)",
            "gold_call": "_oracle_eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.ones(2), 1, 2)",
            "tol": 1e-9,
        },
        # --- Edge: an allyl cation with a charged core ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13))
""",
            "call": "eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.ones(3), 1, 3)",
            "gold_call": "_oracle_eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.ones(3), 1, 3)",
            "tol": 1e-9,
        },
        # --- Edge: a strongly donating nitrogen that pushes the charge-transfer state down ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -1.6]), np.array([0.0, 0.0, -5.0]), np.array([11.13, 11.13, 16.76]))
""",
            "call": "eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2, 2)",
            "gold_call": "_oracle_eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2, 2)",
            "tol": 1e-9,
        },
        # --- Invalid: more singlet roots than a two-electron two-orbital unit possesses ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]))
def _exception_code(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "_exception_code(lambda: eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.ones(2), 1, 9))",
            "gold_call": "_exception_code(lambda: _oracle_eom_ccsd_singlet_energies(M[0].copy(), M[1].copy(), np.ones(2), 1, 9))",
        },
    ]
