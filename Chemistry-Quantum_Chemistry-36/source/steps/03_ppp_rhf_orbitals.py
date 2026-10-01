"""
Closed-shell restricted Hartree-Fock orbitals of a PPP pi system with atomic core charges.

The PPP Hamiltonian of a pi system with site basis mu, spin sigma and occupation operators n_mu = n_mu,up + n_mu,down is

  H = sum_{mu,nu,sigma} h_mu,nu a+_mu,sigma a_nu,sigma + sum_mu gamma_mu,mu n_mu,up n_mu,down + sum_{mu<nu} gamma_mu,nu (n_mu - Z_mu)(n_nu - Z_nu)

where Z_mu is the number of pi electrons the atom contributes in its neutral form, its core charge. A carbon in a C=C bond has Z = 1, while a pyrrole-type nitrogen that donates its lone pair to the pi system has Z = 2. The constant core-core repulsion is part of H as written, so energies computed from it are total pi energies.

For a closed-shell state with n_occ doubly occupied orbitals the restricted Hartree-Fock (RHF) solution follows from the zero-differential-overlap Fock matrix

  F_mu,mu = h_mu,mu + (1/2) P_mu,mu gamma_mu,mu + sum_{nu != mu} (P_nu,nu - Z_nu) gamma_mu,nu
  F_mu,nu = h_mu,nu - (1/2) P_mu,nu gamma_mu,nu   (mu != nu)

with the density matrix P = 2 C_occ C_occ^T built from the n_occ orbitals of lowest energy. The equations are iterated to self-consistency, the aufbau solution being the one wanted, and the procedure is converged when no element of P changes by more than 1e-12 between iterations.

The orbitals are returned as the columns of the site-by-orbital coefficient matrix C in order of increasing orbital energy. An eigenvector is only defined up to sign, so each column is made unique by requiring its component of largest magnitude to be positive. Components whose magnitudes differ by less than 1e-8 count as tied, and among tied components the one with the lowest site index decides. Such ties are exact, not accidental, in chains that are symmetric under reversal, such as butadiene or the allyl cation, where rounding makes the two equal magnitudes differ in the last few digits.

Returns
-------
numpy.ndarray of shape (n_sites + 1, n_sites): row 0 holds the RHF orbital energies in eV in increasing order and rows 1 to n_sites hold the coefficient matrix C, column k being orbital k
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ppp_rhf_orbitals(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                     n_occ: int) -> "np.ndarray":
    '''Converged closed-shell RHF orbital energies and phase-fixed orbital coefficients.

    Parameters
    ----------
    h : numpy.ndarray
        Symmetric one-electron matrix of shape (n_sites, n_sites), in eV.
    gamma : numpy.ndarray
        Symmetric electron-repulsion matrix of shape (n_sites, n_sites), in eV, whose
        diagonal holds the one-centre repulsions.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals; must satisfy 1 <= n_occ < n_sites.

    Returns
    -------
    orbitals : numpy.ndarray
        Real array of shape (n_sites + 1, n_sites). Row 0 is the list of orbital
        energies in increasing order; rows 1 to n_sites form the coefficient matrix
        C with orbital k in column k, its largest-magnitude component positive.

    Raises
    ------
    ValueError
        If h or gamma is not a square symmetric matrix of the same size, if
        core_charges does not have length n_sites, or if n_occ is not an integer
        with 1 <= n_occ < n_sites.
    '''
    return orbitals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_ppp_rhf_orbitals(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                             n_occ: int) -> "np.ndarray":
    import numpy as np
    H, G, Z, nocc = _check_ppp_system(h, gamma, core_charges, n_occ)
    e, C = _rhf_iterations(H, G, Z, nocc)
    return np.vstack([e[None, :], C])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark enamine-like chromophore, pyrrole-type nitrogen with Z = 2 ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]))
""",
            "call": "ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2)",
            "gold_call": "_oracle_ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2)",
            "tol": 1e-9,
        },
        # --- Normal: butadiene, four electrons in four orbitals ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.35, 1.46, 1.35]), np.array([-2.6, -2.2, -2.6]), np.zeros(4), np.full(4, 11.13))
""",
            "call": "ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.ones(4), 2)",
            "gold_call": "_oracle_ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.ones(4), 2)",
            "tol": 1e-9,
        },
        # --- Boundary: a two-atom polar unit with a single occupied orbital ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]))
""",
            "call": "ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.ones(2), 1)",
            "gold_call": "_oracle_ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.ones(2), 1)",
            "tol": 1e-9,
        },
        # --- Edge: an allyl cation, two electrons on three carbons, where the charge is not neutralised ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13))
""",
            "call": "ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.ones(3), 1)",
            "gold_call": "_oracle_ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.ones(3), 1)",
            "tol": 1e-9,
        },
        # --- Edge: a strongly donating terminal nitrogen that pulls the HOMO onto one site ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -1.6]), np.array([0.0, 0.0, -5.0]), np.array([11.13, 11.13, 16.76]))
""",
            "call": "ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2)",
            "gold_call": "_oracle_ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2)",
            "tol": 1e-9,
        },
        # --- Invalid: as many occupied orbitals as sites leaves no virtual space ---
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
            "call": "_exception_code(lambda: ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.ones(2), 2))",
            "gold_call": "_exception_code(lambda: _oracle_ppp_rhf_orbitals(M[0].copy(), M[1].copy(), np.ones(2), 2))",
        },
    ]
