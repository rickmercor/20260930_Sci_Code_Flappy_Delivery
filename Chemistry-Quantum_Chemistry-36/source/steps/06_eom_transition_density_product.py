"""
Product of the two EOM-CCSD site transition densities between the ground state and one singlet excited state.

Because the EOM-CCSD problem is not symmetric, the ground-to-excited and excited-to-ground transition moments are different quantities, and only products of the two are uniquely defined. Both are built from the similarity-transformed site occupation operator exp(-T) n_mu exp(T) and from the left and right eigenvectors of the projected transformed Hamiltonian of the previous step.

For the chosen excited state the right eigenvector R, including its component on the reference determinant, and the left eigenvector L are normalised against each other so that their dot product is one; any further scaling or sign change of R, compensated in L, cancels in the product returned here. The ground state has the trivial right eigenvector, the reference alone, and a left eigenvector whose reference component is one and whose excited components are the Lambda amplitudes of the stationary CC energy functional.

The two transition densities on site mu are then

  rho^{10}_mu : the matrix element of exp(-T) n_mu exp(T) between the excited left eigenvector (bra) and the reference determinant (ket)
  rho^{01}_mu : the matrix element of exp(-T) n_mu exp(T) between the ground-state left eigenvector (1, Lambda) (bra) and the excited right eigenvector (ket)

The first is the moment for the transition from the excited state down to the ground state, the second the moment for the transition from the ground state up to the excited state. The excited state is selected as the root-th lowest singlet excited state, counting from zero, in the ordering of the previous step.

Returns
-------
numpy.ndarray of shape (n_sites, n_sites): the outer product rho^{10}_mu rho^{01}_nu of the two EOM-CCSD site transition densities of the selected singlet state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def eom_transition_density_product(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                                   n_occ: int, root: int) -> "np.ndarray":
    '''Outer product of the EOM-CCSD down and up site transition densities.

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
    root : int
        Index, counting from zero, of the singlet excited state in increasing order
        of EOM-CCSD excitation energy.

    Returns
    -------
    product : numpy.ndarray
        Real array of shape (n_sites, n_sites) whose element [mu, nu] is
        rho^{10}_mu * rho^{01}_nu for the selected state.

    Raises
    ------
    ValueError
        If h or gamma is not a square symmetric matrix of the same size, if
        core_charges does not have length n_sites, if n_occ is not an integer with
        1 <= n_occ < n_sites, or if root is negative or not smaller than the number
        of singlet excited states available.
    '''
    return product

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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
    omegas = _oracle_eom_ccsd_singlet_energies(H, G, Z, nocc, int(root) + 1)
    E0 = _oracle_ccsd_energy(H, G, Z, nocc)
    orb = _oracle_ppp_rhf_orbitals(H, G, Z, nocc)
    cc = _ccsd_state(H, G, Z, orb[1:], list(range(nocc)))
    roots, M, lg = _eom_singlets(cc, E0)
    k = int(np.argmin([abs(x[0] - omegas[int(root)]) for x in roots]))
    omega, r, l = roots[k]
    e0 = np.zeros(M.shape[0])
    e0[0] = 1.0
    rho10 = _site_transition_density(cc, l, e0)
    rho01 = _site_transition_density(cc, lg, r)
    return dict(omega=omega, r=r, l=l, lg=lg, rho10=rho10, rho01=rho01, cc=cc, C=orb[1:], E0=E0)


def _oracle_eom_transition_density_product(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                                           n_occ: int, root: int) -> "np.ndarray":
    import numpy as np
    H, G, Z, nocc = _check_ppp_system(h, gamma, core_charges, n_occ)
    st = _monomer_transition(H, G, Z, nocc, root)
    return np.outer(st["rho10"], st["rho01"])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the bright state of the benchmark C=C-N monomer ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]))
""",
            "call": "eom_transition_density_product(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2, 0)",
            "gold_call": "_oracle_eom_transition_density_product(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2, 0)",
            "tol": 1e-9,
        },
        # --- Normal: butadiene, not exact at CCSD, second singlet ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.35, 1.46, 1.35]), np.array([-2.6, -2.2, -2.6]), np.zeros(4), np.full(4, 11.13))
""",
            "call": "eom_transition_density_product(M[0].copy(), M[1].copy(), np.ones(4), 2, 1)",
            "gold_call": "_oracle_eom_transition_density_product(M[0].copy(), M[1].copy(), np.ones(4), 2, 1)",
            "tol": 1e-9,
        },
        # --- Normal: an allyl cation, whose lowest singlet has a charged core ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13))
""",
            "call": "eom_transition_density_product(M[0].copy(), M[1].copy(), np.ones(3), 1, 0)",
            "gold_call": "_oracle_eom_transition_density_product(M[0].copy(), M[1].copy(), np.ones(3), 1, 0)",
            "tol": 1e-9,
        },
        # --- Boundary: a two-electron polar unit ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]))
""",
            "call": "eom_transition_density_product(M[0].copy(), M[1].copy(), np.ones(2), 1, 0)",
            "gold_call": "_oracle_eom_transition_density_product(M[0].copy(), M[1].copy(), np.ones(2), 1, 0)",
            "tol": 1e-9,
        },
        # --- Edge: the second singlet of a strongly donating C=C-N unit ---
        {
            "setup": """import numpy as np
M = _oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -1.6]), np.array([0.0, 0.0, -5.0]), np.array([11.13, 11.13, 16.76]))
""",
            "call": "eom_transition_density_product(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2, 1)",
            "gold_call": "_oracle_eom_transition_density_product(M[0].copy(), M[1].copy(), np.array([1.0, 1.0, 2.0]), 2, 1)",
            "tol": 1e-9,
        },
        # --- Invalid: a negative root index ---
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
            "call": "_exception_code(lambda: eom_transition_density_product(M[0].copy(), M[1].copy(), np.ones(2), 1, -1))",
            "gold_call": "_exception_code(lambda: _oracle_eom_transition_density_product(M[0].copy(), M[1].copy(), np.ones(2), 1, -1))",
        },
    ]
