"""
Step 04 - Self-consistent orbital dressing by the two-parameter regularisation operator.

The regularised perturbation theory used here keeps the Hartree-Fock determinant as the reference but moves part of the correlation into the zeroth-order Hamiltonian. The Fock operator f is supplemented by a one-particle operator W that has no occupied-virtual block and is built from first-order doubles amplitudes. Work in spin orbitals, each restricted spatial orbital carrying an alpha and a beta spin orbital, with antisymmetrised integrals <pq||rs> = <pq|rs> - <pq|sr> in physicists' notation, occupied indices i, j, k, virtual indices a, b, c, and amplitudes t_ij^ab = <ij||ab> / (e_a + e_b - e_i - e_j), where the e are the current zeroth-order orbital energies. Then

W_ij = -(A0/8) * sum over k, a, b of ( <ik||ab> t_jk^ab + <jk||ab> t_ik^ab )

W_ab = -(B0/8) * sum over i, j, c of ( <ac||ij> t_ij^bc + <bc||ij> t_ij^ac )

Here A0 regularises the occupied block and B0 the virtual block, and A0 = B0 = 0 gives back Møller-Plesset partitioning.

The dressing is iterated to self-consistency. Start from the canonical Hartree-Fock orbitals and energies. In each cycle express the integrals in the current orbitals, form t with the current energies, build W, and diagonalise f + W separately inside the occupied block and inside the virtual block, with f the Hartree-Fock Fock operator written in the current orbitals; f itself is never altered. The eigenvalues become the new zeroth-order energies and the eigenvectors rotate the orbitals. Stop when the second-order energy E(2) = -(1/4) * sum of <ij||ab> t_ij^ab changes by less than 1e-12 hartree and f + W is diagonal to better than 1e-11. The alpha and beta blocks are identical, so the dressed orbitals stay restricted and each dressed spatial orbital has one energy.

Returns
-------
numpy.ndarray of shape (n + 1, n): dressed spatial orbital energies in hartree (row 0) and the sign-fixed dressed orbital coefficients (rows 1 to n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def dressed_orbitals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list, n_electrons: int, A0: float, B0: float) -> np.ndarray:
    '''Self-consistently dressed orbitals of the two-parameter regularised partitioning.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : array_like
        Nuclear charges, one positive value per atom.
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.
    n_electrons : int
        Number of electrons, a positive even integer.
    A0 : float
        Regularisation parameter of the occupied block of W.
    B0 : float
        Regularisation parameter of the virtual block of W.

    Returns
    -------
    dressed : numpy.ndarray
        Array of shape (n + 1, n). Row 0 holds the converged dressed spatial
        orbital energies in hartree, occupied ones first and each block in
        ascending order (every occupied level lies below every virtual one).
        Rows 1 to n hold the dressed orbitals as columns expanded in the basis
        functions, each sign-fixed so that its largest-magnitude entry is
        positive (the lowest index decides between entries equal to a
        relative 1e-8).

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, charges does not hold one positive finite value per atom,
        the basis description is malformed (empty, mismatched lengths,
        non-positive or non-finite exponents, non-finite or all-zero
        coefficients), n_electrons is not a positive even integer or leaves
        no virtual orbital, the smallest eigenvalue of the overlap matrix is
        below 1e-10, the SCF iterations do not converge within 1000 cycles,
        A0 or B0 is not finite, the dressed occupied and virtual levels
        overlap during the iterations, or the dressing does not converge
        within 2000 cycles.
    '''
    return dressed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _spin_orbital_antisymmetrized(eri_mo):
    """<pq||rs> for spin orbitals p = 2k + s (s = 0 alpha, 1 beta) from chemist integrals (kl|mn)."""
    import numpy as np
    n = eri_mo.shape[0]
    k = np.arange(2 * n) // 2
    s = np.arange(2 * n) % 2
    g = eri_mo[np.ix_(k, k, k, k)].transpose(0, 2, 1, 3)
    same = (s[:, None, None, None] == s[None, None, :, None]) & (s[None, :, None, None] == s[None, None, None, :])
    g = np.where(same, g, 0.0)
    return g - g.transpose(0, 1, 3, 2)


def _pair_denominators(eo, ev):
    """e_a + e_b - e_i - e_j as an (o, o, v, v) array."""
    import numpy as np
    return ev[None, None, :, None] + ev[None, None, None, :] - eo[:, None, None, None] - eo[None, :, None, None]


def _oracle_dressed_orbitals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                             coefficients: list, n_electrons: int, A0: float, B0: float) -> np.ndarray:
    import numpy as np
    A0 = float(A0)
    B0 = float(B0)
    if not (np.isfinite(A0) and np.isfinite(B0)):
        raise ValueError("A0 and B0 must be finite")
    ref = _oracle_rhf_canonical_orbitals(coords, charges, exponents, coefficients, n_electrons)
    e, C = ref[0], ref[1:]
    eri = _oracle_electron_repulsion_integrals(coords, exponents, coefficients)
    eri_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, C, C, C, C, optimize=True)
    g = _spin_orbital_antisymmetrized(eri_mo)
    n = e.size
    nocc = int(n_electrons) // 2
    no = 2 * nocc
    eps = np.repeat(e, 2)
    Ro = np.eye(nocc)
    Rv = np.eye(n - nocc)
    eo = eps[:no].copy()
    ev = eps[no:].copy()
    E_old = None
    for it in range(2000):
        Uo = np.kron(Ro, np.eye(2))
        Uv = np.kron(Rv, np.eye(2))
        goovv = np.einsum('pqrs,pi,qj,ra,sb->ijab', g[:no, :no, no:, no:], Uo, Uo, Uv, Uv, optimize=True)
        t = goovv / _pair_denominators(eo, ev)
        E2 = -0.25 * np.sum(goovv * t)
        X = np.einsum('ikab,jkab->ij', goovv, t)
        Y = np.einsum('ijac,ijbc->ab', goovv, t)
        Woo = -(A0 / 8.0) * (X + X.T)
        Wvv = -(B0 / 8.0) * (Y + Y.T)
        Ho = (Uo.T @ np.diag(eps[:no]) @ Uo + Woo)[0::2, 0::2]
        Hv = (Uv.T @ np.diag(eps[no:]) @ Uv + Wvv)[0::2, 0::2]
        off = max(np.max(np.abs(Ho - np.diag(np.diag(Ho)))), np.max(np.abs(Hv - np.diag(np.diag(Hv)))))
        converged = E_old is not None and abs(E2 - E_old) < 1.0e-13 and off < 1.0e-12
        wo, Qo = np.linalg.eigh(Ho)
        wv, Qv = np.linalg.eigh(Hv)
        if not (np.all(np.isfinite(wo)) and np.all(np.isfinite(wv))) or wv[0] <= wo[-1]:
            raise ValueError("the dressed occupied and virtual levels overlap; the dressing has no solution here")
        Ro = Ro @ Qo
        Rv = Rv @ Qv
        eo = np.repeat(wo, 2)
        ev = np.repeat(wv, 2)
        E_old = E2
        if converged:
            break
    else:
        raise ValueError("the self-consistent dressing did not converge")
    R = np.zeros((n, n))
    R[:nocc, :nocc] = Ro
    R[nocc:, nocc:] = Rv
    Cd = _positive_phase_columns(C @ R)
    return np.vstack([np.concatenate([wo, wv])[None, :], Cd])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: target cluster with occupied and virtual regularisation ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.60, 0.00, 0.00], [1.90, 2.40, 0.20], [0.20, 2.50, 0.00]])
charges = np.array([1.0, 1.0, 1.0, 1.0])
n_electrons = 4
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
A0 = -0.75
B0 = 0.5
""",
            "call": "dressed_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "gold_call": "_oracle_dressed_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "tol": 1e-07,
        },
        # --- Boundary: A0 = B0 = 0, the dressing must return the canonical orbitals ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.62, 0.00, 0.00], [0.70, 1.52, 0.05]])
charges = np.array([1.0, 1.0, 1.0])
n_electrons = 2
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
A0 = 0.0
B0 = 0.0
""",
            "call": "dressed_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "gold_call": "_oracle_dressed_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "tol": 1e-07,
        },
        # --- Edge: occupied-only regularisation, the virtual block stays canonical ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.50, 0.00, 0.00], [2.90, 1.20, 0.00], [4.60, 0.90, 0.00]])
charges = np.array([1.0, 1.0, 1.0, 1.0])
n_electrons = 4
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
A0 = 3.8
B0 = 0.0
""",
            "call": "dressed_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "gold_call": "_oracle_dressed_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "tol": 1e-07,
        },
        # --- Normal: strong virtual regularisation of opposite sign ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [0.00, 0.00, 1.46]])
charges = np.array([2.0, 1.0])
n_electrons = 2
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
A0 = 1.0
B0 = -3.0
""",
            "call": "dressed_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "gold_call": "_oracle_dressed_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "tol": 1e-07,
        },
    ]
