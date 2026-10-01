"""
Step 03 - Closed-shell restricted Hartree-Fock reference and its canonical orbitals.

The correlated treatment that follows starts from the closed-shell restricted Hartree-Fock determinant. Solve the Roothaan-Hall equations F C = S C e self-consistently, with the Fock matrix built from the core Hamiltonian and the density of the n_electrons/2 lowest orbitals, each doubly occupied. Start from the orbitals of the core Hamiltonian and iterate until the density matrix changes by less than about 1e-10; convergence acceleration such as DIIS helps but is not required. For every system used here this procedure reaches the lowest closed-shell solution.

The canonical orbitals are the eigenvectors of the converged Fock matrix, normalised in the overlap metric and listed in ascending order of orbital energy. An eigenvector is defined only up to sign, so fix each one by making its largest-magnitude coefficient positive; if two coefficients of the same orbital agree in magnitude to a relative 1e-8, the one with the lower basis-function index decides.

Returns
-------
numpy.ndarray of shape (n + 1, n): canonical orbital energies in hartree (row 0) and the sign-fixed orbital coefficients (rows 1 to n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def rhf_canonical_orbitals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list, n_electrons: int) -> np.ndarray:
    '''Canonical orbitals of the closed-shell restricted Hartree-Fock determinant.

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

    Returns
    -------
    orbitals : numpy.ndarray
        Array of shape (n + 1, n). Row 0 holds the canonical orbital energies
        in hartree in ascending order. Rows 1 to n hold the coefficient matrix
        C, whose column p expands orbital p in the basis functions, each
        column sign-fixed so that its largest-magnitude entry is positive (the
        lowest index decides between entries equal to a relative 1e-8).

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, charges does not hold one positive finite value per atom,
        the basis description is malformed (empty, mismatched lengths,
        non-positive or non-finite exponents, non-finite or all-zero
        coefficients), n_electrons is not a positive even integer or leaves
        no virtual orbital, the smallest eigenvalue of the overlap matrix is
        below 1e-10, or the SCF iterations do not converge within 1000
        cycles.
    '''
    return orbitals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _positive_phase_columns(M):
    """Flip each column so that its largest-magnitude entry (first one among near-ties) is positive."""
    import numpy as np
    M = np.array(M, dtype=float)
    for j in range(M.shape[1]):
        col = np.abs(M[:, j])
        k = int(np.flatnonzero(col >= col.max() * (1.0 - 1.0e-8))[0])
        if M[k, j] < 0.0:
            M[:, j] = -M[:, j]
    return M


def _closed_shell_count(n_electrons, n_basis):
    import numpy as np
    ne = float(n_electrons)
    if not np.isfinite(ne) or ne != int(ne) or int(ne) < 2 or int(ne) % 2 != 0:
        raise ValueError("n_electrons must be a positive even integer")
    nocc = int(ne) // 2
    if nocc >= n_basis:
        raise ValueError("the basis must leave at least one virtual orbital")
    return nocc


def _closed_shell_fock(H, eri, D):
    """Closed-shell Fock matrix H + 2J - K for the density D = C_occ C_occ^T."""
    import numpy as np
    return H + 2.0 * np.einsum('pqrs,rs->pq', eri, D) - np.einsum('prqs,rs->pq', eri, D)


def _oracle_rhf_canonical_orbitals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                                   coefficients: list, n_electrons: int) -> np.ndarray:
    import numpy as np
    SH = _oracle_one_electron_integrals(coords, charges, exponents, coefficients)
    S, H = SH[0], SH[1]
    eri = _oracle_electron_repulsion_integrals(coords, exponents, coefficients)
    n = S.shape[0]
    nocc = _closed_shell_count(n_electrons, n)
    s, U = np.linalg.eigh(S)
    if s[0] < 1.0e-10:
        raise ValueError("the basis is numerically linearly dependent")
    X = U @ np.diag(s ** -0.5) @ U.T
    e, Cp = np.linalg.eigh(X.T @ H @ X)
    C = X @ Cp
    D = C[:, :nocc] @ C[:, :nocc].T
    F_hist, R_hist = [], []
    for it in range(1000):
        F = _closed_shell_fock(H, eri, D)
        F_hist.append(F)
        R_hist.append(F @ D @ S - S @ D @ F)
        F_hist, R_hist = F_hist[-8:], R_hist[-8:]
        m = len(F_hist)
        if m > 1:
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0.0
            for a in range(m):
                for b in range(m):
                    B[a, b] = np.sum(R_hist[a] * R_hist[b])
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            w = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            F = sum(wi * Fi for wi, Fi in zip(w, F_hist))
        e, Cp = np.linalg.eigh(X.T @ F @ X)
        C = X @ Cp
        D_new = C[:, :nocc] @ C[:, :nocc].T
        change = np.max(np.abs(D_new - D))
        D = D_new
        if change < 1.0e-12 and it > 1:
            break
    else:
        raise ValueError("the SCF iterations did not converge")
    e, Cp = np.linalg.eigh(X.T @ _closed_shell_fock(H, eri, D) @ X)
    C = _positive_phase_columns(X @ Cp)
    return np.vstack([e[None, :], C])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the four-electron target cluster ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.60, 0.00, 0.00], [1.90, 2.40, 0.20], [0.20, 2.50, 0.00]])
charges = np.array([1.0, 1.0, 1.0, 1.0])
n_electrons = 4
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "rhf_canonical_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons)",
            "gold_call": "_oracle_rhf_canonical_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons)",
            "tol": 1e-07,
        },
        # --- Boundary: two electrons, a single doubly occupied orbital ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.62, 0.00, 0.00], [0.70, 1.52, 0.05]])
charges = np.array([1.0, 1.0, 1.0])
n_electrons = 2
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "rhf_canonical_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons)",
            "gold_call": "_oracle_rhf_canonical_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons)",
            "tol": 1e-07,
        },
        # --- Edge: strongly polar two-centre cation with unequal charges ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [0.00, 0.00, 1.46]])
charges = np.array([2.0, 1.0])
n_electrons = 2
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "rhf_canonical_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons)",
            "gold_call": "_oracle_rhf_canonical_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons)",
            "tol": 1e-07,
        },
        # --- Normal: an open zigzag chain in a minimal basis ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.50, 0.00, 0.00], [2.90, 1.20, 0.00], [4.60, 0.90, 0.00]])
charges = np.array([1.0, 1.0, 1.0, 1.0])
n_electrons = 4
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
""",
            "call": "rhf_canonical_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons)",
            "gold_call": "_oracle_rhf_canonical_orbitals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons)",
            "tol": 1e-07,
        },
    ]
