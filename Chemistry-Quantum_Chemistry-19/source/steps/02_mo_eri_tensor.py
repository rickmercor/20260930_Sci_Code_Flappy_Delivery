"""
This step diagonalizes a converged Fock matrix, orders the canonical orbitals by increasing orbital energy, fixes the arbitrary sign of every orbital so that its coefficient of largest magnitude is positive, and returns the two-electron integrals in physicist (Dirac) notation, bracket p q bar r s, equal to the double sum over centres mu and nu of C(mu, p) C(nu, q) C(mu, r) C(nu, s) V(mu, nu). In this notation electron one carries the first and third index and electron two the second and fourth index, so that the chemist integral (p r | q s) and the Dirac integral bracket p q bar r s are the same number. Getting this index convention right is essential: the expressions of the later steps distinguish bracket i c bar k q from bracket i k bar c q, and for the PPP integrals these are different numbers.

Many-body perturbation theory is formulated in the basis of canonical Hartree-Fock orbitals. For the PPP model the only two-electron integrals in the site basis are density-density terms: the Ohno matrix element V between two centres multiplies the product of their populations, and U acts on the centre itself. Transforming to the molecular orbital basis spreads these few numbers over a full four-index tensor, which is what the screening and the self-energy expressions of the later steps consume.

Returns
-------
np.ndarray of shape (n, n, n, n): the Dirac two-electron integrals bracket p q bar r s in the phased canonical orbitals, in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mo_eri_tensor(fock, ohno):
    '''Two-electron integrals in the canonical orbital basis (Dirac notation).

    Parameters
    ----------
    fock : array_like of float, shape (n, n)
        Symmetric Fock matrix in the site basis, in eV.
    ohno : array_like of float, shape (n, n)
        Symmetric Ohno interaction matrix in the site basis (U on the
        diagonal), in eV.

    Returns
    -------
    eri : np.ndarray, shape (n, n, n, n)
        eri[p, q, r, s] = bracket p q bar r s in the canonical orbitals of
        fock, ordered by increasing orbital energy, each orbital phased so
        that its largest-magnitude site coefficient is positive.
        Raises ValueError if the matrices are not square, of the same size,
        symmetric within 1e-10, and finite.
    '''
    n = np.asarray(fock).shape[0]
    eri = np.zeros((n, n, n, n))
    return eri

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_mo_eri_tensor(fock, ohno):
    F = np.asarray(fock, dtype=float)
    V = np.asarray(ohno, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape != V.shape or F.shape[0] < 1:
        raise ValueError("fock and ohno must be square matrices of the same size")
    if not (np.all(np.isfinite(F)) and np.all(np.isfinite(V))):
        raise ValueError("fock and ohno must be finite")
    if np.max(np.abs(F - F.T)) > 1e-10 or np.max(np.abs(V - V.T)) > 1e-10:
        raise ValueError("fock and ohno must be symmetric")
    _, C = np.linalg.eigh(F)
    C = C.copy()
    for p in range(C.shape[1]):
        k = np.argmax(np.abs(C[:, p]))
        if C[k, p] < 0.0:
            C[:, p] *= -1.0
    return np.einsum("mp,nq,mr,ns,mn->pqrs", C, C, C, C, V, optimize=True)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def _ohno(lengths, U):
    pos = np.concatenate([[0.0], np.cumsum(lengths)])
    r = np.abs(pos[:, None] - pos[None, :])
    return 14.397 / np.sqrt((14.397 / U) ** 2 + r ** 2)
def _fock(lengths, hops, site, U, n_occ):
    V = _ohno(lengths, U)
    n = len(site)
    h = np.zeros((n, n))
    for k in range(n - 1):
        h[k, k + 1] = h[k + 1, k] = hops[k]
    for m in range(n):
        h[m, m] = site[m] - (np.sum(V[m]) - V[m, m])
    _, C = np.linalg.eigh(h)
    P = 2.0 * np.dot(C[:, :n_occ], C[:, :n_occ].T)
    for _ in range(2000):
        F = h + np.diag(np.dot(V, np.diag(P))) - 0.5 * P * V
        _, C = np.linalg.eigh(F)
        Pn = 2.0 * np.dot(C[:, :n_occ], C[:, :n_occ].T)
        if np.max(np.abs(Pn - P)) < 1e-12:
            P = Pn
            break
        P = 0.5 * (P + Pn)
    return h + np.diag(np.dot(V, np.diag(P))) - 0.5 * P * V, V
F6, V6 = _fock([1.34, 1.47, 1.36, 1.46, 1.38], [-2.23, -1.62, -2.02, -2.05, -2.10], [-14.2, -11.2, -10.5, -11.9, -10.4, -10.4], 14.0, 3)
F4, V4 = _fock([1.35, 1.45, 1.37], [-2.4, -1.9, -2.3], [-12.6, -11.0, -11.5, -10.2], 11.0, 2)
"""
    return [
        # --- Normal: the reference chain ---
        {
            "setup": setup,
            "call": "mo_eri_tensor(F6, V6)",
            "gold_call": "_oracle_mo_eri_tensor(F6, V6)",
        },
        # --- Normal: four-centre chain ---
        {
            "setup": setup,
            "call": "mo_eri_tensor(F4, V4)",
            "gold_call": "_oracle_mo_eri_tensor(F4, V4)",
        },
        # --- Boundary: a diagonal Fock matrix (orbitals are the sites themselves,
        #     so only density-density integrals survive) ---
        {
            "setup": setup + "Fd = np.diag([-13.0, -11.0, -9.0, -7.0])\n",
            "call": "mo_eri_tensor(Fd, V4)",
            "gold_call": "_oracle_mo_eri_tensor(Fd, V4)",
        },
        # --- Edge: a two-centre problem where the phase rule decides the sign of
        #     the mixed integrals ---
        {
            "setup": setup + "F2 = np.array([[-11.0, -2.5], [-2.5, -12.4]])\nV2 = np.array([[10.0, 5.9], [5.9, 10.0]])\n",
            "call": "mo_eri_tensor(F2, V2)",
            "gold_call": "_oracle_mo_eri_tensor(F2, V2)",
        },
        # --- Invalid: non-symmetric Fock matrix ---
        {
            "setup": setup + """
def run_model():
    try:
        mo_eri_tensor(np.array([[-11.0, -2.5], [-2.0, -11.0]]), np.array([[10.0, 5.9], [5.9, 10.0]]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mo_eri_tensor(np.array([[-11.0, -2.5], [-2.0, -11.0]]), np.array([[10.0, 5.9], [5.9, 10.0]]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: size mismatch ---
        {
            "setup": setup + """
def run_model():
    try:
        mo_eri_tensor(F4, V6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mo_eri_tensor(F4, V6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
