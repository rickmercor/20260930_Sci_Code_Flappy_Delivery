"""
Step 03 - Canonical closed-shell RHF orbitals.

Closed-shell restricted Hartree-Fock orbitals and orbital energies.

The mean-field determinant is the reference of the whole correlated treatment:
the coupled-cluster excitation operators, the particle-hole labels of every
configuration, and the one- and two-electron integrals that drive the real-time
propagation are all expressed in the canonical orbitals of a closed-shell
restricted Hartree-Fock calculation.

The integrals come from the two previous steps for the given nuclei and basis.
With n_occ doubly occupied orbitals the density matrix is D = 2 C_occ C_occ^T
and the Fock matrix is F = H + J - K/2, where H is the core Hamiltonian
(kinetic plus nuclear attraction), J_mn = sum_ls (mn|ls) D_ls and
K_mn = sum_ls (ml|ns) D_ls. The Roothaan-Hall equations F C = S C e are solved
in the symmetrically orthogonalised basis S^(-1/2). The target is the
lowest-energy closed-shell solution with aufbau occupation; starting from the
eigenvectors of the core Hamiltonian and accelerating with DIIS reaches it for
every system in this task, including stretched geometries where other,
higher-lying closed-shell solutions exist. Iterate until the largest change of
any density matrix element is below 1e-10 and the electronic energy changes by
less than 1e-12 hartree, then diagonalise the converged Fock matrix once more
to obtain the canonical orbitals.

Canonical orbitals are unique only up to sign, so a sign convention is part of
the output: each orbital (column) is multiplied by -1 if needed so that its
first coefficient of magnitude above 1e-8, scanning the basis functions in
order, is positive. Orbital energies are returned in ascending order, and the
columns of the coefficient matrix follow the same order and satisfy
C^T S C = 1.

Returns
-------
numpy.ndarray of shape (n_bf + 1, n_bf): row 0 ascending orbital energies in hartree, rows 1 to n_bf the sign-fixed coefficient matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def rhf_canonical_orbitals(coords: np.ndarray, charges: np.ndarray, exponents: list, coefficients: list, n_occ: int) -> np.ndarray:
    '''Canonical closed-shell RHF orbital energies and coefficients.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,).
    exponents : list
        One 1D array of primitive exponents per contracted s shell, the same
        shells on every atom.
    coefficients : list
        One 1D array of contraction coefficients per shell (normalised
        primitives), in the same order as exponents.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n_bf.

    Returns
    -------
    orbitals : np.ndarray
        Array of shape (n_bf + 1, n_bf). Row 0 holds the orbital energies in
        ascending order (hartree); rows 1 to n_bf hold the coefficient matrix C,
        column k being orbital k, normalised so that C^T S C = 1 and signed so
        that the first coefficient of magnitude above 1e-8 in each column is
        positive.

    Raises
    ------
    ValueError
        If the geometry or basis is invalid (the conditions of the integral
        steps), if n_occ is not an integer in [1, n_bf], if the overlap matrix
        is not positive definite, or if the self-consistent field does not
        converge within 1000 iterations.
    '''
    return orbitals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _rhf_solve(S, H, eri, n_occ):
    """Converged closed-shell RHF: (electronic energy, orbital energies, C)."""
    n = S.shape[0]
    s_val, s_vec = np.linalg.eigh(S)
    if s_val[0] <= 1e-10:
        raise ValueError("overlap matrix is not positive definite")
    X = s_vec @ np.diag(s_val ** -0.5) @ s_vec.T

    def _fock(D):
        J = np.einsum('mnls,ls->mn', eri, D)
        K = np.einsum('mlns,ls->mn', eri, D)
        return H + J - 0.5 * K

    _, cp = np.linalg.eigh(X.T @ H @ X)
    C = X @ cp
    D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    e_old = None
    focks, errs = [], []
    converged = False
    for _ in range(1000):
        F = _fock(D)
        focks.append(F)
        errs.append(F @ D @ S - S @ D @ F)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        m = len(focks)
        B = -np.ones((m + 1, m + 1))
        B[m, m] = 0.0
        for x in range(m):
            for y in range(m):
                B[x, y] = np.sum(errs[x] * errs[y])
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0
        try:
            w = np.linalg.solve(B, rhs)
            F_use = sum(w[k] * focks[k] for k in range(m))
        except np.linalg.LinAlgError:
            F_use = F
        _, cp = np.linalg.eigh(X.T @ F_use @ X)
        C = X @ cp
        D_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        e_new = 0.5 * float(np.sum(D_new * (H + _fock(D_new))))
        if e_old is not None and np.max(np.abs(D_new - D)) < 1e-10 and abs(e_new - e_old) < 1e-12:
            D = D_new
            converged = True
            break
        D = D_new
        e_old = e_new
    if not converged:
        raise ValueError("RHF did not converge in 1000 iterations")
    F = _fock(D)
    eps, cp = np.linalg.eigh(X.T @ F @ X)
    C = X @ cp
    for k in range(n):
        lead = C[np.abs(C[:, k]) > 1e-8, k]
        if lead.size and lead[0] < 0.0:
            C[:, k] = -C[:, k]
    e_elec = 0.5 * float(np.sum(D * (H + F)))
    return e_elec, eps, C


def _oracle_rhf_canonical_orbitals(coords: np.ndarray, charges: np.ndarray, exponents: list, coefficients: list, n_occ: int) -> np.ndarray:
    ints = _oracle_s_type_one_electron_integrals(coords, charges, exponents, coefficients)
    eri = _oracle_s_type_electron_repulsion(coords, exponents, coefficients)
    n_bf = ints.shape[1]
    if isinstance(n_occ, bool) or not isinstance(n_occ, (int, np.integer)) or not 1 <= int(n_occ) <= n_bf:
        raise ValueError("n_occ must be an integer between 1 and n_bf")
    _, eps, C = _rhf_solve(ints[0], ints[1] + ints[2], eri, int(n_occ))
    return np.vstack([eps[None, :], C])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: four-atom chain at 1.8 bohr centred on the origin, split-valence basis ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, 1.8 * (k - 1.5)] for k in range(4)])
charges = np.ones(4)
exponents = [np.array([18.7311370, 2.8253944, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "rhf_canonical_orbitals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients), 2)",
            "gold_call": "_oracle_rhf_canonical_orbitals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients), 2)",
            "tol": 1e-7,
        },
        # --- Boundary: stretched chain at 3.4 bohr, where higher closed-shell solutions also exist ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, 3.4 * k] for k in range(4)])
charges = np.ones(4)
exponents = [np.array([18.7311370, 2.8253944, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "rhf_canonical_orbitals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients), 2)",
            "gold_call": "_oracle_rhf_canonical_orbitals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients), 2)",
            "tol": 1e-7,
        },
        # --- Edge: two-electron linear H3+ with unequal spacings in a minimal basis ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.65], [0.0, 0.0, 3.55]])
charges = np.ones(3)
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
""",
            "call": "rhf_canonical_orbitals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients), 1)",
            "gold_call": "_oracle_rhf_canonical_orbitals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients), 1)",
            "tol": 1e-7,
        },
        # --- Invalid: more occupied orbitals than basis functions ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
charges = np.ones(2)
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
def run_model():
    try:
        rhf_canonical_orbitals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients), 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_rhf_canonical_orbitals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients), 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
