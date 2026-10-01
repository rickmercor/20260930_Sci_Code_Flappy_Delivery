"""
Read an electronic energy out of a workspace matrix. SPECIFICATION: solve the generalised eigenvalue problem in which M plays the role of the effective Hamiltonian and S the metric, that is find the coefficients and eigenvalues satisfying M c = S c e. Order the eigenvalues ascending and keep the n_occ eigenvectors with the smallest ones. The density matrix is twice the product of that block of coefficients with its own transpose. The Coulomb matrix has element (m, n) equal to the sum over l and s of the repulsion element (m, n, l, s) times the density element (l, s), and the exchange matrix has element (m, n) equal to the sum over l and s of the repulsion element (m, l, s, n) times the density element (l, s). The Fock matrix is hcore plus the Coulomb matrix minus one half of the exchange matrix. Return one half of the sum over all elements of the density matrix times the sum of hcore and the Fock matrix. Nuclear repulsion is not included.

This is the ordinary closed-shell energy expression, but it is being evaluated at a density that never went through a self-consistent cycle, which is the whole point: the workspace matrix is asked to supply orbitals directly. The expression is variational about the true solution, so it is forgiving of small orbital errors and punishing of large ones, which is what makes it a usable score. Occupying the lowest eigenvalues is an assumption rather than a theorem here, because the workspace matrix is not a Fock matrix and its spectrum carries no guarantee.

Returns
-------
float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def program_energy(M: "np.ndarray", S: "np.ndarray", hcore: "np.ndarray", eri: "np.ndarray", n_occ: "int") -> "float":
    """Read an electronic energy out of a workspace matrix. SPECIFICATION: solve the generalised eigenvalue problem in which M plays the role of the effective Hamiltonian and S the metric, that is find the coefficients and eigenvalues satisfying M c = S c e. Order the eigenvalues ascending and keep the n_occ eigenvectors with the smallest ones. The density matrix is twice the product of that block of coefficients with its own transpose. The Coulomb matrix has element (m, n) equal to the sum over l and s of the repulsion element (m, n, l, s) times the density element (l, s), and the exchange matrix has element (m, n) equal to the sum over l and s of the repulsion element (m, l, s, n) times the density element (l, s). The Fock matrix is hcore plus the Coulomb matrix minus one half of the exchange matrix. Return one half of the sum over all elements of the density matrix times the sum of hcore and the Fock matrix. Nuclear repulsion is not included.

    Parameters
    ----------
    M : numpy.ndarray of shape (n, n) and real dtype
        The symmetrised workspace matrix whose orbitals are used.
    S : numpy.ndarray of shape (n, n) and real dtype
        The overlap matrix.
    hcore : numpy.ndarray of shape (n, n) and real dtype
        The core Hamiltonian.
    eri : numpy.ndarray of shape (n, n, n, n) and real dtype
        The two-electron repulsion tensor in chemists' notation.
    n_occ : int
        The number of doubly occupied orbitals, at least 1 and at most n.

    Returns
    -------
    float.

    Raises
    ------
    ValueError: if n_occ is not between one and the size of the basis, if the matrix shapes are inconsistent, if the generalised eigenvalue problem cannot be solved, or if the energy is not finite.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _oracle_program_energy(M: "np.ndarray", S: "np.ndarray", hcore: "np.ndarray", eri: "np.ndarray", n_occ: "int") -> "float":
    M = np.asarray(M, dtype=float)
    S = np.asarray(S, dtype=float)
    hcore = np.asarray(hcore, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n = M.shape[0]
    n_occ = int(n_occ)
    if n_occ < 1 or n_occ > n:
        raise ValueError("n_occ outside the basis")
    if S.shape != (n, n) or hcore.shape != (n, n) or eri.shape != (n, n, n, n):
        raise ValueError("inconsistent matrix shapes")
    try:
        w, C = eigh(M, S)
    except Exception:
        raise ValueError("the generalised eigenvalue problem could not be solved")
    Co = C[:, :n_occ]
    P = 2.0*Co @ Co.T
    J = np.einsum("mnls,ls->mn", eri, P)
    K = np.einsum("mlsn,ls->mn", eri, P)
    e = 0.5*float(np.sum(P*(hcore + hcore + J - 0.5*K)))
    if not np.isfinite(e):
        raise ValueError("non-finite electronic energy")
    return e

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base_0 = """from copy import deepcopy as _dc
import numpy as np
TOY = {'X': [(0, [1.9, 0.44, 0.14], [0.16, 0.53, 0.45])], 'Y': [(0, [4.1, 0.95, 0.31], [0.15, 0.54, 0.44]), (1, [0.83, 0.29, 0.11], [0.16, 0.61, 0.39])]}
TOY_G = _dc(TOY)
TT = basis_table(['X', 'Y'], TOY)
TCO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]])
TCO_G = _dc(TCO)
TCORE = core_integrals(TT, TCO, [2.0, 4.0])
TERI = electron_repulsion(TT, TCO)
"""
    return [
        {
            "setup": base_0 + """TDS = descriptor_matrices(TT, TCO, TERI)
TOPS = operand_table(TCORE, TDS)
THC = TCORE[1] + TCORE[2]
TWM = workspace_matrix(THC, [117, 102], TOPS)
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE_G = _oracle_core_integrals(TT_G, TCO_G, [2.0, 4.0])
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)
TDS_G = _oracle_descriptor_matrices(TT_G, TCO_G, TERI_G)
TOPS_G = _oracle_operand_table(TCORE_G, TDS_G)
TWM_G = _oracle_workspace_matrix((TCORE_G[1] + TCORE_G[2]), [117, 102], TOPS_G)""",
            'call': 'program_energy(TWM,TCORE[0],THC,TERI,3)',
            'gold_call': '_oracle_program_energy(TWM_G,TCORE_G[0],(TCORE_G[1] + TCORE_G[2]),TERI_G,3)',
        },
        {
            "setup": base_0 + """THC = TCORE[1] + TCORE[2]
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE_G = _oracle_core_integrals(TT_G, TCO_G, [2.0, 4.0])
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)""",
            'call': 'program_energy(THC,TCORE[0],THC,TERI,3)',
            'gold_call': '_oracle_program_energy((TCORE_G[1] + TCORE_G[2]),TCORE_G[0],(TCORE_G[1] + TCORE_G[2]),TERI_G,3)',
        },
        {
            "setup": base_0 + """TDS = descriptor_matrices(TT, TCO, TERI)
TOPS = operand_table(TCORE, TDS)
THC = TCORE[1] + TCORE[2]
TWM = workspace_matrix(THC, [117, 102], TOPS)
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE_G = _oracle_core_integrals(TT_G, TCO_G, [2.0, 4.0])
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)
TDS_G = _oracle_descriptor_matrices(TT_G, TCO_G, TERI_G)
TOPS_G = _oracle_operand_table(TCORE_G, TDS_G)
TWM_G = _oracle_workspace_matrix((TCORE_G[1] + TCORE_G[2]), [117, 102], TOPS_G)""",
            'call': 'program_energy(TWM,TCORE[0],THC,TERI,1)',
            'gold_call': '_oracle_program_energy(TWM_G,TCORE_G[0],(TCORE_G[1] + TCORE_G[2]),TERI_G,1)',
        },
        {
            "setup": base_0 + """TDS = descriptor_matrices(TT, TCO, TERI)
TOPS = operand_table(TCORE, TDS)
THC = TCORE[1] + TCORE[2]
TWM = workspace_matrix(THC, [117, 102], TOPS)
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE_G = _oracle_core_integrals(TT_G, TCO_G, [2.0, 4.0])
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)
TDS_G = _oracle_descriptor_matrices(TT_G, TCO_G, TERI_G)
TOPS_G = _oracle_operand_table(TCORE_G, TDS_G)
TWM_G = _oracle_workspace_matrix((TCORE_G[1] + TCORE_G[2]), [117, 102], TOPS_G)

def _c():
    try:
        program_energy(TWM, TCORE[0], THC, TERI, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _g():
    try:
        _oracle_program_energy(TWM_G, TCORE_G[0], (TCORE_G[1] + TCORE_G[2]), TERI_G, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
    ]
