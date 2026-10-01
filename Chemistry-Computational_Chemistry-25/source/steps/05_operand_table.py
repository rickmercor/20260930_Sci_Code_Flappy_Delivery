"""
Collect the eleven matrices the operations are allowed to use, in their fixed order. SPECIFICATION: core is the array of shape (3, n, n) returned by the one-electron step and descriptors the array of shape (4, n, n) returned by the previous step. The result is a real array of shape (11, n, n) whose entries are, in this order: the overlap matrix; the kinetic matrix; the nuclear attraction matrix; the inverse square root of the overlap matrix; the distance matrix; the element-wise reciprocal of the distance matrix, defined to be zero at every position where the distance itself is zero; the square root of the overlap matrix; and then the three repulsion slices in the order they arrive, followed by the matrix inverse of the overlap matrix. Both the square root and the inverse square root are the symmetric ones, obtained by diagonalising the overlap matrix and raising its eigenvalues to the power one half or minus one half while keeping its eigenvectors.

Fixing this order once is what gives every operation a stable number, and the numbers are the whole vocabulary of the search: a sequence of two integers is a candidate theory of the electronic structure. Two of the eleven entries are singular objects handled by convention rather than by algebra. The reciprocal distance matrix would be infinite on every block of orbitals sharing an atom, and the square roots of the overlap matrix are defined through its eigendecomposition rather than element-wise, so they are genuine matrix functions and not tables of square roots.

Returns
-------
numpy.ndarray of shape (11, n, n) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def operand_table(core: "np.ndarray", descriptors: "np.ndarray") -> "np.ndarray":
    """Collect the eleven matrices the operations are allowed to use, in their fixed order. SPECIFICATION: core is the array of shape (3, n, n) returned by the one-electron step and descriptors the array of shape (4, n, n) returned by the previous step. The result is a real array of shape (11, n, n) whose entries are, in this order: the overlap matrix; the kinetic matrix; the nuclear attraction matrix; the inverse square root of the overlap matrix; the distance matrix; the element-wise reciprocal of the distance matrix, defined to be zero at every position where the distance itself is zero; the square root of the overlap matrix; and then the three repulsion slices in the order they arrive, followed by the matrix inverse of the overlap matrix. Both the square root and the inverse square root are the symmetric ones, obtained by diagonalising the overlap matrix and raising its eigenvalues to the power one half or minus one half while keeping its eigenvectors.

    Parameters
    ----------
    core : numpy.ndarray of shape (3, n, n) and real dtype
        The overlap, kinetic-energy and nuclear-attraction matrices, in that order.
    descriptors : numpy.ndarray of shape (4, n, n) and real dtype
        The distance matrix and the three repulsion slices (mm|nn), (mn|mn) and (mn|nn),
        in that order.

    Returns
    -------
    numpy.ndarray of shape (11, n, n) and real dtype.

    Raises
    ------
    ValueError: if core does not have shape (3, n, n), if descriptors does not have shape (4, n, n), or if the overlap matrix is singular or not positive definite.
    """
    return [[[0.0]]]  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _sym_power(S, power):
    w, U = np.linalg.eigh(np.asarray(S, dtype=float))
    if np.min(w) <= 0.0:
        raise ValueError("overlap matrix is not positive definite")
    return (U*np.power(w, power)) @ U.T


def _safe_inverse(A):
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("operand is not square")
    try:
        B = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("operand has no inverse")
    if not np.all(np.isfinite(B)):
        raise ValueError("operand has no finite inverse")
    return B


def _oracle_operand_table(core: "np.ndarray", descriptors: "np.ndarray") -> "np.ndarray":
    core = np.asarray(core, dtype=float)
    descriptors = np.asarray(descriptors, dtype=float)
    if core.ndim != 3 or core.shape[0] != 3:
        raise ValueError("core must have shape (3, n, n)")
    n = core.shape[1]
    if descriptors.shape != (4, n, n):
        raise ValueError("descriptors must have shape (4, n, n)")
    S, T, V = core
    D, mmnn, mnmn, mnnn = descriptors
    Dinv = np.where(D == 0.0, 0.0, 1.0/np.where(D == 0.0, 1.0, D))
    return np.stack([S, T, V, _sym_power(S, -0.5), D, Dinv, _sym_power(S, 0.5),
                     mmnn, mnmn, mnnn, _safe_inverse(S)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base_0 = """from copy import deepcopy as _dc
import numpy as np
BD = {'Li': [(0, [16.119575, 2.9362007, 0.7946505], [0.15432897, 0.53532814, 0.44463454]), (0, [0.6362897, 0.1478601, 0.0480887], [-0.09996723, 0.39951283, 0.70011547]), (1, [0.6362897, 0.1478601, 0.0480887], [0.15591627, 0.60768372, 0.39195739])], 'F': [(0, [166.67913, 30.360812, 8.2168207], [0.15432897, 0.53532814, 0.44463454]), (0, [6.4648032, 1.5022812, 0.4885885], [-0.09996723, 0.39951283, 0.70011547]), (1, [6.4648032, 1.5022812, 0.4885885], [0.15591627, 0.60768372, 0.39195739])]}
BD_G = _dc(BD)
CO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6 / 0.52917721092]])
CO_G = _dc(CO)
TB = basis_table(['Li', 'F'], BD)
CORE = core_integrals(TB, CO, [3.0, 9.0])
ERI = electron_repulsion(TB, CO)
DS = descriptor_matrices(TB, CO, ERI)
TB_G = _oracle_basis_table(['Li', 'F'], BD_G)
CORE_G = _oracle_core_integrals(TB_G, CO_G, [3.0, 9.0])
ERI_G = _oracle_electron_repulsion(TB_G, CO_G)
DS_G = _oracle_descriptor_matrices(TB_G, CO_G, ERI_G)"""
    base_1 = """from copy import deepcopy as _dc
import numpy as np
TOY = {'X': [(0, [1.9, 0.44, 0.14], [0.16, 0.53, 0.45])], 'Y': [(0, [4.1, 0.95, 0.31], [0.15, 0.54, 0.44]), (1, [0.83, 0.29, 0.11], [0.16, 0.61, 0.39])]}
TOY_G = _dc(TOY)
TT = basis_table(['X', 'Y'], TOY)
"""
    return [
        {
            "setup": base_0,
            'call': 'operand_table(CORE,DS)',
            'gold_call': '_oracle_operand_table(CORE_G,DS_G)',
        },
        {
            "setup": base_1 + """TCO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]])
TCO_G = _dc(TCO)
TCORE = core_integrals(TT, TCO, [2.0, 4.0])
TERI = electron_repulsion(TT, TCO)
TDS = descriptor_matrices(TT, TCO, TERI)
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE_G = _oracle_core_integrals(TT_G, TCO_G, [2.0, 4.0])
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)
TDS_G = _oracle_descriptor_matrices(TT_G, TCO_G, TERI_G)""",
            'call': 'operand_table(TCORE,TDS)',
            'gold_call': '_oracle_operand_table(TCORE_G,TDS_G)',
        },
        {
            "setup": base_1 + """TCO2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
TCO2_G = _dc(TCO2)
TCORE2 = core_integrals(TT, TCO2, [2.0, 4.0])
TERI2 = electron_repulsion(TT, TCO2)
TDS2 = descriptor_matrices(TT, TCO2, TERI2)
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE2_G = _oracle_core_integrals(TT_G, TCO2_G, [2.0, 4.0])
TERI2_G = _oracle_electron_repulsion(TT_G, TCO2_G)
TDS2_G = _oracle_descriptor_matrices(TT_G, TCO2_G, TERI2_G)""",
            'call': 'operand_table(TCORE2,TDS2)',
            'gold_call': '_oracle_operand_table(TCORE2_G,TDS2_G)',
        },
        {
            "setup": base_1 + """TCO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]])
TCO_G = _dc(TCO)
TCORE = core_integrals(TT, TCO, [2.0, 4.0])
TERI = electron_repulsion(TT, TCO)
TDS = descriptor_matrices(TT, TCO, TERI)
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE_G = _oracle_core_integrals(TT_G, TCO_G, [2.0, 4.0])
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)
TDS_G = _oracle_descriptor_matrices(TT_G, TCO_G, TERI_G)

def _c():
    try:
        operand_table(TCORE, TDS[:3])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _g():
    try:
        _oracle_operand_table(TCORE_G, TDS_G[:3])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
    ]
