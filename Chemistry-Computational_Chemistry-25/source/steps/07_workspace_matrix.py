"""
Run a whole program and return the symmetrised workspace matrix. SPECIFICATION: the workspace matrix starts as hcore and each entry of program, an integer operation id, is applied to it in order using the rules of the previous step. Once every operation has been applied the matrix is symmetrised by taking L to be its lower triangle including the diagonal and replacing it by L plus the transpose of L minus the diagonal of L, so that the final matrix agrees with the unsymmetrised one on and below the diagonal and is symmetric. An empty program leaves the workspace matrix at hcore before that symmetrisation. The result is a real symmetric array with the same shape as hcore.

Symmetry survives much of the catalogue: adding, scaling or applying an element-wise function to a symmetric matrix leaves it symmetric, and so does combining it element-wise with a symmetric operand. What does not survive is a matrix product, which of two symmetric matrices is not symmetric in general, nor an element-wise combination with the one operand that is itself asymmetric or with its matrix inverse. Every program in the search opens with a matrix product, so the workspace matrix reaching this step is in general no longer symmetric, although particular programs, and the empty program, can return a symmetric matrix. Once the workspace matrix has lost its symmetry the eigenvalue problem that follows is no longer guaranteed real. Reflecting the lower triangle is the cheapest repair, and it is not a neutral one, because it throws away the upper triangle rather than averaging the two. Which half survives is fixed by the order the orbitals were laid out in, so this single line ties the answer back to the very first step.

Returns
-------
numpy.ndarray of shape (n, n) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def workspace_matrix(hcore: "np.ndarray", program: "list", operands: "np.ndarray") -> "np.ndarray":
    """Run a whole program and return the symmetrised workspace matrix. SPECIFICATION: the workspace matrix starts as hcore and each entry of program, an integer operation id, is applied to it in order using the rules of the previous step. Once every operation has been applied the matrix is symmetrised by taking L to be its lower triangle including the diagonal and replacing it by L plus the transpose of L minus the diagonal of L, so that the final matrix agrees with the unsymmetrised one on and below the diagonal and is symmetric. An empty program leaves the workspace matrix at hcore before that symmetrisation. The result is a real symmetric array with the same shape as hcore.

    Parameters
    ----------
    hcore : numpy.ndarray of shape (n, n) and real dtype
        The core Hamiltonian, which the workspace matrix starts as.
    program : list of int
        The operation ids to apply, in order. May be empty.
    operands : numpy.ndarray of shape (11, n, n) and real dtype
        The eleven operand matrices, in the order fixed by the operand table.

    Returns
    -------
    numpy.ndarray of shape (n, n) and real dtype.

    Raises
    ------
    ValueError: if hcore is not square, or if any operation in program fails for the reasons listed for the previous step.
    """
    return [[0.0]]  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _oracle_workspace_matrix(hcore: "np.ndarray", program: "list", operands: "np.ndarray") -> "np.ndarray":
    M = np.asarray(hcore, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("hcore must be a square matrix")
    for op_id in program:
        M = _oracle_apply_operation(M, op_id, operands)
    L = np.tril(M)
    return L + L.T - np.diag(np.diag(L))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base_0 = """from copy import deepcopy as _dc
import numpy as np
TOY = {'X': [(0, [1.9, 0.44, 0.14], [0.16, 0.53, 0.45])], 'Y': [(0, [4.1, 0.95, 0.31], [0.15, 0.54, 0.44]), (1, [0.83, 0.29, 0.11], [0.16, 0.61, 0.39])]}
TOY_G = _dc(TOY)
TT = basis_table(['X', 'Y'], TOY)
"""
    return [
        {
            "setup": base_0 + """TCO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]])
TCO_G = _dc(TCO)
TCORE = core_integrals(TT, TCO, [2.0, 4.0])
TERI = electron_repulsion(TT, TCO)
TDS = descriptor_matrices(TT, TCO, TERI)
TOPS = operand_table(TCORE, TDS)
THC = TCORE[1] + TCORE[2]
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE_G = _oracle_core_integrals(TT_G, TCO_G, [2.0, 4.0])
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)
TDS_G = _oracle_descriptor_matrices(TT_G, TCO_G, TERI_G)
TOPS_G = _oracle_operand_table(TCORE_G, TDS_G)""",
            'call': 'workspace_matrix(THC,[117,102],TOPS)',
            'gold_call': '_oracle_workspace_matrix((TCORE_G[1] + TCORE_G[2]),[117,102],TOPS_G)',
        },
        {
            "setup": base_0 + """TCO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]])
TCO_G = _dc(TCO)
TCORE = core_integrals(TT, TCO, [2.0, 4.0])
TERI = electron_repulsion(TT, TCO)
TDS = descriptor_matrices(TT, TCO, TERI)
TOPS = operand_table(TCORE, TDS)
THC = TCORE[1] + TCORE[2]
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE_G = _oracle_core_integrals(TT_G, TCO_G, [2.0, 4.0])
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)
TDS_G = _oracle_descriptor_matrices(TT_G, TCO_G, TERI_G)
TOPS_G = _oracle_operand_table(TCORE_G, TDS_G)""",
            'call': 'workspace_matrix(THC,[60,8,129,130],TOPS)',
            'gold_call': '_oracle_workspace_matrix((TCORE_G[1] + TCORE_G[2]),[60,8,129,130],TOPS_G)',
        },
        {
            "setup": base_0 + """TCO2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
TCO2_G = _dc(TCO2)
TCORE2 = core_integrals(TT, TCO2, [2.0, 4.0])
TERI2 = electron_repulsion(TT, TCO2)
TDS2 = descriptor_matrices(TT, TCO2, TERI2)
TOPS2 = operand_table(TCORE2, TDS2)
THC2 = TCORE2[1] + TCORE2[2]
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE2_G = _oracle_core_integrals(TT_G, TCO2_G, [2.0, 4.0])
TERI2_G = _oracle_electron_repulsion(TT_G, TCO2_G)
TDS2_G = _oracle_descriptor_matrices(TT_G, TCO2_G, TERI2_G)
TOPS2_G = _oracle_operand_table(TCORE2_G, TDS2_G)""",
            'call': 'workspace_matrix(THC2,[],TOPS2)',
            'gold_call': '_oracle_workspace_matrix((TCORE2_G[1] + TCORE2_G[2]),[],TOPS2_G)',
        },
        {
            "setup": base_0 + """TCO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]])
TCO_G = _dc(TCO)
TCORE = core_integrals(TT, TCO, [2.0, 4.0])
TERI = electron_repulsion(TT, TCO)
TDS = descriptor_matrices(TT, TCO, TERI)
TOPS = operand_table(TCORE, TDS)
THC = TCORE[1] + TCORE[2]
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TCORE_G = _oracle_core_integrals(TT_G, TCO_G, [2.0, 4.0])
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)
TDS_G = _oracle_descriptor_matrices(TT_G, TCO_G, TERI_G)
TOPS_G = _oracle_operand_table(TCORE_G, TDS_G)

def _c():
    try:
        workspace_matrix(THC[:2], [117], TOPS)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _g():
    try:
        _oracle_workspace_matrix((TCORE_G[1] + TCORE_G[2])[:2], [117], TOPS_G)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
    ]
