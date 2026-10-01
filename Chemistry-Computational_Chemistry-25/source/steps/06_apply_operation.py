"""
Apply one catalogue operation to the workspace matrix. SPECIFICATION: operands is the array of shape (11, n, n) from the previous step and A(k) denotes its k-th entry, counting from zero. Operations are numbered as follows. Ids 0 to 15 combine M with a constant c, adding for ids 0 to 3, subtracting for 4 to 7, multiplying for 8 to 11 and dividing for 12 to 15, with c running over 0.5, 1, 3 and 4 in that order inside each block of four. Ids 16 to 59 combine M element-wise with A(k), adding for 16 to 26, subtracting for 27 to 37, multiplying for 38 to 48 and dividing for 49 to 59, with k running from 0 to 10 in that order inside each block of eleven. Ids 60 to 70 replace M by the matrix product of M with A(k), with k running from 0 to 10. Ids 71 to 114 repeat the four element-wise combinations but against the matrix inverse of A(k), in the same arrangement, so 71 to 81 add, 82 to 92 subtract, 93 to 103 multiply and 104 to 114 divide. Ids 115 to 125 replace M by the matrix product of M with the matrix inverse of A(k). Id 126 applies the exponential to every element, 127 the natural logarithm, 128 raises every element to the power one half, 129 squares every element, and 130 applies the exponential of the negative of every element. Division and the element-wise functions act entry by entry; only the products named as matrix products, and the inverses, are matrix operations. The result is the transformed matrix, of the same shape as M.

The catalogue is deliberately blind to physics. It mixes matrices that have different units and different symmetry, it offers both an element-wise product and a matrix product with the same operand, and it offers constants with no dimensions at all. That is the point: the search is meant to find combinations no one would write down, and the only filter is whether the resulting matrix has useful eigenvectors. An operation that asks for the inverse of a matrix that has none is not an error to be patched around but a verdict on the candidate, so it has to be signalled rather than smoothed over.

Returns
-------
numpy.ndarray of shape (n, n) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_operation(M: "np.ndarray", op_id: "int", operands: "np.ndarray") -> "np.ndarray":
    """Apply one catalogue operation to the workspace matrix. SPECIFICATION: operands is the array of shape (11, n, n) from the previous step and A(k) denotes its k-th entry, counting from zero. Operations are numbered as follows. Ids 0 to 15 combine M with a constant c, adding for ids 0 to 3, subtracting for 4 to 7, multiplying for 8 to 11 and dividing for 12 to 15, with c running over 0.5, 1, 3 and 4 in that order inside each block of four. Ids 16 to 59 combine M element-wise with A(k), adding for 16 to 26, subtracting for 27 to 37, multiplying for 38 to 48 and dividing for 49 to 59, with k running from 0 to 10 in that order inside each block of eleven. Ids 60 to 70 replace M by the matrix product of M with A(k), with k running from 0 to 10. Ids 71 to 114 repeat the four element-wise combinations but against the matrix inverse of A(k), in the same arrangement, so 71 to 81 add, 82 to 92 subtract, 93 to 103 multiply and 104 to 114 divide. Ids 115 to 125 replace M by the matrix product of M with the matrix inverse of A(k). Id 126 applies the exponential to every element, 127 the natural logarithm, 128 raises every element to the power one half, 129 squares every element, and 130 applies the exponential of the negative of every element. Division and the element-wise functions act entry by entry; only the products named as matrix products, and the inverses, are matrix operations. The result is the transformed matrix, of the same shape as M.

    Parameters
    ----------
    M : numpy.ndarray of shape (n, n) and real dtype
        The workspace matrix the operation acts on.
    op_id : int
        The operation id, an integer from 0 to 130 inclusive.
    operands : numpy.ndarray of shape (11, n, n) and real dtype
        The eleven operand matrices, in the order fixed by the operand table.

    Returns
    -------
    numpy.ndarray of shape (n, n) and real dtype.

    Raises
    ------
    ValueError: if op_id is outside 0 to 130, if M is not square, if operands does not have shape (11, n, n) matching M, if an operation needs the inverse of a matrix that has none, or if the result holds a non-finite entry.
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


def _library():
    """(kind, payload) for operation ids 0..130, in the fixed order of the contract."""
    lib = []
    for op in "+-*/":
        for c in (0.5, 1.0, 3.0, 4.0):
            lib.append(("const", (op, c)))
    for op in "+-*/":
        for k in range(11):
            lib.append(("amat", (op, k)))
    for k in range(11):
        lib.append(("amul", k))
    for op in "+-*/":
        for k in range(11):
            lib.append(("ainv", (op, k)))
    for k in range(11):
        lib.append(("ainvmul", k))
    lib.append(("unary", "exp"))
    lib.append(("unary", "ln"))
    lib.append(("power", 0.5))
    lib.append(("power", 2.0))
    lib.append(("unary", "expneg"))
    return lib


def _oracle_apply_operation(M: "np.ndarray", op_id: "int", operands: "np.ndarray") -> "np.ndarray":
    lib = _library()
    op_id = int(op_id)
    if op_id < 0 or op_id >= len(lib):
        raise ValueError("op_id outside the library")
    M = np.asarray(M, dtype=float)
    operands = np.asarray(operands, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    if operands.shape != (11,) + M.shape:
        raise ValueError("operands must have shape (11, n, n) matching M")
    kind, pay = lib[op_id]
    with np.errstate(all="ignore"):
        if kind == "const":
            o, c = pay
            out = M + c if o == "+" else M - c if o == "-" else M*c if o == "*" else M/c
        elif kind == "amat":
            o, k = pay
            X = operands[k]
            out = M + X if o == "+" else M - X if o == "-" else M*X if o == "*" else M/X
        elif kind == "amul":
            out = M @ operands[pay]
        elif kind == "ainvmul":
            out = M @ _safe_inverse(operands[pay])
        elif kind == "ainv":
            o, k = pay
            X = _safe_inverse(operands[k])
            out = M + X if o == "+" else M - X if o == "-" else M*X if o == "*" else M/X
        elif kind == "unary":
            out = np.exp(M) if pay == "exp" else np.log(M) if pay == "ln" else np.exp(-M)
        else:
            out = np.power(M, pay)
    out = np.asarray(out, dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("operation produced a non-finite workspace matrix")
    return out

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
            'call': 'apply_operation(THC,117,TOPS)',
            'gold_call': '_oracle_apply_operation((TCORE_G[1] + TCORE_G[2]),117,TOPS_G)',
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
            'call': 'apply_operation(THC,102,TOPS)',
            'gold_call': '_oracle_apply_operation((TCORE_G[1] + TCORE_G[2]),102,TOPS_G)',
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
            'call': 'apply_operation(THC2,38,TOPS2)',
            'gold_call': '_oracle_apply_operation((TCORE2_G[1] + TCORE2_G[2]),38,TOPS2_G)',
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
            'call': 'apply_operation(THC,126,TOPS)',
            'gold_call': '_oracle_apply_operation((TCORE_G[1] + TCORE_G[2]),126,TOPS_G)',
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
        apply_operation(THC, 131, TOPS)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _g():
    try:
        _oracle_apply_operation((TCORE_G[1] + TCORE_G[2]), 131, TOPS_G)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
        # --- Invalid: op 119, missing inverse of the singular distance matrix; the discard contract raises ValueError ---
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
        apply_operation(THC, 119, TOPS)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _g():
    try:
        _oracle_apply_operation((TCORE_G[1] + TCORE_G[2]), 119, TOPS_G)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
        # --- Invalid: op 127, non-finite logarithm of entries <= 0; the discard contract raises ValueError ---
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
        apply_operation(THC, 127, TOPS)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _g():
    try:
        _oracle_apply_operation((TCORE_G[1] + TCORE_G[2]), 127, TOPS_G)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
    ]
