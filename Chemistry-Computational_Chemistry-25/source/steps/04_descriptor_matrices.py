"""
Build the four geometry and repulsion descriptors the search draws on. SPECIFICATION: table and coords are as before and eri is the tensor returned by the previous step. The distance matrix holds, at (i, j), the distance in bohr between the nucleus carrying orbital i and the nucleus carrying orbital j, so its entries are zero whenever the two orbitals sit on the same atom. The three repulsion slices are, at (m, n), the tensor elements (m, m, n, n), then (m, n, m, n), then (m, n, n, n). The result is a real array of shape (4, n, n) holding, in this order, the distance matrix and those three slices.

The Coulomb and exchange matrices of a normal calculation cannot appear here, because both are contractions of the repulsion tensor with the density and the density is exactly what is not yet known. Fixed slices of the tensor are the way out: they carry two-electron information, they cost nothing once the tensor exists, and they depend only on the geometry. The distance matrix is the only place where the internuclear separation enters the algebra directly rather than through an integral, which is what lets a single expression follow a bond as it stretches.

Returns
-------
numpy.ndarray of shape (4, n, n) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def descriptor_matrices(table: "np.ndarray", coords: "np.ndarray", eri: "np.ndarray") -> "np.ndarray":
    """Build the four geometry and repulsion descriptors the search draws on. SPECIFICATION: table and coords are as before and eri is the tensor returned by the previous step. The distance matrix holds, at (i, j), the distance in bohr between the nucleus carrying orbital i and the nucleus carrying orbital j, so its entries are zero whenever the two orbitals sit on the same atom. The three repulsion slices are, at (m, n), the tensor elements (m, m, n, n), then (m, n, m, n), then (m, n, n, n). The result is a real array of shape (4, n, n) holding, in this order, the distance matrix and those three slices.

    Parameters
    ----------
    table : numpy.ndarray of shape (n, 10) and real dtype
        The contracted-orbital table, one row per orbital.
    coords : numpy.ndarray of shape (2, 3) and real dtype
        Cartesian nuclear positions in bohr, one row per atom.
    eri : numpy.ndarray of shape (n, n, n, n) and real dtype
        The two-electron repulsion tensor in chemists' notation.

    Returns
    -------
    numpy.ndarray of shape (4, n, n) and real dtype.

    Raises
    ------
    ValueError: if the basis table does not have ten columns, or if eri does not have shape (n, n, n, n) for the n rows of the table.
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


def _unpack(table):
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 10:
        raise ValueError("basis table must have ten columns")
    return (table[:, 0].astype(int), table[:, 1:4].astype(int),
            table[:, 4:7], table[:, 7:10])


def _oracle_descriptor_matrices(table: "np.ndarray", coords: "np.ndarray", eri: "np.ndarray") -> "np.ndarray":
    centers, lxyz, exps, coefs = _unpack(table)
    coords = np.asarray(coords, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n = len(centers)
    if eri.shape != (n, n, n, n):
        raise ValueError("eri does not match the basis size")
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(coords[centers[i]] - coords[centers[j]])
    return np.stack([D, np.einsum("mmnn->mn", eri), np.einsum("mnmn->mn", eri),
                     np.einsum("mnnn->mn", eri)])

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
ERI = electron_repulsion(TB, CO)
TB_G = _oracle_basis_table(['Li', 'F'], BD_G)
ERI_G = _oracle_electron_repulsion(TB_G, CO_G)"""
    base_1 = """from copy import deepcopy as _dc
import numpy as np
TOY = {'X': [(0, [1.9, 0.44, 0.14], [0.16, 0.53, 0.45])], 'Y': [(0, [4.1, 0.95, 0.31], [0.15, 0.54, 0.44]), (1, [0.83, 0.29, 0.11], [0.16, 0.61, 0.39])]}
TOY_G = _dc(TOY)
TT = basis_table(['X', 'Y'], TOY)
"""
    return [
        {
            "setup": base_0,
            'call': 'descriptor_matrices(TB,CO,ERI)',
            'gold_call': '_oracle_descriptor_matrices(TB_G,CO_G,ERI_G)',
        },
        {
            "setup": base_1 + """TCO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]])
TCO_G = _dc(TCO)
TERI = electron_repulsion(TT, TCO)
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)""",
            'call': 'descriptor_matrices(TT,TCO,TERI)',
            'gold_call': '_oracle_descriptor_matrices(TT_G,TCO_G,TERI_G)',
        },
        {
            "setup": base_1 + """TCO2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
TCO2_G = _dc(TCO2)
TERI2 = electron_repulsion(TT, TCO2)
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TERI2_G = _oracle_electron_repulsion(TT_G, TCO2_G)""",
            'call': 'descriptor_matrices(TT,TCO2,TERI2)',
            'gold_call': '_oracle_descriptor_matrices(TT_G,TCO2_G,TERI2_G)',
        },
        {
            "setup": base_1 + """TCO = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.6]])
TCO_G = _dc(TCO)
TERI = electron_repulsion(TT, TCO)
TT_G = _oracle_basis_table(['X', 'Y'], TOY_G)
TERI_G = _oracle_electron_repulsion(TT_G, TCO_G)

def _c():
    try:
        descriptor_matrices(TT, TCO, TERI[:, :, :, :1])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _g():
    try:
        _oracle_descriptor_matrices(TT_G, TCO_G, TERI_G[:, :, :, :1])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
    ]
