"""
Through-space Ohno interaction matrix of a planar zigzag polyene chain built from its bond lengths and bond angle.

A pi-electron chain is specified by the positions of its carbon atoms. Atoms 1 to N (array indices 0 to N-1) are joined by bonds 1 to N-1, bond k joining atoms k and k+1; the odd-numbered bonds (array bond index 0, 2, 4, ...) are the double bonds. In the planar zigzag the bonds alternate in direction: the first bond points at +beta from a fixed axis, the second at -beta, and so on, with beta = (180 degrees - bond angle)/2, so that a bond angle of 180 degrees gives a straight chain. Every bond has the length of its type.

The screened Coulomb interaction between the pi electrons on two different atoms is the Ohno form V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2), with U and V in eV, the through-space distance r_ij in angstrom and eps the relative permittivity; it tends to U at short distance and to the screened Coulomb law at long distance. The on-site repulsion is treated separately, so the diagonal of the matrix returned here is zero.

Returns
-------
numpy.ndarray of shape (n_sites, n_sites): the symmetric Ohno interaction matrix V_ij in eV with a zero diagonal
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def ohno_interaction_matrix(n_sites: int, bond_double: float, bond_single: float, bond_angle_deg: float, U: float, eps_r: float) -> np.ndarray:
    '''Ohno interaction matrix of a planar zigzag chain.

    Parameters
    ----------
    n_sites : int
        Number of carbon atoms N, an even integer >= 2.
    bond_double : float
        Length of the double bonds (bonds 1, 3, 5, ...) in angstrom, positive.
    bond_single : float
        Length of the single bonds (bonds 2, 4, 6, ...) in angstrom, positive.
    bond_angle_deg : float
        Bond angle at every atom in degrees, in (0, 180]; 180 gives a straight chain.
    U : float
        Hubbard parameter in eV, positive.
    eps_r : float
        Relative permittivity of the Ohno form, positive.

    Returns
    -------
    V : numpy.ndarray
        Shape (n_sites, n_sites), V[i, j] = U / sqrt(1 + (U eps_r r_ij / 14.397)^2) for
        i != j with r_ij the through-space distance in angstrom, and V[i, i] = 0.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 2, a bond length is not positive,
        bond_angle_deg is outside (0, 180], or U or eps_r is not positive.
    '''
    return V

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _check_even_chain(n_sites):
    if not isinstance(n_sites, (int, np.integer)) or isinstance(n_sites, bool):
        raise ValueError("n_sites must be an integer")
    if n_sites < 2 or n_sites % 2 != 0:
        raise ValueError("n_sites must be an even integer >= 2")


def _oracle_ohno_interaction_matrix(n_sites: int, bond_double: float, bond_single: float, bond_angle_deg: float, U: float, eps_r: float) -> np.ndarray:
    _check_even_chain(n_sites)
    for name, x in (("bond_double", bond_double), ("bond_single", bond_single)):
        if not (np.isfinite(x) and x > 0.0):
            raise ValueError(name + " must be positive")
    if not (np.isfinite(bond_angle_deg) and 0.0 < bond_angle_deg <= 180.0):
        raise ValueError("bond_angle_deg must lie in (0, 180]")
    if not (np.isfinite(U) and U > 0.0):
        raise ValueError("U must be positive")
    if not (np.isfinite(eps_r) and eps_r > 0.0):
        raise ValueError("eps_r must be positive")
    beta = math.radians(180.0 - bond_angle_deg) / 2.0
    pos = np.zeros((n_sites, 2))
    for k in range(n_sites - 1):
        L = bond_double if k % 2 == 0 else bond_single
        ang = beta if k % 2 == 0 else -beta
        pos[k + 1] = pos[k] + L * np.array([math.cos(ang), math.sin(ang)])
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    V = U / np.sqrt(1.0 + (U * eps_r * r / 14.397) ** 2)
    np.fill_diagonal(V, 0.0)
    return V

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the ten-atom benchmark zigzag ---
        {
            "setup": """import numpy as np
""",
            "call": "ohno_interaction_matrix(10, 1.35, 1.46, 120.0, 4.0, 2.3)",
            "gold_call": "_oracle_ohno_interaction_matrix(10, 1.35, 1.46, 120.0, 4.0, 2.3)",
            "tol": 1e-09,
        },
        # --- Normal: a straight six-atom chain with equal bond lengths ---
        {
            "setup": """import numpy as np
""",
            "call": "ohno_interaction_matrix(6, 1.40, 1.40, 180.0, 8.0, 2.0)",
            "gold_call": "_oracle_ohno_interaction_matrix(6, 1.40, 1.40, 180.0, 8.0, 2.0)",
            "tol": 1e-09,
        },
        # --- Normal: a strongly bent eight-atom chain with a weak screening ---
        {
            "setup": """import numpy as np
""",
            "call": "ohno_interaction_matrix(8, 1.30, 1.50, 100.0, 5.0, 1.8)",
            "gold_call": "_oracle_ohno_interaction_matrix(8, 1.30, 1.50, 100.0, 5.0, 1.8)",
            "tol": 1e-09,
        },
        # --- Boundary: the two-atom chain, a single double bond ---
        {
            "setup": """import numpy as np
""",
            "call": "ohno_interaction_matrix(2, 1.35, 1.46, 120.0, 4.0, 2.3)",
            "gold_call": "_oracle_ohno_interaction_matrix(2, 1.35, 1.46, 120.0, 4.0, 2.3)",
            "tol": 1e-09,
        },
        # --- Invalid: an odd number of atoms ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        ohno_interaction_matrix(5, 1.35, 1.46, 120.0, 4.0, 2.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_ohno_interaction_matrix(5, 1.35, 1.46, 120.0, 4.0, 2.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: a zero bond angle ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        ohno_interaction_matrix(6, 1.35, 1.46, 0.0, 4.0, 2.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_ohno_interaction_matrix(6, 1.35, 1.46, 0.0, 4.0, 2.3)
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
