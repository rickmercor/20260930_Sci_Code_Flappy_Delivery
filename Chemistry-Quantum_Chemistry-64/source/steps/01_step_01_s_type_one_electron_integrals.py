"""
Step 01 - One-electron and z-coordinate integrals over contracted s-type Gaussians.

One-electron integrals over contracted s-type Gaussian functions, including the
matrix of the electronic z coordinate.

Every calculation in this task starts from the same small set of atom-centred
basis functions. Each atom carries an identical list of contracted s shells. A
shell is a fixed linear combination of primitive Gaussians exp(-alpha r^2); each
primitive is first normalised to unit self-overlap, (2 alpha / pi)^(3/4), the
tabulated contraction coefficients are applied to these normalised primitives,
and the contracted function is then rescaled so that its own overlap is exactly
one. Basis functions are ordered atom by atom, and within an atom in the order
the shells are listed.

For two s primitives with exponents a and b on centres A and B, write p = a + b,
mu = a b / p and P = (a A + b B) / p. The overlap is
(pi / p)^(3/2) exp(-mu |A - B|^2), the kinetic energy integral is
mu (3 - 2 mu |A - B|^2) times the overlap, and the attraction to a nucleus of
charge Z at C is -Z (2 pi / p) exp(-mu |A - B|^2) F0(p |P - C|^2), where
F0(t) = (1/2) sqrt(pi / t) erf(sqrt(t)) and F0(0) = 1. The product of two s
Gaussians is an s Gaussian centred at P, so the integral of the electron's z
coordinate between them is the overlap times the z component of P. Contracted
integrals are the coefficient-weighted sums of the primitive ones, and the
nuclear attraction matrix V sums the attraction to every nucleus. The z matrix
refers to the coordinate origin of the input positions. Everything is in atomic
units (bohr and hartree).

The z matrix is what couples the electrons to a z-polarised electric field in
the length gauge later on, so it is returned together with the three matrices
of the field-free Hamiltonian.

Returns
-------
numpy.ndarray of shape (4, n_bf, n_bf): the overlap, kinetic, nuclear attraction and electronic z-coordinate matrices in atomic units
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.special import erf


def s_type_one_electron_integrals(coords: np.ndarray, charges: np.ndarray, exponents: list, coefficients: list) -> np.ndarray:
    '''Overlap, kinetic, nuclear attraction and z-coordinate matrices over contracted s functions.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3). Every atom carries the
        same list of shells.
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,), each non-negative.
    exponents : list
        One 1D array of primitive exponents (bohr^-2) per contracted shell.
    coefficients : list
        One 1D array of contraction coefficients per shell, referring to
        normalised primitives, in the same order as exponents.

    Returns
    -------
    integrals : np.ndarray
        Array of shape (4, n_bf, n_bf) holding the overlap matrix S, the
        kinetic energy matrix T, the nuclear attraction matrix V and the matrix
        Z of the electron's z coordinate (bohr), in that order, with
        n_bf = n_atoms * len(exponents).

    Raises
    ------
    ValueError
        If coords is not a finite (n_atoms, 3) array, if charges does not have
        shape (n_atoms,) or holds a negative or non-finite value, if exponents
        and coefficients are empty or differ in length, if any shell has
        mismatched or empty arrays, or if any exponent is not positive.
    '''
    return integrals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf


def _boys0(t):
    """Zeroth-order Boys function, elementwise, with the t -> 0 series."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    small = t < 1e-10
    out[small] = 1.0 - t[small] / 3.0
    big = ~small
    out[big] = 0.5 * np.sqrt(np.pi / t[big]) * erf(np.sqrt(t[big]))
    return out


def _contracted_shells(coords, exponents, coefficients):
    """List of (centre, exponents, normalised coefficients) per basis function."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 1:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must be finite")
    if len(exponents) == 0 or len(exponents) != len(coefficients):
        raise ValueError("exponents and coefficients must be non-empty lists of equal length")
    shells = []
    for exps, coefs in zip(exponents, coefficients):
        a = np.asarray(exps, dtype=float).ravel()
        c = np.asarray(coefs, dtype=float).ravel()
        if a.size == 0 or a.size != c.size:
            raise ValueError("each shell needs matching, non-empty exponent and coefficient arrays")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(c))) or np.any(a <= 0.0):
            raise ValueError("exponents must be positive and all values finite")
        d = c * (2.0 * a / np.pi) ** 0.75
        pab = a[:, None] + a[None, :]
        self_overlap = float(np.sum(d[:, None] * d[None, :] * (np.pi / pab) ** 1.5))
        if self_overlap <= 0.0:
            raise ValueError("a contracted shell has non-positive self-overlap")
        shells.append((a, d / np.sqrt(self_overlap)))
    basis = []
    for centre in coords:
        for a, d in shells:
            basis.append((centre, a, d))
    return basis


def _oracle_s_type_one_electron_integrals(coords: np.ndarray, charges: np.ndarray, exponents: list, coefficients: list) -> np.ndarray:
    basis = _contracted_shells(coords, exponents, coefficients)
    coords = np.asarray(coords, dtype=float)
    charges = np.asarray(charges, dtype=float)
    if charges.shape != (coords.shape[0],) or not np.all(np.isfinite(charges)) or np.any(charges < 0.0):
        raise ValueError("charges must be a finite, non-negative array of shape (n_atoms,)")
    n = len(basis)
    out = np.zeros((4, n, n))
    for i, (A, a, da) in enumerate(basis):
        for j, (B, b, db) in enumerate(basis):
            p = a[:, None] + b[None, :]
            mu = a[:, None] * b[None, :] / p
            ab2 = float(np.sum((A - B) ** 2))
            prim_s = (np.pi / p) ** 1.5 * np.exp(-mu * ab2)
            dd = da[:, None] * db[None, :]
            P = (a[:, None, None] * A + b[None, :, None] * B) / p[:, :, None]
            out[0, i, j] = np.sum(dd * prim_s)
            out[1, i, j] = np.sum(dd * mu * (3.0 - 2.0 * mu * ab2) * prim_s)
            v = 0.0
            for C, Z in zip(coords, charges):
                pc2 = np.sum((P - C) ** 2, axis=2)
                v -= Z * np.sum(dd * (2.0 * np.pi / p) * np.exp(-mu * ab2) * _boys0(p * pc2))
            out[2, i, j] = v
            out[3, i, j] = np.sum(dd * prim_s * P[:, :, 2])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: four-atom chain centred on the origin, split-valence basis ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, -2.7], [0.0, 0.0, -0.9], [0.0, 0.0, 0.9], [0.0, 0.0, 2.7]])
charges = np.ones(4)
exponents = [np.array([18.7311370, 2.8253944, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "s_type_one_electron_integrals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "gold_call": "_oracle_s_type_one_electron_integrals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "tol": 1e-10,
        },
        # --- Boundary: one atom, one primitive, off-origin; coefficient scale must not matter ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.3, -0.2, 0.7]])
charges = np.array([2.0])
exponents = [np.array([0.75])]
coefficients = [np.array([3.5])]
""",
            "call": "s_type_one_electron_integrals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "gold_call": "_oracle_s_type_one_electron_integrals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "tol": 1e-10,
        },
        # --- Edge: off-axis atoms, unequal charges, a ghost centre (charge 0) and a coincident pair ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, 0.0], [1.1, 0.4, -0.3], [1.1, 0.4, -0.3]])
charges = np.array([1.0, 3.0, 0.0])
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
""",
            "call": "s_type_one_electron_integrals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "gold_call": "_oracle_s_type_one_electron_integrals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "tol": 1e-10,
        },
        # --- Invalid: a non-positive exponent ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, 0.0]])
charges = np.array([1.0])
exponents = [np.array([1.0, -0.5])]
coefficients = [np.array([0.5, 0.5])]
def run_model():
    try:
        s_type_one_electron_integrals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_s_type_one_electron_integrals(copy.deepcopy(coords), copy.deepcopy(charges), copy.deepcopy(exponents), copy.deepcopy(coefficients))
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
