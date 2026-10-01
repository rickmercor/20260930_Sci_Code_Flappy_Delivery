"""
Step 02 - Electron repulsion integrals over contracted s-type Gaussians.

Two-electron repulsion integrals over contracted s-type Gaussian functions.

The electron-electron interaction enters the mean-field reference, the
correlated ground state and every step of the real-time propagation through
the four-index repulsion integrals in chemist's notation,
(ij|kl) = integral phi_i(1) phi_j(1) (1/r12) phi_k(2) phi_l(2) d1 d2, with real
basis functions. The basis is the one built for the one-electron integrals:
identical contracted s shells on every atom, normalised primitives, each
contracted function rescaled to unit self-overlap, and functions ordered atom
by atom and then shell by shell.

For four s primitives with exponents a, b, c, d on centres A, B, C, D, write
p = a + b, q = c + d, P = (a A + b B) / p and Q = (c C + d D) / q. The
primitive integral is
2 pi^(5/2) / (p q sqrt(p + q)) exp(-a b |A - B|^2 / p) exp(-c d |C - D|^2 / q)
F0(p q |P - Q|^2 / (p + q)), with the same zeroth-order Boys function F0 used
for the nuclear attraction. The contracted integral is the coefficient-weighted
sum over all primitive quartets. The result carries the full eight-fold
permutational symmetry of real orbitals: (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij).

Returns
-------
numpy.ndarray of shape (n_bf, n_bf, n_bf, n_bf): electron repulsion integrals (ij|kl) in hartree, chemist notation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def s_type_electron_repulsion(coords: np.ndarray, exponents: list, coefficients: list) -> np.ndarray:
    '''Four-index electron repulsion integrals (ij|kl) over contracted s functions.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3). Every atom carries the
        same list of shells.
    exponents : list
        One 1D array of primitive exponents (bohr^-2) per contracted shell.
    coefficients : list
        One 1D array of contraction coefficients per shell, referring to
        normalised primitives, in the same order as exponents.

    Returns
    -------
    eri : np.ndarray
        Array of shape (n_bf, n_bf, n_bf, n_bf) in hartree, eri[i, j, k, l] =
        (ij|kl) in chemist's notation, n_bf = n_atoms * len(exponents).

    Raises
    ------
    ValueError
        If coords is not a finite (n_atoms, 3) array, if exponents and
        coefficients are empty or differ in length, if any shell has
        mismatched or empty arrays, or if any exponent is not positive.
    '''
    return eri

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_s_type_electron_repulsion(coords: np.ndarray, exponents: list, coefficients: list) -> np.ndarray:
    basis = _contracted_shells(coords, exponents, coefficients)
    n = len(basis)
    pairs = []
    for i in range(n):
        A, a, da = basis[i]
        for j in range(i + 1):
            B, b, db = basis[j]
            p = a[:, None] + b[None, :]
            k_ab = np.exp(-a[:, None] * b[None, :] / p * float(np.sum((A - B) ** 2)))
            P = (a[:, None, None] * A + b[None, :, None] * B) / p[:, :, None]
            pairs.append((i, j, p.ravel(), P.reshape(-1, 3), (da[:, None] * db[None, :] * k_ab).ravel()))
    eri = np.zeros((n, n, n, n))
    for x, (i, j, p, P, w1) in enumerate(pairs):
        for (k, l, q, Q, w2) in pairs[: x + 1]:
            pq = p[:, None] * q[None, :]
            ps = p[:, None] + q[None, :]
            pq2 = np.sum((P[:, None, :] - Q[None, :, :]) ** 2, axis=2)
            val = float(np.sum(w1[:, None] * w2[None, :] * 2.0 * np.pi ** 2.5
                               / (pq * np.sqrt(ps)) * _boys0(pq / ps * pq2)))
            for (r, s, t, u) in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                                 (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)):
                eri[r, s, t, u] = val
    return eri

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: two-atom split-valence basis at a stretched separation ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, -0.9], [0.0, 0.0, 0.9]])
exponents = [np.array([18.7311370, 2.8253944, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "s_type_electron_repulsion(copy.deepcopy(coords), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "gold_call": "_oracle_s_type_electron_repulsion(copy.deepcopy(coords), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "tol": 1e-10,
        },
        # --- Boundary: one primitive on one centre; the coefficient scale must not matter ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[1.0, 2.0, 3.0]])
exponents = [np.array([0.42])]
coefficients = [np.array([7.0])]
""",
            "call": "s_type_electron_repulsion(copy.deepcopy(coords), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "gold_call": "_oracle_s_type_electron_repulsion(copy.deepcopy(coords), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "tol": 1e-10,
        },
        # --- Edge: four centres not on one line, one three-primitive shell per centre ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [1.4000001, 0.9, 0.0], [0.2, -0.6, 1.1]])
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
""",
            "call": "s_type_electron_repulsion(copy.deepcopy(coords), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "gold_call": "_oracle_s_type_electron_repulsion(copy.deepcopy(coords), copy.deepcopy(exponents), copy.deepcopy(coefficients))",
            "tol": 1e-10,
        },
        # --- Invalid: two exponent shells but only one coefficient array ---
        {
            "setup": """import numpy as np
import copy
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
exponents = [np.array([18.7311370, 2.8253944, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733])]
def run_model():
    try:
        s_type_electron_repulsion(copy.deepcopy(coords), copy.deepcopy(exponents), copy.deepcopy(coefficients))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_s_type_electron_repulsion(copy.deepcopy(coords), copy.deepcopy(exponents), copy.deepcopy(coefficients))
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
