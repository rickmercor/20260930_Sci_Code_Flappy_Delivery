"""
Step 02 - Two-electron repulsion integrals over the same contracted s-type Gaussians.

Electron-electron repulsion enters through the integrals (pq|rs), the Coulomb interaction between the charge distribution phi_p phi_q of electron 1 and phi_r phi_s of electron 2, written in chemists' notation. Over s-type Gaussians every primitive integral has a closed form: each Gaussian product collapses onto a single centre by the product theorem, and the Coulomb interaction between the two resulting distributions is again a zeroth-order Boys function of their separation.

With real functions the array has eight-fold permutational symmetry, (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq) and so on, which can be used to avoid repeated work. The basis, its normalisation and the ordering of the functions are exactly those of the one-electron integrals.

Returns
-------
numpy.ndarray of shape (n, n, n, n): electron repulsion integrals (pq|rs) in hartree, chemists' notation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def electron_repulsion_integrals(coords: npt.ArrayLike, exponents: list, coefficients: list) -> np.ndarray:
    '''Two-electron repulsion integrals over contracted s-type Gaussians.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.

    Returns
    -------
    eri : numpy.ndarray
        Array of shape (n, n, n, n) holding (pq|rs) in hartree, chemists'
        notation, with the basis ordered atom by atom and shell by shell.

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, exponents and coefficients are empty or differ in length,
        the exponent and coefficient arrays of a shell differ in length, an
        exponent is not positive and finite, or the coefficients of a shell
        are not finite or are all zero.
    '''
    return eri

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_electron_repulsion_integrals(coords: npt.ArrayLike, exponents: list,
                                         coefficients: list) -> np.ndarray:
    import numpy as np
    xyz, functions = _s_basis_functions(coords, exponents, coefficients)
    n = len(functions)
    pair = {}
    for p in range(n):
        A, ea, da = functions[p]
        for q in range(n):
            B, eb, db = functions[q]
            gam = ea[:, None] + eb[None, :]
            K = np.exp(-ea[:, None] * eb[None, :] / gam * float(np.dot(A - B, A - B)))
            P = (ea[:, None, None] * A + eb[None, :, None] * B) / gam[..., None]
            pair[(p, q)] = (gam, K * da[:, None] * db[None, :], P)
    eri = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(p + 1):
            g1, w1, P1 = pair[(p, q)]
            for r in range(n):
                for s in range(r + 1):
                    if r * (r + 1) // 2 + s > p * (p + 1) // 2 + q:
                        continue
                    g2, w2, P2 = pair[(r, s)]
                    G1 = g1[:, :, None, None]
                    G2 = g2[None, None, :, :]
                    PQ2 = np.sum((P1[:, :, None, None, :] - P2[None, None, :, :, :]) ** 2, axis=-1)
                    val = np.sum(w1[:, :, None, None] * w2[None, None, :, :]
                                 * 2.0 * np.pi ** 2.5 / (G1 * G2 * np.sqrt(G1 + G2))
                                 * _boys_zero(G1 * G2 / (G1 + G2) * PQ2))
                    for idx in ((p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                                (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)):
                        eri[idx] = val
    return eri

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the four-atom target cluster in the split-valence basis ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.60, 0.00, 0.00], [1.90, 2.40, 0.20], [0.20, 2.50, 0.00]])
charges = np.array([1.0, 1.0, 1.0, 1.0])
n_electrons = 4
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "electron_repulsion_integrals(np.array(coords), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "gold_call": "_oracle_electron_repulsion_integrals(np.array(coords), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "tol": 1e-09,
        },
        # --- Boundary: a two-centre minimal-basis case ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.40, 0.00, 0.00]])
charges = np.array([1.0, 1.0])
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
""",
            "call": "electron_repulsion_integrals(np.array(coords), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "gold_call": "_oracle_electron_repulsion_integrals(np.array(coords), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "tol": 1e-09,
        },
        # --- Edge: a single atom, only one-centre integrals ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00]])
charges = np.array([1.0])
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "electron_repulsion_integrals(np.array(coords), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "gold_call": "_oracle_electron_repulsion_integrals(np.array(coords), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "tol": 1e-09,
        },
    ]
