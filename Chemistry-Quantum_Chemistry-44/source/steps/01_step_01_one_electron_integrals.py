"""
Step 01 - Overlap and core-Hamiltonian matrices of a hydrogen cluster in a contracted s-type Gaussian basis.

Every calculation in this problem uses basis sets made only of s-type Gaussian functions, which is all a hydrogen cluster needs at the 6-31G or STO-3G level. Each atom carries the same list of contracted shells. A shell is a fixed linear combination of normalised primitive Gaussians exp(-alpha r^2) centred on the atom, with the listed exponents and contraction coefficients, and the contracted function is then rescaled to unit self-overlap. Basis functions are ordered atom by atom, following the rows of the coordinate array, and within an atom in the order the shells are listed.

For s functions the overlap, kinetic-energy and nuclear-attraction integrals all follow from the Gaussian product theorem. The nuclear-attraction integral needs the zeroth-order Boys function, F0(t) = (1/2) sqrt(pi/t) erf(sqrt(t)), which tends to 1 as t goes to zero. The core Hamiltonian is the kinetic-energy matrix plus the attraction of an electron to every nucleus, each nucleus weighted by its charge. Atomic units are used throughout: coordinates in bohr, energies in hartree.

Returns
-------
numpy.ndarray of shape (2, n, n): the overlap matrix and the core-Hamiltonian matrix in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def one_electron_integrals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list) -> np.ndarray:
    '''Overlap and core-Hamiltonian matrices over contracted s-type Gaussians.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : array_like
        Nuclear charges, one positive value per atom.
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.

    Returns
    -------
    one_electron : numpy.ndarray
        Array of shape (2, n, n) with n = n_atoms * len(exponents). Element
        [0] is the overlap matrix and element [1] the core Hamiltonian, kinetic
        energy plus nuclear attraction, in hartree. Basis functions are ordered
        atom by atom and, within an atom, shell by shell.

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, charges does not hold one positive finite value per atom,
        exponents and coefficients are empty or differ in length, the
        exponent and coefficient arrays of a shell differ in length, an
        exponent is not positive and finite, or the coefficients of a shell
        are not finite or are all zero.
    '''
    return one_electron

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _boys_zero(t):
    """Boys function of order zero, F0(t) = (1/2) sqrt(pi / t) erf(sqrt(t))."""
    import numpy as np
    from scipy.special import erf
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    small = t < 1.0e-12
    out[small] = 1.0 - t[small] / 3.0
    big = ~small
    out[big] = 0.5 * np.sqrt(np.pi / t[big]) * erf(np.sqrt(t[big]))
    return out


def _s_basis_functions(coords, exponents, coefficients):
    """Validate the geometry and basis; return the normalised contracted s functions, atom-major."""
    import numpy as np
    xyz = np.asarray(coords, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 1 or not np.all(np.isfinite(xyz)):
        raise ValueError("coords must be a finite array of shape (n_atoms, 3)")
    if xyz.shape[0] > 1:
        dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        if np.min(dist[np.triu_indices(xyz.shape[0], 1)]) < 1.0e-6:
            raise ValueError("two nuclei coincide")
    if len(exponents) < 1 or len(exponents) != len(coefficients):
        raise ValueError("exponents and coefficients must be non-empty sequences of equal length")
    shells = []
    for e, c in zip(exponents, coefficients):
        e = np.atleast_1d(np.asarray(e, dtype=float))
        c = np.atleast_1d(np.asarray(c, dtype=float))
        if e.ndim != 1 or e.shape != c.shape:
            raise ValueError("each shell needs one-dimensional exponents and coefficients of equal length")
        if not (np.all(np.isfinite(e)) and np.all(e > 0.0) and np.all(np.isfinite(c)) and np.any(c != 0.0)):
            raise ValueError("exponents must be positive and finite, coefficients finite and not all zero")
        shells.append((e, c))
    functions = []
    for centre in xyz:
        for e, c in shells:
            d = c * (2.0 * e / np.pi) ** 0.75
            norm2 = np.sum(d[:, None] * d[None, :] * (np.pi / (e[:, None] + e[None, :])) ** 1.5)
            functions.append((centre.copy(), e, d / np.sqrt(norm2)))
    return xyz, functions


def _oracle_one_electron_integrals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                                   coefficients: list) -> np.ndarray:
    import numpy as np
    xyz, functions = _s_basis_functions(coords, exponents, coefficients)
    Z = np.atleast_1d(np.asarray(charges, dtype=float))
    if Z.shape != (xyz.shape[0],) or not np.all(np.isfinite(Z)) or np.any(Z <= 0.0):
        raise ValueError("charges must hold one positive finite nuclear charge per atom")
    n = len(functions)
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n))
    for p in range(n):
        A, ea, da = functions[p]
        for q in range(n):
            B, eb, db = functions[q]
            gam = ea[:, None] + eb[None, :]
            mu = ea[:, None] * eb[None, :] / gam
            AB2 = float(np.dot(A - B, A - B))
            K = np.exp(-mu * AB2)
            P = (ea[:, None, None] * A + eb[None, :, None] * B) / gam[..., None]
            dd = da[:, None] * db[None, :]
            s_prim = (np.pi / gam) ** 1.5 * K
            S[p, q] = np.sum(dd * s_prim)
            T[p, q] = np.sum(dd * mu * (3.0 - 2.0 * mu * AB2) * s_prim)
            for C, zc in zip(xyz, Z):
                PC2 = np.sum((P - C) ** 2, axis=-1)
                V[p, q] -= zc * np.sum(dd * 2.0 * np.pi / gam * K * _boys_zero(gam * PC2))
    return np.stack([S, T + V])

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
            "call": "one_electron_integrals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "gold_call": "_oracle_one_electron_integrals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "tol": 1e-09,
        },
        # --- Boundary: unequal nuclear charges, a helium-hydride-like cation in a minimal basis ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [0.00, 0.00, 1.46]])
charges = np.array([2.0, 1.0])
n_electrons = 2
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
""",
            "call": "one_electron_integrals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "gold_call": "_oracle_one_electron_integrals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "tol": 1e-09,
        },
        # --- Edge: a single atom, no two-centre terms at all ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00]])
charges = np.array([1.0])
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "one_electron_integrals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "gold_call": "_oracle_one_electron_integrals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "tol": 1e-09,
        },
        # --- Normal: a puckered six-atom ring in a minimal basis ---
        {
            "setup": """import numpy as np
coords = np.array([[1.90, 0.00, 0.00], [1.10, 1.65, 0.10], [-0.95, 1.65, 0.20],
                   [-1.75, 0.00, 0.30], [-0.95, -1.65, 0.40], [1.10, -1.65, 0.50]])
charges = np.ones(6)
n_electrons = 6
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
""",
            "call": "one_electron_integrals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "gold_call": "_oracle_one_electron_integrals(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients])",
            "tol": 1e-09,
        },
    ]
