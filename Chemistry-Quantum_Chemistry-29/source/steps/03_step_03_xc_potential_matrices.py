"""
Step 03: Basis-set matrices of the basis-free spin-resolved exchange-correlation potentials. Matrix representations of the basis-free spin-resolved exchange-correlation potentials in a Gaussian basis.

A local, multiplicative one-electron potential v(r) enters a basis-set calculation only through its matrix elements V_kl = <f_k | v | f_l>, the integrals of v times the product of two basis functions. When the potential is known exactly in real space, for example the exchange-correlation potential of a numerically converged atom, its matrix in a finite basis is obtained by quadrature, and in a spin-polarized atom each spin has its own potential and its own matrix. Such matrices are the typical input of inverse problems in density functional theory: Kohn-Sham inversion, density embedding and optimized effective potential methods all produce or consume potentials that are known only through their matrices in a finite basis. A K by K symmetric matrix holds only K(K + 1)/2 numbers, so it does not fix the potential pointwise, and in atom-centred Gaussian basis sets steep functions near the nucleus give the largest matrix elements, while diffuse functions carry the information about the potential far from the nucleus, where the spin polarization of an open-shell atom is largest.

Returns
-------
numpy.ndarray of shape (2, K, K): matrices of the basis-free up-spin and down-spin LSD exchange-correlation potentials in a normalized s-Gaussian basis, in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def xc_potential_matrices(Z: float, n_up: int, n_down: int, exponents: "np.ndarray") -> "np.ndarray":
    '''Matrices of the basis-free LSD exchange-correlation potentials of both spins in a normalized s-Gaussian basis.

    Parameters
    ----------
    Z : float
        Nuclear charge in units of e, as in radial_lsd_spin_densities.
    n_up : int
        Number of occupied up-spin s orbitals, as in radial_lsd_spin_densities.
    n_down : int
        Number of occupied down-spin s orbitals, as in radial_lsd_spin_densities.
    exponents : np.ndarray
        1-D array of K distinct, finite, positive exponents alpha_k in bohr^-2 defining the normalized basis functions
        f_k(r) = (2 alpha_k / pi)^(3/4) exp(-alpha_k r^2), in the order given.

    Returns
    -------
    result : np.ndarray
        Array of shape (2, K, K) in hartree, each block symmetric: result[s][k, l] is the integral over all space of
        f_k(r) v_xc,s(r) f_l(r), with s = 0 for the up-spin and s = 1 for the down-spin potential of
        lsd_exchange_correlation evaluated at the basis-free self-consistent spin densities of
        radial_lsd_spin_densities, with no density cutoff. Each element is converged to better than 1e-9 hartree.

    Raises
    ------
    ValueError
        Under the conditions of radial_lsd_spin_densities, or if the exponents are not a 1-D array of distinct finite
        positive numbers.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _sgauss_check(exponents):
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("exponents must be a 1-D array of finite positive numbers")
    if np.unique(a).size != a.size:
        raise ValueError("exponents must be distinct")
    return a

def _radial_grid(exponents):
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    r_min = 1e-5 / np.sqrt(a.max())
    r_max = max(np.sqrt(200.0 / a.min()), 80.0)
    x = np.linspace(np.log(r_min), np.log(r_max), 6001)
    r = np.exp(x)
    w = 4.0 * np.pi * r ** 3 * (x[1] - x[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return r, w

def _basis_on_grid(exponents, r):
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    return (2.0 * a[:, None] / np.pi) ** 0.75 * np.exp(-np.minimum(a[:, None] * r[None, :] ** 2, 745.0))

def _xc_on_grid(Z, n_up, n_down, r):
    import numpy as np
    rho = _oracle_radial_lsd_spin_densities(Z, n_up, n_down, r)
    return _oracle_lsd_exchange_correlation(np.maximum(rho[0], 0.0), np.maximum(rho[1], 0.0))

def _oracle_xc_potential_matrices(Z: float, n_up: int, n_down: int, exponents: "np.ndarray") -> "np.ndarray":
    import numpy as np
    a = _sgauss_check(exponents)
    r, w = _radial_grid(a)
    G = _basis_on_grid(a, r)
    xc = _xc_on_grid(Z, n_up, n_down, r)
    V_up = (G * (w * xc[1])) @ G.T
    V_down = (G * (w * xc[2])) @ G.T
    return np.stack([0.5 * (V_up + V_up.T), 0.5 * (V_down + V_down.T)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: lithium in a 14-function even-tempered basis ---
        {"setup": "import numpy as np\nex = 0.03 * 2.35 ** np.arange(14)\n",
         "call": "xc_potential_matrices(3.0, 2, 1, ex.copy())",
         "gold_call": "_oracle_xc_potential_matrices(3.0, 2, 1, ex.copy())", "tol": 1e-8},
        # --- Normal: the Be+ cation with a steep core and a compact valence shell ---
        {"setup": "import numpy as np\nex = 0.05 * 2.4 ** np.arange(13)\n",
         "call": "xc_potential_matrices(4.0, 2, 1, ex.copy())",
         "gold_call": "_oracle_xc_potential_matrices(4.0, 2, 1, ex.copy())", "tol": 1e-8},
        # --- Boundary: closed-shell helium, where the two potentials coincide ---
        {"setup": "import numpy as np\nex = 0.02 * 3.0 ** np.arange(10)\n",
         "call": "xc_potential_matrices(2.0, 1, 1, ex.copy())",
         "gold_call": "_oracle_xc_potential_matrices(2.0, 1, 1, ex.copy())", "tol": 1e-8},
        # --- Edge: hydrogen in a single function, with an empty down-spin channel ---
        {"setup": "import numpy as np\nex = np.array([0.3])\n",
         "call": "xc_potential_matrices(1.0, 1, 0, ex.copy())",
         "gold_call": "_oracle_xc_potential_matrices(1.0, 1, 0, ex.copy())", "tol": 1e-8},
        # --- Error: a non-positive exponent ---
        {"setup": "import numpy as np\ndef _probe(fn):\n    try:\n        fn(3.0, 2, 1, np.array([0.5, 0.0, 4.0]))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(xc_potential_matrices)", "gold_call": "_probe(_oracle_xc_potential_matrices)"},
    ]
