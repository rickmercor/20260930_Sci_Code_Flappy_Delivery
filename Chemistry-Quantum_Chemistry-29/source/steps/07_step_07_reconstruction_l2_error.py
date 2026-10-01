"""
Step 07: L2 error of the reconstructed spin-resolved exchange-correlation potential (orchestrator). Real-space L2 error of a spin-resolved exchange-correlation potential reconstructed from its basis-set matrix (orchestrator).

The quality of a potential recovered from its matrix is judged in real space by the L2 distance between the reconstruction and the true potential, the square root of the integral of their squared difference over all space. For a numerically converged LSD atom the true exchange-correlation potential of each spin is known pointwise from the basis-free spin densities, so the error of an importance-selected reconstruction can be measured exactly. That error falls as the selection threshold is tightened and more products are admitted, but it stays finite in a finite basis. For an atom it is dominated by the region far from the nucleus, where the densities, and with them the LSD potentials, decay more slowly than the most diffuse basis-function products can describe, while the matrix elements that drive the selection are dominated by the steep functions near the nucleus. In an open-shell atom the minority-spin potential in that tail depends strongly on how the correlation functional interpolates between unpolarized and fully polarized electron gases. The error depends on the pipeline as a whole: the functional and the self-consistent spin densities fix the potential, its matrix fixes the selection, and the selected products fix the reconstructed function.

Returns
-------
float, real-space L2 error of the reconstructed LSD exchange-correlation potential of the requested spin in hartree bohr^(3/2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstruction_l2_error(Z: float, n_up: int, n_down: int, exponents: "np.ndarray", spin: int, eps: float) -> float:
    '''L2 error of the importance-selected reconstruction of a self-consistent LSD exchange-correlation potential.

    Parameters
    ----------
    Z : float
        Nuclear charge in units of e, as in radial_lsd_spin_densities.
    n_up : int
        Number of occupied up-spin s orbitals, as in radial_lsd_spin_densities.
    n_down : int
        Number of occupied down-spin s orbitals, as in radial_lsd_spin_densities.
    exponents : np.ndarray
        1-D array of K distinct positive exponents in bohr^-2 of the normalized s-Gaussian basis.
    spin : int
        0 for the up-spin and 1 for the down-spin exchange-correlation potential.
    eps : float
        Selection threshold in hartree, finite and > 0, as in importance_selected_products.

    Returns
    -------
    result : float
        ||v_R - v_xc,s|| = (integral over all space of (v_R(r) - v_xc,s(r))^2 d^3r)^(1/2) in hartree bohr^(3/2), as a
        Python float, where v_xc,s is the LSD potential of spin s at the basis-free self-consistent spin densities
        (radial_lsd_spin_densities and lsd_exchange_correlation, no density cutoff) and v_R is the reconstruction of
        reconstructed_potential from the matrix of spin s from xc_potential_matrices, with the same basis and threshold.
        Converged to better than 1e-7.

    Raises
    ------
    ValueError
        Under the conditions of radial_lsd_spin_densities, xc_potential_matrices or importance_selected_products, or if spin is not 0 or 1.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_reconstruction_l2_error(Z: float, n_up: int, n_down: int, exponents: "np.ndarray", spin: int, eps: float) -> float:
    import numpy as np
    if isinstance(spin, bool) or spin not in (0, 1):
        raise ValueError("spin must be 0 or 1")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    a = _sgauss_check(exponents)
    V = _oracle_xc_potential_matrices(Z, n_up, n_down, a)[spin]
    r, w = _radial_grid(a)
    v = _xc_on_grid(Z, n_up, n_down, r)[1 + spin]
    vr = _oracle_reconstructed_potential(V, a, eps, r)
    return float(np.sqrt(np.sum(w * (vr - v) ** 2)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the down-spin potential of lithium, 14 even-tempered functions ---
        {"setup": "import numpy as np\nex = 0.03 * 2.35 ** np.arange(14)\n",
         "call": "reconstruction_l2_error(3.0, 2, 1, ex.copy(), 1, 1e-5)",
         "gold_call": "_oracle_reconstruction_l2_error(3.0, 2, 1, ex.copy(), 1, 1e-5)", "tol": 1e-6},
        # --- Normal: the up-spin potential of the Be+ cation ---
        {"setup": "import numpy as np\nex = 0.05 * 2.4 ** np.arange(13)\n",
         "call": "reconstruction_l2_error(4.0, 2, 1, ex.copy(), 0, 1e-5)",
         "gold_call": "_oracle_reconstruction_l2_error(4.0, 2, 1, ex.copy(), 0, 1e-5)", "tol": 1e-6},
        # --- Boundary: closed-shell helium with a loose threshold in a diffuse basis ---
        {"setup": "import numpy as np\nex = 0.02 * 3.0 ** np.arange(10)\n",
         "call": "reconstruction_l2_error(2.0, 1, 1, ex.copy(), 1, 1e-5)",
         "gold_call": "_oracle_reconstruction_l2_error(2.0, 1, 1, ex.copy(), 1, 1e-5)", "tol": 1e-6},
        # --- Edge: a threshold that admits no product, so the error is the norm of the potential itself ---
        {"setup": "import numpy as np\nex = np.array([0.3, 1.0, 3.5, 12.0, 45.0])\n",
         "call": "reconstruction_l2_error(3.0, 2, 1, ex.copy(), 1, 50.0)",
         "gold_call": "_oracle_reconstruction_l2_error(3.0, 2, 1, ex.copy(), 1, 50.0)", "tol": 1e-6},
        # --- Error: a spin label other than 0 or 1 ---
        {"setup": "import numpy as np\ndef _probe(fn):\n    try:\n        fn(3.0, 2, 1, 0.05 * 2.2 ** np.arange(8), 2, 1e-6)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(reconstruction_l2_error)", "gold_call": "_probe(_oracle_reconstruction_l2_error)"},
    ]
