"""
Step 06: Reconstructed real-space potential. Real-space potential reconstructed from its matrix with importance-selected products.

Once the subset of products is fixed, the reconstructed potential is an explicit function: a linear combination of the selected products with the coefficients that reproduce their matrix elements exactly. Enlarging the subset can only reduce the least-squares distance between this function and the true potential when the expansion functions are the products themselves, because the reconstruction is then the orthogonal projection of the true potential onto the span of the selected products. The reconstruction is nevertheless not the true potential. A finite matrix is unchanged by any potential component orthogonal to all products, so the pointwise shape is fixed only in the region the basis products can describe, and a potential whose tail decays more slowly than the most diffuse products cannot be followed there. The coefficients of nearly dependent products are large and of alternating sign, so the function must be evaluated from the full linear combination rather than from any truncated or smoothed version of it.

Returns
-------
numpy.ndarray with the shape of radii: reconstructed potential v_R(r) in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstructed_potential(V: "np.ndarray", exponents: "np.ndarray", eps: float, radii: "np.ndarray") -> "np.ndarray":
    '''Reconstructed local potential v_R(r) at given distances from the nucleus.

    Parameters
    ----------
    V : np.ndarray
        Symmetric (K, K) potential matrix in hartree, as in importance_selected_products.
    exponents : np.ndarray
        1-D array of the K distinct positive exponents of the normalized s-Gaussian basis, in bohr^-2.
    eps : float
        Selection threshold in hartree, finite and > 0, as in importance_selected_products.
    radii : np.ndarray
        1-D array of finite distances r >= 0 in bohr.

    Returns
    -------
    result : np.ndarray
        Array with the shape of radii, in hartree: v_R(r) = sum over (k, l) in R of b_kl f_k(r) f_l(r), with R the pairs
        selected by importance_selected_products for (V, exponents, eps) and b the coefficients that reproduce V exactly
        on those pairs. It is 0 everywhere when no pair is selected.

    Raises
    ------
    ValueError
        Under the conditions of importance_selected_products, or if radii contains a negative or non-finite value.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_reconstructed_potential(V: "np.ndarray", exponents: "np.ndarray", eps: float, radii: "np.ndarray") -> "np.ndarray":
    import numpy as np
    r = np.asarray(radii, dtype=float)
    if not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("radii must be finite and non-negative")
    pairs = _oracle_importance_selected_products(V, exponents, eps)
    chosen, coef, iu, ju = _select_products(V, exponents, eps)
    a = np.asarray(exponents, dtype=float)
    n = (2.0 * a / np.pi) ** 0.75
    if pairs.shape[0] == 0:
        return np.zeros_like(r)
    i, j = pairs[:, 0], pairs[:, 1]
    pa = a[i] + a[j]
    pn = n[i] * n[j]
    flat = r.reshape(-1)
    vals = np.exp(-np.minimum(pa[None, :] * flat[:, None] ** 2, 745.0)) @ (pn * coef)
    return vals.reshape(r.shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    common = ("import numpy as np\n"
              "def vmat_gauss(ex, cs, gs):\n"
              "    n = (2 * ex / np.pi) ** 0.75\n"
              "    p = ex[:, None] + ex[None, :]\n"
              "    return sum(c * np.outer(n, n) * (np.pi / (p + g)) ** 1.5 for c, g in zip(cs, gs))\n"
              "def vmat_erf(ex, q, mu):\n"
              "    n = (2 * ex / np.pi) ** 0.75\n"
              "    p = ex[:, None] + ex[None, :]\n"
              "    return -q * np.outer(n, n) * 2 * np.pi / p * np.sqrt(mu / (p + mu))\n"
              "rr = np.array([0.0, 0.05, 0.3, 0.8, 1.5, 2.5, 4.0, 6.0])\n")
    return [
        # --- Normal: screened nuclear attraction, 12 functions ---
        {"setup": common + "ex = 0.03 * 2.1 ** np.arange(12)\nV = vmat_erf(ex, 4.0, 2.5)\n",
         "call": "reconstructed_potential(V.copy(), ex.copy(), 0.001, rr.copy())",
         "gold_call": "_oracle_reconstructed_potential(V.copy(), ex.copy(), 0.001, rr.copy())", "tol": 1e-06},
        # --- Normal: a potential that lies inside the product span is recovered ---
        {"setup": common + "ex = np.array([0.5, 1.0, 1.5, 2.0, 2.5])\nV = vmat_gauss(ex, [-1.3, 0.4], [2.0, 3.0])\n",
         "call": "reconstructed_potential(V.copy(), ex.copy(), 1e-07, rr.copy())",
         "gold_call": "_oracle_reconstructed_potential(V.copy(), ex.copy(), 1e-07, rr.copy())", "tol": 1e-07},
        # --- Normal: a two-Gaussian well in a 20-function basis ---
        {"setup": common + "ex = 0.03 * 2.1 ** np.arange(20)\nV = vmat_gauss(ex, [-2.0, -0.7], [0.9, 0.12])\n",
         "call": "reconstructed_potential(V.copy(), ex.copy(), 0.0001, rr.copy())",
         "gold_call": "_oracle_reconstructed_potential(V.copy(), ex.copy(), 0.0001, rr.copy())", "tol": 1e-06},
        # --- Boundary: a loose threshold with few products ---
        {"setup": common + "ex = 0.08 * 2.6 ** np.arange(9)\nV = vmat_gauss(ex, [-2.0, -0.7], [0.9, 0.12])\n",
         "call": "reconstructed_potential(V.copy(), ex.copy(), 0.03, rr.copy())",
         "gold_call": "_oracle_reconstructed_potential(V.copy(), ex.copy(), 0.03, rr.copy())", "tol": 1e-08},
        # --- Edge: a threshold above every matrix element gives the zero potential ---
        {"setup": common + "ex = np.array([0.4, 3.0])\nV = vmat_gauss(ex, [-1.0], [1.0])\n",
         "call": "reconstructed_potential(V.copy(), ex.copy(), 10.0, rr.copy())",
         "gold_call": "_oracle_reconstructed_potential(V.copy(), ex.copy(), 10.0, rr.copy())", "tol": 1e-12},
        # --- Error: a negative distance ---
        {"setup": common + "ex = np.array([0.4, 3.0])\nV = vmat_gauss(ex, [-1.0], [1.0])\n"
                  "def _probe(fn):\n    try:\n        fn(V.copy(), ex.copy(), 1e-6, np.array([1.0, -0.1]))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(reconstructed_potential)", "gold_call": "_probe(_oracle_reconstructed_potential)"},
    ]
