"""
Step 04: Overlap matrix of basis-function products. Overlap matrix of the products of pairs of basis functions.

A local potential can be expanded in functions built from the orbital basis itself, the simplest choice being the products f_k f_l of pairs of basis functions. Requiring that such an expansion reproduce given matrix elements <f_i f_j | v> leads to a linear system whose matrix holds the overlaps <f_i f_j | f_k f_l> of those products, which are four-function integrals. A basis of K functions has K(K + 1)/2 distinct products, and for Gaussian basis sets these products are numerically, and sometimes exactly, linearly dependent, so the full product overlap matrix is extremely ill-conditioned or singular. For s-type Gaussians a product of two functions is again an s-type Gaussian whose exponent is the sum of the two exponents, which makes the overlaps elementary but also makes the dependence visible: two pairs with the same exponent sum give identical product functions up to a constant factor.

Returns
-------
numpy.ndarray of shape (M, M), M = K(K+1)/2: overlaps of the unnormalized products of normalized s Gaussians, pairs i <= j in row-major order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def product_overlap_matrix(exponents: "np.ndarray") -> "np.ndarray":
    '''Overlap matrix of the unnormalized products of normalized s Gaussians.

    Parameters
    ----------
    exponents : np.ndarray
        1-D array of K distinct, finite, positive exponents alpha_k in bohr^-2 defining the normalized basis functions
        f_k(r) = (2 alpha_k / pi)^(3/4) exp(-alpha_k r^2).

    Returns
    -------
    result : np.ndarray
        Symmetric array of shape (M, M), M = K(K + 1)/2, with element (p, q) equal to the integral over all space of
        f_i(r) f_j(r) f_k(r) f_l(r), where p labels the pair (i, j) and q the pair (k, l). Pairs with i <= j are ordered
        row by row: (0, 0), (0, 1), ..., (0, K-1), (1, 1), (1, 2), ..., (K-1, K-1). The products are not renormalized.

    Raises
    ------
    ValueError
        If the exponents are not a 1-D array of distinct finite positive numbers.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_product_overlap_matrix(exponents: "np.ndarray") -> "np.ndarray":
    import numpy as np
    a = _sgauss_check(exponents)
    n = (2.0 * a / np.pi) ** 0.75
    iu, ju = np.triu_indices(a.size)
    pa = a[iu] + a[ju]
    pn = n[iu] * n[ju]
    return pn[:, None] * pn[None, :] * (np.pi / (pa[:, None] + pa[None, :])) ** 1.5

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: an even-tempered basis of 8 functions (36 products) ---
        {"setup": "import numpy as np\nex = 0.03 * 2.1 ** np.arange(8)\n",
         "call": "product_overlap_matrix(ex.copy())",
         "gold_call": "_oracle_product_overlap_matrix(ex.copy())", "tol": 1e-10},
        # --- Normal: exponents with equal pair sums, 1 + 4 = 2 + 3, give proportional products ---
        {"setup": "import numpy as np\nex = np.array([1.0, 2.0, 3.0, 4.0])\n",
         "call": "product_overlap_matrix(ex.copy())",
         "gold_call": "_oracle_product_overlap_matrix(ex.copy())", "tol": 1e-10},
        # --- Boundary: widely spread exponents, from very diffuse to very steep ---
        {"setup": "import numpy as np\nex = np.array([2.0e-3, 0.7, 5.0e4])\n",
         "call": "product_overlap_matrix(ex.copy())",
         "gold_call": "_oracle_product_overlap_matrix(ex.copy())", "tol": 1e-10},
        # --- Edge: a single basis function ---
        {"setup": "import numpy as np\nex = np.array([0.8])\n",
         "call": "product_overlap_matrix(ex.copy())",
         "gold_call": "_oracle_product_overlap_matrix(ex.copy())", "tol": 1e-12},
        # --- Error: a negative exponent ---
        {"setup": "import numpy as np\ndef _probe(fn):\n    try:\n        fn(np.array([0.5, -2.0]))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(product_overlap_matrix)", "gold_call": "_probe(_oracle_product_overlap_matrix)"},
    ]
