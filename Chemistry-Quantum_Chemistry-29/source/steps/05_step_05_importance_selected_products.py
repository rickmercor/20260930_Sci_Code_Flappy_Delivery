"""
Step 05: Importance-based selection of basis-function products. Importance-based selection of basis-function products that reproduce a potential matrix.

A potential expanded in a subset R of basis-function products, with coefficients fixed by requiring that its matrix elements equal the given ones for every product in R, reproduces not only those elements but every element whose product is a linear combination of the products in R. Linear dependencies among products therefore help rather than hurt: once the right independent products are fixed, the dependent matrix elements follow automatically. Classical approaches choose the subset for numerical linear independence alone, by singular value truncation or pivoted Cholesky decomposition of the product overlap matrix, and so may keep unimportant products and drop important ones. The alternative is to grow the subset one product at a time in order of importance, measured by the magnitude of the part of the matrix that the current expansion does not yet reproduce, in the spirit of matching pursuit. A product whose matrix element is already reproduced is linearly dependent on the chosen ones and is never picked, so the selection filters itself, and the coefficients always come from a square, nonsingular system restricted to the chosen products. Adding several products per iteration instead risks choosing nearly dependent products together and destroys this stability.

Returns
-------
numpy.ndarray of int, shape (M_R, 2): zero-based pairs (i, j), i <= j, in the order they are selected
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def importance_selected_products(V: "np.ndarray", exponents: "np.ndarray", eps: float) -> "np.ndarray":
    '''Products of basis-function pairs chosen one at a time by the largest unreproduced matrix element.

    Parameters
    ----------
    V : np.ndarray
        Symmetric (K, K) matrix in hartree of a local potential v in the normalized basis f_k of product_overlap_matrix,
        V_ij = <f_i | v | f_j>.
    exponents : np.ndarray
        1-D array of the K distinct positive exponents of that basis, in bohr^-2.
    eps : float
        Convergence threshold in hartree, finite and > 0.

    Returns
    -------
    result : np.ndarray
        Integer array of shape (M_R, 2): the selected pairs (i, j), i <= j, zero-based, in the order selected. With the
        selected set R, the trial potential v_R(r) = sum over (k, l) in R of b_kl f_k(r) f_l(r) uses plain, unnormalized
        products, and its coefficients b solve exactly the square system sum over (k, l) in R of
        <f_i f_j | f_k f_l> b_kl = V_ij for all (i, j) in R, so its matrix V^R reproduces V on every selected pair.
        Starting from an empty R (v_R = 0), each iteration computes the residuals |V^R_ij - V_ij| over all pairs i <= j;
        if the largest residual is below eps the selection stops, otherwise the pair with the largest residual is added
        (the first in the row-by-row pair order of product_overlap_matrix if several are equal) and b is recomputed.
        Exactly one pair is added per iteration, and the selection also stops once every pair has been selected. If
        the largest element of |V| is already below eps the result has shape (0, 2).

    Raises
    ------
    ValueError
        If V is not a finite symmetric (K, K) array matching the exponents, the exponents are invalid, or eps is not
        finite and positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _select_products(V, exponents, eps):
    import numpy as np
    a = _sgauss_check(exponents)
    V = np.asarray(V, dtype=float)
    K = a.size
    if V.shape != (K, K) or not np.all(np.isfinite(V)) or not np.allclose(V, V.T, rtol=1e-12, atol=1e-14):
        raise ValueError("V must be a finite symmetric matrix matching the basis")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    W = _oracle_product_overlap_matrix(a)
    iu, ju = np.triu_indices(K)
    target = V[iu, ju]
    chosen = []
    coef = np.zeros(0)
    fit = np.zeros_like(target)
    while len(chosen) < target.size:
        res = np.abs(fit - target)
        q = int(np.argmax(res))
        if res[q] < eps:
            break
        chosen.append(q)
        coef = np.linalg.solve(W[np.ix_(chosen, chosen)], target[chosen])
        fit = W[:, chosen] @ coef
    return np.array(chosen, dtype=int), coef, iu, ju

def _oracle_importance_selected_products(V: "np.ndarray", exponents: "np.ndarray", eps: float) -> "np.ndarray":
    import numpy as np
    chosen, coef, iu, ju = _select_products(V, exponents, eps)
    if chosen.size == 0:
        return np.zeros((0, 2), dtype=int)
    return np.stack([iu[chosen], ju[chosen]], axis=1).astype(int)

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
              "    return -q * np.outer(n, n) * 2 * np.pi / p * np.sqrt(mu / (p + mu))\n")
    return [
        # --- Normal: a screened nuclear attraction in an even-tempered basis of 12 functions ---
        {"setup": common + "ex = 0.03 * 2.1 ** np.arange(12)\nV = vmat_erf(ex, 4.0, 2.5)\n",
         "call": "importance_selected_products(V.copy(), ex.copy(), 0.0001)",
         "gold_call": "_oracle_importance_selected_products(V.copy(), ex.copy(), 0.0001)", "tol": 1e-12},
        # --- Normal: a two-Gaussian well in a sparser basis ---
        {"setup": common + "ex = 0.05 * 2.5 ** np.arange(10)\nV = vmat_gauss(ex, [-2.0, -0.7], [0.9, 0.12])\n",
         "call": "importance_selected_products(V.copy(), ex.copy(), 0.0001)",
         "gold_call": "_oracle_importance_selected_products(V.copy(), ex.copy(), 0.0001)", "tol": 1e-12},
        # --- Boundary: exact product dependencies, 15 products with only 9 distinct exponent sums ---
        {"setup": common + "ex = np.array([0.5, 1.0, 1.5, 2.0, 2.5])\nV = vmat_gauss(ex, [-1.3, 0.4], [0.35, 3.0])\n",
         "call": "importance_selected_products(V.copy(), ex.copy(), 1e-06)",
         "gold_call": "_oracle_importance_selected_products(V.copy(), ex.copy(), 1e-06)", "tol": 1e-12},
        # --- Normal at scale: 26 functions and 351 products ---
        {"setup": common + "ex = 0.02 * 2.0 ** np.arange(26)\nV = vmat_gauss(ex, [-2.0, -0.7], [0.9, 0.12])\n",
         "call": "importance_selected_products(V.copy(), ex.copy(), 0.0001)",
         "gold_call": "_oracle_importance_selected_products(V.copy(), ex.copy(), 0.0001)", "tol": 1e-12},
        # --- Edge: a threshold that stops after the first product ---
        {"setup": common + "ex = np.array([0.4, 3.0])\nV = vmat_gauss(ex, [-1.0], [1.0])\n"
                  "t = 0.999 * np.sort(np.abs(V[np.triu_indices(2)]))[-2]\n",
         "call": "importance_selected_products(V.copy(), ex.copy(), t)",
         "gold_call": "_oracle_importance_selected_products(V.copy(), ex.copy(), t)", "tol": 1e-12},
        # --- Error: a non-positive threshold ---
        {"setup": common + "ex = np.array([0.4, 3.0])\nV = vmat_gauss(ex, [-1.0], [1.0])\n"
                  "def _probe(fn):\n    try:\n        fn(V.copy(), ex.copy(), 0.0)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(importance_selected_products)", "gold_call": "_probe(_oracle_importance_selected_products)"},
    ]
