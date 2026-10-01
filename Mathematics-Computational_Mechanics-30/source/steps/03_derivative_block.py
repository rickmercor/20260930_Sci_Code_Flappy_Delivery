"""
Assemble the derivative-operator matrices of Eqs (56) and (57) for the x and y directions and stack them as a (2 n_T, n_p) array, x rows first, where the n_T = n_p - 1 test functions are the non-constant monomials of the basis in basis order (the constant test gives an identically zero row and is omitted). family is the (n_F, 5) array of step 1, basis the (n_F, n_p) basis evaluated at the neighbours, basis0 the basis evaluated at the node itself. Each row is a sum over the family of the source's per-bond basis factor, times the bond component divided by the bond length to the power q, times the test function increment between neighbour and node, times V_j. Raise ValueError if family is not (n_F, 5), if basis is not (n_F, n_p) with basis0 of length n_p, or if delta or q is not positive.

The discrete derivative operator of bond-based peridynamics applied to a test function, with the fitted weight field inserted, is linear in the basis coefficients. The source fixes how the weight field enters each bond, and that choice is what these rows encode.

Returns
-------
ndarray of float64 with shape (2 n_T, n_p).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def derivative_block(family, basis, basis0, delta, q):
    """ndarray of float64 with shape (2 n_T, n_p)."""
    return np.zeros((2 * (np.shape(basis)[1] - 1), np.shape(basis)[1]), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_derivative_block(family, basis, basis0, delta, q):
    """Eqs (56)-(57): the x- and y-derivative operator matrices, stacked as (2*n_T, n_p).

    Row (l, k): sum_j  (p(xi_j) + p(0))/2  *  xi_{j,l} / |xi_j|^q  *  (phi_k(xi_j) - phi_k(0))  * V_j
    for the n_T = 5 non-constant scaled monomial tests (the constant test gives an
    identically zero row and is omitted). The bond-symmetrized average (p(xi_j) + p(0))/2
    is what makes the fitted weights enter every bond symmetrically.
    """
    import numpy as np
    fam = np.asarray(family, dtype=np.float64)
    P = np.asarray(basis, dtype=np.float64)
    p0 = np.asarray(basis0, dtype=np.float64).reshape(-1)
    if fam.ndim != 2 or fam.shape[1] != 5:
        raise ValueError("family must be (n_F, 5)")
    if P.ndim != 2 or P.shape[0] != fam.shape[0] or P.shape[1] != p0.size or P.shape[1] < 2:
        raise ValueError("basis must be (n_F, n_p) and basis0 must have n_p entries")
    if delta <= 0.0 or q <= 0.0:
        raise ValueError("delta and q must be positive")
    Pav = 0.5 * (P + p0[None, :])
    dphi = (P - p0[None, :])[:, 1:]            # the n_p - 1 non-constant tests
    r = fam[:, 3]
    V = fam[:, 4]
    blocks = []
    for l in (1, 2):                            # xi_x, xi_y columns
        kern = fam[:, l] / r ** q * V
        blocks.append((Pav[:, None, :] * (kern[:, None] * dphi)[:, :, None]).sum(axis=0))
    return np.vstack(blocks)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nfamily = np.array([[1.0, 0.25, 0.0, 0.25, 0.0625], [2.0, 0.0, 0.25, 0.25, 0.0625], [3.0, 0.25, 0.25, 0.3535533905932738, 0.0625], [4.0, -0.25, 0.0, 0.25, 0.0625]])\ndelta, q = 0.5, 3\nbasis = _oracle_scaled_monomial_basis(family[:, 1:3], delta)\nbasis0 = _oracle_scaled_monomial_basis(np.zeros((1, 2)), delta)[0]",
            "call": "derivative_block(family, basis, basis0, delta, q)",
            "gold_call": "_oracle_derivative_block(family, basis, basis0, delta, q)",
        },
        {
            "setup": "import numpy as np\nfamily = np.array([[7.0, 0.125, 0.0, 0.125, 0.015625]])\ndelta, q = 0.25, 3\nbasis = _oracle_scaled_monomial_basis(family[:, 1:3], delta)\nbasis0 = _oracle_scaled_monomial_basis(np.zeros((1, 2)), delta)[0]",
            "call": "derivative_block(family, basis, basis0, delta, q)",
            "gold_call": "_oracle_derivative_block(family, basis, basis0, delta, q)",
        },
        {
            "setup": "import numpy as np\nfamily = np.array([[0.0, 0.375, 0.5, 0.625, 0.0625], [1.0, -0.375, 0.5, 0.625, 0.0625], [2.0, 0.0, -0.625, 0.625, 0.0625]])\ndelta, q = 0.625, 2\nbasis = _oracle_scaled_monomial_basis(family[:, 1:3], delta)\nbasis0 = _oracle_scaled_monomial_basis(np.zeros((1, 2)), delta)[0]",
            "call": "derivative_block(family, basis, basis0, delta, q)",
            "gold_call": "_oracle_derivative_block(family, basis, basis0, delta, q)",
        },
    ]
