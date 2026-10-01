"""
Assemble the energy-operator matrix of Eq (59) for the vector test functions formed by multiplying each non-constant scalar test by the two Cartesian unit vectors, and return it as a (2 n_T, n_p) array, x-lifted tests first, tests in basis order. Each row is a sum over the family of the source's per-bond basis factor, times the square of the test increment times the bond component, divided by the bond length to the power q, times V_j; leave out any constant prefactor common to a row and its target. Raise ValueError under the same conditions as the derivative block.

The bond-based strain energy applied to an affine test deformation is quadratic in the deformation and linear in the weight coefficients. Matching it alongside the derivative operators is what the source's least-squares problem does.

Returns
-------
ndarray of float64 with shape (2 n_T, n_p).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def energy_block(family, basis, basis0, delta, q):
    """ndarray of float64 with shape (2 n_T, n_p)."""
    return np.zeros((2 * (np.shape(basis)[1] - 1), np.shape(basis)[1]), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_energy_block(family, basis, basis0, delta, q):
    """Eq (59): the energy operator rows for the vector tests phi_k e_l, stacked (2*n_T, n_p).

    Row (l, k): sum_j  (p(xi_j) + p(0))/2  *  ((phi_k(xi_j) - phi_k(0)) * xi_{j,l})^2 / |xi_j|^q  * V_j
    x-lifted tests first, then y-lifted, for the 5 non-constant scaled monomials. Any
    constant prefactor (the micromodulus, the 1/4 of the source's eq 45) is common to the
    row and its target and is removed by the per-row normalization of step 6.
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
    dphi = (P - p0[None, :])[:, 1:]
    r = fam[:, 3]
    V = fam[:, 4]
    blocks = []
    for l in (1, 2):
        kern = fam[:, l] ** 2 / r ** q * V
        blocks.append((Pav[:, None, :] * (kern[:, None] * dphi ** 2)[:, :, None]).sum(axis=0))
    return np.vstack(blocks)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nfamily = np.array([[1.0, 0.25, 0.0, 0.25, 0.0625], [2.0, 0.0, 0.25, 0.25, 0.0625], [3.0, 0.25, 0.25, 0.3535533905932738, 0.0625], [4.0, -0.25, 0.0, 0.25, 0.0625]])\ndelta, q = 0.5, 3\nbasis = _oracle_scaled_monomial_basis(family[:, 1:3], delta)\nbasis0 = _oracle_scaled_monomial_basis(np.zeros((1, 2)), delta)[0]",
            "call": "energy_block(family, basis, basis0, delta, q)",
            "gold_call": "_oracle_energy_block(family, basis, basis0, delta, q)",
        },
        {
            "setup": "import numpy as np\nfamily = np.array([[7.0, 0.125, 0.0, 0.125, 0.015625]])\ndelta, q = 0.25, 3\nbasis = _oracle_scaled_monomial_basis(family[:, 1:3], delta)\nbasis0 = _oracle_scaled_monomial_basis(np.zeros((1, 2)), delta)[0]",
            "call": "energy_block(family, basis, basis0, delta, q)",
            "gold_call": "_oracle_energy_block(family, basis, basis0, delta, q)",
        },
        {
            "setup": "import numpy as np\nfamily = np.array([[0.0, 0.375, 0.5, 0.625, 0.0625], [1.0, -0.375, 0.5, 0.625, 0.0625], [2.0, 0.0, -0.625, 0.625, 0.0625]])\ndelta, q = 0.625, 2\nbasis = _oracle_scaled_monomial_basis(family[:, 1:3], delta)\nbasis0 = _oracle_scaled_monomial_basis(np.zeros((1, 2)), delta)[0]",
            "call": "energy_block(family, basis, basis0, delta, q)",
            "gold_call": "_oracle_energy_block(family, basis, basis0, delta, q)",
        },
    ]
