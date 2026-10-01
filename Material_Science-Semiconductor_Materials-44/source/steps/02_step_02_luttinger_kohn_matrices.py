"""
Build the internal-space matrices of the Luttinger-Kohn kinetic energy, grouped by momentum dependence.

The kinetic energy of a hole in the cubic valence band of cuprous oxide is the Luttinger-Kohn Hamiltonian, equivalently the Suzuki-Hensel Hamiltonian, written for a quasispin I = 1 coupled to the hole spin S = 1/2. Its terms fall into three groups according to how they depend on momentum: one internal-space matrix multiplies the isotropic squared momentum, one matrix per Cartesian axis multiplies that axis's squared momentum component, and one matrix per unordered pair of distinct axes multiplies the symmetrised product of those two momentum components. The Luttinger parameters gamma_i and the quasispin-spin parameters eta_i enter as in the standard cubic form, and the symmetrised product is {A, B} = (A B + B A) / 2. Every block is Hermitian, but they are not all real: keep the array complex. Return the seven matrices stacked along the first axis, ordered isotropic, then axial x, y, z, then the pair matrices for (x, y), (y, z) and (z, x). They are dimensionless; the factor hbar^2 / (2 m0) is applied later.

Returns
-------
numpy.ndarray of shape (7, 6, 6), complex, dimensionless
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def luttinger_kohn_matrices(gamma1: float, gamma2: float, gamma3: float, eta1: float, eta2: float, eta3: float) -> "np.ndarray":
    '''Internal-space matrices of the Luttinger-Kohn kinetic energy.

    Parameters
    ----------
    gamma1, gamma2, gamma3 : float
        Luttinger parameters. Must be finite.
    eta1, eta2, eta3 : float
        Quasispin-spin coupling parameters. Must be finite.

    Returns
    -------
    numpy.ndarray
        Complex array of shape (7, 6, 6), dimensionless, each block Hermitian,
        stacked as isotropic, axial x, axial y, axial z, pair xy, pair yz, pair zx.
        The isotropic and axial blocks are real; the xy and yz pair blocks are purely
        imaginary.

    Raises
    ------
    ValueError
        If any argument is not finite.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _spin_operators():
    """Quasispin I (I=1) and hole spin S (S=1/2) in the ordered product basis."""
    root2 = np.sqrt(2.0)
    ix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float) / root2
    iy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / root2
    iz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2.0
    sy = np.array([[0, -1j], [1j, 0]]) / 2.0
    sz = np.diag([1.0, -1.0]).astype(complex) / 2.0
    e3, e2 = np.eye(3), np.eye(2)
    return ([np.kron(m, e2) for m in (ix, iy, iz)],
            [np.kron(e3, m) for m in (sx, sy, sz)])

def _oracle_luttinger_kohn_matrices(gamma1: float, gamma2: float, gamma3: float, eta1: float, eta2: float, eta3: float) -> "np.ndarray":
    vals = [float(v) for v in (gamma1, gamma2, gamma3, eta1, eta2, eta3)]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("all six band parameters must be finite")
    g1, g2, g3, n1, n2, n3 = vals
    iop, sop = _spin_operators()
    i_dot_s = sum(iop[a] @ sop[a] for a in range(3))
    blocks = [(g1 + 4.0 * g2) * np.eye(6) + 2.0 * (n1 + 2.0 * n2) * i_dot_s]
    for a in range(3):
        blocks.append(-6.0 * g2 * (iop[a] @ iop[a]) - 12.0 * n2 * (iop[a] @ sop[a]))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        anti = (iop[a] @ iop[b] + iop[b] @ iop[a]) / 2.0
        blocks.append(-12.0 * g3 * anti - 12.0 * n3 * (iop[a] @ sop[b] + iop[b] @ sop[a]))
    return np.stack(blocks)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "luttinger_kohn_matrices(1.76, 0.7532, -0.3668, -0.020, -0.0037, -0.0337)",
            "gold_call": "_oracle_luttinger_kohn_matrices(1.76, 0.7532, -0.3668, -0.020, -0.0037, -0.0337)",
        },
        {
            "setup": "import numpy as np",
            "call": "luttinger_kohn_matrices(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)",
            "gold_call": "_oracle_luttinger_kohn_matrices(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "luttinger_kohn_matrices(2.5, -1.25, 0.75, 0.3, -0.45, 0.2)",
            "gold_call": "_oracle_luttinger_kohn_matrices(2.5, -1.25, 0.75, 0.3, -0.45, 0.2)",
        },
    ]
