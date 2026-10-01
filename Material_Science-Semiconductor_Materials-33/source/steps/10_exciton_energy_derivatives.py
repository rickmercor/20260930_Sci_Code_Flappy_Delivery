"""
Compute thickness derivatives of a simple lowest exciton eigenvalue.

The direct interaction and its first two ordinary thickness derivatives

define H(h) = diag(E_c-E_v) - D(h), with fixed bands. Curvature includes

the changing normalized exciton eigenstate. Excited-state degeneracies are

allowed; the lowest eigenvalue must be isolated.

Returns
-------
real array (3,) containing energy and its first two thickness derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exciton_energy_derivatives(
    energies: "np.ndarray",
    direct_derivatives: "np.ndarray",
    gap_tolerance: float = 1e-9,
) -> "np.ndarray":
    """Return the lowest excitation energy, slope and curvature.

    Parameters
    ----------
    energies : "np.ndarray"
        Finite real array (K, 2), K >= 1, with positive rowwise direct gaps.
        The band energies do not depend on the differentiation parameter.
    direct_derivatives : "np.ndarray"
        Finite complex array (3, K, K) containing D, dD/dh and d^2D/dh^2.
        Each is Hermitian to entrywise absolute tolerance 1e-10, with
        roundoff within that bound symmetrized. Units are eV, eV/angstrom
        and eV/angstrom^2. The mesh weight is already included.
    gap_tolerance : float
        Positive finite number in eV. For K > 1, the difference of the
        lowest two eigenvalues of H must be greater than this number.

    Returns
    -------
    derivatives : "np.ndarray"
        Real array (3,) containing E0, dE0/dh and d^2E0/dh^2, with units
        eV, eV/angstrom and eV/angstrom^2. The two derivatives are ordinary
        analytic derivatives of the simple lowest eigenvalue of
        H(h)=diag(E_c-E_v)-D(h); include the eigenstate response. No
        finite differences, clipping, extra spin or extra mesh factor.
        For K=1 the eigenstate-response contribution is zero.
        The order-0 value is obtained using lowest_exciton_energy.

    Raises
    ------
    ValueError
        If any shape, finiteness, realness, positive direct gap,
        Hermiticity or positive tolerance requirement is violated, or the
        lowest excitation is not isolated by more than gap_tolerance.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exciton_energy_derivatives(
    energies: "np.ndarray",
    direct_derivatives: "np.ndarray",
    gap_tolerance: float = 1e-9,
) -> "np.ndarray":
    """Evaluate the spectral response with the reduced resolvent."""
    e = _finite_array(energies, float, "energies")
    ds = _finite_array(direct_derivatives, complex, "direct_derivatives")
    tol = _finite_scalar(gap_tolerance, "gap_tolerance")
    if e.ndim != 2 or e.shape[1] != 2 or len(e) < 1:
        raise ValueError("energies must have shape (K, 2)")
    if ds.shape != (3, len(e), len(e)) or tol <= 0:
        raise ValueError("invalid derivative shape or gap tolerance")
    if not np.allclose(ds, ds.conj().transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError("all direct derivatives must be Hermitian")
    ds = (ds + ds.conj().transpose(0, 2, 1)) / 2
    energy = _oracle_lowest_exciton_energy(e, ds[0])
    h0 = np.diag(e[:, 1] - e[:, 0]) - ds[0]
    values, states = np.linalg.eigh(h0)
    if len(e) > 1 and values[1] - values[0] <= tol:
        raise ValueError("lowest excitation is not sufficiently isolated")
    ground = states[:, 0]
    h1, h2 = -ds[1], -ds[2]
    slope = np.vdot(ground, h1 @ ground).real
    curvature = np.vdot(ground, h2 @ ground).real
    amplitudes = states[:, 1:].conj().T @ h1 @ ground
    curvature += 2 * np.sum(np.abs(amplitudes) ** 2 / (energy - values[1:]))
    return np.array([energy, slope, curvature], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific and boundary fixtures."""
    return [
        {
            "setup": """
import numpy as np

e = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
h0 = np.diag([0.4, 1.1, 1.1]).astype(complex)
h1 = np.array([[0.3, 0.2j, -0.1], [-0.2j, -0.2, 0.15j], [-0.1, -0.15j, 0.4]])
h2 = np.array([[0.2, 0.03, 0.04j], [0.03, -0.1, 0.02], [-0.04j, 0.02, 0.3]])
ds = np.stack((2 * np.eye(3) - h0, -h1, -h2))
tol = 1e-9
""",
            "call": """
exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
            "gold_call": """
_oracle_exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
        },
        {
            "setup": """
import numpy as np

e = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
h0 = np.diag([0.4, 1.1, 1.1]).astype(complex)
h1 = np.array([[0.3, 0.2j, -0.1], [-0.2j, -0.2, 0.15j], [-0.1, -0.15j, 0.4]])
h2 = np.array([[0.2, 0.03, 0.04j], [0.03, -0.1, 0.02], [-0.04j, 0.02, 0.3]])
ds = np.stack((2 * np.eye(3) - h0, -h1, -h2))
tol = 1e-9
rng = np.random.default_rng(1021)
u = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
ds = np.array([u @ matrix @ u.conj().T for matrix in ds])
""",
            "call": """
exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
            "gold_call": """
_oracle_exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
        },
        {
            "setup": """
import numpy as np

e = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
h0 = np.diag([0.4, 1.1, 1.1]).astype(complex)
h1 = np.array([[0.3, 0.2j, -0.1], [-0.2j, -0.2, 0.15j], [-0.1, -0.15j, 0.4]])
h2 = np.array([[0.2, 0.03, 0.04j], [0.03, -0.1, 0.02], [-0.04j, 0.02, 0.3]])
ds = np.stack((2 * np.eye(3) - h0, -h1, -h2))
tol = 1e-9
h0[1, 1] = 0.4001
ds[0] = 2 * np.eye(3) - h0
""",
            "call": """
exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
            "gold_call": """
_oracle_exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
        },
        {
            "setup": """
import numpy as np

e = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
h0 = np.diag([0.4, 1.1, 1.1]).astype(complex)
h1 = np.array([[0.3, 0.2j, -0.1], [-0.2j, -0.2, 0.15j], [-0.1, -0.15j, 0.4]])
h2 = np.array([[0.2, 0.03, 0.04j], [0.03, -0.1, 0.02], [-0.04j, 0.02, 0.3]])
ds = np.stack((2 * np.eye(3) - h0, -h1, -h2))
tol = 1e-9
e = np.array([[-1.0, 2.0]])
ds = np.array([[[0.7]], [[-0.3]], [[0.2]]])
""",
            "call": """
exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
            "gold_call": """
_oracle_exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
        },
        {
            "setup": """
import numpy as np

e = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
h0 = np.diag([0.4, 1.1, 1.1]).astype(complex)
h1 = np.array([[0.3, 0.2j, -0.1], [-0.2j, -0.2, 0.15j], [-0.1, -0.15j, 0.4]])
h2 = np.array([[0.2, 0.03, 0.04j], [0.03, -0.1, 0.02], [-0.04j, 0.02, 0.3]])
ds = np.stack((2 * np.eye(3) - h0, -h1, -h2))
tol = 1e-9
ds[1] = -np.eye(3) * 0.2
ds[2] = np.eye(3) * 0.1
""",
            "call": """
exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
            "gold_call": """
_oracle_exciton_energy_derivatives(e.copy(), ds.copy(), tol)
""",
        },
        {
            "setup": """
import numpy as np

e = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
h0 = np.diag([0.4, 1.1, 1.1]).astype(complex)
h1 = np.array([[0.3, 0.2j, -0.1], [-0.2j, -0.2, 0.15j], [-0.1, -0.15j, 0.4]])
h2 = np.array([[0.2, 0.03, 0.04j], [0.03, -0.1, 0.02], [-0.04j, 0.02, 0.3]])
ds = np.stack((2 * np.eye(3) - h0, -h1, -h2))
tol = 1e-9

ds[0] = 2 * np.eye(3) - np.diag([0.4, 0.4, 1.0])


def rejected(fn):
    try:
        fn(e, ds, tol)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": """
rejected(exciton_energy_derivatives)
""",
            "gold_call": """
rejected(_oracle_exciton_energy_derivatives)
""",
        },
    ]
