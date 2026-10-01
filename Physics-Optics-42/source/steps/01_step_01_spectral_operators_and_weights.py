"""
Construct tensor-product spectral derivatives and normalized volume weights.

The lateral field is quasiperiodic and is differentiated through its shifted Fourier orders.  The vertical coordinate uses Chebyshev-Lobatto collocation, while the volume weights integrate the same interpolating polynomial. These operators provide the common numerical representation for every later stage.

Returns
-------
omplex128 ndarray of shape (3, N, N) containing Dx, Dz, and the diagonal normalized quadrature matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_operators_and_weights(nx: int, nz: int, d: float, h: float, alpha: float) -> "np.ndarray":
    '''Construct full-grid quasiperiodic derivative and quadrature operators.

    Parameters
    ----------
    nx : int
        Even number of equispaced lateral nodes, at least 4.
    nz : int
        Chebyshev polynomial degree, at least 2; there are nz + 1 nodes.
    d : float
        Positive lateral period in the common length unit.
    h : float
        Positive slab half-height in the common length unit.
    alpha : float
        Real Bloch wavenumber in inverse-length units.

    Returns
    -------
    operators : np.ndarray
        Complex128 array of shape (3, N, N), N = nx * (nz + 1).
        operators[0] is the x derivative, operators[1] the z derivative,
        and operators[2] a real diagonal matrix of normalized tensor-product
        quadrature weights.  Fields are flattened in C order with index
        j * (nz + 1) + r, x_j = j*d/nx, and
        z_r = h*cos(pi*r/nz), so r = 0 is the top boundary.  The diagonal
        weights sum to one and approximate (2*h*d)^(-1) times the volume
        integral.

    Raises
    ------
    ValueError
        If nx is odd or less than 4, nz is less than 2, d or h is not
        positive and finite, or alpha is not finite.
    '''
    return operators

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spectral_operators_and_weights(nx: int, nz: int, d: float, h: float, alpha: float) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not isinstance(nz, (int, np.integer)) or isinstance(nz, (bool, np.bool_)) or nz < 2:
        raise ValueError("nz must be an integer at least 2")
    if not np.isfinite(d) or d <= 0.0 or not np.isfinite(h) or h <= 0.0:
        raise ValueError("d and h must be positive and finite")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")

    x = np.arange(nx, dtype=float) * (float(d) / nx)
    p = np.arange(-nx // 2, nx // 2, dtype=int)
    alpha_p = float(alpha) + 2.0 * np.pi * p / float(d)
    qmat = np.exp(-1j * alpha_p[:, None] * x[None, :]) / nx
    qinv = np.exp(1j * x[:, None] * alpha_p[None, :])
    dx_1d = qinv @ np.diag(1j * alpha_p) @ qmat

    r = np.arange(nz + 1, dtype=int)
    zeta = np.cos(np.pi * r / nz)
    c = np.ones(nz + 1, dtype=float)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** r
    differences = zeta[:, None] - zeta[None, :]
    d_cheb = (c[:, None] / c[None, :]) / (differences + np.eye(nz + 1))
    d_cheb -= np.diag(np.sum(d_cheb, axis=1))
    dz_1d = d_cheb / float(h)

    identity_x = np.eye(nx, dtype=complex)
    identity_z = np.eye(nz + 1, dtype=complex)
    dx = np.kron(dx_1d, identity_z)
    dz = np.kron(identity_x, dz_1d)

    cosine_moments = np.cos(
        np.pi * np.arange(nz + 1)[:, None] * np.arange(nz + 1)[None, :] / nz
    )
    moments = np.zeros(nz + 1, dtype=float)
    for order in range(0, nz + 1, 2):
        moments[order] = 2.0 * float(h) / (1.0 - order * order)
    weights_z = np.linalg.solve(cosine_moments, moments)
    normalized_weights = np.tile(weights_z / (2.0 * float(h) * nx), nx)
    weights = np.diag(normalized_weights.astype(complex))
    return np.stack((dx, dz, weights)).astype(np.complex128)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
nx, nz, d, h, alpha = 6, 5, 1.2, 0.8, 0.37
""",
            "call": "spectral_operators_and_weights(nx, nz, d, h, alpha)",
            "gold_call": "_oracle_spectral_operators_and_weights(nx, nz, d, h, alpha)",
            "tol": 2e-11,
        },
        {
            "setup": """import numpy as np
nx, nz, d, h, alpha = 4, 2, 0.9, 0.55, -0.2
""",
            "call": "spectral_operators_and_weights(nx, nz, d, h, alpha)",
            "gold_call": "_oracle_spectral_operators_and_weights(nx, nz, d, h, alpha)",
            "tol": 2e-11,
        },
        {
            "setup": """import numpy as np
nx, nz, d, h, alpha = 8, 4, 1.0, 1.1, 0.0
""",
            "call": "spectral_operators_and_weights(nx, nz, d, h, alpha)",
            "gold_call": "_oracle_spectral_operators_and_weights(nx, nz, d, h, alpha)",
            "tol": 2e-11,
        },
        {
            "setup": """def catch_value_error(fn):
    try:
        fn(5, 4, 1.0, 0.8, 0.1)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(spectral_operators_and_weights)",
            "gold_call": "catch_value_error(_oracle_spectral_operators_and_weights)",
        },
    ]
