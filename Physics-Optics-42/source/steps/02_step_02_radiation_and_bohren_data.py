"""
Construct radiation branches and circular-channel incident boundary data.

Outgoing diffraction orders require a fixed square-root branch.  A unit transverse-electric incident wave is also decomposed into its two circular Beltrami channels, producing different complex top-boundary forcing even though both channels share the same exterior vacuum.

Returns
-------
complex128 ndarray of shape (nx, 5) containing alpha_p, gamma_p, the DtN multiplier, and the two channel forcings
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radiation_and_bohren_data(nx: int, d: float, h: float, wavelength: float, theta: float) -> "np.ndarray":
    '''Return ordered diffraction data and unit-TE circular-channel forcing.

    Parameters
    ----------
    nx : int
        Even Fourier-node count, at least 4.
    d : float
        Positive lateral period.
    h : float
        Positive slab half-height.
    wavelength : float
        Positive vacuum wavelength.
    theta : float
        Finite incidence angle in radians, with abs(theta) < pi/2.

    Returns
    -------
    data : np.ndarray
        Complex128 array of shape (nx, 5), with rows ordered by
        p = -nx/2, ..., nx/2-1.  The columns are alpha_p, gamma_p,
        -i*gamma_p, the LCP top-forcing coefficient, and the RCP
        top-forcing coefficient.  Propagating gamma_p uses the nonnegative
        real root and evanescent gamma_p uses +i times the positive root.
        Only p = 0 has nonzero forcing for the incident field normalized by
        E_y = 1 and H_y = 0.

    Raises
    ------
    ValueError
        If dimensions or positive parameters are invalid, theta is outside
        the stated range, or any retained order is at a Wood anomaly, defined
        by abs(k0^2 - alpha_p^2) <= 64*eps*k0^2.
    '''
    return data

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_radiation_and_bohren_data(nx: int, d: float, h: float, wavelength: float, theta: float) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not np.isfinite(d) or d <= 0.0 or not np.isfinite(h) or h <= 0.0:
        raise ValueError("d and h must be positive and finite")
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength must be positive and finite")
    if not np.isfinite(theta) or abs(float(theta)) >= 0.5 * np.pi:
        raise ValueError("theta must be finite with abs(theta) < pi/2")

    k0 = 2.0 * np.pi / float(wavelength)
    alpha = k0 * np.sin(float(theta))
    p = np.arange(-nx // 2, nx // 2, dtype=int)
    alpha_p = alpha + 2.0 * np.pi * p / float(d)
    radicand = k0 * k0 - alpha_p * alpha_p
    wood_tol = 64.0 * np.finfo(float).eps * k0 * k0
    if np.any(np.abs(radicand) <= wood_tol):
        raise ValueError("a retained Fourier order lies at a Wood anomaly")
    gamma_p = np.where(
        radicand > 0.0,
        np.sqrt(np.maximum(radicand, 0.0)).astype(complex),
        1j * np.sqrt(np.maximum(-radicand, 0.0)),
    )
    data = np.zeros((nx, 5), dtype=np.complex128)
    data[:, 0] = alpha_p
    data[:, 1] = gamma_p
    data[:, 2] = -1j * gamma_p
    zero_slot = nx // 2
    phase = np.exp(-1j * gamma_p[zero_slot] * float(h))
    data[zero_slot, 3] = 2j * gamma_p[zero_slot] * phase * 0.5
    data[zero_slot, 4] = 2j * gamma_p[zero_slot] * phase * (0.5j)
    return data

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
nx, d, h, wavelength, theta = 8, 0.8, 0.95, 0.7, np.pi/6
""",
            "call": "radiation_and_bohren_data(nx, d, h, wavelength, theta)",
            "gold_call": "_oracle_radiation_and_bohren_data(nx, d, h, wavelength, theta)",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
nx, d, h, wavelength, theta = 4, 0.73, 0.4, 0.91, 0.0
""",
            "call": "radiation_and_bohren_data(nx, d, h, wavelength, theta)",
            "gold_call": "_oracle_radiation_and_bohren_data(nx, d, h, wavelength, theta)",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
nx, d, h, wavelength, theta = 10, 0.41, 1.2, 0.83, -0.31
""",
            "call": "radiation_and_bohren_data(nx, d, h, wavelength, theta)",
            "gold_call": "_oracle_radiation_and_bohren_data(nx, d, h, wavelength, theta)",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
def catch_value_error(fn):
    try:
        fn(4, 1.0, 0.8, 1.0, 0.0)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(radiation_and_bohren_data)",
            "gold_call": "catch_value_error(_oracle_radiation_and_bohren_data)",
        },
    ]
