"""
Build the envelope and arbitrary-center circular-channel coefficients.

At a nonzero deformation center the reference material factor is spatially varying.  The two circular channels have opposite envelope signs and different background factors, which prevents reducing one channel to a reused copy of the other.

Returns
-------
float64 ndarray of shape (3, nx*(nz+1)) containing X, rho_center, and rho_1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def centered_channel_profiles(sign: int, nx: int, nz: int, d: float, h: float, t: float, sharpness: float, wavelength: float, chi_bar: float, chi_amplitude: float, lateral_scale: float, delta_center: float) -> "np.ndarray":
    '''Construct the smooth envelope and one circular channel's coefficients.

    Parameters
    ----------
    sign : int
        +1 for the left-circular channel or -1 for the right-circular channel.
    nx, nz : int
        Even lateral node count and Chebyshev degree, using the same nodes as
        spectral_operators_and_weights.
    d, h : float
        Positive period and half-height.
    t : float
        Positive bump half-thickness with t < h.
    sharpness : float
        Positive inverse-length tanh sharpness.
    wavelength : float
        Positive vacuum wavelength.
    chi_bar : float
        Finite background chirality in length units.
    chi_amplitude : float
        Finite envelope amplitude in length units.
    lateral_scale : float
        Finite multiplier of the four fixed lateral harmonics.
    delta_center : float
        Finite global deformation value about which the local series is formed.

    Returns
    -------
    profiles : np.ndarray
        Float64 array of shape (3, N), flattened in C order.  Row zero is the
        envelope X, row one the centered material factor rho_c, and row two
        the local linear coefficient rho_1.  The envelope uses the exact four
        harmonics and vertical asymmetry stated in the problem.

    Raises
    ------
    ValueError
        If sign is not +1 or -1, grid or positive geometric inputs are
        invalid, t is not below h, a finite scalar is required but absent, or
        the centered profile violates max(abs(k0*(chi_bar-delta_center*X))) < 1.
    '''
    return profiles

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_centered_channel_profiles(sign: int, nx: int, nz: int, d: float, h: float, t: float, sharpness: float, wavelength: float, chi_bar: float, chi_amplitude: float, lateral_scale: float, delta_center: float) -> "np.ndarray":
    """Reference implementation."""
    if sign not in (-1, 1) or isinstance(sign, (bool, np.bool_)):
        raise ValueError("sign must equal +1 or -1")
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not isinstance(nz, (int, np.integer)) or isinstance(nz, (bool, np.bool_)) or nz < 2:
        raise ValueError("nz must be an integer at least 2")
    if not np.isfinite(d) or d <= 0.0 or not np.isfinite(h) or h <= 0.0:
        raise ValueError("d and h must be positive and finite")
    if not np.isfinite(t) or t <= 0.0 or t >= h:
        raise ValueError("t must satisfy 0 < t < h")
    if not np.isfinite(sharpness) or sharpness <= 0.0 or not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("sharpness and wavelength must be positive and finite")
    scalars = (chi_bar, chi_amplitude, lateral_scale, delta_center)
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("chirality and deformation scalars must be finite")

    x = np.arange(nx, dtype=float)[:, None] * (float(d) / nx)
    z = (float(h) * np.cos(np.pi * np.arange(nz + 1) / nz))[None, :]
    phi = 0.5 * (
        np.tanh(float(sharpness) * (z + float(t)))
        - np.tanh(float(sharpness) * (z - float(t)))
    )
    harmonics = (
        0.23 * np.cos(2.0 * np.pi * x / float(d))
        - 0.19 * np.sin(4.0 * np.pi * x / float(d))
        + 0.13 * np.cos(6.0 * np.pi * x / float(d))
        + 0.07 * np.sin(8.0 * np.pi * x / float(d))
    )
    lateral = 1.0 + float(lateral_scale) * harmonics
    envelope = float(chi_amplitude) * phi * (1.0 + 0.14 * z / float(h)) * lateral
    envelope = envelope.reshape(-1).astype(float)
    k0 = 2.0 * np.pi / float(wavelength)
    centered_chi = float(chi_bar) - float(delta_center) * envelope
    if np.max(np.abs(k0 * centered_chi)) >= 1.0:
        raise ValueError("the centered chirality violates abs(k0*chi) < 1")
    rho_center = 1.0 - sign * k0 * float(chi_bar) + sign * k0 * float(delta_center) * envelope
    rho_one = sign * k0 * envelope
    return np.stack((envelope, rho_center, rho_one)).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    common = """import numpy as np
nx, nz, d, h, t = 8, 6, 0.8, 0.95, 0.5
sharpness, wavelength = 8.0, 0.7
chi_bar, chi_amplitude, delta_center = 0.006, 0.04, 0.25
"""
    return [
        {
            "setup": common + """sign, lateral_scale = 1, 1.0
""",
            "call": "centered_channel_profiles(sign, nx, nz, d, h, t, sharpness, wavelength, chi_bar, chi_amplitude, lateral_scale, delta_center)",
            "gold_call": "_oracle_centered_channel_profiles(sign, nx, nz, d, h, t, sharpness, wavelength, chi_bar, chi_amplitude, lateral_scale, delta_center)",
            "tol": 2e-13,
        },
        {
            "setup": common + """sign, lateral_scale = -1, 1.0
""",
            "call": "centered_channel_profiles(sign, nx, nz, d, h, t, sharpness, wavelength, chi_bar, chi_amplitude, lateral_scale, delta_center)",
            "gold_call": "_oracle_centered_channel_profiles(sign, nx, nz, d, h, t, sharpness, wavelength, chi_bar, chi_amplitude, lateral_scale, delta_center)",
            "tol": 2e-13,
        },
        {
            "setup": common + """sign, lateral_scale = 1, 0.0
""",
            "call": "centered_channel_profiles(sign, nx, nz, d, h, t, sharpness, wavelength, chi_bar, chi_amplitude, lateral_scale, delta_center)",
            "gold_call": "_oracle_centered_channel_profiles(sign, nx, nz, d, h, t, sharpness, wavelength, chi_bar, chi_amplitude, lateral_scale, delta_center)",
            "tol": 2e-13,
        },
        {
            "setup": common + """sign, lateral_scale = 0, 1.0
def catch_value_error(fn):
    try:
        fn(sign, nx, nz, d, h, t, sharpness, wavelength, chi_bar, chi_amplitude, lateral_scale, delta_center)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(centered_channel_profiles)",
            "gold_call": "catch_value_error(_oracle_centered_channel_profiles)",
        },
    ]
