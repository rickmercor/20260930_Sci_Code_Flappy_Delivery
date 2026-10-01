"""
Free-space numerical aperture the synthesis system must supply.

The coupler conserves the transverse wave number, so a spectral component at w
leaves the synthesis system at an angle phi(w) with sin(phi) = c*|k_x(w)|/w,
using the V law of step 04. The required numerical aperture is the largest
value of sin(phi) over the band, that is between its two edge frequencies.
Since |k_x| grows linearly away from the apex while the divisor grows only as
w, the maximum sits at the band edge farthest from the apex. Band in vacuum
WAVELENGTH: 780.0 nm centre, 96.0 nm wide (732.0-828.0 nm) - symmetric in
wavelength, so NOT in frequency. c = 2.99792458e8 m/s, w = 2*pi*nu.

Returns
-------
float: numerical aperture, dimensionless.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def required_numerical_aperture(alpha_deg: float) -> float:
    """Free-space numerical aperture the synthesis system must supply.

    Args:
        alpha_deg: opening angle in degrees.

    Returns:
        float: numerical aperture, dimensionless.

    Raises:
        ValueError: if alpha_deg is not finite, or |alpha_deg| is not strictly between 0 and 90.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_C = 2.99792458e8
_EPS_INF, _WP, _GD = 3.70, 1.3600e16, 3.0000e13
_F1, _W1, _G1 = 0.2000, 6.0000e15, 1.0000e15
_N_O, _N_E = 1.520, 1.495
_D_C, _EPS_S, _EPS_SUB = 12.00, 1.0, 3.67 ** 2
_LAM_O, _DLAM, _D_M = 780.0, 96.0, 55.00   # nm
_NU_O = _C / (_LAM_O * 1.0e-9) / 1.0e12                    # THz; band NOT symmetric here
_NU_L = _C / ((_LAM_O + _DLAM / 2.0) * 1.0e-9) / 1.0e12
_NU_U = _C / ((_LAM_O - _DLAM / 2.0) * 1.0e-9) / 1.0e12


def _omega(nu_thz):
    """rad/s from THz."""
    return 2.0 * np.pi * nu_thz * 1.0e12

def _k_x(nu_thz, alpha_deg):
    """Transverse wave number of the V-shaped spectrum, rad/m."""
    ta = np.tan(np.deg2rad(alpha_deg))
    nu_ref = _NU_U if ta < 0.0 else _NU_L
    return _omega(nu_thz - nu_ref) / (_C * ta)


def _oracle_required_numerical_aperture(alpha_deg: float) -> float:
    """Reference implementation of required_numerical_aperture."""
    if not np.isfinite(alpha_deg) or not 0.0 < abs(alpha_deg) < 90.0:
        raise ValueError("alpha_deg must be finite with 0 < |alpha| < 90 degrees")
    return max(abs(_k_x(nu, alpha_deg)) * _C / _omega(nu)
               for nu in (_NU_L, _NU_U))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: superluminal branch near the luminal angle
            "setup": 'import numpy as np',
            "call": 'required_numerical_aperture(36.0)',
            "gold_call": '_oracle_required_numerical_aperture(36.0)',
        },
        {
            # normal: subluminal branch, same |alpha|
            "setup": 'import numpy as np',
            "call": 'required_numerical_aperture(-36.0)',
            "gold_call": '_oracle_required_numerical_aperture(-36.0)',
        },
        {
            # boundary: a steep V, needing little aperture
            "setup": 'import numpy as np',
            "call": 'required_numerical_aperture(80.0)',
            "gold_call": '_oracle_required_numerical_aperture(80.0)',
        },
    ]
