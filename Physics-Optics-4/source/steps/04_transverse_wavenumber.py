"""
Transverse wave number k_x of the V-shaped spatiotemporal spectrum, in rad/um.

The spectral support projects onto the (k_x, w/c) plane as a V whose two
straight sides make an angle alpha with the k_x axis, so w - w_ref =
c*|k_x|*tan(alpha) for a reference frequency fixed by the branch. Diffraction-
free propagation requires |k_x| to be in one-to-one correspondence with w
across the band, which puts the apex of the V at an EDGE of the band and not
at its centre: for the subluminal branch tan(alpha) < 0 and w_ref is the UPPER
edge frequency; for the superluminal branch tan(alpha) > 0 and w_ref is the
LOWER edge frequency. Both edges come from the wavelength span, and neither is
w_o plus or minus half a bandwidth. Hence k_x(w) = (w - w_ref)/(c*tan(alpha)),
non-negative across the band and vanishing only at the apex. Note that k_x
does NOT vanish at the band centre. Return k_x in rad/um. Band in vacuum
WAVELENGTH: 780.0 nm centre, 96.0 nm wide (732.0-828.0 nm) - symmetric in
wavelength, so NOT in frequency. c = 2.99792458e8 m/s, w = 2*pi*nu.

Returns
-------
float: transverse wave number in rad/um.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transverse_wavenumber(nu_thz: float, alpha_deg: float) -> float:
    """Transverse wave number k_x of the V-shaped spatiotemporal spectrum, in
    rad/um.

    Args:
        nu_thz: ordinary frequency in THz.
        alpha_deg: opening angle in degrees; negative is the subluminal branch, positive the superluminal one.

    Returns:
        float: transverse wave number in rad/um.

    Raises:
        ValueError: if nu_thz is not finite and positive; or if alpha_deg is not finite, or |alpha_deg| is not strictly between 0 and 90.
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


def _oracle_transverse_wavenumber(nu_thz: float, alpha_deg: float) -> float:
    """Reference implementation of transverse_wavenumber."""
    if not np.isfinite(nu_thz) or nu_thz <= 0.0:
        raise ValueError("nu_thz must be a finite positive frequency in THz")
    if not np.isfinite(alpha_deg) or not 0.0 < abs(alpha_deg) < 90.0:
        raise ValueError("alpha_deg must be finite with 0 < |alpha| < 90 degrees")
    return _k_x(nu_thz, alpha_deg) * 1.0e-6

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: superluminal branch at the band centre
            "setup": 'import numpy as np',
            "call": 'transverse_wavenumber(385.0, 36.0)',
            "gold_call": '_oracle_transverse_wavenumber(385.0, 36.0)',
        },
        {
            # normal: superluminal branch above the centre
            "setup": 'import numpy as np',
            "call": 'transverse_wavenumber(400.0, 36.0)',
            "gold_call": '_oracle_transverse_wavenumber(400.0, 36.0)',
        },
        {
            # boundary: the subluminal branch at the same frequency and |alpha|, which only the referencing separates
            "setup": 'import numpy as np',
            "call": 'transverse_wavenumber(400.0, -36.0)',
            "gold_call": '_oracle_transverse_wavenumber(400.0, -36.0)',
        },
    ]
