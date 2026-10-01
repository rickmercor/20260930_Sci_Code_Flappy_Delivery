"""
Axial group index of the ST-SPP at the band-centre frequency.

The axial group velocity is v = d(w)/d(k_z) at the band centre taken ALONG the
spectral curve, so k_x varies with w. Differentiating k_z(w) = sqrt(k_SPP(w)^2
- k_x(w)^2) at w_o gives exactly n_ST = (ko*n_spp - kx/tan(alpha)) / sqrt(ko^2
- kx^2), with ko = Re(k_SPP) at w_o and kx = k_x(w_o) from the V law of step
04 - NOT half the angular bandwidth over c*tan(alpha), which holds only for a
band symmetric in frequency. The two k_x terms do different jobs. The kx^2
under the root is the AXIAL PROJECTION: on the light-cone, k_z falls short of
k_SPP by exactly that. The kx/tan(alpha) in the numerator is the TRANSVERSE
DERIVATIVE, k_x dk_x/dw along the curve, absent when k_x is held fixed.
Dropping just one does NOT return n_spp: at the luminal angle, dropping the
projection gives 0.996846 and dropping the derivative 1.112484, against n_spp
= 1.108976; only removing both recovers it. The form is exact, not a
truncation. The root is real only while |kx| < ko, a minimum opening angle
|tan(alpha)| > |w_o - w_ref|/(c*ko); at or below it raise ValueError rather
than return NaN. Both upstream values are passed in, so the mode solve is not
repeated. Band in vacuum WAVELENGTH: 780.0 nm centre, 96.0 nm wide
(732.0-828.0 nm) - symmetric in wavelength, so NOT in frequency. c =
2.99792458e8 m/s, w = 2*pi*nu.

Returns
-------
float: axial group index c/v, dimensionless.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def st_group_index(alpha_deg: float, k_spp_o_rad_um: float, n_spp: float) -> float:
    """Axial group index of the ST-SPP at the band-centre frequency.

    Args:
        alpha_deg: opening angle in degrees; negative subluminal, positive superluminal.
        k_spp_o_rad_um: Re(k_SPP) at the band centre in rad/um, from step 01.
        n_spp: plane-wave group index at the band centre, from step 03.

    Returns:
        float: axial group index c/v, dimensionless.

    Raises:
        ValueError: if alpha_deg is not finite, or |alpha_deg| is not strictly between 0 and 90; or if k_spp_o_rad_um is not finite and positive; or if n_spp is not finite and positive; or if |alpha_deg| is at or below the minimum opening angle, where |k_x(omega_o)| >= k_spp_o_rad_um and the square root is not real.
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


def _oracle_st_group_index(alpha_deg: float, k_spp_o_rad_um: float, n_spp: float) -> float:
    """Reference implementation of st_group_index."""
    if not np.isfinite(alpha_deg) or not 0.0 < abs(alpha_deg) < 90.0:
        raise ValueError("alpha_deg must be finite with 0 < |alpha| < 90 degrees")
    if not np.isfinite(k_spp_o_rad_um) or k_spp_o_rad_um <= 0.0:
        raise ValueError("k_spp_o_rad_um must be finite and positive")
    if not np.isfinite(n_spp) or n_spp <= 0.0:
        raise ValueError("n_spp must be finite and positive")
    ta = np.tan(np.deg2rad(alpha_deg))
    ko = k_spp_o_rad_um * 1.0e6
    kx = _k_x(_NU_O, alpha_deg)
    if abs(kx) >= ko:
        raise ValueError("|alpha_deg| is at or below the minimum opening angle")
    return (ko * n_spp - kx / ta) / np.sqrt(ko * ko - kx * kx)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: superluminal branch near the luminal angle
            "setup": 'import numpy as np',
            "call": 'st_group_index(36.0, 8.334000, 1.110769)',
            "gold_call": '_oracle_st_group_index(36.0, 8.334000, 1.110769)',
        },
        {
            # normal: subluminal branch, same magnitude of alpha
            "setup": 'import numpy as np',
            "call": 'st_group_index(-36.0, 8.334000, 1.110769)',
            "gold_call": '_oracle_st_group_index(-36.0, 8.334000, 1.110769)',
        },
        {
            # boundary: a steep V, close to the plane-wave mode
            "setup": 'import numpy as np',
            "call": 'st_group_index(80.0, 8.334000, 1.110769)',
            "gold_call": '_oracle_st_group_index(80.0, 8.334000, 1.110769)',
        },
        {
            # edge: 3.15 deg, below the 3.207-deg minimum here; must raise ValueError
            "setup": 'import numpy as np\ndef _raises_value_error(f, *args):\n    try:\n        f(*args)\n    except ValueError:\n        return 1.0\n    return 0.0',
            "call": '_raises_value_error(st_group_index, 3.15, 8.334000, 1.110769)',
            "gold_call": '_raises_value_error(_oracle_st_group_index, 3.15, 8.334000, 1.110769)',
        },
    ]
