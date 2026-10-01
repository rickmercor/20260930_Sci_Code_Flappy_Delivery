"""
Axial wave number k_z of the ST-SPP spectral curve, in rad/um.

The spectral support lies on the surface-mode light-cone k_x^2 + k_z^2 =
Re(k_SPP)^2, built from the real part of the step-01 root. The forward branch
is k_z(w) = +sqrt(Re(k_SPP)^2 - k_x(w)^2) with k_x from the V law of step 04.
Return k_z in rad/um. k_z is real only while |k_x| < Re(k_SPP) at the
frequency asked for, and |k_x| = |w - w_ref|/(c*|tan(alpha)|), so that sets a
minimum opening angle: |tan(alpha)| must exceed |w - w_ref|/(c*Re(k_SPP)). At
or below it raise ValueError rather than return a non-finite value. Converge
rather than fix scheme parameters: tolerance is 1e-9 absolute, so solve the
root to machine precision and keep derivatives stable to 1e-12 relative.
Parameters: vacuum eps_s = 1 / UNIAXIAL coating 12.00 nm, n_o = 1.520, n_e =
1.495, axis normal / metal film 55.00 nm nominal / substrate eps_sub = 3.67^2.
Metal, exp(-i w t): eps_m = eps_inf - wp^2/(w^2 + 1j*gd*w) + f1*w1^2/(w1^2 -
w^2 - 1j*g1*w), eps_inf = 3.70, wp = 1.3600e16, gd = 3.0000e13, f1 = 0.2000,
w1 = 6.0000e15, g1 = 1.0000e15 rad/s. Band 732.0-828.0 nm in vacuum
WAVELENGTH, symmetric in wavelength, NOT in frequency. c = 2.99792458e8 m/s, w
= 2*pi*nu.

Returns
-------
float: axial wave number in rad/um.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def axial_wavenumber(nu_thz: float, alpha_deg: float, d_m_nm: float) -> float:
    """Axial wave number k_z of the ST-SPP spectral curve, in rad/um.

    Args:
        nu_thz: ordinary frequency in THz.
        alpha_deg: opening angle in degrees.
        d_m_nm: metal-film thickness in nm.

    Returns:
        float: axial wave number in rad/um.

    Raises:
        ValueError: if nu_thz is not finite and positive; or if alpha_deg is not finite, or |alpha_deg| is not strictly between 0 and 90; or if d_m_nm is not finite and positive; or if |alpha_deg| is at or below the minimum opening angle at nu_thz, where |k_x(nu_thz)| >= Re(k_SPP(nu_thz)) and k_z is not real.
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

def _eps_metal(nu_thz):
    """Drude + one Lorentz, exp(-i w t)."""
    w = complex(_omega(nu_thz))
    return (_EPS_INF - _WP * _WP / (w * w + 1j * _GD * w)
            + _F1 * _W1 * _W1 / (_W1 * _W1 - w * w - 1j * _G1 * w))

def _bare_beta(nu_thz):
    """Uncoated metal-vacuum SPP wave number, rad/m; a starting point only."""
    k0 = _omega(nu_thz) / _C
    em = _eps_metal(nu_thz)
    return k0 * np.sqrt(_EPS_S * em / (_EPS_S + em))

def _kappa_bound(beta, eps, k0):
    """Bound medium: root with positive real part."""
    k = np.sqrt(complex(beta) ** 2 - eps * k0 * k0)
    return k if k.real >= 0.0 else -k


def _kappa_out(beta, eps, k0):
    """Substrate: outgoing (radiating) root, Im(kappa) <= 0."""
    k = np.sqrt(complex(beta) ** 2 - eps * k0 * k0)
    return k if k.imag <= 0.0 else -k


def _mode_residual(beta, nu_thz, d_m_nm):
    """Four-layer TM mode condition; zero on a mode. d_m_nm None = semi-infinite."""
    k0 = _omega(nu_thz) / _C
    e0, e2, e3 = _EPS_S, _eps_metal(nu_thz), _EPS_SUB
    eo, ee = _N_O * _N_O, _N_E * _N_E
    k_0 = _kappa_bound(beta, e0, k0)
    k_1 = np.sqrt((eo / ee) * complex(beta) ** 2 - eo * k0 * k0)   # uniaxial coating
    k_2 = np.sqrt(complex(beta) ** 2 - e2 * k0 * k0)
    k_3 = _kappa_out(beta, e3, k0)
    n0, n1, n2 = k_0 / e0, k_1 / eo, k_2 / e2   # ordinary eps for the coating
    Y = k_3 / e3
    t2 = 1.0 + 0j if d_m_nm is None else np.tanh(k_2 * d_m_nm * 1.0e-9)
    Y = n2 * (Y + n2 * t2) / (n2 + Y * t2)
    t1 = np.tanh(k_1 * _D_C * 1.0e-9)
    Y = n1 * (Y + n1 * t1) / (n1 + Y * t1)
    return Y + n0

def _newton(f, x0):
    """Complex Newton, central-difference derivative, to machine precision."""
    x = complex(x0)
    for _ in range(200):
        h = x * 1.0e-7
        d = (f(x + h) - f(x - h)) / (2.0 * h)
        if d == 0:
            break
        step = f(x) / d
        x = x - step
        if abs(step) <= 1.0e-15 * abs(x):
            break
    return x


def _mode_beta(nu_thz, d_m_nm, n_cont=16):
    """Complex beta (rad/m), the branch connected to the uncoated semi-infinite SPP."""
    b = _newton(lambda x: _mode_residual(x, nu_thz, None), _bare_beta(nu_thz))
    if d_m_nm is None:
        return b
    for i in range(1, n_cont + 1):
        dm = 5000.0 + (d_m_nm - 5000.0) * i / n_cont
        b = _newton(lambda x, dm=dm: _mode_residual(x, nu_thz, dm), b)
    return b

def _k_x(nu_thz, alpha_deg):
    """Transverse wave number of the V-shaped spectrum, rad/m."""
    ta = np.tan(np.deg2rad(alpha_deg))
    nu_ref = _NU_U if ta < 0.0 else _NU_L
    return _omega(nu_thz - nu_ref) / (_C * ta)


def _oracle_axial_wavenumber(nu_thz: float, alpha_deg: float, d_m_nm: float) -> float:
    """Reference implementation of axial_wavenumber."""
    if not np.isfinite(nu_thz) or nu_thz <= 0.0:
        raise ValueError("nu_thz must be a finite positive frequency in THz")
    if not np.isfinite(alpha_deg) or not 0.0 < abs(alpha_deg) < 90.0:
        raise ValueError("alpha_deg must be finite with 0 < |alpha| < 90 degrees")
    if not np.isfinite(d_m_nm) or d_m_nm <= 0.0:
        raise ValueError("d_m_nm must be a finite positive thickness in nm")
    b = _mode_beta(nu_thz, d_m_nm).real
    kx = _k_x(nu_thz, alpha_deg)
    if abs(kx) >= b:
        raise ValueError("|alpha_deg| is at or below the minimum opening angle")
    return np.sqrt(b * b - kx * kx) * 1.0e-6

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: superluminal branch at the band centre, nominal film
            "setup": 'import numpy as np',
            "call": 'axial_wavenumber(385.0, 36.0, 55.00)',
            "gold_call": '_oracle_axial_wavenumber(385.0, 36.0, 55.00)',
        },
        {
            # boundary: 0.93 THz above the superluminal apex, where k_z approaches the full mode wave number
            "setup": 'import numpy as np',
            "call": 'axial_wavenumber(363.0, 36.0, 55.00)',
            "gold_call": '_oracle_axial_wavenumber(363.0, 36.0, 55.00)',
        },
        {
            # normal: the subluminal branch above the centre
            "setup": 'import numpy as np',
            "call": 'axial_wavenumber(400.0, -36.0, 55.00)',
            "gold_call": '_oracle_axial_wavenumber(400.0, -36.0, 55.00)',
        },
        {
            # edge: 3.25 deg, below the 3.302-deg minimum at 385.0 THz; must raise ValueError
            "setup": 'import numpy as np\ndef _raises_value_error(f, *args):\n    try:\n        f(*args)\n    except ValueError:\n        return 1.0\n    return 0.0',
            "call": '_raises_value_error(axial_wavenumber, 385.0, 3.25, 55.00)',
            "gold_call": '_raises_value_error(_oracle_axial_wavenumber, 385.0, 3.25, 55.00)',
        },
    ]
