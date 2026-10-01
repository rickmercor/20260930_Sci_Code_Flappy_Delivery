"""
Group index c*d(Re beta)/d(omega) of the plane-wave pulsed surface mode.

The group index is n_SPP = c*d(Re beta)/d(omega) at the frequency asked for,
with beta the root of step 01. It is NOT the phase index Re(beta)*c/w; on this
stack the two differ by about eight percent and the task's answer turns on the
difference. Differentiate the root numerically. The accuracy matters more here
than anywhere else, because the final answer divides by n_SPP - 1. Converge
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
float: group index, dimensionless.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spp_group_index(nu_thz: float, d_m_nm: float) -> float:
    """Group index c*d(Re beta)/d(omega) of the plane-wave pulsed surface mode.

    Args:
        nu_thz: ordinary frequency in THz.
        d_m_nm: metal-film thickness in nm.

    Returns:
        float: group index, dimensionless.

    Raises:
        ValueError: if nu_thz is not finite and positive; or if d_m_nm is not finite and positive.
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

def _d_re_beta(nu_thz, d_m_nm):
    """d(Re beta)/d(omega): halve the step until two estimates agree to 1e-12 relative."""
    f = lambda x: _mode_beta(x, d_m_nm).real
    prev, est = None, None
    h = nu_thz * 1.0e-2
    for _ in range(12):
        num = f(nu_thz - 2 * h) - 8 * f(nu_thz - h) + 8 * f(nu_thz + h) - f(nu_thz + 2 * h)
        est = num / (12.0 * (2.0 * np.pi * h * 1.0e12))
        if prev is not None and abs(est - prev) <= 1.0e-12 * abs(est):
            break
        prev = est
        h *= 0.5
    return est


def _oracle_spp_group_index(nu_thz: float, d_m_nm: float) -> float:
    """Reference implementation of spp_group_index."""
    if not np.isfinite(nu_thz) or nu_thz <= 0.0:
        raise ValueError("nu_thz must be a finite positive frequency in THz")
    if not np.isfinite(d_m_nm) or d_m_nm <= 0.0:
        raise ValueError("d_m_nm must be a finite positive thickness in nm")
    return _C * _d_re_beta(nu_thz, d_m_nm)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: the nominal film at the band centre
            "setup": 'import numpy as np',
            "call": 'spp_group_index(385.0, 55.00)',
            "gold_call": '_oracle_spp_group_index(385.0, 55.00)',
        },
        {
            # normal: the nominal film at the upper band edge
            "setup": 'import numpy as np',
            "call": 'spp_group_index(409.0, 55.00)',
            "gold_call": '_oracle_spp_group_index(409.0, 55.00)',
        },
        {
            # boundary: a thick film
            "setup": 'import numpy as np',
            "call": 'spp_group_index(385.0, 300.00)',
            "gold_call": '_oracle_spp_group_index(385.0, 300.00)',
        },
    ]
