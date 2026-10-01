#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

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


def film_mode_index(nu_thz: float, d_m_nm: float) -> float:
    """Reference implementation of film_mode_index."""
    if not np.isfinite(nu_thz) or nu_thz <= 0.0:
        raise ValueError("nu_thz must be a finite positive frequency in THz")
    if not np.isfinite(d_m_nm) or d_m_nm <= 0.0:
        raise ValueError("d_m_nm must be a finite positive thickness in nm")
    return _mode_beta(nu_thz, d_m_nm).real * _C / _omega(nu_thz)

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


def propagation_length(nu_thz: float, d_m_nm: float) -> float:
    """Reference implementation of propagation_length."""
    if not np.isfinite(nu_thz) or nu_thz <= 0.0:
        raise ValueError("nu_thz must be a finite positive frequency in THz")
    if not np.isfinite(d_m_nm) or d_m_nm <= 0.0:
        raise ValueError("d_m_nm must be a finite positive thickness in nm")
    return 1.0e6 / (2.0 * _mode_beta(nu_thz, d_m_nm).imag)

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


def spp_group_index(nu_thz: float, d_m_nm: float) -> float:
    """Reference implementation of spp_group_index."""
    if not np.isfinite(nu_thz) or nu_thz <= 0.0:
        raise ValueError("nu_thz must be a finite positive frequency in THz")
    if not np.isfinite(d_m_nm) or d_m_nm <= 0.0:
        raise ValueError("d_m_nm must be a finite positive thickness in nm")
    return _C * _d_re_beta(nu_thz, d_m_nm)

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


def transverse_wavenumber(nu_thz: float, alpha_deg: float) -> float:
    """Reference implementation of transverse_wavenumber."""
    if not np.isfinite(nu_thz) or nu_thz <= 0.0:
        raise ValueError("nu_thz must be a finite positive frequency in THz")
    if not np.isfinite(alpha_deg) or not 0.0 < abs(alpha_deg) < 90.0:
        raise ValueError("alpha_deg must be finite with 0 < |alpha| < 90 degrees")
    return _k_x(nu_thz, alpha_deg) * 1.0e-6

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


def axial_wavenumber(nu_thz: float, alpha_deg: float, d_m_nm: float) -> float:
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


def st_group_index(alpha_deg: float, k_spp_o_rad_um: float, n_spp: float) -> float:
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


def required_numerical_aperture(alpha_deg: float) -> float:
    """Reference implementation of required_numerical_aperture."""
    if not np.isfinite(alpha_deg) or not 0.0 < abs(alpha_deg) < 90.0:
        raise ValueError("alpha_deg must be finite with 0 < |alpha| < 90 degrees")
    return max(abs(_k_x(nu, alpha_deg)) * _C / _omega(nu)
               for nu in (_NU_L, _NU_U))

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

def _k_x(nu_thz, alpha_deg):
    """Transverse wave number of the V-shaped spectrum, rad/m."""
    ta = np.tan(np.deg2rad(alpha_deg))
    nu_ref = _NU_U if ta < 0.0 else _NU_L
    return _omega(nu_thz - nu_ref) / (_C * ta)

def _step_film_mode_index(nu_thz, d_m_nm):
    return _mode_beta(nu_thz, d_m_nm).real * _C / _omega(nu_thz)


def _step_propagation_length(nu_thz, d_m_nm):
    return 1.0e6 / (2.0 * _mode_beta(nu_thz, d_m_nm).imag)


def _step_spp_group_index(nu_thz, d_m_nm):
    return _C * _d_re_beta(nu_thz, d_m_nm)


def _step_transverse_wavenumber(nu_thz, alpha_deg):
    return _k_x(nu_thz, alpha_deg) * 1.0e-6


def _step_axial_wavenumber(nu_thz, alpha_deg, d_m_nm):
    b = _mode_beta(nu_thz, d_m_nm).real
    kx = _k_x(nu_thz, alpha_deg)
    return np.sqrt(b * b - kx * kx) * 1.0e-6


def _step_st_group_index(alpha_deg, k_spp_o_rad_um, n_spp):
    ta = np.tan(np.deg2rad(alpha_deg))
    ko = k_spp_o_rad_um * 1.0e6
    kx = _k_x(_NU_O, alpha_deg)
    A = ko * ko - kx * kx
    return (ko * n_spp - kx / ta) / np.sqrt(A)


def _step_required_numerical_aperture(alpha_deg):
    return max(abs(_k_x(nu, alpha_deg)) * _C / _omega(nu) for nu in (_NU_L, _NU_U))

try:
    st_group_index
except NameError:                      # only when this file is run standalone
    film_mode_index = _step_film_mode_index
    propagation_length = _step_propagation_length
    spp_group_index = _step_spp_group_index
    transverse_wavenumber = _step_transverse_wavenumber
    axial_wavenumber = _step_axial_wavenumber
    st_group_index = _step_st_group_index
    required_numerical_aperture = _step_required_numerical_aperture


def luminal_opening_angle(d_m_nm: float) -> float:
    """Reference implementation of luminal_opening_angle."""
    if not np.isfinite(d_m_nm) or d_m_nm <= 0.0:
        raise ValueError("d_m_nm must be a finite positive thickness in nm")
    nu = _NU_O
    k_spp_o = film_mode_index(nu, d_m_nm) * _omega(nu) / _C * 1.0e-6   # rad/um
    if propagation_length(nu, d_m_nm) <= 0.0:
        raise ValueError("negative propagation length: the incoming substrate branch")
    n_spp = spp_group_index(nu, d_m_nm)
    dw = _omega(_NU_O - _NU_L)
    a_min = np.rad2deg(np.arctan(dw / (_C * k_spp_o * 1.0e6))) + 1.0e-9

    def resid(a):
        return st_group_index(a, k_spp_o, n_spp) - 1.0

    lo = hi = f_lo = None
    prev_a = prev_f = None
    for a in np.linspace(max(6.0, a_min), 89.0, 600):
        if a <= a_min:
            continue
        f = resid(a)
        if prev_f is not None and prev_f * f < 0.0:
            lo, hi, f_lo = prev_a, a, prev_f
            break
        prev_a, prev_f = a, f
    if lo is None:
        raise ValueError("no luminal angle on the superluminal branch")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        f_mid = resid(mid)
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < 1.0e-12:
            break
    alpha = 0.5 * (lo + hi)

    kx = transverse_wavenumber(nu, alpha)
    kz = axial_wavenumber(nu, alpha, d_m_nm)
    if abs(kx * kx + kz * kz - k_spp_o * k_spp_o) > 1.0e-9 * k_spp_o * k_spp_o:
        raise ValueError("the returned angle is off the surface-mode light-cone")
    if not required_numerical_aperture(alpha) > 0.0:
        raise ValueError("non-physical numerical aperture")
    return alpha
SCICODE_GOLD_EOF
