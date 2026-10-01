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
from scipy import special as sp
from scipy import integrate, optimize


def _exciton_units(mu):
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a positive finite reduced mass")
    R_exc = 13.605693 * mu
    a_exc = 0.529177 / mu
    return np.array([R_exc, a_exc, 1.5 * a_exc])


def exciton_units(mu: float) -> np.ndarray:
    return _exciton_units(mu)

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def _laplace_lorentz(x):
    """F(x) = int_0^inf y exp(-x y) / (1 + y^2) dy = -[cos(x) Ci(x) + sin(x) si(x)], si = Si - pi/2."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 40.0
    si_, ci_ = sp.sici(x[small])
    out[small] = -(np.cos(x[small]) * ci_ + np.sin(x[small]) * (si_ - 0.5 * np.pi))
    xl = x[~small]
    out[~small] = 1.0 / xl**2 - 6.0 / xl**4 + 120.0 / xl**6 - 5040.0 / xl**8 + 362880.0 / xl**10
    return out


def _z0_default(R, z0):
    return 2.0 * R / np.sqrt(5.0) if z0 is None else float(z0)


def _check_common(R, distribution, z0=None):
    allowed = ("ribbon", "surface", "homogeneous", "centre")
    if distribution not in allowed:
        raise ValueError("distribution must be one of " + ", ".join(allowed))
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be a positive finite length")
    if z0 is not None and (not np.isfinite(z0) or z0 <= 0.0):
        raise ValueError("z0 must be a positive finite length")



def _bare_potential_fourier(q, R, distribution, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    with np.errstate(divide="ignore", invalid="ignore"):
        if distribution in ("ribbon", "surface"):
            out = 2.0 * sp.k0(q * R)
        elif distribution == "homogeneous":
            x = q * R
            out = np.empty_like(x)
            s = (x < 3e-3)
            out[~s] = 4.0 / x[~s]**2 * (1.0 - x[~s] * sp.k1(x[~s]))
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = -2.0 * (L - 0.5) - 0.25 * x[s]**2 * (L - 1.25)
        else:
            out = 2.0 * _laplace_lorentz(q * _z0_default(R, z0))
    out = np.where(q == 0.0, np.inf, out)
    return out


def bare_potential_fourier(q: "float | np.ndarray", R: float, distribution: str, z0: "float | None" = None) -> np.ndarray:
    return _bare_potential_fourier(q, R, distribution, z0)

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def _laplace_lorentz(x):
    """F(x) = int_0^inf y exp(-x y) / (1 + y^2) dy = -[cos(x) Ci(x) + sin(x) si(x)], si = Si - pi/2."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 40.0
    si_, ci_ = sp.sici(x[small])
    out[small] = -(np.cos(x[small]) * ci_ + np.sin(x[small]) * (si_ - 0.5 * np.pi))
    xl = x[~small]
    out[~small] = 1.0 / xl**2 - 6.0 / xl**4 + 120.0 / xl**6 - 5040.0 / xl**8 + 362880.0 / xl**10
    return out


def _z0_default(R, z0):
    return 2.0 * R / np.sqrt(5.0) if z0 is None else float(z0)


def _check_common(R, distribution, z0=None):
    allowed = ("ribbon", "surface", "homogeneous", "centre")
    if distribution not in allowed:
        raise ValueError("distribution must be one of " + ", ".join(allowed))
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be a positive finite length")
    if z0 is not None and (not np.isfinite(z0) or z0 <= 0.0):
        raise ValueError("z0 must be a positive finite length")



def _nonlocal_form_factor(q, R, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    if distribution == "ribbon":
        if use_nonlocal:
            raise ValueError("the ribbon distribution has no finite nonlocal form factor")
        return 4.0 * np.ones_like(q)
    if not use_nonlocal:
        return np.ones_like(q)
    if distribution == "surface":
        return sp.i0(q * R)
    if distribution == "homogeneous":
        x = q * R
        out = np.ones_like(x)
        s = (x < 3e-3)
        out[~s] = (1.0 - 2.0 * sp.i1(x[~s]) * sp.k1(x[~s])) / (1.0 - x[~s] * sp.k1(x[~s]))
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = ((L - 0.25) + x[s]**2 * (0.25 * L - 5.0 / 24.0)) / ((L - 0.5) + 0.125 * x[s]**2 * (L - 1.25))
        out = np.where(q == 0.0, 1.0, out)
        return out
    z = _z0_default(R, z0)
    return _laplace_lorentz(2.0 * q * z) / _laplace_lorentz(q * z)


def nonlocal_form_factor(q: "float | np.ndarray", R: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    return _nonlocal_form_factor(q, R, distribution, use_nonlocal, z0)

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def _laplace_lorentz(x):
    """F(x) = int_0^inf y exp(-x y) / (1 + y^2) dy = -[cos(x) Ci(x) + sin(x) si(x)], si = Si - pi/2."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 40.0
    si_, ci_ = sp.sici(x[small])
    out[small] = -(np.cos(x[small]) * ci_ + np.sin(x[small]) * (si_ - 0.5 * np.pi))
    xl = x[~small]
    out[~small] = 1.0 / xl**2 - 6.0 / xl**4 + 120.0 / xl**6 - 5040.0 / xl**8 + 362880.0 / xl**10
    return out


def _z0_default(R, z0):
    return 2.0 * R / np.sqrt(5.0) if z0 is None else float(z0)


def _check_common(R, distribution, z0=None):
    allowed = ("ribbon", "surface", "homogeneous", "centre")
    if distribution not in allowed:
        raise ValueError("distribution must be one of " + ", ".join(allowed))
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be a positive finite length")
    if z0 is not None and (not np.isfinite(z0) or z0 <= 0.0):
        raise ValueError("z0 must be a positive finite length")



def _bare_potential_fourier(q, R, distribution, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    with np.errstate(divide="ignore", invalid="ignore"):
        if distribution in ("ribbon", "surface"):
            out = 2.0 * sp.k0(q * R)
        elif distribution == "homogeneous":
            x = q * R
            out = np.empty_like(x)
            s = (x < 3e-3)
            out[~s] = 4.0 / x[~s]**2 * (1.0 - x[~s] * sp.k1(x[~s]))
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = -2.0 * (L - 0.5) - 0.25 * x[s]**2 * (L - 1.25)
        else:
            out = 2.0 * _laplace_lorentz(q * _z0_default(R, z0))
    out = np.where(q == 0.0, np.inf, out)
    return out



def _nonlocal_form_factor(q, R, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    if distribution == "ribbon":
        if use_nonlocal:
            raise ValueError("the ribbon distribution has no finite nonlocal form factor")
        return 4.0 * np.ones_like(q)
    if not use_nonlocal:
        return np.ones_like(q)
    if distribution == "surface":
        return sp.i0(q * R)
    if distribution == "homogeneous":
        x = q * R
        out = np.ones_like(x)
        s = (x < 3e-3)
        out[~s] = (1.0 - 2.0 * sp.i1(x[~s]) * sp.k1(x[~s])) / (1.0 - x[~s] * sp.k1(x[~s]))
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = ((L - 0.25) + x[s]**2 * (0.25 * L - 5.0 / 24.0)) / ((L - 0.5) + 0.125 * x[s]**2 * (L - 1.25))
        out = np.where(q == 0.0, 1.0, out)
        return out
    z = _z0_default(R, z0)
    return _laplace_lorentz(2.0 * q * z) / _laplace_lorentz(q * z)



def _dielectric_function(q, R, alpha, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a non-negative finite polarizability")
    q = np.abs(np.asarray(q, dtype=float))
    x = q * float(R)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if distribution == "surface" and use_nonlocal:
            prod = 2.0 * sp.i0e(x) * sp.k0e(x)          # chi * V_bare evaluated without overflow
        elif distribution == "homogeneous" and use_nonlocal:
            num = 1.0 - 2.0 * sp.i1e(x) * sp.k1e(x)
            small = _nonlocal_form_factor(q, R, distribution, True, z0) * _bare_potential_fourier(q, R, distribution, z0)
            prod = np.where(x < 3e-3, small, 4.0 * num / np.where(x == 0.0, 1.0, x)**2)
        else:
            prod = _nonlocal_form_factor(q, R, distribution, use_nonlocal, z0) * _bare_potential_fourier(q, R, distribution, z0)
        eps = 1.0 + alpha * q**2 * prod
    eps = np.where(q == 0.0, 1.0, eps)
    return eps


def dielectric_function(q: "float | np.ndarray", R: float, alpha: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    return _dielectric_function(q, R, alpha, distribution, use_nonlocal, z0)

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def _laplace_lorentz(x):
    """F(x) = int_0^inf y exp(-x y) / (1 + y^2) dy = -[cos(x) Ci(x) + sin(x) si(x)], si = Si - pi/2."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 40.0
    si_, ci_ = sp.sici(x[small])
    out[small] = -(np.cos(x[small]) * ci_ + np.sin(x[small]) * (si_ - 0.5 * np.pi))
    xl = x[~small]
    out[~small] = 1.0 / xl**2 - 6.0 / xl**4 + 120.0 / xl**6 - 5040.0 / xl**8 + 362880.0 / xl**10
    return out


def _z0_default(R, z0):
    return 2.0 * R / np.sqrt(5.0) if z0 is None else float(z0)


def _check_common(R, distribution, z0=None):
    allowed = ("ribbon", "surface", "homogeneous", "centre")
    if distribution not in allowed:
        raise ValueError("distribution must be one of " + ", ".join(allowed))
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be a positive finite length")
    if z0 is not None and (not np.isfinite(z0) or z0 <= 0.0):
        raise ValueError("z0 must be a positive finite length")



def _bare_potential_fourier(q, R, distribution, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    with np.errstate(divide="ignore", invalid="ignore"):
        if distribution in ("ribbon", "surface"):
            out = 2.0 * sp.k0(q * R)
        elif distribution == "homogeneous":
            x = q * R
            out = np.empty_like(x)
            s = (x < 3e-3)
            out[~s] = 4.0 / x[~s]**2 * (1.0 - x[~s] * sp.k1(x[~s]))
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = -2.0 * (L - 0.5) - 0.25 * x[s]**2 * (L - 1.25)
        else:
            out = 2.0 * _laplace_lorentz(q * _z0_default(R, z0))
    out = np.where(q == 0.0, np.inf, out)
    return out



def _nonlocal_form_factor(q, R, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    if distribution == "ribbon":
        if use_nonlocal:
            raise ValueError("the ribbon distribution has no finite nonlocal form factor")
        return 4.0 * np.ones_like(q)
    if not use_nonlocal:
        return np.ones_like(q)
    if distribution == "surface":
        return sp.i0(q * R)
    if distribution == "homogeneous":
        x = q * R
        out = np.ones_like(x)
        s = (x < 3e-3)
        out[~s] = (1.0 - 2.0 * sp.i1(x[~s]) * sp.k1(x[~s])) / (1.0 - x[~s] * sp.k1(x[~s]))
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = ((L - 0.25) + x[s]**2 * (0.25 * L - 5.0 / 24.0)) / ((L - 0.5) + 0.125 * x[s]**2 * (L - 1.25))
        out = np.where(q == 0.0, 1.0, out)
        return out
    z = _z0_default(R, z0)
    return _laplace_lorentz(2.0 * q * z) / _laplace_lorentz(q * z)



def _dielectric_function(q, R, alpha, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a non-negative finite polarizability")
    q = np.abs(np.asarray(q, dtype=float))
    x = q * float(R)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if distribution == "surface" and use_nonlocal:
            prod = 2.0 * sp.i0e(x) * sp.k0e(x)          # chi * V_bare evaluated without overflow
        elif distribution == "homogeneous" and use_nonlocal:
            num = 1.0 - 2.0 * sp.i1e(x) * sp.k1e(x)
            small = _nonlocal_form_factor(q, R, distribution, True, z0) * _bare_potential_fourier(q, R, distribution, z0)
            prod = np.where(x < 3e-3, small, 4.0 * num / np.where(x == 0.0, 1.0, x)**2)
        else:
            prod = _nonlocal_form_factor(q, R, distribution, use_nonlocal, z0) * _bare_potential_fourier(q, R, distribution, z0)
        eps = 1.0 + alpha * q**2 * prod
    eps = np.where(q == 0.0, 1.0, eps)
    return eps



def _screened_potential_fourier(q, R, alpha, distribution, use_nonlocal=True, z0=None):
    q = np.abs(np.asarray(q, dtype=float))
    vb = _bare_potential_fourier(q, R, distribution, z0)
    eps = _dielectric_function(q, R, alpha, distribution, use_nonlocal, z0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = vb / eps
    out = np.where(q == 0.0, np.inf, out)
    return out


def screened_potential_fourier(q: "float | np.ndarray", R: float, alpha: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    return _screened_potential_fourier(q, R, alpha, distribution, use_nonlocal, z0)

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def _laplace_lorentz(x):
    """F(x) = int_0^inf y exp(-x y) / (1 + y^2) dy = -[cos(x) Ci(x) + sin(x) si(x)], si = Si - pi/2."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 40.0
    si_, ci_ = sp.sici(x[small])
    out[small] = -(np.cos(x[small]) * ci_ + np.sin(x[small]) * (si_ - 0.5 * np.pi))
    xl = x[~small]
    out[~small] = 1.0 / xl**2 - 6.0 / xl**4 + 120.0 / xl**6 - 5040.0 / xl**8 + 362880.0 / xl**10
    return out


def _z0_default(R, z0):
    return 2.0 * R / np.sqrt(5.0) if z0 is None else float(z0)


def _check_common(R, distribution, z0=None):
    allowed = ("ribbon", "surface", "homogeneous", "centre")
    if distribution not in allowed:
        raise ValueError("distribution must be one of " + ", ".join(allowed))
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be a positive finite length")
    if z0 is not None and (not np.isfinite(z0) or z0 <= 0.0):
        raise ValueError("z0 must be a positive finite length")



def _exciton_units(mu):
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a positive finite reduced mass")
    R_exc = 13.605693 * mu
    a_exc = 0.529177 / mu
    return np.array([R_exc, a_exc, 1.5 * a_exc])



def _bare_potential_fourier(q, R, distribution, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    with np.errstate(divide="ignore", invalid="ignore"):
        if distribution in ("ribbon", "surface"):
            out = 2.0 * sp.k0(q * R)
        elif distribution == "homogeneous":
            x = q * R
            out = np.empty_like(x)
            s = (x < 3e-3)
            out[~s] = 4.0 / x[~s]**2 * (1.0 - x[~s] * sp.k1(x[~s]))
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = -2.0 * (L - 0.5) - 0.25 * x[s]**2 * (L - 1.25)
        else:
            out = 2.0 * _laplace_lorentz(q * _z0_default(R, z0))
    out = np.where(q == 0.0, np.inf, out)
    return out



def _nonlocal_form_factor(q, R, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    if distribution == "ribbon":
        if use_nonlocal:
            raise ValueError("the ribbon distribution has no finite nonlocal form factor")
        return 4.0 * np.ones_like(q)
    if not use_nonlocal:
        return np.ones_like(q)
    if distribution == "surface":
        return sp.i0(q * R)
    if distribution == "homogeneous":
        x = q * R
        out = np.ones_like(x)
        s = (x < 3e-3)
        out[~s] = (1.0 - 2.0 * sp.i1(x[~s]) * sp.k1(x[~s])) / (1.0 - x[~s] * sp.k1(x[~s]))
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = ((L - 0.25) + x[s]**2 * (0.25 * L - 5.0 / 24.0)) / ((L - 0.5) + 0.125 * x[s]**2 * (L - 1.25))
        out = np.where(q == 0.0, 1.0, out)
        return out
    z = _z0_default(R, z0)
    return _laplace_lorentz(2.0 * q * z) / _laplace_lorentz(q * z)



def _dielectric_function(q, R, alpha, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a non-negative finite polarizability")
    q = np.abs(np.asarray(q, dtype=float))
    x = q * float(R)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if distribution == "surface" and use_nonlocal:
            prod = 2.0 * sp.i0e(x) * sp.k0e(x)          # chi * V_bare evaluated without overflow
        elif distribution == "homogeneous" and use_nonlocal:
            num = 1.0 - 2.0 * sp.i1e(x) * sp.k1e(x)
            small = _nonlocal_form_factor(q, R, distribution, True, z0) * _bare_potential_fourier(q, R, distribution, z0)
            prod = np.where(x < 3e-3, small, 4.0 * num / np.where(x == 0.0, 1.0, x)**2)
        else:
            prod = _nonlocal_form_factor(q, R, distribution, use_nonlocal, z0) * _bare_potential_fourier(q, R, distribution, z0)
        eps = 1.0 + alpha * q**2 * prod
    eps = np.where(q == 0.0, 1.0, eps)
    return eps



def _screened_potential_fourier(q, R, alpha, distribution, use_nonlocal=True, z0=None):
    q = np.abs(np.asarray(q, dtype=float))
    vb = _bare_potential_fourier(q, R, distribution, z0)
    eps = _dielectric_function(q, R, alpha, distribution, use_nonlocal, z0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = vb / eps
    out = np.where(q == 0.0, np.inf, out)
    return out



def _variational_kernel_integral(lam, R, alpha, mu, distribution, use_nonlocal=True, z0=None):
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("lam must be a positive finite variational parameter")
    a_exc = _exciton_units(mu)[1]
    _check_common(R, distribution, z0)
    kappa = 2.0 * lam / a_exc
    f = lambda t: (1.0 - 3.0 * t * t) / (1.0 + t * t)**3 * float(_screened_potential_fourier(kappa * t, R, alpha, distribution, use_nonlocal, z0))
    pts = [0.0, 1e-6, 1e-3, 0.1, 1.0 / np.sqrt(3.0), 1.0, 3.0, 10.0, 50.0]
    val = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        val += integrate.quad(f, a, b, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    val += integrate.quad(f, pts[-1], np.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    return val


def variational_kernel_integral(lam: float, R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> float:
    return _variational_kernel_integral(lam, R, alpha, mu, distribution, use_nonlocal, z0)

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def _laplace_lorentz(x):
    """F(x) = int_0^inf y exp(-x y) / (1 + y^2) dy = -[cos(x) Ci(x) + sin(x) si(x)], si = Si - pi/2."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 40.0
    si_, ci_ = sp.sici(x[small])
    out[small] = -(np.cos(x[small]) * ci_ + np.sin(x[small]) * (si_ - 0.5 * np.pi))
    xl = x[~small]
    out[~small] = 1.0 / xl**2 - 6.0 / xl**4 + 120.0 / xl**6 - 5040.0 / xl**8 + 362880.0 / xl**10
    return out


def _z0_default(R, z0):
    return 2.0 * R / np.sqrt(5.0) if z0 is None else float(z0)


def _check_common(R, distribution, z0=None):
    allowed = ("ribbon", "surface", "homogeneous", "centre")
    if distribution not in allowed:
        raise ValueError("distribution must be one of " + ", ".join(allowed))
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be a positive finite length")
    if z0 is not None and (not np.isfinite(z0) or z0 <= 0.0):
        raise ValueError("z0 must be a positive finite length")



def _exciton_units(mu):
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a positive finite reduced mass")
    R_exc = 13.605693 * mu
    a_exc = 0.529177 / mu
    return np.array([R_exc, a_exc, 1.5 * a_exc])



def _bare_potential_fourier(q, R, distribution, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    with np.errstate(divide="ignore", invalid="ignore"):
        if distribution in ("ribbon", "surface"):
            out = 2.0 * sp.k0(q * R)
        elif distribution == "homogeneous":
            x = q * R
            out = np.empty_like(x)
            s = (x < 3e-3)
            out[~s] = 4.0 / x[~s]**2 * (1.0 - x[~s] * sp.k1(x[~s]))
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = -2.0 * (L - 0.5) - 0.25 * x[s]**2 * (L - 1.25)
        else:
            out = 2.0 * _laplace_lorentz(q * _z0_default(R, z0))
    out = np.where(q == 0.0, np.inf, out)
    return out



def _nonlocal_form_factor(q, R, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    if distribution == "ribbon":
        if use_nonlocal:
            raise ValueError("the ribbon distribution has no finite nonlocal form factor")
        return 4.0 * np.ones_like(q)
    if not use_nonlocal:
        return np.ones_like(q)
    if distribution == "surface":
        return sp.i0(q * R)
    if distribution == "homogeneous":
        x = q * R
        out = np.ones_like(x)
        s = (x < 3e-3)
        out[~s] = (1.0 - 2.0 * sp.i1(x[~s]) * sp.k1(x[~s])) / (1.0 - x[~s] * sp.k1(x[~s]))
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = ((L - 0.25) + x[s]**2 * (0.25 * L - 5.0 / 24.0)) / ((L - 0.5) + 0.125 * x[s]**2 * (L - 1.25))
        out = np.where(q == 0.0, 1.0, out)
        return out
    z = _z0_default(R, z0)
    return _laplace_lorentz(2.0 * q * z) / _laplace_lorentz(q * z)



def _dielectric_function(q, R, alpha, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a non-negative finite polarizability")
    q = np.abs(np.asarray(q, dtype=float))
    x = q * float(R)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if distribution == "surface" and use_nonlocal:
            prod = 2.0 * sp.i0e(x) * sp.k0e(x)          # chi * V_bare evaluated without overflow
        elif distribution == "homogeneous" and use_nonlocal:
            num = 1.0 - 2.0 * sp.i1e(x) * sp.k1e(x)
            small = _nonlocal_form_factor(q, R, distribution, True, z0) * _bare_potential_fourier(q, R, distribution, z0)
            prod = np.where(x < 3e-3, small, 4.0 * num / np.where(x == 0.0, 1.0, x)**2)
        else:
            prod = _nonlocal_form_factor(q, R, distribution, use_nonlocal, z0) * _bare_potential_fourier(q, R, distribution, z0)
        eps = 1.0 + alpha * q**2 * prod
    eps = np.where(q == 0.0, 1.0, eps)
    return eps



def _screened_potential_fourier(q, R, alpha, distribution, use_nonlocal=True, z0=None):
    q = np.abs(np.asarray(q, dtype=float))
    vb = _bare_potential_fourier(q, R, distribution, z0)
    eps = _dielectric_function(q, R, alpha, distribution, use_nonlocal, z0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = vb / eps
    out = np.where(q == 0.0, np.inf, out)
    return out



def _variational_kernel_integral(lam, R, alpha, mu, distribution, use_nonlocal=True, z0=None):
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("lam must be a positive finite variational parameter")
    a_exc = _exciton_units(mu)[1]
    _check_common(R, distribution, z0)
    kappa = 2.0 * lam / a_exc
    f = lambda t: (1.0 - 3.0 * t * t) / (1.0 + t * t)**3 * float(_screened_potential_fourier(kappa * t, R, alpha, distribution, use_nonlocal, z0))
    pts = [0.0, 1e-6, 1e-3, 0.1, 1.0 / np.sqrt(3.0), 1.0, 3.0, 10.0, 50.0]
    val = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        val += integrate.quad(f, a, b, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    val += integrate.quad(f, pts[-1], np.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    return val



def _variational_binding_energy(lam, R, alpha, mu, distribution, use_nonlocal=True, z0=None):
    R_exc = _exciton_units(mu)[0]
    I = _variational_kernel_integral(lam, R, alpha, mu, distribution, use_nonlocal, z0)
    lam = float(lam)
    return R_exc * (-lam * lam + lam * 4.0 / np.pi * I)


def variational_binding_energy(lam: float, R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> float:
    return _variational_binding_energy(lam, R, alpha, mu, distribution, use_nonlocal, z0)

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def _laplace_lorentz(x):
    """F(x) = int_0^inf y exp(-x y) / (1 + y^2) dy = -[cos(x) Ci(x) + sin(x) si(x)], si = Si - pi/2."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 40.0
    si_, ci_ = sp.sici(x[small])
    out[small] = -(np.cos(x[small]) * ci_ + np.sin(x[small]) * (si_ - 0.5 * np.pi))
    xl = x[~small]
    out[~small] = 1.0 / xl**2 - 6.0 / xl**4 + 120.0 / xl**6 - 5040.0 / xl**8 + 362880.0 / xl**10
    return out


def _z0_default(R, z0):
    return 2.0 * R / np.sqrt(5.0) if z0 is None else float(z0)


def _check_common(R, distribution, z0=None):
    allowed = ("ribbon", "surface", "homogeneous", "centre")
    if distribution not in allowed:
        raise ValueError("distribution must be one of " + ", ".join(allowed))
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be a positive finite length")
    if z0 is not None and (not np.isfinite(z0) or z0 <= 0.0):
        raise ValueError("z0 must be a positive finite length")



def _exciton_units(mu):
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a positive finite reduced mass")
    R_exc = 13.605693 * mu
    a_exc = 0.529177 / mu
    return np.array([R_exc, a_exc, 1.5 * a_exc])



def _bare_potential_fourier(q, R, distribution, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    with np.errstate(divide="ignore", invalid="ignore"):
        if distribution in ("ribbon", "surface"):
            out = 2.0 * sp.k0(q * R)
        elif distribution == "homogeneous":
            x = q * R
            out = np.empty_like(x)
            s = (x < 3e-3)
            out[~s] = 4.0 / x[~s]**2 * (1.0 - x[~s] * sp.k1(x[~s]))
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = -2.0 * (L - 0.5) - 0.25 * x[s]**2 * (L - 1.25)
        else:
            out = 2.0 * _laplace_lorentz(q * _z0_default(R, z0))
    out = np.where(q == 0.0, np.inf, out)
    return out



def _nonlocal_form_factor(q, R, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    q = np.abs(np.asarray(q, dtype=float))
    if np.any(~np.isfinite(q)):
        raise ValueError("q must be finite")
    R = float(R)
    if distribution == "ribbon":
        if use_nonlocal:
            raise ValueError("the ribbon distribution has no finite nonlocal form factor")
        return 4.0 * np.ones_like(q)
    if not use_nonlocal:
        return np.ones_like(q)
    if distribution == "surface":
        return sp.i0(q * R)
    if distribution == "homogeneous":
        x = q * R
        out = np.ones_like(x)
        s = (x < 3e-3)
        out[~s] = (1.0 - 2.0 * sp.i1(x[~s]) * sp.k1(x[~s])) / (1.0 - x[~s] * sp.k1(x[~s]))
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log(0.5 * x[s]) + np.euler_gamma
            out[s] = ((L - 0.25) + x[s]**2 * (0.25 * L - 5.0 / 24.0)) / ((L - 0.5) + 0.125 * x[s]**2 * (L - 1.25))
        out = np.where(q == 0.0, 1.0, out)
        return out
    z = _z0_default(R, z0)
    return _laplace_lorentz(2.0 * q * z) / _laplace_lorentz(q * z)



def _dielectric_function(q, R, alpha, distribution, use_nonlocal=True, z0=None):
    _check_common(R, distribution, z0)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a non-negative finite polarizability")
    q = np.abs(np.asarray(q, dtype=float))
    x = q * float(R)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if distribution == "surface" and use_nonlocal:
            prod = 2.0 * sp.i0e(x) * sp.k0e(x)          # chi * V_bare evaluated without overflow
        elif distribution == "homogeneous" and use_nonlocal:
            num = 1.0 - 2.0 * sp.i1e(x) * sp.k1e(x)
            small = _nonlocal_form_factor(q, R, distribution, True, z0) * _bare_potential_fourier(q, R, distribution, z0)
            prod = np.where(x < 3e-3, small, 4.0 * num / np.where(x == 0.0, 1.0, x)**2)
        else:
            prod = _nonlocal_form_factor(q, R, distribution, use_nonlocal, z0) * _bare_potential_fourier(q, R, distribution, z0)
        eps = 1.0 + alpha * q**2 * prod
    eps = np.where(q == 0.0, 1.0, eps)
    return eps



def _screened_potential_fourier(q, R, alpha, distribution, use_nonlocal=True, z0=None):
    q = np.abs(np.asarray(q, dtype=float))
    vb = _bare_potential_fourier(q, R, distribution, z0)
    eps = _dielectric_function(q, R, alpha, distribution, use_nonlocal, z0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = vb / eps
    out = np.where(q == 0.0, np.inf, out)
    return out



def _variational_kernel_integral(lam, R, alpha, mu, distribution, use_nonlocal=True, z0=None):
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("lam must be a positive finite variational parameter")
    a_exc = _exciton_units(mu)[1]
    _check_common(R, distribution, z0)
    kappa = 2.0 * lam / a_exc
    f = lambda t: (1.0 - 3.0 * t * t) / (1.0 + t * t)**3 * float(_screened_potential_fourier(kappa * t, R, alpha, distribution, use_nonlocal, z0))
    pts = [0.0, 1e-6, 1e-3, 0.1, 1.0 / np.sqrt(3.0), 1.0, 3.0, 10.0, 50.0]
    val = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        val += integrate.quad(f, a, b, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    val += integrate.quad(f, pts[-1], np.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    return val



def _variational_binding_energy(lam, R, alpha, mu, distribution, use_nonlocal=True, z0=None):
    R_exc = _exciton_units(mu)[0]
    I = _variational_kernel_integral(lam, R, alpha, mu, distribution, use_nonlocal, z0)
    lam = float(lam)
    return R_exc * (-lam * lam + lam * 4.0 / np.pi * I)



def _optimise_exciton(R, alpha, mu, distribution, use_nonlocal=True, z0=None, lam_bounds=(0.02, 3.0)):
    lo, hi = float(lam_bounds[0]), float(lam_bounds[1])
    if not (0.0 < lo < hi):
        raise ValueError("lam_bounds must satisfy 0 < lower < upper")
    a_exc = _exciton_units(mu)[1]
    res = optimize.minimize_scalar(lambda l: -_variational_binding_energy(l, R, alpha, mu, distribution, use_nonlocal, z0),
                                   bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    lam0 = float(res.x)
    E_B = -float(res.fun)
    return np.array([lam0, E_B, 1.5 * a_exc / lam0])


def optimise_exciton(R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None, lam_bounds: tuple = (0.02, 3.0)) -> np.ndarray:
    return _optimise_exciton(R, alpha, mu, distribution, use_nonlocal, z0, lam_bounds)

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def nonlocality_correction(R: float, alpha: float, mu: float, distribution: str = 'homogeneous', z0: "float | None" = None) -> float:
    if distribution == "ribbon":
        raise ValueError("the ribbon distribution has no nonlocal variant")
    units = exciton_units(mu)
    q_ref = 1.0
    vb = float(bare_potential_fourier(q_ref, R, distribution, z0))
    chi = float(nonlocal_form_factor(q_ref, R, distribution, True, z0))
    eps = float(dielectric_function(q_ref, R, alpha, distribution, True, z0))
    vs = float(screened_potential_fourier(q_ref, R, alpha, distribution, True, z0))
    if abs(eps - (1.0 + alpha * q_ref**2 * chi * vb)) > 1e-9 * max(1.0, abs(eps)):
        raise ValueError("dielectric function is inconsistent with the form factor and the bare interaction")
    if abs(vs * eps - vb) > 1e-9 * max(1.0, abs(vb)):
        raise ValueError("screened interaction is inconsistent with the bare interaction")
    energies = []
    for use_nl in (False, True):
        out = optimise_exciton(R, alpha, mu, distribution, use_nl, z0)
        lam0, e_b = float(out[0]), float(out[1])
        if abs(out[0] * out[2] - units[2]) > 1e-6:
            raise ValueError("optimiser radius is inconsistent with the excitonic units")
        e_re = float(variational_binding_energy(lam0, R, alpha, mu, distribution, use_nl, z0))
        kern = float(variational_kernel_integral(lam0, R, alpha, mu, distribution, use_nl, z0))
        e_fun = units[0] * (-lam0 * lam0 + lam0 * 4.0 / np.pi * kern)
        if abs(e_re - e_b) > 1e-8 or abs(e_fun - e_b) > 1e-8:
            raise ValueError("energy functional is inconsistent with the optimiser energy")
        energies.append(e_b)
    return 1000.0 * (energies[1] - energies[0])
SCICODE_GOLD_EOF
