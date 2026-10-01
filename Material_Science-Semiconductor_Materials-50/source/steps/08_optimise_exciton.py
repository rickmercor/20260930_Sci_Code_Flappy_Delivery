"""
Variational optimum. The exciton binding energy is the maximum of E_B(lambda) over the

variational parameter; the optimum lambda_0 gives the average electron-hole separation

r_B = 3 a_exc / (2 lambda_0). E_B(lambda) is smooth and unimodal on the physical range, so

a bounded scalar maximisation (golden-section / Brent) on a fixed interval is sufficient;

the default bounds 0.02 <= lambda <= 3 cover excitons from very extended to strongly

compressed. Because E_B is stationary at the optimum, the binding energy is insensitive to

the last digits of lambda_0, whereas lambda_0 and r_B themselves are determined to about

six significant figures. Invalid bounds (lower >= upper or lower <= 0) are rejected.

Variational optimum. The exciton binding energy is the maximum of E_B(lambda) over the
variational parameter; the optimum lambda_0 gives the average electron-hole separation
r_B = 3 a_exc / (2 lambda_0). E_B(lambda) is smooth and unimodal on the physical range, so
a bounded scalar maximisation (golden-section / Brent) on a fixed interval is sufficient;
the default bounds 0.02 <= lambda <= 3 cover excitons from very extended to strongly
compressed. Because E_B is stationary at the optimum, the binding energy is insensitive to
the last digits of lambda_0, whereas lambda_0 and r_B themselves are determined to about
six significant figures. Invalid bounds (lower >= upper or lower <= 0) are rejected.

Returns
-------
np.ndarray of float with shape (3,): [lambda_0, E_B in eV, r_B = 3 a_exc / (2 lambda_0) in Angstrom].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimise_exciton(R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None, lam_bounds: tuple = (0.02, 3.0)) -> np.ndarray:
    '''Maximise E_B(lambda); return the optimum lambda_0, the binding energy and the exciton radius.

    Parameters
    ----------
    R : float
        Wire radius in Angstrom, > 0.
    alpha : float
        Static 1D polarizability in Angstrom^2, >= 0.
    mu : float
        Interband reduced mass in free-electron masses, > 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    use_nonlocal : bool
        Local (False) or partially nonlocal (True) form factor.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).
    lam_bounds : tuple of float
        Search interval (lower, upper) for lambda with 0 < lower < upper.

    Returns
    -------
    out : np.ndarray of float, shape (3,)
        [lambda_0, E_B in eV, r_B in Angstrom] at the variational optimum.
        Raises ValueError for invalid bounds or wire parameters.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_optimise_exciton(R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None, lam_bounds: tuple = (0.02, 3.0)) -> np.ndarray:
    return _optimise_exciton(R, alpha, mu, distribution, use_nonlocal, z0, lam_bounds)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
""",
            "call": "np.round(optimise_exciton(2.28, 12.06, 0.39, 'homogeneous', False) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
            "gold_call": "np.round(_oracle_optimise_exciton(2.28, 12.06, 0.39, 'homogeneous', False) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
        },
        {
            "setup": """import numpy as np
""",
            "call": "np.round(optimise_exciton(2.28, 12.06, 0.39, 'homogeneous', True) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
            "gold_call": "np.round(_oracle_optimise_exciton(2.28, 12.06, 0.39, 'homogeneous', True) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
        },
        {
            "setup": """import numpy as np
""",
            "call": "np.round(optimise_exciton(2.13, 17.43, 0.30, 'centre', True) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
            "gold_call": "np.round(_oracle_optimise_exciton(2.13, 17.43, 0.30, 'centre', True) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
        },
        {
            "setup": """import numpy as np
""",
            "call": "np.round(optimise_exciton(1e-4, 0.0, 0.39, 'surface', False) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
            "gold_call": "np.round(_oracle_optimise_exciton(1e-4, 0.0, 0.39, 'surface', False) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
        },
        {
            "setup": """import numpy as np
""",
            "call": "np.round(optimise_exciton(1.07, 2.18, 0.69, 'surface', True, lam_bounds=(0.1, 2.0)) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
            "gold_call": "np.round(_oracle_optimise_exciton(1.07, 2.18, 0.69, 'surface', True, lam_bounds=(0.1, 2.0)) * np.array([1e4, 1e5, 1e3])) / np.array([1e4, 1e5, 1e3])",
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        optimise_exciton(2.28, 12.06, 0.39, 'homogeneous', True, lam_bounds=(1.0, 0.5))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_optimise_exciton(2.28, 12.06, 0.39, 'homogeneous', True, lam_bounds=(1.0, 0.5))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        optimise_exciton(2.28, 12.06, 0.39, 'ribbon', True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_optimise_exciton(2.28, 12.06, 0.39, 'ribbon', True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
