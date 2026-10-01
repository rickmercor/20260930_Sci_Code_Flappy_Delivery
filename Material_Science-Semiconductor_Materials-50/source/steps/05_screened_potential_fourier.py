"""
Statically screened one-dimensional electron-hole interaction in Fourier space,

Statically screened one-dimensional electron-hole interaction in Fourier space,

  V(q) = V_bare(q) / eps(q) = V_bare(q) / [1 + alpha_1D q^2 chi(q) V_bare(q)].

V(q) carries the same logarithmic divergence as V_bare for q -> 0 (reported as +inf at
q = 0) because eps(0) = 1, and it is reduced relative to V_bare over the intermediate
range of |q| R where the polarisation acts. The screened potential energy in real space
is W(z) = -e^2 int dq / (2 pi) e^(i q z) V(|q|); its long-range part falls below the bare
interaction (antiscreening) because the induced charge integrates to zero. The nonlocal
form factors modify V(q) only at finite |q| R, where V_bare is already small, so the
difference between the local and nonlocal screened potentials is a few ten percent at
|q| R of order one and vanishes in both limits.

Returns
-------
np.ndarray of float with the shape of q: the dimensionless screened interaction V(q) (np.inf at q = 0).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screened_potential_fourier(q: "float | np.ndarray", R: float, alpha: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    '''Screened 1D interaction V(q) = V_bare(q) / eps(q).

    Parameters
    ----------
    q : float or array_like of float
        Longitudinal wave number(s) in 1/Angstrom; only |q| enters.
    R : float
        Wire radius in Angstrom, > 0.
    alpha : float
        Static 1D polarizability alpha_1D in Angstrom^2, >= 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    use_nonlocal : bool
        Local (False) or partially nonlocal (True) form factor.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    v : np.ndarray of float
        V(q) in 1/Angstrom with the shape of q (+inf at q = 0).
        Raises ValueError for invalid distribution, R, z0, alpha or q.
    '''
    return v

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


def _oracle_screened_potential_fourier(q: "float | np.ndarray", R: float, alpha: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    return _screened_potential_fourier(q, R, alpha, distribution, use_nonlocal, z0)

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
q = np.array([0.05, 0.1, 0.5, 1.0, 2.5, 8.0])
""",
            "call": "np.round(screened_potential_fourier(q, 2.28, 12.06, 'homogeneous', True), 10)",
            "gold_call": "np.round(_oracle_screened_potential_fourier(q, 2.28, 12.06, 'homogeneous', True), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.05, 0.1, 0.5, 1.0, 2.5, 8.0])
""",
            "call": "np.round(screened_potential_fourier(q, 2.28, 12.06, 'homogeneous', False), 10)",
            "gold_call": "np.round(_oracle_screened_potential_fourier(q, 2.28, 12.06, 'homogeneous', False), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.1, 1.0, 2.5, 50.0, 400.0])
""",
            "call": "np.round(np.concatenate([screened_potential_fourier(q, 2.28, 12.06, 'surface', True), screened_potential_fourier(q, 2.28, 12.06, 'surface', False)]), 10)",
            "gold_call": "np.round(np.concatenate([_oracle_screened_potential_fourier(q, 2.28, 12.06, 'surface', True), _oracle_screened_potential_fourier(q, 2.28, 12.06, 'surface', False)]), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.1, 1.0, 2.5, 8.0])
""",
            "call": "np.round(np.concatenate([screened_potential_fourier(q, 2.28, 12.06, 'centre', True), screened_potential_fourier(q, 2.28, 12.06, 'centre', False), screened_potential_fourier(q, 3.0, 51.63, 'centre', True, z0=3.0)]), 10)",
            "gold_call": "np.round(np.concatenate([_oracle_screened_potential_fourier(q, 2.28, 12.06, 'centre', True), _oracle_screened_potential_fourier(q, 2.28, 12.06, 'centre', False), _oracle_screened_potential_fourier(q, 3.0, 51.63, 'centre', True, z0=3.0)]), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.3, 1.0, 3.0])
""",
            "call": "np.round(screened_potential_fourier(q, 2.28, 0.0, 'homogeneous', True) - screened_potential_fourier(q, 2.28, 0.0, 'homogeneous', False), 12)",
            "gold_call": "np.round(_oracle_screened_potential_fourier(q, 2.28, 0.0, 'homogeneous', True) - _oracle_screened_potential_fourier(q, 2.28, 0.0, 'homogeneous', False), 12)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "np.isinf(screened_potential_fourier(0.0, 2.28, 12.06, 'homogeneous', True))",
            "gold_call": "np.isinf(_oracle_screened_potential_fourier(0.0, 2.28, 12.06, 'homogeneous', True))",
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        screened_potential_fourier(1.0, 2.28, 12.06, 'ribbon', True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_screened_potential_fourier(1.0, 2.28, 12.06, 'ribbon', True)
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
