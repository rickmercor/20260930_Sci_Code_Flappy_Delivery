"""
Effective one-dimensional dielectric function. With the averaged bare interaction

V_bare(q) and the form factor chi(q) the screening of a quasi-1D semiconductor with

static polarizability alpha_1D (units of length squared) is

Effective one-dimensional dielectric function. With the averaged bare interaction
V_bare(q) and the form factor chi(q) the screening of a quasi-1D semiconductor with
static polarizability alpha_1D (units of length squared) is

  eps(q) = 1 + alpha_1D q^2 chi(q) V_bare(q).

Because no net charge can be induced in a one-dimensional semiconductor, eps(0) = 1 for
every distribution and treatment (the q^2 factor beats the logarithmic divergence of
V_bare). At large |q| R the product chi V_bare decides the limit: eps -> 1 for the local
surface and ribbon models, eps grows linearly in q for the nonlocal surface model, eps
tends to 1 + 4 alpha_1D / R^2 for the homogeneous distribution in both treatments (the
nonlocal form factor returns to 1 there), and to 1 + 2 alpha_1D / z_0^2 (centre, local)
and 1 + alpha_1D / (2 z_0^2) (centre, nonlocal). The nonlocal surface
product I_0(|q| R) 2 K_0(|q| R) and the nonlocal homogeneous product
4 [1 - 2 I_1 K_1] / (q R)^2 are evaluated from exponentially scaled Bessel functions so
that no overflow occurs at large |q| R. A negative polarizability is rejected.

Returns
-------
np.ndarray of float with the shape of q: the dimensionless dielectric function eps(q), equal to 1 at q = 0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dielectric_function(q: "float | np.ndarray", R: float, alpha: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    '''Effective 1D dielectric function eps(q) = 1 + alpha q^2 chi(q) V_bare(q).

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
    eps : np.ndarray of float
        eps(q) with the shape of q (exactly 1 at q = 0).
        Raises ValueError for invalid distribution, R, z0, alpha < 0, non-finite q,
        or the nonlocal ribbon request.
    '''
    return eps

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


def _oracle_dielectric_function(q: "float | np.ndarray", R: float, alpha: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    return _dielectric_function(q, R, alpha, distribution, use_nonlocal, z0)

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
q = np.array([0.0, 0.1, 0.5, 1.0, 2.5, 8.0])
""",
            "call": "np.round(dielectric_function(q, 2.28, 12.06, 'homogeneous', True), 10)",
            "gold_call": "np.round(_oracle_dielectric_function(q, 2.28, 12.06, 'homogeneous', True), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.0, 0.1, 0.5, 1.0, 2.5, 8.0])
""",
            "call": "np.round(dielectric_function(q, 2.28, 12.06, 'homogeneous', False), 10)",
            "gold_call": "np.round(_oracle_dielectric_function(q, 2.28, 12.06, 'homogeneous', False), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.1, 1.0, 2.5, 50.0, 400.0])
""",
            "call": "np.round(dielectric_function(q, 2.28, 12.06, 'surface', True), 8)",
            "gold_call": "np.round(_oracle_dielectric_function(q, 2.28, 12.06, 'surface', True), 8)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.1, 1.0, 2.5, 8.0])
""",
            "call": "np.round(np.concatenate([dielectric_function(q, 2.28, 12.06, 'centre', True), dielectric_function(q, 2.28, 12.06, 'centre', False), dielectric_function(q, np.sqrt(3.33**2 + 3.12**2) / 2, 12.06, 'ribbon', False)]), 10)",
            "gold_call": "np.round(np.concatenate([_oracle_dielectric_function(q, 2.28, 12.06, 'centre', True), _oracle_dielectric_function(q, 2.28, 12.06, 'centre', False), _oracle_dielectric_function(q, np.sqrt(3.33**2 + 3.12**2) / 2, 12.06, 'ribbon', False)]), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([1e-6, 1e-3, 1.0, 1e3])
""",
            "call": "np.round(dielectric_function(q, 2.28, 0.0, 'homogeneous', True), 10)",
            "gold_call": "np.round(_oracle_dielectric_function(q, 2.28, 0.0, 'homogeneous', True), 10)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "np.round(dielectric_function(200.0, 2.28, 12.06, 'homogeneous', False) - 1.0 - 4.0 * 12.06 / 2.28**2, 6)",
            "gold_call": "np.round(_oracle_dielectric_function(200.0, 2.28, 12.06, 'homogeneous', False) - 1.0 - 4.0 * 12.06 / 2.28**2, 6)",
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        dielectric_function(1.0, 2.28, -1.0, 'homogeneous')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_dielectric_function(1.0, 2.28, -1.0, 'homogeneous')
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
