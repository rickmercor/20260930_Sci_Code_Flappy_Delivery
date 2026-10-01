"""
Momentum-space expectation value of the screened interaction in the variational exciton

state. The trial function has the shape of the one-dimensional hydrogen ground state with

the exciton Bohr radius divided by the variational parameter lambda,

Momentum-space expectation value of the screened interaction in the variational exciton
state. The trial function has the shape of the one-dimensional hydrogen ground state with
the exciton Bohr radius divided by the variational parameter lambda,

  Phi(z) = sqrt(2 lambda^3 / a_exc^3) z exp(-lambda |z| / a_exc),

which vanishes at z = 0 as every bound state of the 1D Coulomb problem must. Inserting
the Fourier representation of W(z) into <Phi| W |Phi> and substituting q = 2 lambda t / a_exc
turns the potential energy into -R_exc lambda (4 / pi) I(lambda) with the kernel integral

  I(lambda) = int_0^inf dt (1 - 3 t^2) / (1 + t^2)^3 V(2 lambda t / a_exc).

The kernel is the Fourier transform of |Phi|^2; it changes sign at t = 1 / sqrt(3) and
integrates to zero over t, which removes the logarithmic divergence of V(q -> 0) from the
integral. A 1s-like trial function exp(-lambda |z| / a_exc) would replace the kernel by
1 / (1 + t^2) and overbind strongly. For the bare 1D Coulomb potential V = 1 / |q| ... the
limit alpha -> 0, R -> 0 gives I = pi / 2, so that the potential energy is -2 R_exc lambda.
The integral is evaluated with adaptive quadrature on panels that isolate the logarithmic
singularity at t -> 0 and the sign change of the kernel. A non-positive lambda is rejected.

Returns
-------
float: the dimensionless kernel integral I(lambda).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def variational_kernel_integral(lam: float, R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> float:
    '''Kernel integral I(lambda) = int_0^inf dt (1 - 3 t^2) / (1 + t^2)^3 V(2 lambda t / a_exc).

    Parameters
    ----------
    lam : float
        Variational parameter lambda, > 0.
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

    Returns
    -------
    val : float
        The dimensionless kernel integral I(lambda) (V in 1/Angstrom times the Angstrom
        from the substitution cancels).
        Raises ValueError for lam <= 0 or invalid wire parameters.
    '''
    return val

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


def _oracle_variational_kernel_integral(lam: float, R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> float:
    return _variational_kernel_integral(lam, R, alpha, mu, distribution, use_nonlocal, z0)

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
            "call": "round(variational_kernel_integral(0.34, 2.28, 12.06, 0.39, 'homogeneous', False), 9)",
            "gold_call": "round(_oracle_variational_kernel_integral(0.34, 2.28, 12.06, 0.39, 'homogeneous', False), 9)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "round(variational_kernel_integral(0.35, 2.28, 12.06, 0.39, 'homogeneous', True), 9)",
            "gold_call": "round(_oracle_variational_kernel_integral(0.35, 2.28, 12.06, 0.39, 'homogeneous', True), 9)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "np.round([variational_kernel_integral(1.0, 2.28, 12.06, 0.39, 'surface', True), variational_kernel_integral(0.3, 2.28, 12.06, 0.39, 'centre', True), variational_kernel_integral(0.25, np.sqrt(3.33**2 + 3.12**2) / 2, 12.06, 0.39, 'ribbon', False)], 9)",
            "gold_call": "np.round([_oracle_variational_kernel_integral(1.0, 2.28, 12.06, 0.39, 'surface', True), _oracle_variational_kernel_integral(0.3, 2.28, 12.06, 0.39, 'centre', True), _oracle_variational_kernel_integral(0.25, np.sqrt(3.33**2 + 3.12**2) / 2, 12.06, 0.39, 'ribbon', False)], 9)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "round(variational_kernel_integral(1.0, 1e-4, 0.0, 0.39, 'homogeneous', False) - np.pi / 2, 6)",
            "gold_call": "round(_oracle_variational_kernel_integral(1.0, 1e-4, 0.0, 0.39, 'homogeneous', False) - np.pi / 2, 6)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "round(variational_kernel_integral(2.5, 4.02, 51.63, 0.11, 'surface', False), 9)",
            "gold_call": "round(_oracle_variational_kernel_integral(2.5, 4.02, 51.63, 0.11, 'surface', False), 9)",
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        variational_kernel_integral(0.0, 2.28, 12.06, 0.39, 'homogeneous')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_variational_kernel_integral(0.0, 2.28, 12.06, 0.39, 'homogeneous')
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
        variational_kernel_integral(0.3, 2.28, 12.06, -0.39, 'homogeneous')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_variational_kernel_integral(0.3, 2.28, 12.06, -0.39, 'homogeneous')
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
