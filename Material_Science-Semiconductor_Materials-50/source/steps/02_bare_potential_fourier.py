"""
Averaged bare Coulomb interaction of a quasi-one-dimensional wire in Fourier space.

The wire is a cylinder of radius R along z; the induced charge is distributed laterally

with a normalised distribution f(rho), and the one-dimensional interaction is the

electrostatic potential of a point charge averaged over f. In terms of the longitudinal

wave number q this average of the Helmholtz Green's function K_0(|q| rho) gives, with

N_f the normalisation of f,

Averaged bare Coulomb interaction of a quasi-one-dimensional wire in Fourier space.
The wire is a cylinder of radius R along z; the induced charge is distributed laterally
with a normalised distribution f(rho), and the one-dimensional interaction is the
electrostatic potential of a point charge averaged over f. In terms of the longitudinal
wave number q this average of the Helmholtz Green's function K_0(|q| rho) gives, with
N_f the normalisation of f,

  V_bare(q) = (2 / N_f) int d^2rho f(rho) K_0(|q| rho)   (dimensionless; the factor e^2, which carries the units, is applied in the real-space energy).

Four lateral distributions are used. 'ribbon' and 'surface' put the induced charge on the
boundary of the cross section (for a rectangular ribbon R = sqrt(L^2 + h^2) / 2 is the
effective radius), 'homogeneous' spreads it uniformly over the disc rho <= R, and 'centre'
uses f(rho) = z_0 / (2 pi (rho^2 + z_0^2)^(3/2)) with its maximum on the wire axis and the
cutoff length z_0 = 2 R / sqrt(5) unless given. The corresponding closed forms are

  ribbon, surface:  2 K_0(|q| R)
  homogeneous:      (4 / (q R)^2) [1 - |q| R K_1(|q| R)]
  centre:           -2 [cos(|q| z_0) Ci(|q| z_0) + sin(|q| z_0) si(|q| z_0)],  si(x) = Si(x) - pi / 2,

whose real-space counterparts are the soft-core potentials 1 / sqrt(z^2 + R^2),
(2 / R^2) [sqrt(z^2 + R^2) - |z|] and the truncated potential 1 / (|z| + z_0). All three
diverge logarithmically for q -> 0 (the value at q = 0 is reported as +inf); in the
homogeneous case the exact form loses precision by cancellation for small |q| R and is
replaced below |q| R = 3e-3 by its expansion -2 [L - 1/2] - (|q| R)^2 [L - 5/4] / 4 with
L = ln(|q| R / 2) + gamma_E, accurate there to better than 1e-11. The centre
form is evaluated from the equivalent Laplace integral 2 int_0^inf dy y e^(-|q| z_0 y) / (1 + y^2)
and its large-argument asymptotic series. Only the absolute value of q matters.

Returns
-------
np.ndarray of float with the shape of q: the dimensionless V_bare(q) (np.inf at q = 0).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bare_potential_fourier(q: "float | np.ndarray", R: float, distribution: str, z0: "float | None" = None) -> np.ndarray:
    '''Laterally averaged bare Coulomb interaction V_bare(q) of a quasi-1D wire.

    Parameters
    ----------
    q : float or array_like of float
        Longitudinal wave number(s) in 1/Angstrom; only |q| enters.
    R : float
        Wire radius (or effective ribbon radius) in Angstrom, > 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    v : np.ndarray of float
        V_bare(q) in 1/Angstrom with the shape of q (+inf where q = 0).
        Raises ValueError for an unknown distribution, R <= 0, z0 <= 0 or non-finite q.
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


def _oracle_bare_potential_fourier(q: "float | np.ndarray", R: float, distribution: str, z0: "float | None" = None) -> np.ndarray:
    return _bare_potential_fourier(q, R, distribution, z0)

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
q = np.array([0.05, 0.3, 1.0, 2.5, 8.0])
""",
            "call": "np.round(bare_potential_fourier(q, 2.28, 'homogeneous'), 10)",
            "gold_call": "np.round(_oracle_bare_potential_fourier(q, 2.28, 'homogeneous'), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.05, 0.3, 1.0, 2.5, 8.0])
""",
            "call": "np.round(bare_potential_fourier(q, 2.28, 'surface'), 10)",
            "gold_call": "np.round(_oracle_bare_potential_fourier(q, 2.28, 'surface'), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.05, 0.3, 1.0, 2.5, 8.0, 40.0])
""",
            "call": "np.round(bare_potential_fourier(q, 2.28, 'centre'), 10)",
            "gold_call": "np.round(_oracle_bare_potential_fourier(q, 2.28, 'centre'), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.3, 1.0, 2.5])
""",
            "call": "np.round(bare_potential_fourier(q, 2.28, 'centre', z0=2.28), 10)",
            "gold_call": "np.round(_oracle_bare_potential_fourier(q, 2.28, 'centre', z0=2.28), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([-1.0, 1e-5, 1e-4])
""",
            "call": "np.round(bare_potential_fourier(q, 2.28, 'homogeneous'), 8)",
            "gold_call": "np.round(_oracle_bare_potential_fourier(q, 2.28, 'homogeneous'), 8)",
        },
        {
            "setup": """import numpy as np
""",
            "call": "np.isinf(bare_potential_fourier(0.0, 2.28, 'homogeneous'))",
            "gold_call": "np.isinf(_oracle_bare_potential_fourier(0.0, 2.28, 'homogeneous'))",
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        bare_potential_fourier(1.0, 2.28, 'gaussian')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_bare_potential_fourier(1.0, 2.28, 'gaussian')
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
        bare_potential_fourier(1.0, -2.28, 'surface')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_bare_potential_fourier(1.0, -2.28, 'surface')
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
