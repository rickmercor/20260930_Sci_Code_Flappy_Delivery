"""
Nonlocality form factor of the one-dimensional screening response. Averaging the

lateral integral equation for the potential over the induced-charge distribution f gives

the screened interaction V(q) = V_bare(q) / [1 + alpha_1D q^2 chi(q) V_bare(q)], where the

form factor chi(q) measures how the Green's function K_0(|q| |rho - rho'|) couples

different lateral positions. In the local treatment the kernel is replaced by its value

at the wire axis and chi reduces to the normalisation N_f of the distribution (4 for the

ribbon, whose four edges carry the charge, and 1 otherwise). The partially nonlocal

treatment replaces the unknown lateral potential in the denominator by its lateral

average and evaluates the remaining double integral over f analytically:

Nonlocality form factor of the one-dimensional screening response. Averaging the
lateral integral equation for the potential over the induced-charge distribution f gives
the screened interaction V(q) = V_bare(q) / [1 + alpha_1D q^2 chi(q) V_bare(q)], where the
form factor chi(q) measures how the Green's function K_0(|q| |rho - rho'|) couples
different lateral positions. In the local treatment the kernel is replaced by its value
at the wire axis and chi reduces to the normalisation N_f of the distribution (4 for the
ribbon, whose four edges carry the charge, and 1 otherwise). The partially nonlocal
treatment replaces the unknown lateral potential in the denominator by its lateral
average and evaluates the remaining double integral over f analytically:

  surface:      chi(q) = I_0(|q| R)
  homogeneous:  chi(q) = [1 - 2 I_1(|q| R) K_1(|q| R)] / [1 - |q| R K_1(|q| R)]
  centre:       chi(q) = V_bare(2 q) / V_bare(q)   with the centre-peaked V_bare of the previous step.

All nonlocal factors tend to 1 for q -> 0 or R -> 0, but the homogeneous one does so only
logarithmically: with L = ln(|q| R / 2) + gamma_E it behaves as (L - 1/4) / (L - 1/2) plus
corrections of order (|q| R)^2, so it is still 0.98 at |q| R = 1e-5 (exactly 1 is returned
only at q = 0). Because the exact quotient cancels catastrophically for small |q| R, it is
evaluated below |q| R = 3e-3 from the expansion [(L - 1/4) + (|q| R)^2 (L/4 - 5/24)] /
[(L - 1/2) + (|q| R)^2 (L - 5/4) / 8], accurate there to better than 1e-12. The surface
factor grows without bound; the homogeneous one passes through a minimum of about 0.77
near |q| R = 2.3 and returns to 1 at large |q| R; the centre one decreases to 1/4. The ribbon distribution has no finite
nonlocal form factor (the kernel is logarithmically singular on the edges), so requesting
it is an error.

Returns
-------
np.ndarray of float with the shape of q: the dimensionless form factor chi(q).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonlocal_form_factor(q: "float | np.ndarray", R: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    '''Form factor chi(q) of the 1D screening response.

    Parameters
    ----------
    q : float or array_like of float
        Longitudinal wave number(s) in 1/Angstrom; only |q| enters.
    R : float
        Wire radius in Angstrom, > 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    use_nonlocal : bool
        False returns the local value N_f; True returns the partially nonlocal factor.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    chi : np.ndarray of float
        chi(q) with the shape of q.
        Raises ValueError for an unknown distribution, R <= 0, z0 <= 0, non-finite q,
        or use_nonlocal=True with the ribbon distribution.
    '''
    return chi

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


def _oracle_nonlocal_form_factor(q: "float | np.ndarray", R: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    return _nonlocal_form_factor(q, R, distribution, use_nonlocal, z0)

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
            "call": "np.round(nonlocal_form_factor(q, 2.28, 'homogeneous'), 10)",
            "gold_call": "np.round(_oracle_nonlocal_form_factor(q, 2.28, 'homogeneous'), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.0, 0.1, 0.5, 1.0, 2.5])
""",
            "call": "np.round(nonlocal_form_factor(q, 2.28, 'surface'), 10)",
            "gold_call": "np.round(_oracle_nonlocal_form_factor(q, 2.28, 'surface'), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.1, 0.5, 1.0, 2.5, 8.0])
""",
            "call": "np.round(nonlocal_form_factor(q, 2.28, 'centre'), 10)",
            "gold_call": "np.round(_oracle_nonlocal_form_factor(q, 2.28, 'centre'), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([0.1, 1.0, 5.0])
""",
            "call": "np.round(np.concatenate([nonlocal_form_factor(q, 2.28, 'homogeneous', use_nonlocal=False), nonlocal_form_factor(q, 2.28, 'ribbon', use_nonlocal=False), nonlocal_form_factor(q, 2.28, 'centre', use_nonlocal=False)]), 10)",
            "gold_call": "np.round(np.concatenate([_oracle_nonlocal_form_factor(q, 2.28, 'homogeneous', use_nonlocal=False), _oracle_nonlocal_form_factor(q, 2.28, 'ribbon', use_nonlocal=False), _oracle_nonlocal_form_factor(q, 2.28, 'centre', use_nonlocal=False)]), 10)",
        },
        {
            "setup": """import numpy as np
q = np.array([1e-5, 2e-4, 5e-4, 1e-3, 2e-3])
""",
            "call": "np.round(nonlocal_form_factor(q, 2.28, 'homogeneous'), 10)",
            "gold_call": "np.round(_oracle_nonlocal_form_factor(q, 2.28, 'homogeneous'), 10)",
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        nonlocal_form_factor(1.0, 2.28, 'ribbon', use_nonlocal=True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_nonlocal_form_factor(1.0, 2.28, 'ribbon', use_nonlocal=True)
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
        nonlocal_form_factor(1.0, 2.28, 'centre', z0=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_nonlocal_form_factor(1.0, 2.28, 'centre', z0=0.0)
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
