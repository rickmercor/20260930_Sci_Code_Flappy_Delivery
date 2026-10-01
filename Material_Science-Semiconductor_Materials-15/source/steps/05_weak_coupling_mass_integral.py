"""
Evaluate the dimensionless weak-coupling (LLP) mass integral of the 2D polaron.

The leading polaron-mass renormalisation follows from the same virtual-phonon cloud that lowers the energy, evaluated in the weak-coupling limit. It is a single dimensionless momentum integral, obtained from the general mass integral when the two variational parameters coincide, and multiplies the coupling constant to give the fractional mass enhancement.

Returns
-------
Python float: the dimensionless integral M_LLP
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weak_coupling_mass_integral(sigma_0: float, sigma_t: float) -> float:
    r"""sigma_0: dimensionless ratio r0/rinf (> 0). sigma_t: dimensionless ratio r_t/rinf (> 0).

    Return the dimensionless weak-coupling (LLP) mass integral M_LLP(sigma_0, sigma_t) that sets the
    weak-coupling polaron mass m_LLP = m_e (1 + alpha_m M_LLP). It is the mass integral in the limit
    of coincident variational parameters (v = w).

    Raises:
        ValueError: for non-positive arguments.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import hyp1f1
from scipy.integrate import quad
from scipy.optimize import minimize
np.seterr(all="ignore")

HB = 38.0998        # hbar^2 / 2 m0   [meV nm^2]
KE = 1439.96        # e^2 / 4 pi eps0 [meV nm]

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(name + " must be positive and finite")
    return v

def _vw(v, w):
    vv = float(v); ww = float(w)
    if not (np.isfinite(vv) and np.isfinite(ww)):
        raise ValueError("v, w must be finite")
    if not (ww > 0.0 and vv >= ww):
        raise ValueError("require v >= w > 0")
    return vv, ww

def _omega(s0, p):
    return np.sqrt((1.0 + s0 * p) / (1.0 + p))

def _pgrid(npp=300):
    # Gauss-Legendre nodes mapped to p in [0, inf) via p = s/(1-s)
    gx, gw = np.polynomial.legendre.leggauss(int(npp))
    s = 0.5 * (gx + 1.0)
    return s / (1.0 - s), 0.5 * gw / (1.0 - s) ** 2
def _oracle_weak_coupling_mass_integral(sigma_0: float, sigma_t: float) -> float:
    s0 = _pos(sigma_0, "sigma_0"); st = _pos(sigma_t, "sigma_t")
    def _integrand(p):
        Om = _omega(s0, p)
        return (st ** 3 * (np.sqrt(s0) + 1.0) ** 2 * p ** 3
                / ((1.0 + s0 * p) ** 2 * (1.0 + st ** 2 * p ** 2 / Om) ** 3))
    val, _ = quad(_integrand, 0.0, np.inf, limit=200)
    return float(2.0 * val)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'sigma_0=5.1225\nsigma_t=1.0285\n',
            "call": 'weak_coupling_mass_integral(sigma_0, sigma_t)',
            "gold_call": '_oracle_weak_coupling_mass_integral(sigma_0, sigma_t)',
        },
        {
            "setup": 'sigma_0=1.3795\nsigma_t=0.6617\n',
            "call": 'weak_coupling_mass_integral(sigma_0, sigma_t)',
            "gold_call": '_oracle_weak_coupling_mass_integral(sigma_0, sigma_t)',
        },
        {
            "setup": 'sigma_0=4.700\nsigma_t=2.150\n',
            "call": 'weak_coupling_mass_integral(sigma_0, sigma_t)',
            "gold_call": '_oracle_weak_coupling_mass_integral(sigma_0, sigma_t)',
        },
    ]
