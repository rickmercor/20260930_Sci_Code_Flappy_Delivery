"""
Evaluate the dimensionless weak-coupling momentum integral of the 2D polaron energy.

At weak coupling the polaron energy shift is the second-order perturbative correction from virtual LO phonons. After the angular integration it reduces to a single momentum integral whose integrand carries the wave-vector dependence of both the Frohlich coupling and the LO frequency. The result is a dimensionless function of the two screening/oscillator ratios.

Returns
-------
Python float: the dimensionless integral I_0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weak_coupling_integral(sigma_0: float, sigma_t: float) -> float:
    r"""sigma_0: dimensionless ratio r0/rinf (> 0). sigma_t: dimensionless ratio r_t/rinf (> 0).

    Return the dimensionless weak-coupling (second-order perturbation-theory) integral
    I_0(sigma_0, sigma_t) that sets the leading polaron energy shift E_0 = -alpha_m hbar omega_t I_0.
    The integrand runs over the dimensionless momentum p in [0, inf) and includes the wave-vector
    dependence of the screening and of the LO frequency through Omega_p = sqrt((1+sigma_0 p)/(1+p)).

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
def _oracle_weak_coupling_integral(sigma_0: float, sigma_t: float) -> float:
    s0 = _pos(sigma_0, "sigma_0"); st = _pos(sigma_t, "sigma_t")
    def _integrand(p):
        Om = _omega(s0, p)
        return st * (np.sqrt(s0) + 1.0) ** 2 * p / ((1.0 + p) * (1.0 + s0 * p) * (st ** 2 * p ** 2 / Om + 1.0))
    val, _ = quad(_integrand, 0.0, np.inf, limit=200)
    return float(val)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'sigma_0=5.1225\nsigma_t=1.0285\n',
            "call": 'weak_coupling_integral(sigma_0, sigma_t)',
            "gold_call": '_oracle_weak_coupling_integral(sigma_0, sigma_t)',
        },
        {
            "setup": 'sigma_0=1.3795\nsigma_t=0.6617\n',
            "call": 'weak_coupling_integral(sigma_0, sigma_t)',
            "gold_call": '_oracle_weak_coupling_integral(sigma_0, sigma_t)',
        },
        {
            "setup": 'sigma_0=4.700\nsigma_t=2.150\n',
            "call": 'weak_coupling_integral(sigma_0, sigma_t)',
            "gold_call": '_oracle_weak_coupling_integral(sigma_0, sigma_t)',
        },
    ]
