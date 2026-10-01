"""
Evaluate the dimensionless Feynman mass integral that renormalises the 2D polaron effective mass.

The polaron effective mass is obtained by giving the whole electron-plus-cloud complex a small drift velocity and reading off the quadratic increase of the energy. In the Feynman formalism this produces a mass integral with the same memory kernel as the energy but an extra factor of the imaginary time squared, so it weights the long-time (adiabatic) response more heavily. Evaluated at the energy-minimising variational parameters it gives the mass renormalisation.

Returns
-------
Python float: the dimensionless integral M_F
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def feynman_mass_integral(sigma_0: float, sigma_t: float, v: float, w: float) -> float:
    r"""sigma_0, sigma_t: dimensionless ratios (> 0). v, w: Feynman variational parameters
    (dimensionless, v >= w > 0).

    Return the dimensionless Feynman mass integral M_F(sigma_0, sigma_t; v, w) that sets the polaron
    effective-mass renormalisation m_F = m_e (1 + alpha_m M_F) once evaluated at the energy-minimising
    variational parameters. It is a double integral over imaginary time and dimensionless momentum;
    the imaginary-time part carries an extra quadratic weight relative to the energy integral.

    Raises:
        ValueError: for non-positive sigma or if not v >= w > 0.
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
def _oracle_feynman_mass_integral(sigma_0: float, sigma_t: float, v: float, w: float) -> float:
    s0 = _pos(sigma_0, "sigma_0"); st = _pos(sigma_t, "sigma_t")
    vv, ww = _vw(v, w)
    p, dp = _pgrid(300)
    Om = _omega(s0, p)
    A = st ** 2 * p ** 2 / Om
    a1 = 1.0 + A * ww ** 2 / vv ** 2
    b = A * Om * (vv ** 2 - ww ** 2) / vv ** 3
    c3 = vv / Om
    # imaginary-time integral with tau^2 weight: J2 = sum_n Poisson(n;b) * 2/(a1 + n c3)^3.
    # Convergent Poisson-weighted series for moderate b; Laplace (large-b) asymptotic for the
    # high-momentum tail (b >= 60), where the momentum integrand is already negligible.
    J2 = np.empty_like(p)
    big = b >= 60.0
    xb = a1[big] + b[big] * c3[big]
    J2[big] = 2.0 / xb ** 3 + 12.0 * b[big] * c3[big] ** 2 / xb ** 5
    sm = ~big
    if np.any(sm):
        a1s = a1[sm]; bs = b[sm]; c3s = c3[sm]
        Js = np.zeros_like(a1s); term = np.exp(-bs); n = 0
        while True:
            Js = Js + term * 2.0 / (a1s + n * c3s) ** 3
            n += 1; term = term * bs / n
            if n > int(np.max(bs)) + 80 and float(np.max(term)) < 1e-18:
                break
            if n > 2000:
                break
        J2[sm] = Js
    pref = st ** 3 * (np.sqrt(s0) + 1.0) ** 2 * p ** 3 / ((1.0 + s0 * p) ** 2)
    return float(np.sum(pref * J2 * dp))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'sigma_0=5.1225\nsigma_t=1.0285\nv=7.559\nw=2.978\n',
            "call": 'feynman_mass_integral(sigma_0, sigma_t, v, w)',
            "gold_call": '_oracle_feynman_mass_integral(sigma_0, sigma_t, v, w)',
            "tol": 1e-6,
        },
        {
            "setup": 'sigma_0=1.3795\nsigma_t=0.6617\nv=2.862\nw=2.581\n',
            "call": 'feynman_mass_integral(sigma_0, sigma_t, v, w)',
            "gold_call": '_oracle_feynman_mass_integral(sigma_0, sigma_t, v, w)',
            "tol": 1e-6,
        },
        {
            "setup": 'sigma_0=4.700\nsigma_t=2.150\nv=5.0\nw=3.0\n',
            "call": 'feynman_mass_integral(sigma_0, sigma_t, v, w)',
            "gold_call": '_oracle_feynman_mass_integral(sigma_0, sigma_t, v, w)',
            "tol": 1e-6,
        },
    ]
