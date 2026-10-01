"""
Evaluate the dimensionless electron-phonon integral of the all-coupling Feynman variational energy.

In the Feynman path-integral treatment the electron couples to the phonon field through a retarded (memory) kernel controlled by two variational parameters. Averaging over the Gaussian trial action gives a double integral over imaginary time and momentum; the imaginary-time integral can be carried out in closed form, leaving a single momentum integral. This dimensionless function interpolates between the weak- and strong-coupling limits as the variational parameters vary.

Returns
-------
Python float: the dimensionless integral I_F
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def feynman_integral(sigma_0: float, sigma_t: float, v: float, w: float) -> float:
    r"""sigma_0, sigma_t: dimensionless ratios (> 0). v, w: the two Feynman variational parameters
    (dimensionless, with v >= w > 0).

    Return the dimensionless Feynman electron-phonon integral I_F(sigma_0, sigma_t; v, w) that enters
    the all-coupling variational energy bound E(v,w) = hbar omega_t[(v-w)^2/(2v) - alpha_m I_F]. The
    integral is over the imaginary-time variable and the dimensionless momentum p in [0, inf); the
    memory kernel of the Feynman trial action enters through the momentum- and time-dependent
    exponent.

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
def _oracle_feynman_integral(sigma_0: float, sigma_t: float, v: float, w: float) -> float:
    s0 = _pos(sigma_0, "sigma_0"); st = _pos(sigma_t, "sigma_t")
    vv, ww = _vw(v, w)
    p, dp = _pgrid(300)
    Om = _omega(s0, p)
    A = st ** 2 * p ** 2 / Om
    a1 = 1.0 + A * ww ** 2 / vv ** 2
    b = A * Om * (vv ** 2 - ww ** 2) / vv ** 3
    c3 = vv / Om
    beta = a1 / c3
    # imaginary-time integral in closed form (Kummer-stable confluent hypergeometric)
    J = hyp1f1(1.0, beta + 1.0, -b) / (c3 * beta)
    pref = st * (np.sqrt(s0) + 1.0) ** 2 * p / ((1.0 + p) * (1.0 + s0 * p))
    return float(np.sum(pref * J * dp))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'sigma_0=5.1225\nsigma_t=1.0285\nv=7.559\nw=2.978\n',
            "call": 'feynman_integral(sigma_0, sigma_t, v, w)',
            "gold_call": '_oracle_feynman_integral(sigma_0, sigma_t, v, w)',
        },
        {
            "setup": 'sigma_0=1.3795\nsigma_t=0.6617\nv=2.862\nw=2.581\n',
            "call": 'feynman_integral(sigma_0, sigma_t, v, w)',
            "gold_call": '_oracle_feynman_integral(sigma_0, sigma_t, v, w)',
        },
        {
            "setup": 'sigma_0=4.700\nsigma_t=2.150\nv=5.0\nw=3.0\n',
            "call": 'feynman_integral(sigma_0, sigma_t, v, w)',
            "gold_call": '_oracle_feynman_integral(sigma_0, sigma_t, v, w)',
        },
    ]
