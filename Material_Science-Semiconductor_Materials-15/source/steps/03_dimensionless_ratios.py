"""
Form the two dimensionless screening/oscillator ratios that parameterise the 2D polaron integrals.

Beyond the overall coupling constant, the 2D polaron energy and mass depend on two dimensionless ratios built from the static screening length, the high-frequency screening length and the oscillator length. They enter every subsequent momentum integral through the wave-vector dependence of the screening and of the LO-phonon frequency.

Returns
-------
numpy array [sigma_0, sigma_t]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def dimensionless_ratios(m_e: float, omega_t: float, r0: float, rinf: float) -> np.ndarray:
    r"""m_e: carrier band mass (m0 units, > 0). omega_t: TO phonon energy (meV, > 0).
    r0, rinf: static and high-frequency 2D screening lengths (nm, > 0, r0 > rinf).

    Return a length-2 numpy array [sigma_0, sigma_t] of the two dimensionless ratios that fix the
    shape of the 2D polaron integrals: sigma_0 = r0/rinf and sigma_t = r_t/rinf, with
    r_t = sqrt(hbar/(2 m_e omega_t)) the oscillator length.

    Raises:
        ValueError: for non-positive arguments or if r0 <= rinf.
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
def _oracle_dimensionless_ratios(m_e: float, omega_t: float, r0: float, rinf: float) -> np.ndarray:
    HB = 38.0998
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t")
    R0 = _pos(r0, "r0"); Ri = _pos(rinf, "rinf")
    if not (R0 > Ri):
        raise ValueError("require r0 > rinf")
    rt = float(_oracle_characteristic_lengths(m, wt, 1.0)[1])     # step 1 (r_t is eps-independent)
    return np.array([R0 / Ri, rt / Ri], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'm_e=0.18\nomega_t=11.10\nr0=21.75\nrinf=4.246\n',
            "call": 'dimensionless_ratios(m_e, omega_t, r0, rinf)',
            "gold_call": '_oracle_dimensionless_ratios(m_e, omega_t, r0, rinf)',
        },
        {
            "setup": 'm_e=0.83\nomega_t=172.3\nr0=1.076\nrinf=0.780\n',
            "call": 'dimensionless_ratios(m_e, omega_t, r0, rinf)',
            "gold_call": '_oracle_dimensionless_ratios(m_e, omega_t, r0, rinf)',
        },
        {
            "setup": 'm_e=0.51\nomega_t=74.15\nr0=0.763\nrinf=0.466\n',
            "call": 'dimensionless_ratios(m_e, omega_t, r0, rinf)',
            "gold_call": '_oracle_dimensionless_ratios(m_e, omega_t, r0, rinf)',
        },
    ]
