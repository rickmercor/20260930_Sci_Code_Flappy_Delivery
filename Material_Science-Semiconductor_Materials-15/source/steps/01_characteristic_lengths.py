"""
Compute the effective Bohr radius and the TO-phonon oscillator length of a 2D polar monolayer.

The polaron problem in a polar monolayer is set by two microscopic length scales: the effective Bohr radius, fixed by the carrier band mass and the background dielectric constant, and the oscillator length associated with the transverse-optical phonon frequency and the carrier mass. These lengths, together with the screening lengths, determine the coupling strength.

Returns
-------
numpy array [a_B, r_t] in nm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def characteristic_lengths(m_e: float, omega_t: float, eps: float) -> np.ndarray:
    r"""m_e: carrier band mass in units of the free-electron mass m0 (> 0).
    omega_t: transverse-optical (TO) phonon energy hbar*omega_t in meV (> 0).
    eps: surrounding/background dielectric constant (> 0; use 1.0 for a freestanding monolayer).

    Return a length-2 numpy array [a_B, r_t] (nm): the effective Bohr radius
    a_B = 4 pi eps0 eps hbar^2/(e^2 m_e) and the oscillator length r_t = sqrt(hbar/(2 m_e omega_t))
    set by the TO phonon frequency.

    Raises:
        ValueError: if any argument is not positive and finite.
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
def _oracle_characteristic_lengths(m_e: float, omega_t: float, eps: float) -> np.ndarray:
    HB = 38.0998
    KE = 1439.96
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t"); e = _pos(eps, "eps")
    aB = e * 2.0 * HB / (m * KE)
    rt = np.sqrt(HB / (m * wt))
    return np.array([aB, rt], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'm_e=0.18\nomega_t=11.10\neps=1.0\n',
            "call": 'characteristic_lengths(m_e, omega_t, eps)',
            "gold_call": '_oracle_characteristic_lengths(m_e, omega_t, eps)',
        },
        {
            "setup": 'm_e=0.83\nomega_t=172.3\neps=1.0\n',
            "call": 'characteristic_lengths(m_e, omega_t, eps)',
            "gold_call": '_oracle_characteristic_lengths(m_e, omega_t, eps)',
        },
        {
            "setup": 'm_e=0.31\nomega_t=29.82\neps=2.0\n',
            "call": 'characteristic_lengths(m_e, omega_t, eps)',
            "gold_call": '_oracle_characteristic_lengths(m_e, omega_t, eps)',
        },
    ]
