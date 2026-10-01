"""
Evaluate the effective 2D Frohlich coupling constant that measures the electron-phonon interaction strength.

In two dimensions the Frohlich coupling is wave-vector dependent, so the interaction strength is characterised by an upper bound on that coupling, obtained by replacing the LO frequency with the TO frequency and evaluating the dielectric factor at its maximising wave vector. It combines the ratio of the oscillator length to the effective Bohr radius with a dimensionless factor built from the static and high-frequency screening lengths, and it plays the role of the 3D Frohlich constant.

Returns
-------
Python float: the effective coupling constant alpha_m
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coupling_constant(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> float:
    r"""m_e: carrier band mass (m0 units, > 0). omega_t: TO phonon energy hbar*omega_t (meV, > 0).
    r0: static 2D screening length (nm, > 0). rinf: high-frequency 2D screening length (nm, > 0,
    with r0 > rinf). eps: background dielectric constant (> 0).

    Return the effective dimensionless Frohlich coupling constant alpha_m of the 2D polaron: the
    upper bound on the wave-vector-dependent coupling, obtained by replacing the LO frequency with
    the TO frequency and evaluating the dielectric factor at the wave vector q = 1/sqrt(r0*rinf)
    that maximises it. It is not a q-average and not the coupling evaluated at that wave vector.

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
def _oracle_coupling_constant(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> float:
    HB = 38.0998
    KE = 1439.96
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t")
    R0 = _pos(r0, "r0"); Ri = _pos(rinf, "rinf"); e = _pos(eps, "eps")
    if not (R0 > Ri):
        raise ValueError("require r0 > rinf")
    L = _oracle_characteristic_lengths(m, wt, e)                  # step 1
    aB = float(L[0]); rt = float(L[1])
    return float((rt / aB) * (np.sqrt(R0) - np.sqrt(Ri)) / (np.sqrt(R0) + np.sqrt(Ri)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'm_e=0.18\nomega_t=11.10\nr0=21.75\nrinf=4.246\neps=1.0\n',
            "call": 'coupling_constant(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_coupling_constant(m_e, omega_t, r0, rinf, eps)',
        },
        {
            "setup": 'm_e=0.83\nomega_t=172.3\nr0=1.076\nrinf=0.780\neps=1.0\n',
            "call": 'coupling_constant(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_coupling_constant(m_e, omega_t, r0, rinf, eps)',
        },
        {
            "setup": 'm_e=0.24\nomega_t=17.81\nr0=13.98\nrinf=2.974\neps=1.0\n',
            "call": 'coupling_constant(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_coupling_constant(m_e, omega_t, r0, rinf, eps)',
        },
    ]
