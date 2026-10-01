"""
Assemble the weak-coupling polaron binding energy and LLP effective mass from the coupling constant and the dimensionless integrals.

Combining the effective coupling constant with the weak-coupling energy and mass integrals gives the leading (Landau-Pekar/LLP) estimates of the polaron binding energy and effective mass. These provide the weak-coupling baseline against which the full all-coupling result is compared.

Returns
-------
numpy array [E0 (meV), m_LLP (m0 units)]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def weak_coupling_observables(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> np.ndarray:
    r"""m_e, omega_t, r0, rinf, eps: as in the earlier steps (all > 0; r0 > rinf).

    Return a length-2 numpy array [E0, m_LLP]: the weak-coupling polaron binding energy
    E0 = -alpha_m hbar omega_t I_0 (meV, a negative energy shift) and the weak-coupling (LLP) polaron
    mass m_LLP = m_e (1 + alpha_m M_LLP) (in m0 units). It must reuse the earlier functions.

    Raises:
        ValueError: whenever any function it calls would raise for these arguments.
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
def _oracle_weak_coupling_observables(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> np.ndarray:
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t")
    R0 = _pos(r0, "r0"); Ri = _pos(rinf, "rinf"); e = _pos(eps, "eps")
    am = _oracle_coupling_constant(m, wt, R0, Ri, e)              # step 2
    sr = _oracle_dimensionless_ratios(m, wt, R0, Ri)             # step 3
    s0 = float(sr[0]); st = float(sr[1])
    I0 = _oracle_weak_coupling_integral(s0, st)                  # step 4
    MLLP = _oracle_weak_coupling_mass_integral(s0, st)           # step 5
    if not (I0 > 0.0 and MLLP > 0.0):
        raise ValueError("non-physical weak-coupling integrals")
    E0 = -am * wt * I0
    m_LLP = m * (1.0 + am * MLLP)
    return np.array([E0, m_LLP], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'm_e=0.18\nomega_t=11.10\nr0=21.75\nrinf=4.246\neps=1.0\n',
            "call": 'weak_coupling_observables(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_weak_coupling_observables(m_e, omega_t, r0, rinf, eps)',
        },
        {
            "setup": 'm_e=0.83\nomega_t=172.3\nr0=1.076\nrinf=0.780\neps=1.0\n',
            "call": 'weak_coupling_observables(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_weak_coupling_observables(m_e, omega_t, r0, rinf, eps)',
        },
        {
            "setup": 'm_e=0.24\nomega_t=17.81\nr0=13.98\nrinf=2.974\neps=1.0\n',
            "call": 'weak_coupling_observables(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_weak_coupling_observables(m_e, omega_t, r0, rinf, eps)',
        },
    ]
