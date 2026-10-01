"""
Assemble the all-coupling pipeline and return the full Feynman polaron effective mass of a 2D polar monolayer.

The all-coupling effective mass is obtained by first minimising the Feynman variational energy over the two trial parameters and then evaluating the mass integral at that minimum. Because the mass weights the long-time response, it departs strongly from the weak-coupling (LLP) estimate as the coupling grows, so the full variational solve at the energy minimum is required.

Returns
-------
Python float: the Feynman polaron effective mass m_F (m0 units)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def polaron_effective_mass(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> float:
    r"""m_e: carrier band mass (m0 units, > 0). omega_t: TO phonon energy hbar*omega_t (meV, > 0).
    r0, rinf: static and high-frequency 2D screening lengths (nm, > 0, r0 > rinf).
    eps: background dielectric constant (> 0; 1.0 for a freestanding monolayer).

    The orchestrator. It must call the earlier functions rather than reimplementing them: form the
    coupling constant and dimensionless ratios, minimise the all-coupling Feynman variational energy
    bound E(v,w) = hbar omega_t[(v-w)^2/(2v) - alpha_m I_F(sigma_0,sigma_t;v,w)] over v >= w > 0, and
    return the full Feynman polaron effective mass m_F = m_e (1 + alpha_m M_F) (in m0 units) evaluated
    at the energy-minimising variational parameters (v*, w*).

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
def _oracle_polaron_effective_mass(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> float:
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t")
    R0 = _pos(r0, "r0"); Ri = _pos(rinf, "rinf"); e = _pos(eps, "eps")
    if not (R0 > Ri):
        raise ValueError("require r0 > rinf")
    am = _oracle_coupling_constant(m, wt, R0, Ri, e)             # step 2
    sr = _oracle_dimensionless_ratios(m, wt, R0, Ri)            # step 3
    s0 = float(sr[0]); st = float(sr[1])
    # checkpoint: the weak-coupling baseline must be a bound, mass-enhancing state
    wk = _oracle_weak_coupling_observables(m, wt, R0, Ri, e)    # step 6 (chains 4,5)
    if not (wk[0] < 0.0 and wk[1] > m):
        raise ValueError("non-physical weak-coupling baseline")

    def _bound(v, w):
        if not (v >= w > 1e-6):
            return 1e18
        return (v - w) ** 2 / (2.0 * v) - am * _oracle_feynman_integral(s0, st, v, w)   # step 7

    best = (1e18, 2.0, 1.5)
    for v0 in np.geomspace(1.05, 60.0, 11):
        for w0 in np.linspace(0.6, min(v0, 6.5), 8):
            f = _bound(v0, w0)
            if f < best[0]:
                best = (f, v0, w0)
    res = minimize(lambda x: _bound(x[0], x[1]), [best[1], best[2]],
                   method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10, maxiter=1500))
    if res.fun <= best[0]:
        v, w = float(res.x[0]), float(res.x[1])
    else:
        v, w = best[1], best[2]
    if not (v >= w > 0.0):
        raise ValueError("variational minimisation failed")
    Mf = _oracle_feynman_mass_integral(s0, st, v, w)           # step 8
    if not (Mf > 0.0):
        raise ValueError("non-physical mass integral")
    m_F = m * (1.0 + am * Mf)
    if not np.isfinite(m_F) or m_F <= m:
        raise ValueError("non-physical polaron mass")
    return float(m_F)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'm_e=0.32\nomega_t=18.2\nr0=25.9\nrinf=4.82\neps=1.0\n',
            "call": 'polaron_effective_mass(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_polaron_effective_mass(m_e, omega_t, r0, rinf, eps)',
            "tol": 1e-4,
        },
        {
            "setup": 'm_e=0.18\nomega_t=11.10\nr0=21.75\nrinf=4.246\neps=1.0\n',
            "call": 'polaron_effective_mass(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_polaron_effective_mass(m_e, omega_t, r0, rinf, eps)',
            "tol": 1e-4,
        },
        {
            "setup": 'm_e=0.31\nomega_t=29.82\nr0=10.62\nrinf=2.826\neps=1.0\n',
            "call": 'polaron_effective_mass(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_polaron_effective_mass(m_e, omega_t, r0, rinf, eps)',
            "tol": 1e-4,
        },
        {
            # edge case: weak coupling (alpha_m = 0.65), where v* -> w* and m_F -> m_LLP
            "setup": 'm_e=0.83\nomega_t=172.3\nr0=1.076\nrinf=0.780\neps=1.0\n',
            "call": 'polaron_effective_mass(m_e, omega_t, r0, rinf, eps)',
            "gold_call": '_oracle_polaron_effective_mass(m_e, omega_t, r0, rinf, eps)',
            "tol": 1e-4,
        },
    ]
