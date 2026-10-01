"""
Let phi_tilde(s) = 1 + a_k / sigma_0 on each tenor interval [tau_k, tau_{k+1}) be the normalised displacement profile of Theorem 1 of the source, supplied here as the array levels (one level per interval). For a tenor tau that is one of the grid points, return the eight distinct time-integral functionals of phi_tilde over [0, tau] that appear in the statement of Theorem 1, in the order in which they first appear there, each evaluated exactly (the profile is piecewise constant, so every integrand is a piecewise polynomial). Do not use the single-interval closed forms of Corollary 2 of the source: for a profile that changes level they are not equal to the integrals of Theorem 1.

The displacement enters the expansion of Theorem 1 only through time integrals of its normalised profile. With the piecewise-constant profile of equation (4) the integrals are elementary, but each of them accumulates the profile over all earlier intervals, which is what the closed forms of Corollary 2 miss.

Returns
-------
A float array of length 8 holding the functionals in order of first appearance in the statement of Theorem 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def displacement_functionals(tenors, levels, tau):
    """Return the eight distinct time-integral functionals of the displacement profile."""
    return np.zeros(8)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_displacement_functionals(tenors, levels, tau):
    import numpy as np
    t = np.asarray(tenors, dtype=float)
    lv = np.asarray(levels, dtype=float)
    if t.ndim != 1 or lv.ndim != 1 or len(t) != len(lv) + 1:
        raise ValueError("tenors must hold tau_0 and one entry per level")
    if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("tenors must start at 0 and increase strictly")
    hits = np.nonzero(np.isclose(t, tau, rtol=0.0, atol=1e-15))[0]
    if len(hits) != 1 or hits[0] == 0:
        raise ValueError("tau must be one of the positive tenor grid points")
    n = int(hits[0])
    P = np.polynomial.polynomial
    # running values of the iterated integrals at the left end of the current interval
    I1 = 0.0; I2 = 0.0; K1 = 0.0; L1 = 0.0; M1 = 0.0
    A0 = A1 = A2 = A3 = A4 = B2 = B3a = B3b = 0.0
    for k in range(n):
        t0, h, l = t[k], t[k + 1] - t[k], lv[k]
        phi = np.array([l])                                          # phi~(s)          (in x = s - t0)
        pI1 = np.array([I1, l])                                      # int_0^s phi~
        pI2 = P.polyadd(np.array([I2]), P.polyint(pI1))              # int_0^s int_0^s1 phi~
        pK1 = P.polyadd(np.array([K1]), P.polyint(P.polymul(pI1, phi)))   # int_0^s (int_0^s1 phi~) phi~(s1)
        s_poly = np.array([t0, 1.0])                                 # s
        pL1 = P.polyadd(np.array([L1]), P.polyint(P.polymul(s_poly, phi)))  # int_0^s s1 phi~(s1)
        pM1 = P.polyadd(np.array([M1]), P.polyint(P.polymul(pK1, phi)))     # int_0^s K1(s1) phi~(s1)
        def integ(p):
            return float(P.polyval(h, P.polyint(p)))
        A0 += integ(P.polymul(phi, phi))          # int phi~^2
        A1 += integ(P.polymul(phi, pI1))          # int phi~(s) (int_0^s phi~)
        A2 += integ(pI1)                          # int int_0^s phi~
        A3 += integ(P.polymul(s_poly, phi))       # int s phi~(s)
        A4 += integ(pI2)                          # int int_0^s int_0^s1 phi~
        B2 += integ(P.polymul(pK1, phi))          # int K1(s) phi~(s)
        B3a += integ(P.polymul(pL1, phi))         # int L1(s) phi~(s)
        B3b += integ(P.polymul(pM1, phi))         # int M1(s) phi~(s)
        I1, I2, K1, L1, M1 = (float(P.polyval(h, pI1)), float(P.polyval(h, pI2)),
                              float(P.polyval(h, pK1)), float(P.polyval(h, pL1)),
                              float(P.polyval(h, pM1)))
    return np.array([A0, A1, A2, A3, A4, B2, B3a, B3b], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\ncfg = dict(tenors=[0.0, 1.0/365.0, 2.0/365.0, 3.0/365.0, 4.0/365.0, 5.0/365.0], atm_vols=[0.19, 0.215, 0.205, 0.23, 0.22], vov=0.6, rho=-0.65, eta0=0.4, alpha0=0.10, delta0=0.0741, s0=100.0, m_put=-2.0, m_call=2.0, n_nodes=4000, u_max=20.0)\n"
                  "sh = _oracle_displacement_levels(cfg['tenors'], cfg['atm_vols'])\n"
                  "lv = 1.0 + sh / cfg['atm_vols'][0]",
         "call": "displacement_functionals(cfg['tenors'], lv, cfg['tenors'][-1])",
         "gold_call": "_oracle_displacement_functionals(cfg['tenors'], lv, cfg['tenors'][-1])"},
        {"setup": "import numpy as np",
         "call": "displacement_functionals([0.0, 0.5, 1.0], [1.0, 1.0], 1.0)",
         "gold_call": "_oracle_displacement_functionals([0.0, 0.5, 1.0], [1.0, 1.0], 1.0)"},
        {"setup": "import numpy as np",
         "call": "displacement_functionals([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 0.5], 2.0)",
         "gold_call": "_oracle_displacement_functionals([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 0.5], 2.0)"},
    ]
