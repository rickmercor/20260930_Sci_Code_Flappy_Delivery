#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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
def characteristic_lengths(m_e: float, omega_t: float, eps: float) -> np.ndarray:
    HB = 38.0998
    KE = 1439.96
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t"); e = _pos(eps, "eps")
    aB = e * 2.0 * HB / (m * KE)
    rt = np.sqrt(HB / (m * wt))
    return np.array([aB, rt], dtype=np.float64)

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
def coupling_constant(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> float:
    HB = 38.0998
    KE = 1439.96
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t")
    R0 = _pos(r0, "r0"); Ri = _pos(rinf, "rinf"); e = _pos(eps, "eps")
    if not (R0 > Ri):
        raise ValueError("require r0 > rinf")
    L = characteristic_lengths(m, wt, e)                  # step 1
    aB = float(L[0]); rt = float(L[1])
    return float((rt / aB) * (np.sqrt(R0) - np.sqrt(Ri)) / (np.sqrt(R0) + np.sqrt(Ri)))

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
def dimensionless_ratios(m_e: float, omega_t: float, r0: float, rinf: float) -> np.ndarray:
    HB = 38.0998
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t")
    R0 = _pos(r0, "r0"); Ri = _pos(rinf, "rinf")
    if not (R0 > Ri):
        raise ValueError("require r0 > rinf")
    rt = float(characteristic_lengths(m, wt, 1.0)[1])     # step 1 (r_t is eps-independent)
    return np.array([R0 / Ri, rt / Ri], dtype=np.float64)

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
def weak_coupling_integral(sigma_0: float, sigma_t: float) -> float:
    s0 = _pos(sigma_0, "sigma_0"); st = _pos(sigma_t, "sigma_t")
    def _integrand(p):
        Om = _omega(s0, p)
        return st * (np.sqrt(s0) + 1.0) ** 2 * p / ((1.0 + p) * (1.0 + s0 * p) * (st ** 2 * p ** 2 / Om + 1.0))
    val, _ = quad(_integrand, 0.0, np.inf, limit=200)
    return float(val)

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
def weak_coupling_mass_integral(sigma_0: float, sigma_t: float) -> float:
    s0 = _pos(sigma_0, "sigma_0"); st = _pos(sigma_t, "sigma_t")
    def _integrand(p):
        Om = _omega(s0, p)
        return (st ** 3 * (np.sqrt(s0) + 1.0) ** 2 * p ** 3
                / ((1.0 + s0 * p) ** 2 * (1.0 + st ** 2 * p ** 2 / Om) ** 3))
    val, _ = quad(_integrand, 0.0, np.inf, limit=200)
    return float(2.0 * val)

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
def weak_coupling_observables(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> np.ndarray:
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t")
    R0 = _pos(r0, "r0"); Ri = _pos(rinf, "rinf"); e = _pos(eps, "eps")
    am = coupling_constant(m, wt, R0, Ri, e)              # step 2
    sr = dimensionless_ratios(m, wt, R0, Ri)             # step 3
    s0 = float(sr[0]); st = float(sr[1])
    I0 = weak_coupling_integral(s0, st)                  # step 4
    MLLP = weak_coupling_mass_integral(s0, st)           # step 5
    if not (I0 > 0.0 and MLLP > 0.0):
        raise ValueError("non-physical weak-coupling integrals")
    E0 = -am * wt * I0
    m_LLP = m * (1.0 + am * MLLP)
    return np.array([E0, m_LLP], dtype=np.float64)

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
def feynman_integral(sigma_0: float, sigma_t: float, v: float, w: float) -> float:
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
def feynman_mass_integral(sigma_0: float, sigma_t: float, v: float, w: float) -> float:
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
def polaron_effective_mass(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> float:
    m = _pos(m_e, "m_e"); wt = _pos(omega_t, "omega_t")
    R0 = _pos(r0, "r0"); Ri = _pos(rinf, "rinf"); e = _pos(eps, "eps")
    if not (R0 > Ri):
        raise ValueError("require r0 > rinf")
    am = coupling_constant(m, wt, R0, Ri, e)             # step 2
    sr = dimensionless_ratios(m, wt, R0, Ri)            # step 3
    s0 = float(sr[0]); st = float(sr[1])
    # checkpoint: the weak-coupling baseline must be a bound, mass-enhancing state
    wk = weak_coupling_observables(m, wt, R0, Ri, e)    # step 6 (chains 4,5)
    if not (wk[0] < 0.0 and wk[1] > m):
        raise ValueError("non-physical weak-coupling baseline")

    def _bound(v, w):
        if not (v >= w > 1e-6):
            return 1e18
        return (v - w) ** 2 / (2.0 * v) - am * feynman_integral(s0, st, v, w)   # step 7

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
    Mf = feynman_mass_integral(s0, st, v, w)           # step 8
    if not (Mf > 0.0):
        raise ValueError("non-physical mass integral")
    m_F = m * (1.0 + am * Mf)
    if not np.isfinite(m_F) or m_F <= m:
        raise ValueError("non-physical polaron mass")
    return float(m_F)
SCICODE_GOLD_EOF
