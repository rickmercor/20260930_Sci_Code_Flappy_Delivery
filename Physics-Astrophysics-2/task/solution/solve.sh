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

def binary_invariants(m1: float, m2: float, chi1: float, chi2: float,
                              l: float, kappa1_0: float, kappa2_0: float,
                              gamma_0: float, c: float = 1.0) -> np.ndarray:
    vals = [m1, m2, chi1, chi2, l, kappa1_0, kappa2_0, gamma_0, c]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all inputs must be finite scalars")
    m1, m2, chi1, chi2 = float(m1), float(m2), float(chi1), float(chi2)
    l, c = float(l), float(c)
    k1, k2, g = float(kappa1_0), float(kappa2_0), float(gamma_0)
    if not (m1 > m2 > 0.0):
        raise ValueError("masses must satisfy m1 > m2 > 0")
    if not (0.0 < chi1 <= 1.0 and 0.0 < chi2 <= 1.0):
        raise ValueError("chi1 and chi2 must lie in (0, 1]")
    if not (l > 0.0 and c > 0.0):
        raise ValueError("l and c must be > 0")
    for ang in (k1, k2, g):
        if not (0.0 < ang < np.pi):
            raise ValueError("initial angles must lie in (0, pi)")
    m = m1 + m2
    mu = m1 * m2 / m
    nu = mu / m
    delta1 = 2.0 * nu * (1.0 + 3.0 * m2 / (4.0 * m1))
    delta2 = 2.0 * nu * (1.0 + 3.0 * m1 / (4.0 * m2))
    sigma1 = nu * (1.0 + m2 / m1)
    sigma2 = nu * (1.0 + m1 / m2)
    s1 = chi1 * m1 ** 2 / (mu * m * c)
    s2 = chi2 * m2 ** 2 / (mu * m * c)
    ck1, ck2, cg = np.cos(k1), np.cos(k2), np.cos(g)
    lam = (sigma1 * s1 * ck1 + sigma2 * s2 * ck2) / l
    big1 = cg + (m1 - m2) / m1 * (l / s2) * ck1
    big2 = ck2 + (m2 / m1) * (s1 / s2) * ck1
    j2 = (l ** 2 + s1 ** 2 + s2 ** 2 + 2.0 * l * s1 * ck1
          + 2.0 * l * s2 * ck2 + 2.0 * s1 * s2 * cg)
    return np.array([nu, mu, delta1, delta2, sigma1, sigma2, s1, s2,
                     np.sqrt(j2), lam, big1, big2], dtype=float)

import numpy as np

def quasi_keplerian_elements(h: float, l: float, nu: float,
                                     c: float = 1.0) -> np.ndarray:
    vals = [h, l, nu, c]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all inputs must be finite scalars")
    h, l, nu, c = float(h), float(l), float(nu), float(c)
    if not (h < 0.0):
        raise ValueError("h must be < 0 for a bound orbit")
    if not (l > 0.0 and c > 0.0):
        raise ValueError("l and c must be > 0")
    if not (0.0 < nu <= 0.25):
        raise ValueError("nu must lie in (0, 0.25]")
    a_r = -1.0 / (2.0 * h) * (1.0 - (nu - 7.0) * h / (2.0 * c ** 2))
    er2 = (1.0 + 2.0 * h * l ** 2 - 2.0 * (6.0 - nu) * h / c ** 2
           - 5.0 * (3.0 - nu) * h ** 2 * l ** 2 / c ** 2)
    n = (-2.0 * h) ** 1.5 * (1.0 + (15.0 - nu) * h / (4.0 * c ** 2))
    et2 = (1.0 + 2.0 * h * l ** 2 + 4.0 * (1.0 - nu) * h / c ** 2
           + (17.0 - 7.0 * nu) * h ** 2 * l ** 2 / c ** 2)
    if er2 <= 0.0 or et2 <= 0.0:
        raise ValueError("h and l must describe an eccentric bound orbit "
                         "with e_r^2 > 0 and e_t^2 > 0")
    e_r, e_t = np.sqrt(er2), np.sqrt(et2)
    e_theta = (3.0 * e_r - e_t) / 2.0
    if not (abs(e_theta) < 1.0):
        raise ValueError("e_theta must satisfy |e_theta| < 1")
    d = a_r * np.sqrt(1.0 - e_theta ** 2)
    return np.array([a_r, e_r, n, e_t, e_theta, d], dtype=float)

import numpy as np

def nutation_cubic_roots(m1: float, m2: float, l: float, s1: float,
                                 s2: float, lam: float, big_sigma1: float,
                                 big_sigma2: float) -> np.ndarray:
    vals = [m1, m2, l, s1, s2, lam, big_sigma1, big_sigma2]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all inputs must be finite scalars")
    m1, m2, l, s1, s2 = float(m1), float(m2), float(l), float(s1), float(s2)
    lam, sg1, sg2 = float(lam), float(big_sigma1), float(big_sigma2)
    if not (m1 > m2 > 0.0):
        raise ValueError("masses must satisfy m1 > m2 > 0")
    if not (l > 0.0 and s1 > 0.0 and s2 > 0.0):
        raise ValueError("l, s1 and s2 must be > 0")
    if lam == 1.0:
        raise ValueError("lambda must differ from 1")
    m = m1 + m2
    a3 = 2.0 * (m1 - m2) * m2 / m1 ** 2 * l * s1
    a2 = -(1.0 / m1 ** 2) * ((m1 - m2) ** 2 * l ** 2 + m2 ** 2 * s1 ** 2
                             + m1 ** 2 * s2 ** 2 + 2.0 * m1 * m2 * s1 * s2 * sg1
                             + 2.0 * m1 * (m1 - m2) * l * s2 * sg2)
    a1 = (2.0 * s2 / m1) * ((m1 - m2) * l * sg1 + (m2 * s1 + m1 * s2 * sg1) * sg2)
    a0 = (1.0 - sg1 ** 2 - sg2 ** 2) * s2 ** 2
    roots = np.roots([a3, a2, a1, a0])
    scale = max(1.0, np.max(np.abs(roots.real)))
    if np.max(np.abs(roots.imag)) > 1e-9 * scale:
        raise ValueError("the nutation cubic must have three real roots")
    roots = np.sort(roots.real)
    big_a = 4.5 * m2 * (m1 - m2) / m ** 2 * (1.0 - lam) ** 2 * l * s1
    return np.array([roots[0], roots[1], roots[2], big_a], dtype=float)

import numpy as np

def kepler_eccentric_anomaly(t_grid: np.ndarray, n: float, e_t: float,
                                     t0: float = 0.0) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float)
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("t_grid must be a nonempty 1-D array of finite times")
    for name, v in (("n", n), ("e_t", e_t), ("t0", t0)):
        if not (np.isscalar(v) and np.isfinite(float(v))):
            raise ValueError(f"{name} must be a finite scalar")
    n, e_t, t0 = float(n), float(e_t), float(t0)
    if n <= 0.0:
        raise ValueError("n must be > 0")
    if not (0.0 <= e_t < 1.0):
        raise ValueError("e_t must lie in [0, 1)")
    with np.errstate(over="ignore", invalid="ignore"):
        ell = n * (t - t0)
    if not np.all(np.isfinite(ell)):
        raise ValueError("computed mean anomaly must be finite in float64")
    if e_t == 0.0:
        return ell.copy()

    # Use the trigonometric functions' argument reduction. Reducing by
    # the rounded float64 value of 2*pi would spuriously treat its multiples
    # as exact periapsis passages and can amplify the phase error when e_t is
    # close to one. atan2(sin(ell), cos(ell)) also preserves tiny anomalies.
    reduced = np.arctan2(np.sin(ell), np.cos(ell))
    mean = np.abs(reduced)

    def residual(z):
        # Stable evaluation near periapsis, including e_t close to one.
        z2 = z * z
        z_minus_sin = np.where(
            z < 0.25,
            z * z2 * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 *
                (1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 *
                (1.0 / 39916800.0 - z2 / 6227020800.0))))),
            z - np.sin(z))
        return (1.0 - e_t) * z + e_t * z_minus_sin - mean

    # The positive reduced root lies in [mean, min(mean + e_t, pi)].
    lo = mean.copy()
    hi = np.minimum(mean + e_t, np.pi)
    hi = np.where(mean == 0.0, 0.0, hi)
    eps = np.finfo(float).eps
    for _ in range(80):
        mid = lo + 0.5 * (hi - lo)
        f_mid = residual(mid)
        lo = np.where(f_mid <= 0.0, mid, lo)
        hi = np.where(f_mid >= 0.0, mid, hi)
        if np.all(hi - lo <= 8.0 * eps * np.maximum(1.0, mid)):
            break
    else:
        raise RuntimeError("elliptic Kepler solver did not converge")
    root = lo + 0.5 * (hi - lo)
    if np.any(np.abs(residual(root)) > 32.0 * eps * np.maximum(1.0, mean)):
        raise RuntimeError("elliptic Kepler residual check failed")

    u = ell + (np.copysign(root, reduced) - reduced)
    error_limit = 1e-12 + 8.0 * eps * np.maximum(1.0, np.abs(ell))
    if (not np.all(np.isfinite(u)) or
            np.any(np.abs((u - ell) - e_t * np.sin(u)) > 2.0 * error_limit)):
        raise RuntimeError("unreduced Kepler residual check failed")
    return u

import numpy as np

def angular_anomaly(u: np.ndarray, e_theta: float) -> np.ndarray:
    uu = np.asarray(u, dtype=float)
    if uu.ndim != 1 or uu.size == 0 or not np.all(np.isfinite(uu)):
        raise ValueError("u must be a nonempty 1-D array of finite values")
    if not (np.isscalar(e_theta) and np.isfinite(float(e_theta))):
        raise ValueError("e_theta must be a finite scalar")
    e_theta = float(e_theta)
    if not (0.0 <= e_theta < 1.0):
        raise ValueError("e_theta must lie in [0, 1)")
    b = e_theta / (1.0 + np.sqrt(1.0 - e_theta ** 2))
    return uu + 2.0 * np.arctan(b * np.sin(uu) / (1.0 - b * np.cos(uu)))

import numpy as np
from scipy.special import ellipj, ellipkinc

def _validate_hybrid_args(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                          e_theta, x0, sign0, c):
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("t_grid must be a nonempty 1-D array of finite times")
    vals = [x_minus, x_plus, x_3, big_a, d, n, e_t, e_theta, x0, sign0, c]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all parameters must be finite scalars")
    if not (float(x_minus) < float(x_plus) < float(x_3)):
        raise ValueError("roots must satisfy x_minus < x_plus < x_3")
    if not (float(big_a) > 0.0 and float(d) > 0.0 and float(n) > 0.0
            and float(c) > 0.0):
        raise ValueError("big_a, d, n and c must be > 0")
    if not (0.0 <= float(e_t) < 1.0 and 0.0 <= float(e_theta) < 1.0):
        raise ValueError("e_t and e_theta must lie in [0, 1)")
    if not (float(x_minus) <= float(x0) <= float(x_plus)):
        raise ValueError("x0 must lie in [x_minus, x_plus]")
    if float(sign0) not in (-1.0, 1.0):
        raise ValueError("sign0 must be +1.0 or -1.0")


def _hybrid_upsilon(t, x_minus, x_plus, x_3, big_a, d, n, e_t, e_theta,
                    x0, sign0, c):
    xm, xp, x3 = float(x_minus), float(x_plus), float(x_3)
    big_a, d, n, c = float(big_a), float(d), float(n), float(c)
    e_theta = float(e_theta)
    mpar = (xp - xm) / (x3 - xm)
    sqax = np.sqrt(big_a * (x3 - xm))
    phi0 = np.arcsin(np.sqrt((float(x0) - xm) / (xp - xm)))
    alpha0 = float(sign0) * 2.0 / sqax * ellipkinc(phi0, mpar)
    u = kepler_eccentric_anomaly(t, n, float(e_t))
    v_th = angular_anomaly(u, e_theta)
    return sqax / 2.0 * (alpha0 + (v_th + e_theta * np.sin(v_th))
                         / (n * c ** 2 * d ** 3))


def nutation_angle(t_grid: np.ndarray, x_minus: float, x_plus: float,
                           x_3: float, big_a: float, d: float, n: float,
                           e_t: float, e_theta: float, x0: float,
                           sign0: float, c: float = 1.0) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float)
    _validate_hybrid_args(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                          e_theta, x0, sign0, c)
    ups = _hybrid_upsilon(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                          e_theta, x0, sign0, c)
    mpar = (float(x_plus) - float(x_minus)) / (float(x_3) - float(x_minus))
    sn, _, _, _ = ellipj(ups, mpar)
    return float(x_minus) + (float(x_plus) - float(x_minus)) * sn ** 2

import numpy as np

def companion_angles(cos_kappa1: np.ndarray, m1: float, m2: float,
                             l: float, s1: float, s2: float,
                             big_sigma1: float, big_sigma2: float) -> np.ndarray:
    x = np.asarray(cos_kappa1, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("cos_kappa1 must be a nonempty 1-D array")
    if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1.0):
        raise ValueError("cos_kappa1 entries must be finite with |x| <= 1")
    vals = [m1, m2, l, s1, s2, big_sigma1, big_sigma2]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all parameters must be finite scalars")
    m1, m2, l = float(m1), float(m2), float(l)
    s1, s2 = float(s1), float(s2)
    sg1, sg2 = float(big_sigma1), float(big_sigma2)
    if not (m1 > m2 > 0.0):
        raise ValueError("masses must satisfy m1 > m2 > 0")
    if not (l > 0.0 and s1 > 0.0 and s2 > 0.0):
        raise ValueError("l, s1 and s2 must be > 0")
    ck2 = sg2 - (m2 / m1) * (s1 / s2) * x
    cg = sg1 - (m1 - m2) / m1 * (l / s2) * x
    return np.vstack([ck2, cg])

import numpy as np

def azimuthal_coefficients(m1: float, m2: float, l: float, s1: float,
                                   s2: float, j: float, lam: float,
                                   big_sigma1: float,
                                   big_sigma2: float) -> np.ndarray:
    vals = [m1, m2, l, s1, s2, j, lam, big_sigma1, big_sigma2]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all inputs must be finite scalars")
    m1, m2, l = float(m1), float(m2), float(l)
    s1, s2, j = float(s1), float(s2), float(j)
    lam, sg1, sg2 = float(lam), float(big_sigma1), float(big_sigma2)
    if not (m1 > m2 > 0.0):
        raise ValueError("masses must satisfy m1 > m2 > 0")
    if not (l > 0.0 and s1 > 0.0 and s2 > 0.0 and j > 0.0):
        raise ValueError("l, s1, s2 and j must be > 0")
    m = m1 + m2
    mu = m1 * m2 / m
    nu = mu / m
    delta1 = 2.0 * nu * (1.0 + 3.0 * m2 / (4.0 * m1))
    delta2 = 2.0 * nu * (1.0 + 3.0 * m1 / (4.0 * m2))

    def l_sector(jj):
        a1 = m1 / (m1 - m2) * (jj + l + s2 * sg2) / s1
        b1 = (3.0 * m1 / (4.0 * (m1 ** 2 - m2 ** 2)) / s1 * (1.0 - lam)
              * (m1 * jj ** 2 + m2 * l ** 2 + (m1 + m2) * jj * l
                 - (m1 - m2) * s1 * (s1 + s2 * sg1)
                 + (m2 * l + m1 * jj) * s2 * sg2))
        return a1, b1

    def s1_sector(jj):
        a1 = m1 / m2 / l * (-jj + s1 + s2 * sg1)
        b1 = (m1 / (4.0 * m2) / l
              * ((2.0 * delta2 - 3.0 * mu / m2 * lam) * jj ** 2
                 - (2.0 * delta1 - 3.0 * mu / m1 * lam) * l ** 2
                 - nu * (s1 ** 2 + s2 ** 2 + 2.0 * s1 * s2 * sg1
                         + 2.0 * l * s2 * sg2)
                 - 3.0 * ((2.0 * m1 - m2) / (m1 + m2)) * (1.0 - lam) * jj * s1
                 + 3.0 * (m1 ** 2 - m2 ** 2) / (m1 + m2) ** 2 * (1.0 - lam)
                 * (s1 ** 2 + s1 * s2 * sg1)
                 - 3.0 * ((1.0 - lam) / (m1 + m2))
                 * (m1 * jj * sg1 + m2 * l * sg2) * s2))
        return a1, b1

    a1l, b1l = l_sector(j)
    a2l, b2l = l_sector(-j)
    b3l = j * nu / 2.0
    a1s, b1s = s1_sector(j)
    a2s, b2s = s1_sector(-j)
    b3s = (delta2 - 3.0 * mu / (2.0 * m2) * lam) * j
    return np.array([a1l, b1l, a2l, b2l, b3l, a1s, b1s, a2s, b2s, b3s],
                    dtype=float)

import numpy as np
from scipy.special import ellipj, ellipk, elliprf, elliprj


def _am_continuous(u, mpar):
    """Continuous Jacobi amplitude for arbitrary real u."""
    big_k = ellipk(mpar)
    ncyc = np.floor((u + big_k) / (2.0 * big_k))
    _, _, _, ph = ellipj(u - 2.0 * big_k * ncyc, mpar)
    return ph + np.pi * ncyc


def _ellip_pi(n, phi, mpar):
    """Legendre incomplete Pi(n; phi | m) extended to arbitrary real phi."""
    ncyc = np.floor((phi + np.pi / 2.0) / np.pi)
    phir = phi - np.pi * ncyc
    sp, cp = np.sin(phir), np.cos(phir)
    y = 1.0 - mpar * sp ** 2
    inc = (sp * elliprf(cp ** 2, y, 1.0)
           + (n / 3.0) * sp ** 3 * elliprj(cp ** 2, y, 1.0, 1.0 - n * sp ** 2))
    comp = (elliprf(0.0, 1.0 - mpar, 1.0)
            + (n / 3.0) * elliprj(0.0, 1.0 - mpar, 1.0, 1.0 - n))
    return inc + 2.0 * ncyc * comp


def azimuthal_phase(t_grid: np.ndarray, x_minus: float, x_plus: float,
                            x_3: float, big_a: float, d: float, n: float,
                            e_t: float, e_theta: float, x0: float,
                            sign0: float, alpha1: float, beta1: float,
                            alpha2: float, beta2: float, beta3: float,
                            c: float = 1.0) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float)
    _validate_hybrid_args(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                          e_theta, x0, sign0, c)
    vals = [alpha1, beta1, alpha2, beta2, beta3]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all coefficients must be finite scalars")
    xm, xp, x3 = float(x_minus), float(x_plus), float(x_3)
    a1, b1, a2, b2, b3 = (float(alpha1), float(beta1), float(alpha2),
                          float(beta2), float(beta3))
    for alpha_i, beta_i in ((a1, b1), (a2, b2)):
        if beta_i != 0.0:
            left, right = alpha_i + xm, alpha_i + xp
            if not ((left > 0.0 and right > 0.0) or
                    (left < 0.0 and right < 0.0)):
                raise ValueError("active alpha_i + x must be nonzero "
                                 "throughout [x_minus, x_plus]")
    mpar = (xp - xm) / (x3 - xm)
    sqax = np.sqrt(float(big_a) * (x3 - xm))
    n1 = (xm - xp) / (a1 + xm) if b1 != 0.0 else 0.0
    n2 = (xm - xp) / (a2 + xm) if b2 != 0.0 else 0.0

    def primitive(u):
        ph = _am_continuous(u, mpar)
        value = b3 * u
        if b1 != 0.0:
            value = value + b1 / (a1 + xm) * _ellip_pi(n1, ph, mpar)
        if b2 != 0.0:
            value = value - b2 / (a2 + xm) * _ellip_pi(n2, ph, mpar)
        return (2.0 / sqax) * value

    ups_t = _hybrid_upsilon(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                            e_theta, x0, sign0, c)
    ups_0 = _hybrid_upsilon(np.array([0.0]), x_minus, x_plus, x_3, big_a,
                            d, n, e_t, e_theta, x0, sign0, c)
    result = primitive(ups_t) - primitive(ups_0)[0]
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("azimuthal phase evaluation is nonfinite")
    return result

import numpy as np

def run_hybrid_precession(m1: float, m2: float, chi1: float,
                                  chi2: float, h: float, l: float,
                                  kappa1_0: float, kappa2_0: float,
                                  gamma_0: float, triple_sign: float,
                                  t_final: float, c: float = 1.0) -> np.ndarray:
    if not (np.isscalar(triple_sign) and float(triple_sign) in (-1.0, 1.0)):
        raise ValueError("triple_sign must be +1.0 or -1.0")
    if not (np.isscalar(t_final) and np.isfinite(float(t_final))
            and float(t_final) > 0.0):
        raise ValueError("t_final must be a finite scalar > 0")
    if not (np.isscalar(h) and np.isfinite(float(h)) and float(h) < 0.0):
        raise ValueError("h must be a finite scalar < 0")
    triple_sign, t_final, h = float(triple_sign), float(t_final), float(h)

    inv = binary_invariants(m1, m2, chi1, chi2, l, kappa1_0,
                                    kappa2_0, gamma_0, c)
    nu, mu, delta1, delta2, sig1, sig2, s1, s2, j, lam, big1, big2 = inv
    qk = quasi_keplerian_elements(h, l, nu, c)
    a_r, e_r, n_mm, e_t, e_th, d = qk
    roots = nutation_cubic_roots(m1, m2, l, s1, s2, lam, big1, big2)
    x_m, x_p, x_3, big_a = roots

    x0 = float(np.cos(float(kappa1_0)))
    sign0 = triple_sign * float(np.sign((delta2 - nu / 2.0) * (1.0 - lam)))

    t_end = np.array([t_final], dtype=float)
    u_end = kepler_eccentric_anomaly(t_end, n_mm, e_t)
    v_th_end = angular_anomaly(u_end, e_th)

    cosk1 = nutation_angle(t_end, x_m, x_p, x_3, big_a, d, n_mm,
                                   e_t, e_th, x0, sign0, c)
    comp = companion_angles(cosk1, m1, m2, l, s1, s2, big1, big2)
    cosk2 = comp[0]

    co = azimuthal_coefficients(m1, m2, l, s1, s2, j, lam, big1, big2)
    dphi_l = azimuthal_phase(t_end, x_m, x_p, x_3, big_a, d, n_mm,
                                     e_t, e_th, x0, sign0, *co[0:5], c)
    dphi_s1 = azimuthal_phase(t_end, x_m, x_p, x_3, big_a, d, n_mm,
                                      e_t, e_th, x0, sign0, *co[5:10], c)
    return np.array([u_end[0], v_th_end[0], float(d), float(cosk1[0]),
                     float(cosk2[0]), float(dphi_l[0]),
                     float(dphi_s1[0])], dtype=float)
SCICODE_GOLD_EOF
