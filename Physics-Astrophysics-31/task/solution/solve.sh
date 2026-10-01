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


def _check_grid(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z must be a one-dimensional grid of at least two points")
    if not np.all(np.isfinite(z)):
        raise ValueError("z must be finite")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z must be strictly increasing")
    if z[0] < 0.0:
        raise ValueError("z must start at or above the base")
    return z


def _check_positive(name, value):
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("%s must be finite and strictly positive" % name)
    return v


def _check_nonnegative(name, value):
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError("%s must be finite and non-negative" % name)
    return v


def stratified_background(z: "ArrayLike", n_base: float, H: float,
                                  B0: float, R0: float, zeta0: float,
                                  Lzeta: float, T_base: float, dT: float,
                                  LT: float, A_He: float) -> "np.ndarray":
    z = _check_grid(z)
    n_base = _check_positive("n_base", n_base)
    H = _check_positive("H", H)
    B0 = _check_positive("B0", B0)
    R0 = _check_positive("R0", R0)
    Lzeta = _check_positive("Lzeta", Lzeta)
    LT = _check_positive("LT", LT)
    T_base = _check_positive("T_base", T_base)
    dT = _check_nonnegative("dT", dT)
    A_He = _check_nonnegative("A_He", A_He)
    zeta0 = float(zeta0)
    if not np.isfinite(zeta0) or zeta0 < 1.0:
        raise ValueError("zeta0 must be finite and not less than 1")

    r_sun_mm = 695.7
    m_proton = 1.6726219e-27
    n_H = n_base * np.exp(-z / H)
    rho = m_proton * (1.0 + 4.0 * A_He) * n_H * 1.0e15 / 1.0e-12
    T = T_base + dT * (1.0 - np.exp(-z / LT))
    B = B0 * (r_sun_mm / (z + r_sun_mm)) ** 2
    R = R0 * np.sqrt(B0 / B)
    zeta = (zeta0 - 1.0) * np.exp(-z / Lzeta) + 1.0
    return np.asarray([n_H, rho, T, B, R, zeta], dtype=float)

import numpy as np


def _broadcast_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        shape = np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    return np.broadcast_to(a, shape).astype(float), np.broadcast_to(b, shape).astype(float)


def cross_section_structure(rho_avg: "ArrayLike", zeta: "ArrayLike",
                                    f: float) -> "np.ndarray":
    rho, zt = _broadcast_pair(rho_avg, zeta)
    if not np.all(np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError("rho_avg must be finite and strictly positive")
    if not np.all(np.isfinite(zt)) or np.any(zt < 1.0):
        raise ValueError("zeta must be finite and not less than 1")
    ff = float(f)
    if not np.isfinite(ff) or not (0.0 < ff < 1.0):
        raise ValueError("f must be finite and satisfy 0 < f < 1")

    weight = 1.0 - ff + ff * zt
    rho_e = rho / weight
    rho_i = zt * rho_e
    return np.asarray([rho_e, rho_i], dtype=float)

import numpy as np


def _broadcast_all(*arrays):
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    try:
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    return [np.broadcast_to(a, shape).astype(float) for a in arrs]


def channel_speeds(B: "ArrayLike", rho_avg: "ArrayLike",
                           rho_i: "ArrayLike", rho_e: "ArrayLike") -> "np.ndarray":
    b, ra, ri, re = _broadcast_all(B, rho_avg, rho_i, rho_e)
    for name, arr in (("B", b), ("rho_avg", ra), ("rho_i", ri), ("rho_e", re)):
        if not np.all(np.isfinite(arr)):
            raise ValueError("%s must be finite" % name)
        if np.any(arr <= 0.0):
            raise ValueError("%s must be strictly positive" % name)

    # Gauss and 1e-12 kg m^-3 in, Mm s^-1 out:
    #   v[m/s] = (1e-4 B) / sqrt(mu_0 * 1e-12 * rho), then divide by 1e6
    mu_0 = 4.0e-7 * np.pi
    speed_factor = 1.0e-6 * 1.0e-4 / np.sqrt(1.0e-12 * mu_0)
    v_alfven = speed_factor * b / np.sqrt(ra)
    v_kink = speed_factor * b * np.sqrt(2.0 / (ri + re))
    return np.asarray([v_alfven, v_kink], dtype=float)

import numpy as np


def _broadcast_four(R, zeta, T, B):
    arrs = [np.asarray(a, dtype=float) for a in (R, zeta, T, B)]
    try:
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    return [np.broadcast_to(a, shape).astype(float) for a in arrs]


def perpendicular_correlation_lengths(R: "ArrayLike", zeta: "ArrayLike",
                                              f: float, T: "ArrayLike",
                                              B: "ArrayLike") -> "np.ndarray":
    r, zt, t, b = _broadcast_four(R, zeta, T, B)
    for name, arr in (("R", r), ("zeta", zt), ("T", t), ("B", b)):
        if not np.all(np.isfinite(arr)):
            raise ValueError("%s must be finite" % name)
    for name, arr in (("R", r), ("T", t), ("B", b)):
        if np.any(arr <= 0.0):
            raise ValueError("%s must be strictly positive" % name)
    if np.any(zt <= 1.0):
        raise ValueError("zeta must be strictly greater than 1")
    ff = float(f)
    if not np.isfinite(ff) or not (0.0 < ff < 1.0):
        raise ValueError("f must be finite and satisfy 0 < f < 1")

    inverse_kink = (np.sqrt(2.0) * (zt - 1.0)
                    / (2.0 * r * np.sqrt(5.0 * ff * np.pi))
                    * (1.0 - ff ** 2.5) / (zt + 1.0 - ff) ** 1.5)
    L_kink = 1.0 / inverse_kink
    # 1e8 * sqrt(T[MK] / B[G]) metres, expressed in Mm
    alfven_length_constant_mm = 1.0e8 / 1.0e6
    L_alfven = alfven_length_constant_mm * np.sqrt(t / b)
    return np.asarray([L_kink, L_alfven], dtype=float)

import numpy as np


def _rate_factor():
    """mJ m^-3 to the three halves, divided by Mm and by sqrt(1e-12 kg m^-3),
    expressed in uW m^-3:  (1e-3)**1.5 / (1e6 * sqrt(1e-12)) * 1e6 = 10**1.5"""
    return 10.0 ** 1.5


def _kink_rate(energy, rho_e, L_kink):
    """Scalar form of the transverse-channel closure, in uW m^-3."""
    w = energy if energy > 0.0 else 0.0
    return _rate_factor() * w ** 1.5 / (L_kink * rho_e ** 0.5)


def _alfven_rate(energy, partner, rho_avg, L_alfven):
    """Scalar form of one Alfven population's closure, in uW m^-3."""
    w = energy if energy > 0.0 else 0.0
    p = partner if partner > 0.0 else 0.0
    return 2.0 * _rate_factor() * p ** 0.5 * w / (L_alfven * rho_avg ** 0.5)


def _broadcast_seven(*arrays):
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    try:
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    return [np.broadcast_to(a, shape).astype(float) for a in arrs]


def deposition_rates(W_kink: "ArrayLike", W_out: "ArrayLike",
                             W_in: "ArrayLike", rho_avg: "ArrayLike",
                             rho_e: "ArrayLike", L_kink: "ArrayLike",
                             L_alfven: "ArrayLike") -> "np.ndarray":
    wk, wo, wi, ra, re, lk, la = _broadcast_seven(
        W_kink, W_out, W_in, rho_avg, rho_e, L_kink, L_alfven)
    names = ("W_kink", "W_out", "W_in", "rho_avg", "rho_e", "L_kink", "L_alfven")
    for name, arr in zip(names, (wk, wo, wi, ra, re, lk, la)):
        if not np.all(np.isfinite(arr)):
            raise ValueError("%s must be finite" % name)
    for name, arr in (("rho_avg", ra), ("rho_e", re),
                      ("L_kink", lk), ("L_alfven", la)):
        if np.any(arr <= 0.0):
            raise ValueError("%s must be strictly positive" % name)

    wk = np.clip(wk, 0.0, None)
    wo = np.clip(wo, 0.0, None)
    wi = np.clip(wi, 0.0, None)

    rate_factor = _rate_factor()
    q_kink = rate_factor * wk ** 1.5 / (lk * np.sqrt(re))
    gamma = 2.0 * rate_factor / (la * np.sqrt(ra))
    q_out = gamma * np.sqrt(wi) * wo
    q_in = gamma * np.sqrt(wo) * wi
    return np.asarray([q_kink, q_out, q_in], dtype=float)

import numpy as np


def _uniform_grid(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 4:
        raise ValueError("z must be a one-dimensional grid of at least four points")
    if not np.all(np.isfinite(z)):
        raise ValueError("z must be finite")
    step = np.diff(z)
    if np.any(step <= 0.0):
        raise ValueError("z must be strictly increasing")
    if not np.allclose(step, step[0], rtol=1e-12, atol=0.0):
        raise ValueError("z must be uniformly spaced")
    return z, float(step[0])


def _positive_profile(name, arr, n):
    a = np.asarray(arr, dtype=float)
    if a.ndim != 1 or a.size != n:
        raise ValueError("%s must have one value per grid point" % name)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if np.any(a <= 0.0):
        raise ValueError("%s must be strictly positive" % name)
    return a


def _midpoints(y):
    """Fourth-order values of y at the interval midpoints of a uniform grid."""
    y = np.asarray(y, dtype=float)
    n = y.size
    out = np.empty(n - 1, dtype=float)
    out[1:n - 2] = (-y[0:n - 3] + 9.0 * y[1:n - 2]
                    + 9.0 * y[2:n - 1] - y[3:n]) / 16.0
    out[0] = (5.0 * y[0] + 15.0 * y[1] - 5.0 * y[2] + y[3]) / 16.0
    out[n - 2] = (y[n - 4] - 5.0 * y[n - 3]
                  + 15.0 * y[n - 2] + 5.0 * y[n - 1]) / 16.0
    return out


def transverse_channel_profile(z: "ArrayLike", v_kink: "ArrayLike",
                                       rho_avg: "ArrayLike", rho_e: "ArrayLike",
                                       L_kink: "ArrayLike",
                                       W_base: float) -> "np.ndarray":
    z, step = _uniform_grid(z)
    n = z.size
    v = _positive_profile("v_kink", v_kink, n)
    ra = _positive_profile("rho_avg", rho_avg, n)
    re = _positive_profile("rho_e", rho_e, n)
    lk = _positive_profile("L_kink", L_kink, n)
    w0 = float(W_base)
    if not np.isfinite(w0) or w0 < 0.0:
        raise ValueError("W_base must be finite and non-negative")

    hv, hra, hre, hlk = (_midpoints(v), _midpoints(ra),
                         _midpoints(re), _midpoints(lk))

    # With heights in Mm, speeds in Mm s^-1 and energy densities in mJ m^-3,
    # the height derivative of a flux is 1e-3 times a rate in uW m^-3.
    flux_unit_factor = 1.0e-3

    def _slope(flux, speed, rho_x, length):
        energy = max(flux, 0.0) / speed
        return -flux_unit_factor * _kink_rate(energy, rho_x, length)

    flux = np.empty(n, dtype=float)
    flux[0] = w0 * v[0]
    for i in range(n - 1):
        k1 = _slope(flux[i], v[i], re[i], lk[i])
        k2 = _slope(flux[i] + 0.5 * step * k1, hv[i], hre[i], hlk[i])
        k3 = _slope(flux[i] + 0.5 * step * k2, hv[i], hre[i], hlk[i])
        k4 = _slope(flux[i] + step * k3, v[i + 1], re[i + 1], lk[i + 1])
        flux[i + 1] = flux[i] + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    energy = np.clip(flux, 0.0, None) / v
    zeros = np.zeros(n, dtype=float)
    rate = deposition_rates(energy, zeros, zeros, ra, re, lk,
                                    np.ones(n, dtype=float))[0]
    return np.asarray([energy, rate], dtype=float)

import numpy as np


def alfven_channel_profile(z: "ArrayLike", v_alfven: "ArrayLike",
                                   rho_avg: "ArrayLike", L_alfven: "ArrayLike",
                                   W_out_base: float, W_in_top: float,
                                   tol: float = 1e-15,
                                   itmax: int = 500) -> "np.ndarray":
    z, step = _uniform_grid(z)
    n = z.size
    v = _positive_profile("v_alfven", v_alfven, n)
    ra = _positive_profile("rho_avg", rho_avg, n)
    la = _positive_profile("L_alfven", L_alfven, n)
    w_out0 = float(W_out_base)
    w_in_top = float(W_in_top)
    for name, value in (("W_out_base", w_out0), ("W_in_top", w_in_top)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("%s must be finite and non-negative" % name)
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tol must be finite and strictly positive")
    if not isinstance(itmax, (int, np.integer)) or int(itmax) < 1:
        raise ValueError("itmax must be an integer of at least 1")

    ones = np.ones(n, dtype=float)
    hv, hra, hla = _midpoints(v), _midpoints(ra), _midpoints(la)
    # Same unit bookkeeping as step 06: with heights in Mm, speeds in Mm s^-1
    # and energy densities in mJ m^-3, the height derivative of a flux is 1e-3
    # times a deposition rate expressed in uW m^-3.
    flux_unit_factor = 1.0e-3

    def _out_rate(flux, speed, rho_a, length, partner):
        energy = max(flux, 0.0) / speed
        return _alfven_rate(energy, partner, rho_a, length)

    def _in_rate(flux, speed, rho_a, length, partner):
        energy = max(flux, 0.0) / speed
        return _alfven_rate(energy, partner, rho_a, length)

    w_in = np.full(n, w_in_top, dtype=float)
    w_out = np.full(n, w_out0, dtype=float)
    for _ in range(int(itmax)):
        h_in = _midpoints(w_in)
        flux = np.empty(n, dtype=float)
        flux[0] = w_out0 * v[0]
        for i in range(n - 1):
            k1 = -flux_unit_factor * _out_rate(flux[i], v[i], ra[i], la[i], w_in[i])
            k2 = -flux_unit_factor * _out_rate(flux[i] + 0.5 * step * k1,
                                               hv[i], hra[i], hla[i], h_in[i])
            k3 = -flux_unit_factor * _out_rate(flux[i] + 0.5 * step * k2,
                                               hv[i], hra[i], hla[i], h_in[i])
            k4 = -flux_unit_factor * _out_rate(flux[i] + step * k3, v[i + 1],
                                               ra[i + 1], la[i + 1], w_in[i + 1])
            flux[i + 1] = flux[i] + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        w_out_new = np.clip(flux, 0.0, None) / v

        h_out = _midpoints(w_out_new)
        gflux = np.empty(n, dtype=float)
        gflux[n - 1] = w_in_top * v[n - 1]
        for i in range(n - 1, 0, -1):
            k1 = flux_unit_factor * _in_rate(gflux[i], v[i], ra[i], la[i],
                                             w_out_new[i])
            k2 = flux_unit_factor * _in_rate(gflux[i] - 0.5 * step * k1, hv[i - 1],
                                             hra[i - 1], hla[i - 1], h_out[i - 1])
            k3 = flux_unit_factor * _in_rate(gflux[i] - 0.5 * step * k2, hv[i - 1],
                                             hra[i - 1], hla[i - 1], h_out[i - 1])
            k4 = flux_unit_factor * _in_rate(gflux[i] - step * k3, v[i - 1],
                                             ra[i - 1], la[i - 1], w_out_new[i - 1])
            gflux[i - 1] = gflux[i] - step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        w_in_new = np.clip(gflux, 0.0, None) / v

        scale = np.max(np.abs(w_in_new))
        change = np.max(np.abs(w_in_new - w_in)) / (scale if scale > 0.0 else 1.0)
        w_out, w_in = w_out_new, w_in_new
        if change < tolerance:
            break

    zeros = np.zeros(n, dtype=float)
    rates = deposition_rates(zeros, w_out, w_in, ra, ones, ones, la)
    return np.asarray([w_out, w_in, rates[1] + rates[2]], dtype=float)

import numpy as np


def radiative_loss_profile(n_H: "ArrayLike", T: "ArrayLike",
                                   rho_avg: "ArrayLike", rho_e: "ArrayLike",
                                   rho_i: "ArrayLike", f: float, lambda0: float,
                                   lambda_exponent: float) -> "np.ndarray":
    arrs = [np.asarray(a, dtype=float) for a in (n_H, T, rho_avg, rho_e, rho_i)]
    try:
        shape = np.broadcast_shapes(*[a.shape for a in arrs])
    except ValueError:
        raise ValueError("inputs do not broadcast to a common shape")
    nh, temp, ra, re, ri = [np.broadcast_to(a, shape).astype(float) for a in arrs]

    names = ("n_H", "T", "rho_avg", "rho_e", "rho_i")
    for name, arr in zip(names, (nh, temp, ra, re, ri)):
        if not np.all(np.isfinite(arr)):
            raise ValueError("%s must be finite" % name)
    if np.any(nh < 0.0):
        raise ValueError("n_H must be non-negative")
    if np.any(temp <= 0.0):
        raise ValueError("T must be strictly positive")
    for name, arr in (("rho_avg", ra), ("rho_e", re), ("rho_i", ri)):
        if np.any(arr <= 0.0):
            raise ValueError("%s must be strictly positive" % name)
    ff = float(f)
    if not np.isfinite(ff) or not (0.0 < ff < 1.0):
        raise ValueError("f must be finite and satisfy 0 < f < 1")
    lam0 = float(lambda0)
    if not np.isfinite(lam0) or lam0 <= 0.0:
        raise ValueError("lambda0 must be finite and strictly positive")
    exponent = float(lambda_exponent)
    if not np.isfinite(exponent):
        raise ValueError("lambda_exponent must be finite")

    n_ambient = nh * re / ra
    n_interior = nh * ri / ra
    mean_square = (1.0 - ff) * n_ambient ** 2 + ff * n_interior ** 2
    # (1e15 m^-3)^2 * 1e-35 W m^3 = 1e-5 W m^-3 = 10 uW m^-3
    loss_factor = 10.0
    return loss_factor * mean_square * lam0 * temp ** exponent

import numpy as np


def _finite_profile(name, arr, n):
    a = np.asarray(arr, dtype=float)
    if a.ndim != 1 or a.size != n:
        raise ValueError("%s must have one value per grid point" % name)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    return a


def _cubic_through(y, t):
    return (-y[0] * (t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0
            + y[1] * t * (t - 2.0) * (t - 3.0) / 2.0
            - y[2] * t * (t - 1.0) * (t - 3.0) / 2.0
            + y[3] * t * (t - 1.0) * (t - 2.0) / 6.0)


def first_balance_height(z: "ArrayLike", heating: "ArrayLike",
                                 losses: "ArrayLike") -> float:
    z, step = _uniform_grid(z)
    n = z.size
    heat = _finite_profile("heating", heating, n)
    loss = _finite_profile("losses", losses, n)

    residual = heat - loss
    crossings = np.nonzero(np.sign(residual[:-1]) * np.sign(residual[1:]) < 0.0)[0]
    if crossings.size == 0:
        raise ValueError("the residual never changes sign inside the grid")
    lower = int(crossings[0])
    i0 = int(np.clip(lower - 1, 0, n - 4))
    node_values = residual[i0:i0 + 4]
    t_lo = float(lower - i0)

    a, b = t_lo, t_lo + 1.0
    fa = _cubic_through(node_values, a)
    for _ in range(200):
        mid = 0.5 * (a + b)
        fm = _cubic_through(node_values, mid)
        if (fa < 0.0) != (fm < 0.0):
            b = mid
        else:
            a, fa = mid, fm
    t_star = 0.5 * (a + b)
    return float(z[i0] + t_star * step)

import numpy as np


def two_channel_balance_height(
        n_points: int = 4001, z_top: float = 120.0,
        n_base: float = 1.5, H: float = 42.0, B0: float = 12.5,
        R0: float = 0.8, zeta0: float = 3.6, Lzeta: float = 400.0,
        T_base: float = 0.62, dT: float = 0.83, LT: float = 45.0,
        A_He: float = 0.1, f: float = 0.16, lambda0: float = 1.15,
        lambda_exponent: float = -0.5, W_kink_base: float = 0.9,
        W_alfven_out_base: float = 0.5, W_alfven_in_top: float = 0.025,
        tol: float = 1e-15, itmax: int = 500) -> float:
    if not isinstance(n_points, (int, np.integer)) or int(n_points) < 4:
        raise ValueError("n_points must be an integer of at least 4")
    top = float(z_top)
    if not np.isfinite(top) or top <= 0.0:
        raise ValueError("z_top must be finite and strictly positive")

    z = np.linspace(0.0, top, int(n_points))

    background = stratified_background(
        z, n_base, H, B0, R0, zeta0, Lzeta, T_base, dT, LT, A_He)
    n_H, rho_avg, temperature, field, radius, contrast = background

    components = cross_section_structure(rho_avg, contrast, f)
    rho_e, rho_i = components

    speeds = channel_speeds(field, rho_avg, rho_i, rho_e)
    v_alfven, v_kink = speeds

    lengths = perpendicular_correlation_lengths(
        radius, contrast, f, temperature, field)
    L_kink, L_alfven = lengths

    kink = transverse_channel_profile(
        z, v_kink, rho_avg, rho_e, L_kink, W_kink_base)
    alfven = alfven_channel_profile(
        z, v_alfven, rho_avg, L_alfven, W_alfven_out_base, W_alfven_in_top,
        tol, itmax)

    # the heating is re-formed from the closures of step 05 on the relaxed
    # profiles, so every earlier step is on the path to the answer
    rates = deposition_rates(
        kink[0], alfven[0], alfven[1], rho_avg, rho_e, L_kink, L_alfven)
    heating = rates[0] + rates[1] + rates[2]
    if not np.allclose(heating, kink[1] + alfven[2], rtol=1e-12, atol=0.0):
        raise ValueError("the two channels disagree with the closures of step 05")

    losses = radiative_loss_profile(
        n_H, temperature, rho_avg, rho_e, rho_i, f, lambda0, lambda_exponent)

    if np.any(kink[1] < 0.0) or np.any(alfven[2] < 0.0):
        raise ValueError("a channel returned a negative deposition rate")
    if not heating[0] < losses[0]:
        raise ValueError("heating already exceeds the losses at the base")
    if not heating[-1] > losses[-1]:
        raise ValueError("heating never reaches the losses inside the domain")

    z_star = first_balance_height(z, heating, losses)
    if not (z[0] < z_star < z[-1]):
        raise ValueError("the balance height is not strictly inside the domain")
    return float(z_star)
SCICODE_GOLD_EOF
