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


_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5     # rad fs^-1 per cm^-1


def _s01_check_bath(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    sd = np.atleast_1d(np.asarray(sigma_donor, dtype=float))
    sa = np.atleast_1d(np.asarray(sigma_acceptor, dtype=float))
    ta = np.atleast_1d(np.asarray(tau_fs, dtype=float))
    if sd.size == 0 or sd.size != sa.size or sd.size != ta.size:
        raise ValueError("sigma_donor, sigma_acceptor and tau_fs must be non-empty and equal length")
    if not (np.all(np.isfinite(sd)) and np.all(np.isfinite(sa)) and np.all(np.isfinite(ta))):
        raise ValueError("bath parameters must all be finite")
    if np.any(sd < 0.0) or np.any(sa < 0.0):
        raise ValueError("fluctuation amplitudes must be non-negative")
    if np.any(ta <= 0.0):
        raise ValueError("correlation times must be strictly positive")
    tstat = float(tau_static_fs)
    if not np.isfinite(tstat) or tstat <= 0.0:
        raise ValueError("tau_static_fs must be finite and strictly positive")
    return sd, sa, ta, tstat


def dephasing_lineshape(t_fs, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    sd, sa, ta, tstat = _s01_check_bath(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)
    t = np.atleast_1d(np.asarray(t_fs, dtype=float))
    if not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("times must be finite and non-negative")
    g = np.zeros_like(t)
    for s_d, s_a, tau in zip(sd, sa, ta):
        if tau >= tstat:
            continue
        lam = (s_d * _TWO_PI_C) ** 2 + (s_a * _TWO_PI_C) ** 2
        x = t / tau
        g = g + lam * tau * tau * (np.exp(-x) + x - 1.0)
    return g


#

import numpy as np


def _s02_check(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    sd = np.atleast_1d(np.asarray(sigma_donor, dtype=float))
    sa = np.atleast_1d(np.asarray(sigma_acceptor, dtype=float))
    ta = np.atleast_1d(np.asarray(tau_fs, dtype=float))
    if sd.size == 0 or sd.size != sa.size or sd.size != ta.size:
        raise ValueError("sigma_donor, sigma_acceptor and tau_fs must be non-empty and equal length")
    if not (np.all(np.isfinite(sd)) and np.all(np.isfinite(sa)) and np.all(np.isfinite(ta))):
        raise ValueError("bath parameters must all be finite")
    if np.any(sd < 0.0) or np.any(sa < 0.0):
        raise ValueError("fluctuation amplitudes must be non-negative")
    if np.any(ta <= 0.0):
        raise ValueError("correlation times must be strictly positive")
    tstat = float(tau_static_fs)
    if not np.isfinite(tstat) or tstat <= 0.0:
        raise ValueError("tau_static_fs must be finite and strictly positive")
    return sd, sa, ta, tstat


def static_gap_width(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    sd, sa, ta, tstat = _s02_check(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)
    mask = ta >= tstat
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.sum(sd[mask] ** 2 + sa[mask] ** 2)))


#

import numpy as np


_S03_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def second_order_response(t_fs, sigma_donor, sigma_acceptor, tau_fs,
                                  tau_static_fs, gap_cm):
    gap = float(gap_cm)
    if not np.isfinite(gap):
        raise ValueError("gap_cm must be finite")
    t = np.atleast_1d(np.asarray(t_fs, dtype=float))
    g = dephasing_lineshape(t, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)
    return np.exp(-g) * np.exp(-1j * gap * _S03_TWO_PI_C * t)


#

import numpy as np


_S04_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def _s04_grid(t_max_fs, dt_fs):
    tmax, dt = float(t_max_fs), float(dt_fs)
    if not np.isfinite(tmax) or tmax <= 0.0:
        raise ValueError("t_max_fs must be finite and strictly positive")
    if not np.isfinite(dt) or dt <= 0.0 or dt > tmax:
        raise ValueError("dt_fs must be finite, strictly positive and at most t_max_fs")
    return np.linspace(0.0, tmax, int(round(tmax / dt)) + 1)


def second_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                              gap_cm, j_cm, t_max_fs, dt_fs):
    j = float(j_cm)
    if not np.isfinite(j):
        raise ValueError("j_cm must be finite")
    t = _s04_grid(t_max_fs, dt_fs)
    resp = second_order_response(t, sigma_donor, sigma_acceptor, tau_fs,
                                         tau_static_fs, gap_cm)
    s = np.trapezoid(resp, t)
    j_ang = j * _S04_TWO_PI_C
    return float(2.0 * j_ang * j_ang * s.real * 1.0e3)


#

import numpy as np


_S05_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def _s05_pieces(t1_fs, tw_fs, t3_fs, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    t1 = np.asarray(t1_fs, dtype=float)
    tw = np.asarray(tw_fs, dtype=float)
    t3 = np.asarray(t3_fs, dtype=float)
    try:
        t1b, twb, t3b = np.broadcast_arrays(t1, tw, t3)
    except ValueError:
        raise ValueError("t1_fs, tw_fs and t3_fs must broadcast together")

    def g(x):
        return dephasing_lineshape(x, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)

    g1 = g(t1b)
    g3 = g(t3b)
    gw = g(twb)
    h = -gw + g(t1b + twb) + g(twb + t3b) - g(t1b + twb + t3b)
    return t1b, twb, t3b, g1, g3, h


def nonrephasing_kernel(t1_fs, tw_fs, t3_fs, sigma_donor, sigma_acceptor, tau_fs,
                                tau_static_fs, gap_cm):
    gap = float(gap_cm)
    if not np.isfinite(gap):
        raise ValueError("gap_cm must be finite")
    t1b, twb, t3b, g1, g3, h = _s05_pieces(t1_fs, tw_fs, t3_fs, sigma_donor,
                                               sigma_acceptor, tau_fs, tau_static_fs)
    w = gap * _S05_TWO_PI_C
    return np.exp(-1j * w * (t1b + t3b)) * np.exp(-g1 - g3 + h)


#

import numpy as np


_S06_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def _s06_pieces(t1_fs, tw_fs, t3_fs, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    t1 = np.asarray(t1_fs, dtype=float)
    tw = np.asarray(tw_fs, dtype=float)
    t3 = np.asarray(t3_fs, dtype=float)
    try:
        t1b, twb, t3b = np.broadcast_arrays(t1, tw, t3)
    except ValueError:
        raise ValueError("t1_fs, tw_fs and t3_fs must broadcast together")

    def g(x):
        return dephasing_lineshape(x, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)

    g1 = g(t1b)
    g3 = g(t3b)
    gw = g(twb)
    h = -gw + g(t1b + twb) + g(twb + t3b) - g(t1b + twb + t3b)
    return t1b, twb, t3b, g1, g3, h


def rephasing_kernel(t1_fs, tw_fs, t3_fs, sigma_donor, sigma_acceptor, tau_fs,
                             tau_static_fs, gap_cm):
    gap = float(gap_cm)
    if not np.isfinite(gap):
        raise ValueError("gap_cm must be finite")
    t1b, twb, t3b, g1, g3, h = _s06_pieces(t1_fs, tw_fs, t3_fs, sigma_donor,
                                               sigma_acceptor, tau_fs, tau_static_fs)
    w = gap * _S06_TWO_PI_C
    return np.exp(-1j * w * (t3b - t1b)) * np.exp(-g1 - g3 - h)


#

import numpy as np


_S07_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def _s07_grid(t_max_fs, dt_fs, what):
    tmax, dt = float(t_max_fs), float(dt_fs)
    if not np.isfinite(tmax) or tmax <= 0.0:
        raise ValueError("%s upper limit must be finite and strictly positive" % what)
    if not np.isfinite(dt) or dt <= 0.0 or dt > tmax:
        raise ValueError("%s spacing must be finite, positive and at most its limit" % what)
    return np.linspace(0.0, tmax, int(round(tmax / dt)) + 1)


def waiting_time_profile(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                 gap_cm, j_cm, t_max_fs, dt_fs, tw_max_fs, dtw_fs):
    j = float(j_cm)
    if not np.isfinite(j):
        raise ValueError("j_cm must be finite")
    t = _s07_grid(t_max_fs, dt_fs, "coherence")
    tw = _s07_grid(tw_max_fs, dtw_fs, "waiting")
    t1 = t[:, None, None]
    t3 = t[None, None, :]
    nt = t.size
    chunk = max(1, int(4.0e6 // (nt * nt)))
    out = np.empty(tw.size, dtype=float)
    for a in range(0, tw.size, chunk):
        b = min(a + chunk, tw.size)
        twb = tw[None, a:b, None]
        block = (nonrephasing_kernel(t1, twb, t3, sigma_donor, sigma_acceptor,
                                             tau_fs, tau_static_fs, gap_cm)
                 + rephasing_kernel(t1, twb, t3, sigma_donor, sigma_acceptor,
                                            tau_fs, tau_static_fs, gap_cm))
        out[a:b] = np.trapezoid(np.trapezoid(block, t, axis=2), t, axis=0).real
    j_ang = j * _S07_TWO_PI_C
    return -4.0 * j_ang ** 4 * out * 1.0e6


#

import numpy as np


_S08_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def plateau_constant(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                             gap_cm, j_cm, t_max_fs, dt_fs):
    j = float(j_cm)
    if not np.isfinite(j):
        raise ValueError("j_cm must be finite")
    tmax, dt = float(t_max_fs), float(dt_fs)
    if not np.isfinite(tmax) or tmax <= 0.0:
        raise ValueError("t_max_fs must be finite and strictly positive")
    if not np.isfinite(dt) or dt <= 0.0 or dt > tmax:
        raise ValueError("dt_fs must be finite, strictly positive and at most t_max_fs")
    t = np.linspace(0.0, tmax, int(round(tmax / dt)) + 1)
    resp = second_order_response(t, sigma_donor, sigma_acceptor, tau_fs,
                                         tau_static_fs, gap_cm)
    s = np.trapezoid(resp, t)
    j_ang = j * _S08_TWO_PI_C
    return float(-4.0 * j_ang ** 4 * (s * s + np.abs(s) ** 2).real * 1.0e6)


#

import numpy as np


def fourth_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                              gap_cm, j_cm, t_max_fs, dt_fs, tw_max_fs, dtw_fs):
    profile = waiting_time_profile(sigma_donor, sigma_acceptor, tau_fs,
                                           tau_static_fs, gap_cm, j_cm,
                                           t_max_fs, dt_fs, tw_max_fs, dtw_fs)
    plateau = plateau_constant(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                       gap_cm, j_cm, t_max_fs, dt_fs)
    twmax, dtw = float(tw_max_fs), float(dtw_fs)
    tw_ps = np.linspace(0.0, twmax, int(round(twmax / dtw)) + 1) * 1.0e-3
    return float(np.trapezoid(profile - plateau, tw_ps))


#

import numpy as np


def corrected_transfer_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                    gap_cm, j_cm, t_max_fs, dt_fs, tw_max_fs, dtw_fs,
                                    n_nodes):
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 1:
        raise ValueError("n_nodes must be a positive integer")
    gap0 = float(gap_cm)
    if not np.isfinite(gap0):
        raise ValueError("gap_cm must be finite")
    sigma_static = static_gap_width(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    weights = weights / np.sqrt(2.0 * np.pi)
    total = 0.0
    for x_i, w_i in zip(nodes, weights):
        gap = gap0 + sigma_static * x_i
        k2 = second_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                       gap, j_cm, t_max_fs, dt_fs)
        plateau = plateau_constant(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                           gap, j_cm, t_max_fs, dt_fs)
        expected = -2.0 * k2 * k2
        if abs(plateau - expected) > 1.0e-6 * max(1.0, abs(expected)):
            raise ValueError("plateau does not match -2 k2^2 at gap %g" % gap)
        k4 = fourth_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                       gap, j_cm, t_max_fs, dt_fs, tw_max_fs, dtw_fs)
        total += w_i * (k2 + k4)
    return float(total)


#
SCICODE_GOLD_EOF
