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
from numpy.typing import ArrayLike


def _rs_positive(name, x):
    """Float array of a strictly positive, finite input."""
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return a


def _rs_scalar(name, x, allow_zero=False):
    """Finite scalar that is strictly positive (or non-negative)."""
    if np.ndim(x) != 0:
        raise ValueError(f"{name} must be a scalar")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not np.isfinite(v) or v < 0.0 or (v == 0.0 and not allow_zero):
        kind = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be finite and {kind}")
    return v


def _rs_coupling(beta0, theta0, chi):
    """Radiative coupling constant of the normalised equations."""
    return beta0 / (2.0 * theta0 * theta0 * chi)


def _rs_sinks_kernel(kap, p, q, ap, aq):
    """Sinks for a precomputed per-cell rate factor kap (no validation)."""
    return -ap * kap * p * p, -aq * kap * p * q


def radiative_pressure_sinks(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    alpha_perp: float,
    alpha_par: float,
    beta0: float,
    theta0: float,
    chi: float,
) -> np.ndarray:
    n = _rs_positive("n", n)
    B = _rs_positive("B", B)
    p = _rs_positive("p_perp", p_perp)
    q = _rs_positive("p_par", p_par)
    ap = _rs_scalar("alpha_perp", alpha_perp, allow_zero=True)
    aq = _rs_scalar("alpha_par", alpha_par, allow_zero=True)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    chi = _rs_scalar("chi", chi)
    try:
        n, B, p, q = np.broadcast_arrays(n, B, p, q)
    except ValueError:
        raise ValueError("n, B, p_perp and p_par must broadcast together") from None
    kap = _rs_coupling(beta0, theta0, chi) * B * B / n
    rp, rq = _rs_sinks_kernel(kap, p, q, ap, aq)
    return np.stack((rp, rq))

import math

import numpy as np
from numpy.typing import ArrayLike


def _fr_post_constants():
    """Radiative closure constants (alpha_perp, alpha_par) at and after the onset time."""
    return 0.9, 0.48


def _fr_finite(name, x):
    """Finite real scalar."""
    if np.ndim(x) != 0:
        raise ValueError(f"{name} must be a scalar")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _fr_rate_prefactor(n, B, beta0, theta0, chi):
    """Factor f(n, B) with nubar = f / (2 P_perp + P_par)^(1/3) (no validation)."""
    b0 = math.sqrt(2.0 * theta0 / beta0)
    return chi ** (-1.0 / 3.0) * (B / b0) ** (4.0 / 3.0) * np.cbrt(3.0 * n * theta0)


def _fr_full_rate(n, B, p, q, beta0, theta0, chi):
    """Full scattering rate nubar at the local field and temperature (no validation)."""
    return _fr_rate_prefactor(n, B, beta0, theta0, chi) / np.cbrt(2.0 * p + q)


def _fr_exponents():
    """Fixed adiabatic exponents of (P_perp, P_par) under perpendicular compression."""
    return 8.0 / 5.0, 4.0 / 5.0


def _fr_drive(p, q, B, ux, rp, rq, c_th):
    """Scattering-free rate of change of the threshold excess of a fluid element,
    given the radiative sinks (rp, rq) (no validation)."""
    gp, gq = _fr_exponents()
    return (gp * p - gq * q + c_th * (B * B)) * ux + (rq - rp)


def _fr_switch(d, thr, g, nub):
    """Zero-width switch: nub above the band, 0 below it, clipped balance rate on it."""
    band = 1e-10
    above = d > thr * (1.0 + band)
    below = d < thr * (1.0 - band)
    nu = np.where(above, nub, 0.0)
    on = ~(above | below)
    if np.any(on):
        with np.errstate(divide="ignore", invalid="ignore"):
            nu = np.where(on, np.clip(g / (3.0 * d), 0.0, nub), nu)
    return nu


def regulated_scattering_rate(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    du_dx: ArrayLike,
    t: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    n = _rs_positive("n", n)
    B = _rs_positive("B", B)
    p = _rs_positive("p_perp", p_perp)
    q = _rs_positive("p_par", p_par)
    ux = np.asarray(du_dx, dtype=float)
    if not np.all(np.isfinite(ux)):
        raise ValueError("du_dx must be finite")
    tt = _fr_finite("t", t)
    td = _fr_finite("t_d", t_d)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    chi = _rs_scalar("chi", chi)
    cth = _rs_scalar("c_th", c_th)
    try:
        n, B, p, q, ux = np.broadcast_arrays(n, B, p, q, ux)
    except ValueError:
        raise ValueError("n, B, p_perp, p_par and du_dx must broadcast together") from None
    if tt < td:
        return np.zeros(n.shape)
    ap, aq = _fr_post_constants()
    sinks = radiative_pressure_sinks(n, B, p, q, ap, aq, beta0, theta0, chi)
    g = _fr_drive(p, q, B, ux, sinks[0], sinks[1], cth)
    nub = _fr_full_rate(n, B, p, q, beta0, theta0, chi)
    return _fr_switch(q - p, 0.5 * cth * (B * B), g, nub)

import numpy as np
from numpy.typing import ArrayLike


def _uc_lam(p0, q0, kap, ap, aq, t):
    """Exact scattering-free evolution at fixed n and B."""
    x = kap * ap * p0 * t
    return p0 / (1.0 + x), q0 * np.exp(-(aq / ap) * np.log1p(x))


def _uc_slide(p0, c, kap, ap, aq, t):
    """Exact evolution of P_perp held on the threshold (anisotropy fixed at c)."""
    lam = kap * aq * c / 3.0
    a1 = 1.0 + (2.0 * ap + aq) * p0 / (aq * c)
    return p0 / (1.0 + a1 * np.expm1(lam * t))


def _uc_lam_event(p0, q0, c, kap, ap, aq, T):
    """First time in (0, T] at which a scattering-free cell reaches the threshold
    (arriving below it), or -1 where it does not.

    Newton iteration in y = 1/(1+x), x = kap ap p0 t, on the scattering-free
    threshold excess, which is concave in y for 0 < aq/ap < 1; started at t = 0
    (excess negative and rising) the iterates increase monotonically to the first
    root. The update is written in x to keep full relative precision."""
    th = np.full(p0.shape, -1.0)
    h0 = q0 - p0 - c
    rate = kap * ap * p0
    cand = np.nonzero((h0 < 0.0) & (rate > 0.0) & (ap * p0 > aq * q0))[0]
    if cand.size == 0:
        return th
    p0c, q0c, cc, rc = p0[cand], q0[cand], c[cand], rate[cand]
    r = aq / ap
    x_end = T[cand] * rc
    if 0.0 < r < 1.0:
        x_pk = (ap * p0c / (aq * q0c)) ** (1.0 / (1.0 - r)) - 1.0
        x_end = np.minimum(x_end, x_pk)
    he = q0c * np.exp(-r * np.log1p(x_end)) - p0c / (1.0 + x_end) - cc
    ok = he >= 0.0
    if not ok.any():
        return th
    cand, p0c, q0c, cc, rc, x_end = cand[ok], p0c[ok], q0c[ok], cc[ok], rc[ok], x_end[ok]
    x = np.zeros(cand.size)
    live = np.arange(cand.size)
    for _ in range(200):
        xl = x[live]
        xe = x_end[live]
        s1 = 1.0 + xl
        qp = q0c[live] * np.exp(-r * np.log1p(xl))
        pp = p0c[live] / s1
        h = qp - pp - cc[live]
        hx = (pp - r * qp) / s1
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = h / (s1 * hx)
            xn = (xl - delta) / (1.0 + delta)
        neg = h < 0.0
        inc = hx > 0.0
        prog = neg & inc & (xn > xl)
        cap = prog & (xn >= xe)
        new = np.where(prog & ~cap, xn, xl)
        new = np.where(cap | (neg & ~inc), xe, new)
        x[live] = new
        live = live[prog & ~cap & ((xn - xl) > 4e-16 * xn) & (np.abs(h) > 4e-16 * (qp + pp + cc[live]))]
        if live.size == 0:
            break
    th[cand] = x / rc
    return th


def _uc_rhs(e, dd, kap, ap, aq, pref):
    """Full-rate (above-threshold) right-hand side for E = 2 P_perp + P_par and
    d = P_par - P_perp (no validation)."""
    p = (e - dd) / 3.0
    q = (e + 2.0 * dd) / 3.0
    rp, rq = _rs_sinks_kernel(kap, p, q, ap, aq)
    return 2.0 * rp + rq, rq - rp - 3.0 * (pref / np.cbrt(e)) * dd


def _uc_rk4(E, d, h, kap, ap, aq, pref):
    """One classical RK4 step of the full-rate flow in (E, d)."""
    k1e, k1d = _uc_rhs(E, d, kap, ap, aq, pref)
    hh = 0.5 * h
    k2e, k2d = _uc_rhs(E + hh * k1e, d + hh * k1d, kap, ap, aq, pref)
    k3e, k3d = _uc_rhs(E + hh * k2e, d + hh * k2d, kap, ap, aq, pref)
    k4e, k4d = _uc_rhs(E + h * k3e, d + h * k3d, kap, ap, aq, pref)
    return (E + h * (k1e + 2.0 * k2e + 2.0 * k3e + k4e) / 6.0,
            d + h * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0)


def _uc_act_event(e0, d0, c, kap, ap, aq, pref, hs, d_end):
    """Root s in (0, hs] of d(s) = c for the RK4 map s -> RK4((e0, d0), s), given
    d0 > c >= d(hs) = d_end. Newton iteration with the flow's own rate of d as
    slope, safeguarded by the shrinking bracket. Returns s and E at s."""
    m = e0.size
    a = np.zeros(m)
    b = hs.copy()
    ga = c - d0
    gb = c - d_end
    with np.errstate(divide="ignore", invalid="ignore"):
        s = b - gb * (b - a) / (gb - ga)
    s = np.where((s > a) & (s < b), s, 0.5 * (a + b))
    s_out = hs.copy()
    e_out = np.zeros(m)
    live = np.arange(m)
    for it in range(100):
        sl = s[live]
        e1, d1 = _uc_rk4(e0[live], d0[live], sl, kap[live], ap, aq, pref[live])
        g = c[live] - d1
        rate_d = _uc_rhs(e1, d1, kap[live], ap, aq, pref[live])[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = g / rate_d
        done = (np.abs(g) <= 4e-16 * c[live]) | (np.abs(corr) <= 1e-15 * hs[live])
        if it == 99:
            done[:] = True
        s_out[live[done]] = sl[done]
        e_out[live[done]] = e1[done]
        past = g >= 0.0
        b[live] = np.where(past, sl, b[live])
        a[live] = np.where(past, a[live], sl)
        sn = sl + corr
        al, bl = a[live], b[live]
        s[live] = np.where((sn > al) & (sn < bl), sn, 0.5 * (al + bl))
        live = live[~done]
        if live.size == 0:
            break
    return s_out, e_out


def _uc_active(E, d, c, T, kap, ap, aq, pref, h_act):
    """Full-rate branch for up to time T per cell, stopping where the anisotropy
    falls to the threshold (event located on the RK4 map).
    Returns E, d, time used, hit flag."""
    m = E.size
    E = E.copy()
    d = d.copy()
    t = np.zeros(m)
    used = np.zeros(m)
    hit = np.zeros(m, dtype=bool)
    p = (E - d) / 3.0
    q = (E + 2.0 * d) / 3.0
    nb = pref / np.cbrt(E)
    hs = h_act / (3.0 * nb + kap * (ap + aq) * p + 1e-300)
    rp, rq = _rs_sinks_kernel(kap, p, q, ap, aq)
    den = 3.0 * nb * d - (rq - rp)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_est = np.where(den > 0.0, (d - c) / den, np.inf)
    shorten = (den > 0.0) & (1.5 * t_est > 0.0) & (1.5 * t_est < hs)
    hs = np.where(shorten, 1.5 * t_est, hs)
    hs = np.where(hs <= 0.0, T, hs)
    run = np.arange(m)
    while run.size:
        er, dr, tr, big_t = E[run], d[run], t[run], T[run]
        hr = hs[run]
        last = hr >= big_t - tr
        hr = np.where(last, big_t - tr, hr)
        e1, d1 = _uc_rk4(er, dr, hr, kap[run], ap, aq, pref[run])
        cross = (d1 - c[run]) <= 0.0
        if cross.any():
            ci = run[cross]
            s_ev, e_ev = _uc_act_event(er[cross], dr[cross], c[ci], kap[ci], ap, aq, pref[ci],
                                       hr[cross], d1[cross])
            E[ci] = e_ev
            d[ci] = c[ci]
            used[ci] = t[ci] + s_ev
            hit[ci] = True
        keep = ~cross
        ni = run[keep]
        E[ni] = e1[keep]
        d[ni] = d1[keep]
        fin = last[keep]
        used[ni[fin]] = T[ni[fin]]
        cont = ni[~fin]
        if cont.size:
            t[cont] = t[cont] + hr[keep][~fin]
            pn = (E[cont] - d[cont]) / 3.0
            nbn = pref[cont] / np.cbrt(E[cont])
            hs[cont] = h_act / (3.0 * nbn + kap[cont] * (ap + aq) * pn + 1e-300)
        run = cont
    return E, d, used, hit


def _uc_on_rates(pp, qq, kap, b, c, pref, ap, aq, cth):
    """Step-04 rate at du_dx = 0 on its post-onset branch (the same helpers as its
    oracle) and the full rate nubar, for states at or near the threshold."""
    rp, rq = _rs_sinks_kernel(kap, pp, qq, ap, aq)
    g = _fr_drive(pp, qq, b, 0.0, rp, rq, cth)
    nub = pref / np.cbrt(2.0 * pp + qq)
    return _fr_switch(qq - pp, c, g, nub), nub


def _uc_post_flow(p, q, tau, kap, c, pref, b, ap, aq, cth, h_act):
    """Post-onset local flow over the interval tau (1-D arrays). Branch codes:
    1 = no scattering, 2 = full rate, 4 = held on the threshold; at most 16
    branch segments per cell (the remainder, never reached in practice, is
    integrated without scattering)."""
    lam_br, act_br, slide_br = 1, 2, 4
    maxseg = 16
    E = 2.0 * p + q
    d = q - p
    t_rem = np.full(E.size, float(tau))
    force = np.zeros(E.size, dtype=np.int64)
    tol_on = 1e-12 * (np.abs(E) + c)
    live = np.nonzero(t_rem > 0.0)[0]
    nseg = 0
    while live.size:
        if nseg >= maxseg:
            p0 = (E[live] - d[live]) / 3.0
            q0 = (E[live] + 2.0 * d[live]) / 3.0
            p1, q1 = _uc_lam(p0, q0, kap[live], ap, aq, t_rem[live])
            E[live] = 2.0 * p1 + q1
            d[live] = q1 - p1
            break
        nseg += 1
        h = d[live] - c[live]
        fl = force[live]
        br = np.where(h > 0.0, act_br, lam_br)
        on = (fl == 0) & (np.abs(h) <= tol_on[live])
        if on.any():
            oi = live[on]
            d[oi] = c[oi]
            pp = (E[oi] - c[oi]) / 3.0
            qq = pp + c[oi]
            nu_on, nb_on = _uc_on_rates(pp, qq, kap[oi], b[oi], c[oi], pref[oi], ap, aq, cth)
            br[on] = np.where(nu_on <= 0.0, lam_br, np.where(nu_on >= nb_on, act_br, slide_br))
        forced = fl != 0
        if forced.any():
            br[forced] = fl[forced]
            force[live] = 0

        sel = live[br == lam_br]
        if sel.size:
            p0 = (E[sel] - d[sel]) / 3.0
            q0 = (E[sel] + 2.0 * d[sel]) / 3.0
            tr = t_rem[sel]
            th = _uc_lam_event(p0, q0, c[sel], kap[sel], ap, aq, tr)
            hit = (th > 0.0) & (th < tr)
            tseg = np.where(hit, th, tr)
            p1, q1 = _uc_lam(p0, q0, kap[sel], ap, aq, tseg)
            E[sel] = 2.0 * p1 + q1
            d[sel] = np.where(hit, c[sel], q1 - p1)
            t_rem[sel] = np.where(hit, tr - tseg, 0.0)

        sel = live[br == slide_br]
        if sel.size:
            cs = c[sel]
            ks = kap[sel]
            tr = t_rem[sel]
            p0 = (E[sel] - cs) / 3.0
            t_ev = np.full(sel.size, np.inf)
            is_rel = np.zeros(sel.size, dtype=bool)
            if ap > aq and aq > 0.0:
                p_rel = aq * cs / (ap - aq)
                lam = ks * aq * cs / 3.0
                is_rel = (p0 > p_rel) & (lam > 0.0)
                if is_rel.any():
                    a1 = 1.0 + (2.0 * ap + aq) * p0 / (aq * cs)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        tev = np.log1p((p0 / p_rel - 1.0) / a1) / lam
                    t_ev = np.where(is_rel, tev, np.inf)
            tseg = np.minimum(tr, t_ev)
            p1 = _uc_slide(p0, cs, ks, ap, aq, tseg)
            nu_e, nb_e = _uc_on_rates(p1, p1 + cs, ks, b[sel], cs, pref[sel], ap, aq, cth)
            capped = nu_e >= nb_e
            if capped.any():
                ci = np.nonzero(capped)[0]
                lo = np.zeros(ci.size)
                hi = tseg[ci].copy()
                tol = 1e-15 * tseg[ci]
                run = np.arange(ci.size)
                for _ in range(200):
                    if run.size == 0:
                        break
                    cr = ci[run]
                    mid = 0.5 * (lo[run] + hi[run])
                    pm = _uc_slide(p0[cr], cs[cr], ks[cr], ap, aq, mid)
                    sc = sel[cr]
                    num, nbm = _uc_on_rates(pm, pm + cs[cr], kap[sc], b[sc], c[sc], pref[sc], ap, aq, cth)
                    g = num >= nbm
                    hi[run] = np.where(g, mid, hi[run])
                    lo[run] = np.where(g, lo[run], mid)
                    run = run[(hi[run] - lo[run]) > tol[run]]
                tseg[ci] = hi
                p1[ci] = _uc_slide(p0[ci], cs[ci], ks[ci], ap, aq, hi)
            E[sel] = 3.0 * p1 + cs
            d[sel] = cs
            left = tseg < tr
            force[sel] = np.where(capped, act_br, np.where(left & is_rel, lam_br, 0))
            t_rem[sel] = np.where(capped | left, tr - tseg, 0.0)

        sel = live[br == act_br]
        if sel.size:
            tr = t_rem[sel]
            e1, d1, used, hit = _uc_active(E[sel], d[sel], c[sel], tr, kap[sel], ap, aq, pref[sel], h_act)
            E[sel] = e1
            d[sel] = np.where(hit, c[sel], d1)
            t_rem[sel] = np.where(hit & (used < tr), tr - used, 0.0)

        tr = t_rem[live]
        tr = np.where(tr <= 1e-15 * tau, 0.0, tr)
        t_rem[live] = tr
        live = live[tr > 0.0]
    return (E - d) / 3.0, (E + 2.0 * d) / 3.0


def advance_uniform_cells(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    t_start: float,
    t_end: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    n = _rs_positive("n", n)
    B = _rs_positive("B", B)
    p = _rs_positive("p_perp", p_perp)
    q = _rs_positive("p_par", p_par)
    t0 = _fr_finite("t_start", t_start)
    t1 = _fr_finite("t_end", t_end)
    td = _fr_finite("t_d", t_d)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    chi = _rs_scalar("chi", chi)
    cth = _rs_scalar("c_th", c_th)
    if t1 < t0:
        raise ValueError("t_end must not be earlier than t_start")
    try:
        n, B, p, q = np.broadcast_arrays(n, B, p, q)
    except ValueError:
        raise ValueError("n, B, p_perp and p_par must broadcast together") from None
    shape = n.shape
    n1 = np.array(n, dtype=float).ravel()
    b1 = np.array(B, dtype=float).ravel()
    pa = np.array(p, dtype=float).ravel()
    qa = np.array(q, dtype=float).ravel()
    if n1.size == 0:
        return np.zeros((2,) + shape)
    kap = -radiative_pressure_sinks(n1, b1, 1.0, 1.0, 1.0, 0.0, beta0, theta0, chi)[0]
    t_pre_end = min(t1, td)
    if t0 < t_pre_end:
        pa, qa = _uc_lam(pa, qa, kap, 16.0 / 15.0, 8.0 / 15.0, t_pre_end - t0)
    t_post0 = max(t0, td)
    if t1 > t_post0:
        ap, aq = _fr_post_constants()
        c = 0.5 * cth * (b1 * b1)
        pref = _fr_rate_prefactor(n1, b1, beta0, theta0, chi)
        # full-rate branch: classical RK4 with steps h_act / (local stiffness rate)
        pa, qa = _uc_post_flow(pa, qa, t1 - t_post0, kap, c, pref, b1, ap, aq, cth, h_act=0.005)
    return np.stack((pa.reshape(shape), qa.reshape(shape)))

import math

import numpy as np
from numpy.typing import ArrayLike


def _pb_vec(name, x):
    """One-dimensional array of finite, strictly positive entries."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size < 1:
        raise ValueError(f"{name} must be a one-dimensional array with at least one entry")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return a


def _pb_factors(p, c, pi):
    """Compression factors s > 0 with p s^(8/5) + c s^2 = pi, by Newton from s = 1
    (the left side is convex and increasing in s, so the iterates stay positive)."""
    gp, _ = _fr_exponents()
    s = np.ones_like(p)
    fp = gp * p + 2.0 * c
    for _ in range(100):
        f = p * s ** gp + c * s * s - pi
        fp = gp * p * s ** (gp - 1.0) + 2.0 * c * s
        ds = f / fp
        s = s - ds
        if np.max(np.abs(ds) / s) < 1e-15:
            break
    return s, fp


def _pb_project(mass, n, p, q, length, b0):
    """Pressure-balanced state without validation; returns n, P_perp, P_par and the
    common total pressure."""
    gp, gq = _fr_exponents()
    c = 0.5 * (b0 * n) ** 2
    vol = mass / n
    pi = float(np.sum(vol * (p + c)) / length)
    for _ in range(200):
        s, fp = _pb_factors(p, c, pi)
        g = float(np.sum(vol / s)) - length
        dg = -float(np.sum(vol / (s * s) / fp))
        dpi = g / dg
        pi -= dpi
        if abs(dpi) <= 1e-15 * pi:
            break
    s, _ = _pb_factors(p, c, pi)
    return n * s, p * s ** gp, q * s ** gq, pi


def _pb_state_args(mass, n, p_perp, p_par):
    """Validated element arrays of equal length."""
    m = _pb_vec("mass", mass)
    nn = _pb_vec("n", n)
    p = _pb_vec("p_perp", p_perp)
    q = _pb_vec("p_par", p_par)
    if not (m.size == nn.size == p.size == q.size):
        raise ValueError("mass, n, p_perp and p_par must have the same length")
    return m, nn, p, q


def pressure_balanced_state(
    mass: ArrayLike,
    n: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    L: float,
    beta0: float,
    theta0: float,
) -> np.ndarray:
    m, nn, p, q = _pb_state_args(mass, n, p_perp, p_par)
    if m.size < 1:
        raise ValueError("the slab needs at least one element")
    length = _rs_scalar("L", L)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    b0 = math.sqrt(2.0 * theta0 / beta0)
    n1, p1, q1, pi = _pb_project(m, nn, p, q, length, b0)
    return np.stack((n1, p1, q1, np.full(n1.size, pi)))

import math

import numpy as np
from numpy.typing import ArrayLike


def pressure_balanced_step(
    mass: ArrayLike,
    n: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    t: float,
    dt: float,
    L: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    m, nn, p, q = _pb_state_args(mass, n, p_perp, p_par)
    t0 = _fr_finite("t", t)
    h = _fr_finite("dt", dt)
    if h < 0.0:
        raise ValueError("dt must be non-negative")
    length = _rs_scalar("L", L)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    chi = _rs_scalar("chi", chi)
    td = _fr_finite("t_d", t_d)
    cth = _rs_scalar("c_th", c_th)
    b0 = math.sqrt(2.0 * theta0 / beta0)
    half = 0.5 * h
    pq = advance_uniform_cells(nn, b0 * nn, p, q, t0, t0 + half, beta0, theta0, chi, td, cth)
    n1, p1, q1, pi = _pb_project(m, nn, pq[0], pq[1], length, b0)
    pq = advance_uniform_cells(n1, b0 * n1, p1, q1, t0 + half, t0 + h, beta0, theta0, chi, td, cth)
    return np.stack((n1, pq[0], pq[1], np.full(n1.size, pi)))

import math

import numpy as np
from numpy.typing import ArrayLike


def evolve_pressure_balanced_slab(
    N: int,
    t_end: float,
    L: float,
    seed: ArrayLike,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
    dt_max: float = 0.5,
) -> np.ndarray:
    if isinstance(N, (bool, np.bool_)) or np.ndim(N) != 0:
        raise ValueError("N must be an integer >= 4")
    try:
        nf = float(N)
    except (TypeError, ValueError):
        raise ValueError("N must be an integer >= 4") from None
    if not np.isfinite(nf) or nf != math.floor(nf) or nf < 4:
        raise ValueError("N must be an integer >= 4")
    ncell = int(nf)
    tf = _fr_finite("t_end", t_end)
    if tf < 0.0:
        raise ValueError("t_end must be non-negative")
    length = _rs_scalar("L", L)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    chi = _rs_scalar("chi", chi)
    td = _fr_finite("t_d", t_d)
    cth = _rs_scalar("c_th", c_th)
    dtm = _rs_scalar("dt_max", dt_max)
    try:
        sd = np.array(seed, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("seed must be a numeric (M, 3) array") from None
    if sd.ndim != 2 or sd.shape[1] != 3 or sd.shape[0] < 1 or not np.all(np.isfinite(sd)):
        raise ValueError("seed must be a finite (M, 3) array with M >= 1")
    if np.any(sd[:, 1] != np.round(sd[:, 1])):
        raise ValueError("seed wavenumber indices must be integers")

    x = (np.arange(ncell, dtype=float) + 0.5) * (length / ncell)
    b = np.ones(ncell)
    for amp, k, ph in sd:
        b = b + amp * np.cos(2.0 * np.pi * k * x / length + ph)
    if np.any(b <= 0.0):
        raise ValueError("the seeded profile b must be positive for every element")
    p = theta0 * (1.0 + (1.0 - b * b) / beta0)
    if np.any(p <= 0.0):
        raise ValueError("the initial pressures must be positive for every element")
    n = b.copy()
    mass = n * (length / ncell)
    q = p.copy()
    b0 = math.sqrt(2.0 * theta0 / beta0)
    n, p, q, _ = _pb_project(mass, n, p, q, length, b0)

    marks = [td, tf] if 0.0 < td < tf else [tf]
    a = 0.0
    for mark in marks:
        span = mark - a
        if span > 0.0:
            k = int(math.ceil(span / dtm - 1e-9))
            h = span / k
            for j in range(k):
                st = pressure_balanced_step(mass, n, p, q, a + j * h, h, length, beta0,
                                                    theta0, chi, td, cth)
                n, p, q = st[0], st[1], st[2]
        a = mark
    return np.stack((n, p, q, mass / n))

import math

import numpy as np
from numpy.typing import ArrayLike


def anisotropy_shortfall_measure(
    N: int = 512,
    t_end: float = 16000.0,
    L: float = 1.0,
    beta0: float = 80.0,
    theta0: float = 1000.0,
    chi: float = 2000.0,
    t_d: float = 340.0,
    c_th: float = 1.4,
    dt_max: float = 0.5,
) -> float:
    seed = np.array([[-0.28, 1.0, 0.0], [0.05, 2.0, math.pi / 3.0]])
    n, p, q, w = evolve_pressure_balanced_slab(N, t_end, L, seed, beta0, theta0, chi, t_d,
                                                       c_th, dt_max)
    length = float(L)
    tf = float(t_end)
    if abs(float(np.sum(w)) - length) > 1e-9 * length:
        raise ValueError("consistency gate: the element widths no longer fill the slab")
    mass = n * w
    b0 = math.sqrt(2.0 * float(theta0) / float(beta0))
    B = b0 * n
    bal = pressure_balanced_state(mass, n, p, q, L, beta0, theta0)
    if np.max(np.abs(bal[0] / n - 1.0)) > 1e-2:
        raise ValueError("consistency gate: the final state is far from pressure balance")
    zero = pressure_balanced_step(mass, n, p, q, tf, 0.0, L, beta0, theta0, chi, t_d, c_th)
    if np.max(np.abs(zero[:3] / bal[:3] - 1.0)) > 1e-12:
        raise ValueError("consistency gate: a zero-length step differs from the balanced state")
    ap, aq = (0.9, 0.48) if tf >= float(t_d) else (16.0 / 15.0, 8.0 / 15.0)
    sinks = radiative_pressure_sinks(n, B, p, q, ap, aq, beta0, theta0, chi)
    if not np.all(2.0 * sinks[0] + sinks[1] < 0.0):
        raise ValueError("consistency gate: radiation does not remove energy")
    thr = 0.5 * float(c_th) * B * B
    nu = regulated_scattering_rate(n, B, p, q, 0.0, tf, beta0, theta0, chi, t_d, c_th)
    below = (q - p) < thr * (1.0 - 1e-10)
    if np.any(nu[below] != 0.0):
        raise ValueError("consistency gate: an element below the threshold is scattering")
    later = advance_uniform_cells(n, B, p, q, tf, tf + 1.0, beta0, theta0, chi, t_d, c_th)
    if not np.all(2.0 * later[0] + later[1] <= (2.0 * p + q) * (1.0 + 1e-12)):
        raise ValueError("consistency gate: the local advance creates energy")
    return float(np.sum(w * np.maximum(0.0, 1.0 - (q - p) / thr) ** 2) / length)
SCICODE_GOLD_EOF
