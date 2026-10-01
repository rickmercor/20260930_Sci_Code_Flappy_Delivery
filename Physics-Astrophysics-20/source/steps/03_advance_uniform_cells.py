"""
Exact local evolution of the two pressures in uniform cells.

Exact local evolution of the two pressures in uniform cells.

A cell at rest with no gradients keeps its density and magnetic field; its two
pressures change only through the radiative sinks of step 01 and the
firehose-regulated scattering of step 02 at zero velocity gradient, under the
coefficient schedule of step 02. Because the scattering rate is switched by the
threshold in its zero-width limit, this local initial-value problem has a
piecewise-smooth solution. This step returns that solution at a later time to a
relative accuracy of 1e-11 or better. It is the local (source) part of the slab
evolution of steps 05 and 06.

Returns
-------
The two pressures of every cell at the end time.

Returns
-------
The two pressures of every cell at the end time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Return the two pressures of uniform cells at rest after local evolution.

    Each cell keeps its density ``n`` and field ``B``; only its pressures
    evolve, by the radiative sinks of step 01 plus the regulated scattering of
    step 02 evaluated with ``du_dx = 0``, under the schedule of step 02. The
    coefficient switch happens exactly at ``t_d``: the part of
    [t_start, t_end] before t_d follows the pre-onset rules, the part at or
    after t_d the post-onset rules. The zero-width switch of step 02 is
    followed exactly: a cell on the threshold stays on it while the step-02
    on-threshold rate lies strictly between 0 and nubar; it leaves the
    threshold without scattering once that rate falls to 0, and with the full
    rate nubar once the rate would exceed nubar. A cell whose on-threshold rate
    is exactly 0 evolves without scattering; one whose rate equals nubar
    evolves with the full rate. A cell that reaches the threshold from either
    side continues by the same rule. Input states within
    1e-12 (2 P_perp + P_par + c_th B^2/2) of the threshold count as on it.

    Parameters
    ----------
    n : float or array_like
        Number density of each cell (units n0), constant in time. Strictly
        positive.
    B : float or array_like
        Field strength of each cell (units sqrt(4 pi n0 m_e c^2)), constant in
        time. Strictly positive.
    p_perp : float or array_like
        Perpendicular pressure at ``t_start`` (units n0 m_e c^2). Strictly
        positive.
    p_par : float or array_like
        Parallel pressure at ``t_start`` (units n0 m_e c^2). Strictly
        positive.
    t_start : float
        Start time (units 1/Omega0).
    t_end : float
        End time (units 1/Omega0); not earlier than ``t_start``.
    beta0 : float
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.
    chi : float
        Reference product tau0 Omega0 (step 01). Strictly positive.
    t_d : float
        Onset time of step 02 (units 1/Omega0).
    c_th : float, optional
        Firehose threshold constant of step 02. Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2,) + S``, with ``S`` the broadcast shape of ``n``,
        ``B``, ``p_perp`` and ``p_par``. Row 0 is P_perp(t_end) and row 1 is
        P_par(t_end), the exact solution of the local initial-value problem to
        a relative accuracy of 1e-11 or better.

    Raises
    ------
    ValueError
        If ``n``, ``B``, ``p_perp`` or ``p_par`` has a non-finite or
        non-positive entry, if these four do not broadcast together, if
        ``t_start``, ``t_end`` or ``t_d`` is not a finite scalar, if
        ``t_end < t_start``, or if ``beta0``, ``theta0``, ``chi`` or ``c_th``
        is not a finite, strictly positive scalar.
    """
    return pressures  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_advance_uniform_cells(
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
    kap = -_oracle_radiative_pressure_sinks(n1, b1, 1.0, 1.0, 1.0, 0.0, beta0, theta0, chi)[0]
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return cases with independent candidate and oracle argument graphs."""
    return [{'setup': 'import numpy as np\n'
               'b0 = 0.2236067977499790\n'
               'n = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.8])\n'
               'B = np.array([b0, b0, 1.0, b0, b0, 0.3])\n'
               'p = np.array([1.0, 1.0, 0.85, 1.0, 0.02, 0.6])\n'
               'q = np.array([1.1, 0.0, 0.0, 1.03, 0.04, 0.7])\n'
               'q[1] = p[1] + 0.7 * B[1] * B[1]\n'
               'q[2] = p[2] + 0.7 * B[2] * B[2]\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(advance_uniform_cells, n, B, p, q, 100.0, 130.0, 40.0, 1.0, 2000.0, '
              '100.0)',
      'gold_call': '_independent_call(_oracle_advance_uniform_cells, n, B, p, q, 100.0, 130.0, 40.0, '
                   '1.0, 2000.0, 100.0)'},
     {'setup': 'import numpy as np\n'
               'b0 = 0.2236067977499790\n'
               'n = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.8])\n'
               'B = np.array([b0, b0, 1.0, b0, b0, 0.3])\n'
               'p = np.array([1.0, 1.0, 0.85, 1.0, 0.02, 0.6])\n'
               'q = np.array([1.1, 0.0, 0.0, 1.03, 0.04, 0.7])\n'
               'q[1] = p[1] + 0.7 * B[1] * B[1]\n'
               'q[2] = p[2] + 0.7 * B[2] * B[2]\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(advance_uniform_cells, n, B, p, q, 90.0, 125.0, 40.0, 1.0, 2000.0, '
              '100.0)',
      'gold_call': '_independent_call(_oracle_advance_uniform_cells, n, B, p, q, 90.0, 125.0, 40.0, '
                   '1.0, 2000.0, 100.0)'},
     {'setup': 'import numpy as np\n'
               'b0 = 0.2236067977499790\n'
               'n = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.8])\n'
               'B = np.array([b0, b0, 1.0, b0, b0, 0.3])\n'
               'p = np.array([1.0, 1.0, 0.85, 1.0, 0.02, 0.6])\n'
               'q = np.array([1.1, 0.0, 0.0, 1.03, 0.04, 0.7])\n'
               'q[1] = p[1] + 0.7 * B[1] * B[1]\n'
               'q[2] = p[2] + 0.7 * B[2] * B[2]\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(advance_uniform_cells, n, B, p, q, 0.0, 100.0, 40.0, 1.0, 2000.0, '
              '100.0)',
      'gold_call': '_independent_call(_oracle_advance_uniform_cells, n, B, p, q, 0.0, 100.0, 40.0, '
                   '1.0, 2000.0, 100.0)'},
     {'setup': 'import numpy as np\n'
               'n = np.array([[1.0], [0.6]])\n'
               'B = np.array([[1.0, 1.4, 0.5]])\n'
               'p = np.array([[0.85, 2.0, 0.3]])\n'
               'q = p + 0.7 * B * B\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(advance_uniform_cells, n, B, p, q, 5.0, 65.0, 40.0, 1.0, 2000.0, 0.0)',
      'gold_call': '_independent_call(_oracle_advance_uniform_cells, n, B, p, q, 5.0, 65.0, 40.0, 1.0, '
                   '2000.0, 0.0)'},
     {'setup': 'import numpy as np\n'
               'b0 = 0.2236067977499790\n'
               'n = np.array([1.0, 0.7, 1.2])\n'
               'B = b0 * np.array([1.0, 1.5, 0.8])\n'
               'p = np.array([1.0, 0.5, 1.3])\n'
               'q = np.array([1.6, 0.9, 1.5])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(advance_uniform_cells, n, B, p, q, 0.0, 12.0, 40.0, 1.0, 20.0, 2.0)',
      'gold_call': '_independent_call(_oracle_advance_uniform_cells, n, B, p, q, 0.0, 12.0, 40.0, 1.0, '
                   '20.0, 2.0)'},
     {'setup': 'import numpy as np\n'
               'def _probe_call():\n'
               '    try:\n'
               '        advance_uniform_cells(1.0, 0.2236, 1.0, 1.05, 10.0, 9.0, 40.0, 1.0, 2000.0, '
               '0.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def _probe_gold():\n'
               '    try:\n'
               '        _oracle_advance_uniform_cells(1.0, 0.2236, 1.0, 1.05, 10.0, 9.0, 40.0, 1.0, '
               '2000.0, 0.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_call()',
      'gold_call': '_probe_gold()'},
     {'setup': 'import numpy as np\n'
               'e = np.array([])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(advance_uniform_cells, e, e, e, e, 0.0, 10.0, 40.0, 1.0, 2000.0, '
              '340.0)',
      'gold_call': '_independent_call(_oracle_advance_uniform_cells, e, e, e, e, 0.0, 10.0, 40.0, 1.0, '
                   '2000.0, 340.0)'}]
