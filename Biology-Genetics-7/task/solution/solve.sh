#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def allele_orientations(bx, by, eaf):
    bx = bx.reshape(-1)
    by = by.reshape(-1)
    eaf = eaf.reshape(-1)
    sm = 1.0 - 2.0 * (eaf < 0.5)
    sn = 1.0 - 2.0 * (bx < 0.0)
    g_m = bx * sm
    G_m = by * sm
    g_n = bx * sn
    G_n = by * sn
    out = type(bx)((4, bx.shape[0]))
    out[0] = g_m
    out[1] = G_m
    out[2] = g_n
    out[3] = G_n
    return out

def rivw_slope(bx, by, sx, sy, z, lam, eta):
    lib = __import__(type(bx).__module__)
    from math import erf, exp, pi, sqrt

    _SQRT2 = sqrt(2.0)
    _SQRT2PI = sqrt(2.0 * pi)
    _phi = lambda x: exp(-0.5 * x * x) / _SQRT2PI
    _Phi = lambda x: 0.5 * (1.0 + erf(x / _SQRT2))

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sx = lib.asarray(sx, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    z = lib.asarray(z, dtype=lib.float64).reshape(-1)
    sel = lib.abs(bx / sx + z) - lam > 0.0
    g_rb = lib.zeros_like(bx)
    s2_rb = lib.zeros_like(bx)
    for j in range(bx.size):
        if not sel[j]:
            continue
        ap = -bx[j] / (sx[j] * eta) + lam / eta
        am = -bx[j] / (sx[j] * eta) - lam / eta
        den = 1.0 - _Phi(ap) + _Phi(am)
        if den < 1e-15:
            g_rb[j] = bx[j]
            s2_rb[j] = sx[j] ** 2
            continue
        num = _phi(ap) - _phi(am)
        g_rb[j] = bx[j] - (sx[j] / eta) * (num / den)
        s2_rb[j] = sx[j] ** 2 * (
            1.0
            - (1.0 / eta**2) * (ap * _phi(ap) - am * _phi(am)) / den
            + (1.0 / eta**2) * (num / den) ** 2
        )
        s2_rb[j] = max(s2_rb[j], 1e-18)
    w = lib.zeros_like(bx)
    w[sel] = 1.0 / lib.maximum(sy[sel] ** 2, 1e-30)
    den = lib.sum(w * (g_rb**2 - s2_rb))
    if abs(den) < 1e-18:
        return 0.0
    return float(lib.sum(w * g_rb * by) / den)

def mei_z_score(bx, by, sx, sy, z, lam, eta):
    lib = __import__(type(bx).__module__)
    from math import erf, exp, pi, sqrt

    _SQRT2 = sqrt(2.0)
    _SQRT2PI = sqrt(2.0 * pi)
    _phi = lambda x: exp(-0.5 * x * x) / _SQRT2PI
    _Phi = lambda x: 0.5 * (1.0 + erf(x / _SQRT2))

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sx = lib.asarray(sx, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    z = lib.asarray(z, dtype=lib.float64).reshape(-1)
    sel = lib.abs(bx / sx + z) - lam > 0.0
    g_rb = lib.zeros_like(bx)
    s2_rb = lib.zeros_like(bx)
    for j in range(bx.size):
        if not sel[j]:
            continue
        ap = -bx[j] / (sx[j] * eta) + lam / eta
        am = -bx[j] / (sx[j] * eta) - lam / eta
        den = 1.0 - _Phi(ap) + _Phi(am)
        if den < 1e-15:
            g_rb[j] = bx[j]
            s2_rb[j] = sx[j] ** 2
            continue
        num = _phi(ap) - _phi(am)
        g_rb[j] = bx[j] - (sx[j] / eta) * (num / den)
        s2_rb[j] = sx[j] ** 2 * (
            1.0
            - (1.0 / eta**2) * (ap * _phi(ap) - am * _phi(am)) / den
            + (1.0 / eta**2) * (num / den) ** 2
        )
        s2_rb[j] = max(s2_rb[j], 1e-18)
    idx = lib.where(sel)[0]
    if idx.size < 2:
        return 0.0
    w = 1.0 / lib.maximum(sy[idx] ** 2, 1e-30)
    g = g_rb[idx]
    G = by[idx]
    s2 = s2_rb[idx]
    ww = lib.zeros_like(bx)
    ww[sel] = 1.0 / lib.maximum(sy[sel] ** 2, 1e-30)
    den = lib.sum(ww * (g_rb**2 - s2_rb))
    beta_r = float(lib.sum(ww * g_rb * by) / den) if abs(den) > 1e-18 else 0.0
    sw_g2 = lib.sum(w * (g**2 - s2))
    sw_gG = lib.sum(w * g * G)
    sw_G = lib.sum(w * G)
    sw_g = lib.sum(w * g)
    lam_rc = sw_g2 * sw_G - sw_gG * sw_g + lib.sum((w**2) * s2 * G)
    u = (G - beta_r * g) * sw_g2 - (g * G - beta_r * (g**2 - s2)) * sw_g
    v = lib.sum((w**2) * u**2)
    if v <= 0.0:
        return 0.0
    return float(lam_rc / sqrt(v))

def mei_combined_stat(bx, by, sx, sy, eaf, z, lam, eta):
    lib = __import__(type(bx).__module__)
    from math import erf, exp, pi, sqrt

    _SQRT2 = sqrt(2.0)
    _SQRT2PI = sqrt(2.0 * pi)
    _phi = lambda x: exp(-0.5 * x * x) / _SQRT2PI
    _Phi = lambda x: 0.5 * (1.0 + erf(x / _SQRT2))

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sx = lib.asarray(sx, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    eaf = lib.asarray(eaf, dtype=lib.float64).reshape(-1)
    z = lib.asarray(z, dtype=lib.float64).reshape(-1)
    g_m = lib.where(eaf < 0.5, -bx, bx)
    G_m = lib.where(eaf < 0.5, -by, by)
    z_m = lib.where(eaf < 0.5, -z, z)
    g_n = lib.where(bx < 0.0, -bx, bx)
    G_n = lib.where(bx < 0.0, -by, by)
    z_n = lib.where(bx < 0.0, -z, z)

    zs = []
    for gx, Gy, zx in ((g_m, G_m, z_m), (g_n, G_n, z_n)):
        sel = lib.abs(gx / sx + zx) - lam > 0.0
        g_rb = lib.zeros_like(gx)
        s2_rb = lib.zeros_like(gx)
        for j in range(gx.size):
            if not sel[j]:
                continue
            ap = -gx[j] / (sx[j] * eta) + lam / eta
            am = -gx[j] / (sx[j] * eta) - lam / eta
            den = 1.0 - _Phi(ap) + _Phi(am)
            if den < 1e-15:
                g_rb[j] = gx[j]
                s2_rb[j] = sx[j] ** 2
                continue
            num = _phi(ap) - _phi(am)
            g_rb[j] = gx[j] - (sx[j] / eta) * (num / den)
            s2_rb[j] = sx[j] ** 2 * (
                1.0
                - (1.0 / eta**2) * (ap * _phi(ap) - am * _phi(am)) / den
                + (1.0 / eta**2) * (num / den) ** 2
            )
            s2_rb[j] = max(s2_rb[j], 1e-18)
        idx = lib.where(sel)[0]
        if idx.size < 2:
            zs.append(0.0)
            continue
        w = 1.0 / lib.maximum(sy[idx] ** 2, 1e-30)
        g = g_rb[idx]
        G = Gy[idx]
        s2 = s2_rb[idx]
        ww = lib.zeros_like(gx)
        ww[sel] = 1.0 / lib.maximum(sy[sel] ** 2, 1e-30)
        den = lib.sum(ww * (g_rb**2 - s2_rb))
        beta_r = float(lib.sum(ww * g_rb * Gy) / den) if abs(den) > 1e-18 else 0.0
        sw_g2 = lib.sum(w * (g**2 - s2))
        sw_gG = lib.sum(w * g * G)
        lam_rc = sw_g2 * lib.sum(w * G) - sw_gG * lib.sum(w * g) + lib.sum((w**2) * s2 * G)
        u = (G - beta_r * g) * sw_g2 - (g * G - beta_r * (g**2 - s2)) * lib.sum(w * g)
        v = lib.sum((w**2) * u**2)
        if v <= 0.0:
            zs.append(0.0)
        else:
            zs.append(float(lam_rc / sqrt(v)))
    return float(max(abs(zs[0]), abs(zs[1])))

def adaptive_alasso_weights(bx, by, nu):
    lib = __import__(type(bx).__module__)

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    theta0 = float(lib.median(by / bx))
    alpha0 = by - theta0 * bx
    return 1.0 / lib.maximum(lib.abs(alpha0), 1e-12) ** nu

def alasso_valid_flags(bx, by, sy, lam_n, nu):
    lib = __import__(type(bx).__module__)

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    w = 1.0 / lib.maximum(sy**2, 1e-30)
    theta0 = float(lib.median(by / bx))
    omega = 1.0 / lib.maximum(lib.abs(by - theta0 * bx), 1e-12) ** nu
    alpha = lib.zeros_like(bx)
    theta = theta0
    for _ in range(400):
        resid = by - theta * bx
        thresh = lam_n * omega / w
        alpha_new = lib.sign(resid) * lib.maximum(lib.abs(resid) - thresh, 0.0)
        den = lib.sum(w * bx**2)
        theta_new = float(lib.sum(w * bx * (by - alpha_new)) / den) if abs(den) > 1e-18 else theta
        if lib.max(lib.abs(alpha_new - alpha)) < 1e-12 and abs(theta_new - theta) < 1e-12:
            alpha = alpha_new
            break
        alpha, theta = alpha_new, theta_new
    return (lib.abs(alpha) <= 1e-10).astype(lib.float64)

def ratio_invse_weights(bx, by, sx, sy):
    lib = __import__(type(bx).__module__)

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sx = lib.asarray(sx, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    r = by / bx
    var = (sy**2) / (bx**2) + (by**2) * (sx**2) / (bx**4)
    se = lib.sqrt(lib.maximum(var, 1e-30))
    return lib.vstack([r, 1.0 / se])

def mr_quantile_ace(bx, sx, by, sy, eaf, z, rivw_lam, eta, alasso_lam, alasso_nu, mei_crit):
    lib = __import__(type(bx).__module__)
    from math import sqrt

    bx = lib.asarray(bx, dtype=lib.float64).reshape(-1)
    sx = lib.asarray(sx, dtype=lib.float64).reshape(-1)
    by = lib.asarray(by, dtype=lib.float64).reshape(-1)
    sy = lib.asarray(sy, dtype=lib.float64).reshape(-1)
    eaf = lib.asarray(eaf, dtype=lib.float64).reshape(-1)
    z = lib.asarray(z, dtype=lib.float64).reshape(-1)

    ori = allele_orientations(bx, by, eaf)
    g_m, G_m, g_n, G_n = ori[0], ori[1], ori[2], ori[3]
    z_m = lib.where(eaf < 0.5, -z, z)
    z_n = lib.where(bx < 0.0, -z, z)

    rivw_slope(g_n, G_n, sx, sy, z_n, rivw_lam, eta)
    mei_z_score(g_m, G_m, sx, sy, z_m, rivw_lam, eta)
    mei_z_score(g_n, G_n, sx, sy, z_n, rivw_lam, eta)
    zc = mei_combined_stat(bx, by, sx, sy, eaf, z, rivw_lam, eta)

    sel = lib.abs(g_n / sx + z_n) - rivw_lam > 0.0
    if zc > mei_crit and lib.any(sel):
        adaptive_alasso_weights(g_n[sel], G_n[sel], alasso_nu)
        flags = alasso_valid_flags(
            g_n[sel], G_n[sel], sy[sel], alasso_lam, alasso_nu
        )
        mask = lib.zeros(bx.size, dtype=bool)
        mask[lib.where(sel)[0][flags > 0.5]] = True
    else:
        mask = sel
    if lib.count_nonzero(mask) < 2:
        mask = sel

    rw = ratio_invse_weights(g_n[mask], G_n[mask], sx[mask], sy[mask])
    r, w = rw[0], rw[1]
    p = float(r.size)
    tau = 0.5
    prev_ll = -lib.inf
    theta = 0.0
    lam = 1.0
    for _ in range(80):
        order = lib.argsort(r, kind="mergesort")
        rs, ws = r[order], w[order]
        cw = lib.cumsum(ws)
        idx = int(lib.searchsorted(cw, tau * cw[-1], side="left"))
        idx = min(max(idx, 0), rs.size - 1)
        theta = float(rs[idx])
        u = r - theta
        rho = lib.where(u >= 0.0, tau * u, (tau - 1.0) * u)
        lam = p / max(lib.sum(w * rho), 1e-18)
        a = lam * lib.sum(w * (r - theta))
        tau = 0.5 - a / (2.0 * (2.0 * p + sqrt(a * a + 4.0 * p * p)))
        tau = min(max(tau, 1e-8), 1.0 - 1e-8)
        u = r - theta
        rho = lib.where(u >= 0.0, tau * u, (tau - 1.0) * u)
        ll = (
            lib.sum(lib.log(w))
            + p * lib.log(max(lam * tau * (1.0 - tau), 1e-300))
            - lam * lib.sum(w * rho)
        )
        if abs(ll - prev_ll) < 1e-12:
            break
        prev_ll = ll
    return float(theta)
SCICODE_GOLD_EOF
