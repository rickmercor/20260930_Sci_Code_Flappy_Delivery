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


def _cw_check_float(name, value, positive):
    """Return value as a finite float, optionally requiring it to be positive."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s must be finite%s" % (name, " and positive" if positive else ""))
    return value


def _cw_gain_dispersion(omega, Gamma, alpha):
    """Real and imaginary parts of the scaled medium response at detuning omega."""
    shift = omega + alpha * Gamma
    den = Gamma ** 2 + shift ** 2
    h1 = (Gamma ** 2 * (1.0 - alpha ** 2) + 2.0 * alpha * Gamma * shift) / den
    h2 = (-2.0 * alpha * Gamma ** 2 + Gamma * (1.0 - alpha ** 2) * shift) / den
    return h1, h2


def _cw_frequency(k, Gamma, alpha, sigma):
    """Root of the continuous-wave dispersion relation closest to omega = k."""
    omega = k
    for _ in range(200):
        h1, h2 = _cw_gain_dispersion(omega, Gamma, alpha)
        if h1 <= 0.0:
            raise ValueError("the wavenumber lies outside the gain band, no lasing solution")
        new = k - sigma * h2 / h1
        if abs(new - omega) <= 1e-16 * max(1.0, abs(new)):
            omega = new
            break
        omega = new
    else:
        raise ValueError("the dispersion relation did not converge")
    # polish with Newton steps on f(w) = w - k + sigma*h2/h1
    for _ in range(5):
        h1, h2 = _cw_gain_dispersion(omega, Gamma, alpha)
        d = 1e-7 * max(1.0, abs(omega))
        h1p, h2p = _cw_gain_dispersion(omega + d, Gamma, alpha)
        h1m, h2m = _cw_gain_dispersion(omega - d, Gamma, alpha)
        f = omega - k + sigma * h2 / h1
        fp = 1.0 + sigma * (h2p / h1p - h2m / h1m) / (2.0 * d)
        step = f / fp
        omega -= step
        if abs(step) <= 1e-17 * max(1.0, abs(omega)):
            break
    return omega


def cw_emission_state(k: float, Gamma: float, alpha: float, sigma: float, mu: float) -> "np.ndarray":
    k = _cw_check_float("k", k, False)
    Gamma = _cw_check_float("Gamma", Gamma, True)
    alpha = _cw_check_float("alpha", alpha, False)
    sigma = _cw_check_float("sigma", sigma, True)
    mu = _cw_check_float("mu", mu, True)
    omega = _cw_frequency(k, Gamma, alpha, sigma)
    h1, h2 = _cw_gain_dispersion(omega, Gamma, alpha)
    d0 = 1.0 / h1
    x = mu - 1.0 / h1
    if not x > 0.0:
        raise ValueError("mu does not exceed the lasing threshold of this continuous wave")
    f0 = np.sqrt(x)
    p0 = (-h1 + 1j * h2) * d0 * f0
    return np.array([omega, d0, x, p0.real, p0.imag], dtype=float)

import numpy as np


def _erf_check(name, value, positive, nonneg=False):
    """Return value as a finite float with the requested sign constraint."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0) or (nonneg and value < 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def effective_rabi_frequency(X: float, Gamma: float, alpha: float, b: float) -> "np.ndarray":
    X = _erf_check("X", X, False, nonneg=True)
    Gamma = _erf_check("Gamma", Gamma, True)
    alpha = _erf_check("alpha", alpha, False)
    b = _erf_check("b", b, True)
    f = np.sqrt(X)
    cp = 1.0 + 1j * alpha
    cm = 1.0 - 1j * alpha
    jac = np.array([[-Gamma * cp, 0.0, -Gamma * cp ** 2 * f],
                    [0.0, -Gamma * cm, -Gamma * cm ** 2 * f],
                    [0.5 * b * f, 0.5 * b * f, -b]], dtype=complex)
    ev = np.linalg.eigvals(jac)
    scale = max(1.0, float(np.max(np.abs(ev))))
    cplx = ev[np.abs(ev.imag) > 1e-10 * scale]
    if cplx.size < 2:
        raise ValueError("no complex-conjugate eigenvalue pair: overdamped medium")
    pick = cplx[np.argmax(np.abs(cplx.imag))]
    return np.array([abs(pick.imag), abs(pick.real)], dtype=float)

import numpy as np


def _sb_check(name, value, positive):
    """Return value as a finite float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def _sb_matrix(kn, k, omega, d0, f0, p0, Gamma, alpha, sigma, b):
    """Linear evolution matrix of (dF_n, dF*_-n, dP_n, dP*_-n, dD_n)."""
    cp = 1.0 + 1j * alpha
    cm = 1.0 - 1j * alpha
    m = np.zeros((5, 5), dtype=complex)
    m[0, 0] = -sigma + 1j * (k - omega + kn)
    m[0, 2] = -sigma
    m[1, 1] = -sigma - 1j * (k - omega - kn)
    m[1, 3] = -sigma
    m[2, 0] = -Gamma * cp ** 2 * d0
    m[2, 2] = -1j * omega - Gamma * cp
    m[2, 4] = -Gamma * cp ** 2 * f0
    m[3, 1] = -Gamma * cm ** 2 * d0
    m[3, 3] = 1j * omega - Gamma * cm
    m[3, 4] = -Gamma * cm ** 2 * np.conj(f0)
    m[4, 0] = 0.5 * b * np.conj(p0)
    m[4, 1] = 0.5 * b * p0
    m[4, 2] = 0.5 * b * np.conj(f0)
    m[4, 3] = 0.5 * b * f0
    m[4, 4] = -b
    return m


def sideband_growth_rate(kn: float, k: float, Gamma: float, alpha: float, sigma: float, b: float, mu: float) -> float:
    kn = _sb_check("kn", kn, False)
    b = _sb_check("b", b, True)
    omega, d0, x, p_re, p_im = cw_emission_state(k, Gamma, alpha, sigma, mu)
    f0 = np.sqrt(x)
    p0 = p_re + 1j * p_im
    m = _sb_matrix(kn, float(k), omega, d0, f0, p0, float(Gamma), float(alpha), float(sigma), b)
    return float(np.max(np.linalg.eigvals(m).real))

import numpy as np
import scipy.linalg
from scipy.optimize import brentq, minimize_scalar


def _pg_check(name, value, positive):
    """Return value as a finite float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def _pg_slope(kn, state, Gamma, alpha, sigma, b):
    """d Re(lambda_max)/d kn for the k = 0 wave, from the left and right eigenvectors of the sideband matrix."""
    omega, d0, x, p_re, p_im = state
    m = _sb_matrix(kn, 0.0, omega, d0, np.sqrt(x), p_re + 1j * p_im, float(Gamma), float(alpha), float(sigma), float(b))
    lam, vl, vr = scipy.linalg.eig(m, left=True, right=True)
    j = int(np.argmax(lam.real))
    w, v = vl[:, j], vr[:, j]
    return float((1j * (np.conj(w[0]) * v[0] + np.conj(w[1]) * v[1]) / (np.conj(w) @ v)).real)


def parametric_gain_maximum(Gamma: float, alpha: float, sigma: float, b: float, mu: float, tau_d_ps: float, f_max_ghz: float) -> "np.ndarray":
    tau = _pg_check("tau_d_ps", tau_d_ps, True) * 1e-12
    fmax = _pg_check("f_max_ghz", f_max_ghz, True)
    to_kn = lambda f_ghz: 2.0 * np.pi * f_ghz * 1e9 * tau

    def _rate(f_ghz):
        return sideband_growth_rate(to_kn(f_ghz), 0.0, Gamma, alpha, sigma, b, mu)

    grid = np.linspace(fmax / 4000.0, fmax, 4000)
    vals = np.array([_rate(f) for f in grid])
    i = int(np.argmax(vals))
    if not vals[i] > 0.0:
        raise ValueError("the parametric gain is nowhere positive on the searched interval")
    f_best, g_best = grid[i], vals[i]
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, grid.size - 1)]
    state = cw_emission_state(0.0, Gamma, alpha, sigma, mu)
    slope = lambda f_ghz: _pg_slope(to_kn(f_ghz), state, Gamma, alpha, sigma, b)
    if 0 < i < grid.size - 1 and slope(lo) > 0.0 > slope(hi):
        f_root = brentq(slope, lo, hi, xtol=1e-13 * fmax, rtol=4.0 * np.finfo(float).eps, maxiter=200)
        g_root = _rate(f_root)
        if g_root >= g_best:
            f_best, g_best = f_root, g_root
    else:
        res = minimize_scalar(lambda f: -_rate(f), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-10 * fmax, "maxiter": 500})
        if -res.fun >= g_best:
            f_best, g_best = res.x, -res.fun
    return np.array([f_best, g_best / tau * 1e-9], dtype=float)

import numpy as np
from scipy.optimize import brentq


def _on_check(name, value, positive):
    """Return value as a finite float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def _on_kn(n, L_mm, n_group, tau_d_ps):
    """Scaled wavenumber offset of cavity sideband n."""
    vg = 299792458.0 / n_group
    return 2.0 * np.pi * n * vg * (tau_d_ps * 1e-12) / (L_mm * 1e-3)


def sideband_onset_pump(n: int, L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, mu_min: float, mu_max: float) -> float:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be an integer >= 1")
    L_mm = _on_check("L_mm", L_mm, True)
    n_group = _on_check("n_group", n_group, True)
    tau_d_ps = _on_check("tau_d_ps", tau_d_ps, True)
    mu_min = _on_check("mu_min", mu_min, True)
    mu_max = _on_check("mu_max", mu_max, True)
    if not mu_min < mu_max:
        raise ValueError("mu_min must be smaller than mu_max")
    kn = _on_kn(int(n), L_mm, n_group, tau_d_ps)
    rate = lambda mu: sideband_growth_rate(kn, 0.0, Gamma, alpha, sigma, b, mu)
    if rate(mu_min) > 0.0:
        raise ValueError("the sideband is already unstable at mu_min")
    grid = np.arange(mu_min, mu_max + 1e-12, 0.005)
    if grid[-1] < mu_max:
        grid = np.append(grid, mu_max)
    prev = rate(grid[0])
    for lo, hi in zip(grid[:-1], grid[1:]):
        cur = rate(hi)
        if prev <= 0.0 < cur:
            return float(brentq(rate, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=500))
        prev = cur
    raise ValueError("the sideband does not become unstable by mu_max")

import numpy as np
from scipy.optimize import brentq


def _hs_check(name, value, positive):
    """Return value as a finite float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def _hs_kn(n, L_mm, n_group, tau_d_ps):
    """Scaled wavenumber offset of cavity sideband n."""
    vg = 299792458.0 / n_group
    return 2.0 * np.pi * n * vg * (tau_d_ps * 1e-12) / (L_mm * 1e-3)


def harmonic_order_switch(L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, n_max: int, mu_min: float, mu_max: float) -> "np.ndarray":
    if isinstance(n_max, bool) or not isinstance(n_max, (int, np.integer)) or int(n_max) < 2:
        raise ValueError("n_max must be an integer >= 2")
    n_max = int(n_max)
    mu_max = _hs_check("mu_max", mu_max, True)
    onsets = {}
    for n in range(1, n_max + 1):
        try:
            onsets[n] = sideband_onset_pump(n, L_mm, n_group, tau_d_ps, Gamma, alpha, sigma, b, mu_min, mu_max)
        except ValueError as exc:
            if "does not become unstable" in str(exc):
                continue
            raise
    if not onsets:
        raise ValueError("no sideband becomes unstable inside the pump window")
    n_c = min(onsets, key=lambda n: (onsets[n], n))
    mu_c = onsets[n_c]
    kns = [_hs_kn(n, L_mm, n_group, tau_d_ps) for n in range(1, n_max + 1)]

    def _rates(mu):
        return np.array([sideband_growth_rate(kn, 0.0, Gamma, alpha, sigma, b, mu) for kn in kns])

    def _rivals(mu):
        r = _rates(mu)
        best_other = max((n for n in range(1, n_max + 1) if n != n_c), key=lambda n: r[n - 1])
        return r[best_other - 1] - r[n_c - 1], best_other

    step = 0.005
    lo = mu_c
    while lo < mu_max:
        hi = min(lo + step, mu_max)
        d_hi, who = _rivals(hi)
        if d_hi > 0.0:
            kn_new = kns[who - 1]
            kn_old = kns[n_c - 1]
            diff = lambda mu: (sideband_growth_rate(kn_new, 0.0, Gamma, alpha, sigma, b, mu)
                               - sideband_growth_rate(kn_old, 0.0, Gamma, alpha, sigma, b, mu))
            mu_star = brentq(diff, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=500)
            return np.array([mu_c, float(n_c), mu_star, float(who)], dtype=float)
        lo = hi
    raise ValueError("the selected sideband does not change before mu_max")

import numpy as np


def erf_gain_offset_at_switch(L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, n_max: int, mu_min: float, mu_max: float, f_max_ghz: float) -> float:
    switch = harmonic_order_switch(L_mm, n_group, tau_d_ps, Gamma, alpha, sigma, b, n_max, mu_min, mu_max)
    mu_star = float(switch[2])
    x = float(cw_emission_state(0.0, Gamma, alpha, sigma, mu_star)[2])
    erf_scaled = float(effective_rabi_frequency(x, Gamma, alpha, b)[0])
    erf_ghz = erf_scaled / (2.0 * np.pi * float(tau_d_ps) * 1e-12) * 1e-9
    peak = parametric_gain_maximum(Gamma, alpha, sigma, b, mu_star, tau_d_ps, f_max_ghz)
    return float(peak[0] - erf_ghz)
SCICODE_GOLD_EOF
