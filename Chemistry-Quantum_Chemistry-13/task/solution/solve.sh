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


def lead_selfenergies(w, bias_v, temp_k, gamma_left, gamma_right, frac_left, e_fermi):
    _KB_EV = 8.617333262e-5
    if not (gamma_left > 0.0) or not (gamma_right > 0.0):
        raise ValueError("hybridisations must be positive")
    if temp_k == 0.0:
        raise ValueError("temp_k must be non-zero")
    if not (0.0 <= frac_left <= 1.0):
        raise ValueError("frac_left must lie in [0, 1]")
    w = np.asarray(w, dtype=float)
    mu_l = e_fermi + frac_left * bias_v
    mu_r = e_fermi - (1.0 - frac_left) * bias_v
    f_l = 0.5 * (1.0 - np.tanh(np.clip((w - mu_l) / (2.0 * _KB_EV * temp_k), -400.0, 400.0)))
    f_r = 0.5 * (1.0 - np.tanh(np.clip((w - mu_r) / (2.0 * _KB_EV * temp_k), -400.0, 400.0)))
    sig_lesser = 1j * (gamma_left * f_l + gamma_right * f_r)
    sig_greater = -1j * (gamma_left * (1.0 - f_l) + gamma_right * (1.0 - f_r))
    return np.vstack([sig_lesser, sig_greater])

import numpy as np


def mean_field_occupation(w, dw, eps, gamma_total, sigma_lesser, u_coupling, q_core):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    if not (gamma_total > 0.0):
        raise ValueError("gamma_total must be positive")
    if u_coupling < 0.0:
        raise ValueError("u_coupling must be non-negative")
    w = np.asarray(w, dtype=float)
    sl = np.asarray(sigma_lesser)

    def occ_of(nv):
        gr = 1.0 / (w - eps - u_coupling * (nv - q_core) + 0.5j * gamma_total)
        gl = gr * sl * np.conj(gr)
        return float((-1j * gl).sum().real * dw / (2.0 * np.pi))

    def resid(nv):
        return occ_of(nv) - nv

    xs = np.linspace(0.0, 1.0, 241)
    vs = np.array([resid(x) for x in xs])
    root = None
    for i in range(len(xs) - 1):
        if vs[i] == 0.0:
            root = float(xs[i])
            break
        if vs[i] * vs[i + 1] < 0.0:
            lo, hi, flo = xs[i], xs[i + 1], vs[i]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fm = resid(mid)
                if flo * fm <= 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fm
                if hi - lo < 1e-15:
                    break
            root = float(0.5 * (lo + hi))
            break
    if root is None:
        raise ValueError("mean-field equation has no root in [0, 1]")
    return np.array([root, u_coupling * (root - q_core)], dtype=float)

import numpy as np


def keldysh_greens(w, eps_shifted, gamma_total, sigma_lesser, sigma_greater,
                           sig_c_ret, sig_c_lesser, sig_c_greater):
    if not (gamma_total > 0.0):
        raise ValueError("gamma_total must be positive")
    w = np.asarray(w, dtype=float)
    n = len(w)
    arrs = [np.asarray(a) for a in (sigma_lesser, sigma_greater, sig_c_ret,
                                    sig_c_lesser, sig_c_greater)]
    for a in arrs:
        if a.shape[-1] != n:
            raise ValueError("all self-energy arrays must match the length of w")
    sl, sg, scr, scl, scg = arrs
    g_ret = 1.0 / (w - eps_shifted - scr + 0.5j * gamma_total)
    g_adv = np.conj(g_ret)
    g_lesser = g_ret * (sl + scl) * g_adv
    g_greater = g_ret * (sg + scg) * g_adv
    return np.vstack([g_ret, g_lesser, g_greater])

import numpy as np


def _fft_linear_convolve(a, b):
    m = len(a) + len(b) - 1
    size = 1
    while size < m:
        size *= 2
    return np.fft.ifft(np.fft.fft(a, size) * np.fft.fft(b, size))[:m]


def _shifted_correlation(a, b, dw):
    """(1/2pi) * int dw' a(w') b(w' - w), as an exact index shift on this grid."""
    n = len(a)
    half = n // 2
    full = _fft_linear_convolve(np.asarray(a, dtype=complex),
                               np.asarray(b, dtype=complex)[::-1])
    return full[np.arange(n) + (n - 1 - half)] * dw / (2.0 * np.pi)


def polarisation_bubbles(g_lesser, g_greater, dw):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    gl = np.asarray(g_lesser, dtype=complex)
    gg = np.asarray(g_greater, dtype=complex)
    if gl.shape != gg.shape:
        raise ValueError("g_lesser and g_greater must have the same shape")
    pi_lesser = -1j * _shifted_correlation(gl, gg, dw)
    pi_greater = -1j * _shifted_correlation(gg, gl, dw)
    return np.vstack([pi_lesser, pi_greater])

import numpy as np


def _fft_linear_convolve(a, b):
    m = len(a) + len(b) - 1
    size = 1
    while size < m:
        size *= 2
    return np.fft.ifft(np.fft.fft(a, size) * np.fft.fft(b, size))[:m]


def _shifted_convolution(a, b, dw):
    """(1/2pi) * int dw' a(w') b(w - w'), as an exact index shift on this grid."""
    n = len(a)
    half = n // 2
    full = _fft_linear_convolve(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex))
    return full[np.arange(n) + half] * dw / (2.0 * np.pi)


def correlation_selfenergies(g_lesser, g_greater, pi_lesser, pi_greater, u_coupling, dw):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    if u_coupling < 0.0:
        raise ValueError("u_coupling must be non-negative")
    arrs = [np.asarray(a, dtype=complex) for a in (g_lesser, g_greater, pi_lesser, pi_greater)]
    if len({a.shape[-1] for a in arrs}) != 1:
        raise ValueError("all input arrays must have the same length")
    gl, gg, pl, pg = arrs
    s_lesser = 1j * u_coupling ** 2 * _shifted_convolution(gl, pl, dw)
    s_greater = 1j * u_coupling ** 2 * _shifted_convolution(gg, pg, dw)
    return np.vstack([1j * s_lesser.imag, 1j * s_greater.imag])

import numpy as np


def _fft_linear_convolve(a, b):
    m = len(a) + len(b) - 1
    size = 1
    while size < m:
        size *= 2
    return np.fft.ifft(np.fft.fft(a, size) * np.fft.fft(b, size))[:m]


def retarded_from_keldysh(sigma_lesser, sigma_greater, w, dw):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    sl = np.asarray(sigma_lesser, dtype=complex)
    sg = np.asarray(sigma_greater, dtype=complex)
    wv = np.asarray(w, dtype=float)
    if not (len(sl) == len(sg) == len(wv)):
        raise ValueError("all input arrays must have the same length")
    n = len(wv)
    num = sg - sl
    offsets = np.arange(2 * n - 1) - (n - 1)
    kernel = np.zeros(2 * n - 1, dtype=complex)
    nonzero = offsets != 0
    kernel[nonzero] = 1.0 / (offsets[nonzero] * dw)
    principal = _fft_linear_convolve(num, kernel)[(n - 1) + np.arange(n)] * dw
    return 1j * principal / (2.0 * np.pi) + 0.5 * num

import numpy as np


def self_consistent_greens(w, dw, eps_shifted, gamma_total, sigma_lesser,
                                   sigma_greater, u_coupling, tol, max_iter):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    if not (gamma_total > 0.0):
        raise ValueError("gamma_total must be positive")
    if u_coupling < 0.0:
        raise ValueError("u_coupling must be non-negative")
    if not (tol > 0.0):
        raise ValueError("tol must be positive")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be positive")
    wv = np.asarray(w, dtype=float)
    n = len(wv)
    sl = np.asarray(sigma_lesser, dtype=complex)
    sg = np.asarray(sigma_greater, dtype=complex)
    zero = np.zeros(n, dtype=complex)

    g = keldysh_greens(wv, eps_shifted, gamma_total, sl, sg, zero, zero, zero)
    g_l, g_g = g[1], g[2]
    mix = 0.5
    for _ in range(int(max_iter)):
        pi = polarisation_bubbles(g_l, g_g, dw)
        sc = correlation_selfenergies(g_l, g_g, pi[0], pi[1], u_coupling, dw)
        sc_ret = retarded_from_keldysh(sc[0], sc[1], wv, dw)
        nxt = keldysh_greens(wv, eps_shifted, gamma_total, sl, sg,
                                     sc_ret, sc[0], sc[1])
        new_l, new_g = nxt[1], nxt[2]
        change = max(float(np.abs(new_l - g_l).max()), float(np.abs(new_g - g_g).max()))
        g_l = g_l + mix * (new_l - g_l)
        g_g = g_g + mix * (new_g - g_g)
        g_l = 1j * g_l.imag
        g_g = 1j * g_g.imag
        if change < tol:
            break
    return np.vstack([g_l, g_g])

import numpy as np


def noise_response_kms(pi_lesser, pi_greater):
    pl = np.asarray(pi_lesser, dtype=complex)
    pg = np.asarray(pi_greater, dtype=complex)
    if pl.shape != pg.shape:
        raise ValueError("pi_lesser and pi_greater must have the same shape")
    s_plus = np.maximum((1j * pg).real, 0.0)
    s_minus = np.maximum((1j * pl).real, 0.0)
    noise = 0.5 * (s_plus + s_minus)
    response = 0.5 * (s_plus - s_minus)
    ratio = np.zeros_like(s_minus)
    good = s_minus > 1e-4 * s_minus.max()
    ratio[good] = s_plus[good] / s_minus[good]
    return np.vstack([s_plus, s_minus, noise, response, ratio])

import numpy as np


def _fft_linear_convolve(a, b):
    m = len(a) + len(b) - 1
    size = 1
    while size < m:
        size *= 2
    return np.fft.ifft(np.fft.fft(a, size) * np.fft.fft(b, size))[:m]


def dispersion_energy(pi_lesser_a, pi_greater_a, pi_lesser_b, pi_greater_b,
                              w, dw, u_coupling):
    if not (dw > 0.0):
        raise ValueError("dw must be positive")
    if u_coupling < 0.0:
        raise ValueError("u_coupling must be non-negative")
    arrs = [np.asarray(a, dtype=complex) for a in
            (pi_lesser_a, pi_greater_a, pi_lesser_b, pi_greater_b)]
    wv = np.asarray(w, dtype=float)
    if len({a.shape[-1] for a in arrs} | {len(wv)}) != 1:
        raise ValueError("all propagator arrays and w must have the same length")
    pla, pga, plb, pgb = arrs
    n = len(wv)
    numerator = _fft_linear_convolve(pla, plb) - _fft_linear_convolve(pga, pgb)
    total = np.arange(2 * n - 1) - n
    keep = total != 0
    acc = (numerator[keep] / (total[keep] * dw)).sum()
    return float((-(u_coupling ** 2) * acc * dw * dw / (2.0 * np.pi) ** 2).real)

import numpy as np


def final_answer(eps=-0.75, u_coupling=0.90, bias_v=2.0, temp_k=300.0,
                         gamma_left=0.050, gamma_right=0.030, frac_left=0.70, e_fermi=0.0,
                         q_core=1.0, w_max=10.0, n=4096, tol=1e-11, max_iter=4000):
    if not (w_max > 0.0) or int(n) < 2 or int(n) % 2 != 0:
        raise ValueError("the frequency grid needs a positive half-width and an even point count")
    if not (gamma_left > 0.0 and gamma_right > 0.0) or u_coupling < 0.0 or temp_k == 0.0:
        raise ValueError("hybridisations must be positive, the coupling non-negative and the temperature non-zero")

    dw = 2.0 * w_max / n
    w = (np.arange(n) - n // 2) * dw
    gamma_total = gamma_left + gamma_right

    sig = lead_selfenergies(w, bias_v, temp_k, gamma_left, gamma_right, frac_left, e_fermi)
    sig_lesser, sig_greater = sig[0], sig[1]

    mf = mean_field_occupation(w, dw, eps, gamma_total, sig_lesser, u_coupling, q_core)
    shift = float(mf[1])
    eps_shifted = eps + shift

    g = self_consistent_greens(w, dw, eps_shifted, gamma_total, sig_lesser, sig_greater,
                                       u_coupling, tol, max_iter)
    g_lesser, g_greater = g[0], g[1]

    pi = polarisation_bubbles(g_lesser, g_greater, dw)
    pi_lesser, pi_greater = pi[0], pi[1]

    # Re-derive the correlation self-energy and the Green's functions from the converged
    # solution, and read off the spectral weights. These exercise the remaining steps on the
    # converged state and are checked against the contracts those steps document.
    sc = correlation_selfenergies(g_lesser, g_greater, pi_lesser, pi_greater,
                                          u_coupling, dw)
    sc_ret = retarded_from_keldysh(sc[0], sc[1], w, dw)
    chk = keldysh_greens(w, eps_shifted, gamma_total, sig_lesser, sig_greater,
                                 sc_ret, sc[0], sc[1])
    weights = noise_response_kms(pi_lesser, pi_greater)
    if not (np.all(np.isfinite(chk)) and np.all(np.isfinite(weights))):
        raise RuntimeError("the converged solution produced a non-finite Green's function or spectral weight")
    if weights[:2].min() < -1e-12 or np.abs(weights[2] + weights[3] - weights[0]).max() > 1e-9:
        raise RuntimeError("the converged spectral weights violate the step 08 contract")

    return dispersion_energy(pi_lesser, pi_greater, pi_lesser, pi_greater,
                                     w, dw, u_coupling)
SCICODE_GOLD_EOF
