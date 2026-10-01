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


def spectral_wavenumbers(N: "int | Sequence[int]",
                                 L: "float | Sequence[float]") -> "np.ndarray":
    _N = np.atleast_1d(np.asarray(N))
    if _N.size not in (1, 2):
        raise ValueError("N must be an integer or a length-2 sequence (Nx, Ny)")
    if not all(float(v).is_integer() for v in _N.ravel()):
        raise ValueError("every grid size must be an integer")
    nv = np.broadcast_to(_N.astype(float), (2,)).astype(int)
    if np.any(nv < 2):
        raise ValueError("every grid size must be >= 2")
    _L = np.atleast_1d(np.asarray(L, float))
    if _L.size not in (1, 2):
        raise ValueError("L must be a scalar or a length-2 vector of box lengths")
    if not np.all(np.isfinite(_L)) or np.any(_L <= 0.0):
        raise ValueError("all box lengths must be finite and positive")
    Lv = np.broadcast_to(np.asarray(L, float), (2,))
    k = [2.0 * np.pi * np.fft.fftfreq(int(nv[a]), d=float(Lv[a]) / int(nv[a]))
         for a in range(2)]
    return k[0][:, None] ** 2 + k[1][None, :] ** 2


def _geom(phi, L):
    nv = phi.shape
    Lv = np.broadcast_to(np.asarray(L, float), (2,))
    hx, hy = float(Lv[0]) / nv[0], float(Lv[1]) / nv[1]
    return nv, Lv, hx, hy

import numpy as np


def pfc_free_energy(phi: "np.ndarray", L: "float | Sequence[float]",
                            alpha: float) -> float:
    phi = np.asarray(phi, float)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional array")
    nv, Lv, hx, hy = _geom(phi, L)
    ksq = spectral_wavenumbers(nv, Lv)
    ph = np.fft.fft2(phi)
    pw = np.abs(ph) ** 2
    scale = hx * hy / (nv[0] * nv[1])
    grad2 = scale * np.sum(ksq * pw)            # \int |grad phi|^2 = -\int phi Lap phi
    lap2 = scale * np.sum(ksq ** 2 * pw)        # \int (Lap phi)^2
    bulk = hx * hy * np.sum(0.25 * phi ** 4 + 0.5 * float(alpha) * phi ** 2)
    return float(bulk - grad2 + 0.5 * lap2)

import numpy as np


def h_minus1_norm_sq(v: "np.ndarray", L: "float | Sequence[float]") -> float:
    v = np.asarray(v, float)
    if v.ndim != 2:
        raise ValueError("v must be a two-dimensional array")
    nv, Lv, hx, hy = _geom(v, L)
    ksq = spectral_wavenumbers(nv, Lv)
    vh = np.fft.fft2(v - v.mean())
    acc = np.zeros_like(ksq)
    nz = ksq > 0.0
    acc[nz] = np.abs(vh[nz]) ** 2 / ksq[nz]
    return float(hx * hy / (nv[0] * nv[1]) * np.sum(acc))

import numpy as np


def _Fprime(x, A):
    return x ** 3 - A * x


def bdf2_decoupled_fields(phi_n: "np.ndarray", phi_nm1: "np.ndarray", gamma: float,
                                  tau: float, L: "float | Sequence[float]", alpha: float,
                                  A: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    phi_n = np.asarray(phi_n, float)
    phi_nm1 = np.asarray(phi_nm1, float)
    if phi_n.shape != phi_nm1.shape:
        raise ValueError("phi_n and phi_nm1 must have the same shape")
    gamma = float(gamma)
    tau = float(tau)
    if gamma < 0.0:
        raise ValueError("the step ratio gamma must be non-negative")
    if not (tau > 0.0):
        raise ValueError("the time step tau must be positive")
    nv, Lv, hx, hy = _geom(phi_n, L)
    ksq = spectral_wavenumbers(nv, Lv)
    # symbol of Delta * Lop, with Lop = Delta^2 + 2 Delta + (alpha + A) I
    dl = -ksq ** 3 + 2.0 * ksq ** 2 - (float(alpha) + float(A)) * ksq
    a_sym = (1.0 + 2.0 * gamma) / (1.0 + gamma) - tau * dl
    phi_bar = phi_n + gamma * (phi_n - phi_nm1)
    rhs_p = (1.0 + gamma) * phi_n - gamma ** 2 / (1.0 + gamma) * phi_nm1
    p = np.real(np.fft.ifft2(np.fft.fft2(rhs_p) / a_sym))
    q = np.real(np.fft.ifft2(-ksq * np.fft.fft2(_Fprime(phi_bar, float(A))) / a_sym))
    return phi_bar, p, q

import numpy as np


def _Fbulk(x, A):
    return 0.25 * x ** 4 - 0.5 * A * x ** 2


def _S(eta):
    return eta * (2.0 - eta)


def solve_multiplier(p: "np.ndarray", q: "np.ndarray", phi_n: "np.ndarray", phi_bar: "np.ndarray",
                             tau: float, theta: float, eta_n: float,
                             A: float, L: "float | Sequence[float]", tol: float=1e-13,
                             max_iter: int=100) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    phi_n = np.asarray(phi_n, float)
    phi_bar = np.asarray(phi_bar, float)
    tau = float(tau)
    theta = float(theta)
    eta_n = float(eta_n)
    A = float(A)
    if theta <= 0.0:
        raise ValueError("the penalty parameter theta must be positive")
    nv, Lv, hx, hy = _geom(p, L)
    w0 = hx * hy
    fpb = _Fprime(phi_bar, A)
    int_Fn = w0 * np.sum(_Fbulk(phi_n, A))

    def _g_dg(eta):
        s = _S(eta)
        w = p + s * tau * q
        g = (w0 * np.sum(_Fbulk(w, A)) - int_Fn
             - s * w0 * np.sum(fpb * (w - phi_n))
             + theta * (eta * eta - eta_n * eta_n))
        dgds = (tau * w0 * np.sum(_Fprime(w, A) * q)
                - w0 * np.sum(fpb * (w - phi_n))
                - s * tau * w0 * np.sum(fpb * q))
        return g, dgds * (2.0 - 2.0 * eta) + 2.0 * theta * eta

    # Newton from eta^n. The iteration selects the branch continuously connected
    # to eta = 1; it is NOT guaranteed to be the only root of g, so the starting
    # point is part of the scheme's specification, not an implementation detail.
    # Any breakdown is reported, never papered over by returning the last iterate.
    eta = eta_n
    converged = False
    for _ in range(int(max_iter)):
        g, dg = _g_dg(eta)
        if not np.isfinite(g) or not np.isfinite(dg):
            raise RuntimeError("the scalar residual or its derivative is "
                               "non-finite; the multiplier solve broke down")
        if dg == 0.0:
            raise RuntimeError("the Newton derivative vanished; the scalar "
                               "equation is not solvable at this state")
        d = g / dg
        eta = eta - d
        if not np.isfinite(eta):
            raise RuntimeError("the Newton iterate left the finite range; the "
                               "multiplier solve broke down")
        if abs(d) <= tol:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            "Newton did not converge to tol=%g in %d iterations; the last "
            "iterate is not a root of the scalar equation" % (tol, int(max_iter)))
    return float(eta)

import numpy as np


def pfc_bdf2_step(phi_n: "np.ndarray", phi_nm1: "np.ndarray", eta_n: float,
                          gamma: float, tau: float, L: "float | Sequence[float]", alpha: float,
                          A: float, theta: float) -> "tuple[np.ndarray, float]":
    phi_bar, p, q = bdf2_decoupled_fields(phi_n, phi_nm1, gamma, tau,
                                                  L, alpha, A)
    eta = solve_multiplier(p, q, np.asarray(phi_n, float), phi_bar,
                                   tau, theta, eta_n, A, L)
    return p + _S(eta) * float(tau) * q, eta

import numpy as np


def adaptive_time_step(tau_n: float, E_n: float, E_nm1: float,
                               tau_min: float, tau_max: float,
                               beta: float) -> float:
    gamma_max = 4.86454          # the source's maximal admissible step ratio
    tau_n = float(tau_n)
    tau_min = float(tau_min)
    tau_max = float(tau_max)
    beta = float(beta)
    if not (tau_n > 0.0):
        raise ValueError("tau_n must be positive")
    if not (0.0 < tau_min <= tau_max):
        raise ValueError("require 0 < tau_min <= tau_max")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    edot = (float(E_n) - float(E_nm1)) / tau_n
    tau_ad = tau_max / np.sqrt(1.0 + beta * edot * edot)
    return float(max(tau_min, min(tau_max, tau_ad, gamma_max * tau_n)))

import numpy as np


def _seed_field(N, L):
    nv = np.broadcast_to(np.atleast_1d(np.asarray(N)).astype(int), (2,))
    Lv = np.broadcast_to(np.asarray(L, float), (2,))
    x = np.arange(nv[0]) * (Lv[0] / nv[0])
    y = np.arange(nv[1]) * (Lv[1] / nv[1])
    s = (2.0 * np.pi * x / Lv[0])[:, None]
    t = (2.0 * np.pi * y / Lv[1])[None, :]
    return (-0.27
            + 0.06 * np.cos(16.0 * s + 0.3) * np.cos(9.0 * t)
            + 0.05 * np.sin(11.0 * s) * np.cos(13.0 * t + 0.7)
            + 0.04 * np.cos(7.0 * s - 14.0 * t + 1.1)
            + 0.03 * np.sin(19.0 * s + 5.0 * t) * np.sin(6.0 * t)
            + 0.02 * np.cos(23.0 * s) * np.sin(21.0 * t + 0.4))


def penalized_lm_pfc_answer(
        thetas: "Sequence[float]"=(0.5, 5.0, 50.0, 500.0),
        N: "int | Sequence[int]"=64,
        L: "float | Sequence[float]"=100.53096491487338,
        alpha: float=0.75, A: float=0.37, n_steps: int=60,
        tau_min: float=1e-4, tau_max: float=1.0, beta: float=4.0) -> float:
    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    thetas = [float(t) for t in np.atleast_1d(np.asarray(thetas, float))]
    Lv = np.broadcast_to(np.asarray(L, float), (2,))
    phi0 = _seed_field(N, Lv)
    total = 0.0
    for theta in thetas:
        tau1 = float(tau_min)
        phi_prev = phi0
        phi_cur, eta = pfc_bdf2_step(phi0, phi0, 1.0, 0.0, tau1,
                                             Lv, alpha, A, theta)
        tau_cur = tau1
        E_prev = pfc_free_energy(phi0, Lv, alpha)
        E_cur = pfc_free_energy(phi_cur, Lv, alpha)
        for _ in range(n_steps - 1):
            tau_new = adaptive_time_step(tau_cur, E_cur, E_prev,
                                                 tau_min, tau_max, beta)
            gamma = tau_new / tau_cur
            phi_new, eta = pfc_bdf2_step(phi_cur, phi_prev, eta, gamma,
                                                 tau_new, Lv, alpha, A, theta)
            phi_prev, phi_cur = phi_cur, phi_new
            tau_cur = tau_new
            E_prev, E_cur = E_cur, pfc_free_energy(phi_cur, Lv, alpha)
        tau_next = adaptive_time_step(tau_cur, E_cur, E_prev,
                                              tau_min, tau_max, beta)
        g_next = tau_next / tau_cur
        bdf = (g_next ** 1.5 / (2.0 * (1.0 + g_next))
               * h_minus1_norm_sq(phi_cur - phi_prev, Lv) / tau_cur)
        e_mod = bdf + E_cur + theta * eta * eta
        total += e_mod
    return float(total)
SCICODE_GOLD_EOF
