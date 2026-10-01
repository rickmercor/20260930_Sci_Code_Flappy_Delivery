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


# Oracle implementation for public function: analytic_strip_halfwidth
def analytic_strip_halfwidth(eps_R: float, d_eps: float) -> float:
    if not (eps_R > d_eps > 0.0):
        raise ValueError("requires eps_R > d_eps > 0")
    return float(np.arccosh(eps_R / d_eps))

import numpy as np


# Oracle implementation for public function: layer_transfer_matrix
def layer_transfer_matrix(eps_m: complex, eps_R: float, w0tau: float) -> "np.ndarray":
    eps_m = complex(eps_m)
    if eps_m == 0 or eps_R <= 0.0 or w0tau <= 0.0:
        raise ValueError("need eps_m != 0, eps_R > 0, w0tau > 0")
    z = np.sqrt(eps_R / eps_m)
    c = np.cos(w0tau * z)
    s = np.sin(w0tau * z)
    return np.array([[c, -1j * s / z], [-1j * z * s, c]], dtype=complex)

import numpy as np


# Oracle implementation for public function: approximant_cell_trace
def approximant_cell_trace(phi: float, h: float, p: int, q: int, eps_R: float,
                                   d_eps: float, w0tau: float) -> complex:
    if int(q) != q or int(p) != p or q < 1:
        raise ValueError("p, q must be integers with q >= 1")
    if np.gcd(int(p), int(q)) != 1:
        raise ValueError("p and q must be coprime")
    if abs(h) >= analytic_strip_halfwidth(eps_R, d_eps):
        raise ValueError("h outside the nonsingular analytic strip")
    M = np.eye(2, dtype=complex)
    for j in range(int(q)):
        eps = eps_R + d_eps * np.sin(2.0 * np.pi * (phi + j * p / q) + 1j * h)
        M = layer_transfer_matrix(eps, eps_R, w0tau) @ M
    return complex(np.trace(M))

import numpy as np


# Oracle implementation for public function: trace_fourier_coefficients
def trace_fourier_coefficients(h0: float, p: int, q: int, eps_R: float, d_eps: float,
                                       w0tau: float, n_max: int, K: int) -> "np.ndarray":
    if n_max < 0 or K < 2 * n_max + 2:
        raise ValueError("need n_max >= 0 and K >= 2 * n_max + 2")
    phis = np.arange(K) / (K * q)
    F = np.array([approximant_cell_trace(x, h0, p, q, eps_R, d_eps, w0tau) for x in phis])
    return np.array([np.mean(F * np.exp(2j * np.pi * n * q * phis)) for n in range(n_max + 1)])

import numpy as np


# Oracle implementation for public function: coefficient_intercepts
def coefficient_intercepts(coeffs: "np.ndarray", q: int, h0: float) -> "np.ndarray":
    coeffs = np.asarray(coeffs, dtype=complex)
    if np.any(np.abs(coeffs) == 0.0):
        raise ValueError("zero Fourier coefficient has no logarithmic intercept")
    n = np.arange(len(coeffs))
    return np.log(np.abs(coeffs)) / q - n * h0

import numpy as np


# Oracle implementation for public function: integrated_dominant_order
def integrated_dominant_order(b: "np.ndarray", h_a: float, h_b: float) -> float:
    b = np.asarray(b, dtype=float)
    if not h_b > h_a:
        raise ValueError("need h_b > h_a")
    orders = np.arange(len(b))
    pts = [h_a, h_b]
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            x = (b[i] - b[j]) / (j - i)
            if h_a < x < h_b:
                pts.append(x)
    pts = np.sort(np.array(pts))
    total = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (lo + hi)
        total += int(np.argmax(b + orders * mid)) * (hi - lo)
    return float(total)

import numpy as np


# Oracle implementation for public function: run_ptqc_spectral_contrast
def run_ptqc_spectral_contrast(w1tau: float, w2tau: float, eps_R: float, d_eps: float,
                                       p: int, q: int, h0: float, h_a: float, h_b: float, N: int,
                                       n_max: int, K: int) -> float:
    h_star = analytic_strip_halfwidth(eps_R, d_eps)
    if not (0.0 <= h_a < h_b < h_star) or not (abs(h0) < h_star):
        raise ValueError("interval and reference height must lie inside the nonsingular strip")
    if N <= 0:
        raise ValueError("N must be positive")
    integrals = []
    for w0tau in (w1tau, w2tau):
        coeffs = trace_fourier_coefficients(h0, p, q, eps_R, d_eps, w0tau, n_max, K)
        b = coefficient_intercepts(coeffs, q, h0)
        integrals.append(integrated_dominant_order(b, h_a, h_b))
    return float(20.0 * N / np.log(10.0) * (integrals[1] - integrals[0]))
SCICODE_GOLD_EOF
