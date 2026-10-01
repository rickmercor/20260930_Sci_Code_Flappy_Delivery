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


def shifted_rk_matrix(order: int) -> "np.ndarray":
    if order == 2:
        return np.array([[1.0, 0.0],
                         [0.5, 0.5]], dtype=float)
    if order == 3:
        return np.array([[1.0, 0.0, 0.0],
                         [0.25, 0.25, 0.0],
                         [1.0/6.0, 1.0/6.0, 2.0/3.0]], dtype=float)
    if order == 4:
        return np.array([[0.5, 0.0, 0.0, 0.0],
                         [0.0, 0.5, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]], dtype=float)
    raise ValueError("order must be one of 2, 3, or 4")

import numpy as np


def rothe_vms_fourier_coefficients(K_star: float, a_star: float, kappa_star: float, tau_star: float) -> "np.ndarray":
    K = float(K_star)
    a = float(a_star)
    k = float(kappa_star)
    tau = float(tau_star)
    c1 = np.cos(K)
    c2 = np.cos(2.0 * K)
    s1 = np.sin(K)
    s2 = np.sin(2.0 * K)
    C = (1.0 - tau) * (c2 + 26.0 * c1 + 33.0)

    lambda1 = (
        20.0 * tau * k * c2 + 40.0 * tau * k * c1 - 60.0 * tau * k
        + 1j * (-5.0 * tau * a * s2 - 50.0 * tau * a * s1)
    ) / C

    lambda2 = (
        (-40.0 * tau * k + 20.0 * k) * c2
        + (-80.0 * tau * k + 40.0 * k) * c1
        + 120.0 * tau * k - 60.0 * k
        + 1j * ((10.0 * tau * a - 5.0 * a) * s2
                + (100.0 * tau * a - 50.0 * a) * s1)
    ) / C

    lambda3 = (
        (20.0 * tau * a * a + 120.0 * tau * k * k) * c2
        + (40.0 * tau * a * a - 480.0 * tau * k * k) * c1
        + 360.0 * tau * k * k - 60.0 * tau * a * a
        + 1j * (-120.0 * tau * k * a * s2 + 240.0 * tau * k * a * s1)
    ) / C

    return np.array([lambda1, lambda2, lambda3], dtype=complex)

import numpy as np


def rothe_vms_amplification(order: int, K_star: float, a_star: float, kappa_star: float, tau_star: float) -> complex:
    alpha = shifted_rk_matrix(order)
    lambda1, lambda2, lambda3 = rothe_vms_fourier_coefficients(K_star, a_star, kappa_star, tau_star)
    s = int(order)
    stage = np.ones(s, dtype=complex)
    stage[1] = 1.0 + (lambda1 + lambda2) * alpha[0, 0]

    for i in range(3, s + 1):
        row = i - 2
        sum_alpha = np.sum(alpha[row, :i-1])
        sum_alpha_zeta = np.dot(alpha[row, :i-1], stage[:i-1])
        nested = 0.0j
        for j in range(2, i):
            nested += alpha[row, j-1] * np.dot(alpha[j-2, :j-1], stage[:j-1])
        stage[i-1] = 1.0 + lambda1 * sum_alpha + lambda2 * sum_alpha_zeta + lambda3 * nested

    nested_final = 0.0j
    for i in range(2, s + 1):
        nested_final += alpha[-1, i-1] * np.dot(alpha[i-2, :i-1], stage[:i-1])

    zeta = (1.0
            + lambda1 * np.sum(alpha[-1, :])
            + lambda2 * np.dot(alpha[-1, :], stage)
            + lambda3 * nested_final)
    return complex(zeta)

import numpy as np


def vertical_vms_fourier_symbol(K_star: float, a_star: float, kappa_star: float) -> "np.ndarray":
    K = float(K_star)
    a = float(a_star)
    k = float(kappa_star)
    tau_diamond_star = (4.0 + 4.0 * a * a + 144.0 * k * k) ** (-0.5)
    c1 = np.cos(K)
    c2 = np.cos(2.0 * K)
    s1 = np.sin(K)
    s2 = np.sin(2.0 * K)
    t = tau_diamond_star

    numerator = (
        (20.0 * t * a * a + 20.0 * k + 120.0 * t * k * k) * c2
        + (40.0 * t * a * a + 40.0 * k - 480.0 * t * k * k) * c1
        - 60.0 * t * a * a - 60.0 * k + 360.0 * t * k * k
        - 1j * ((5.0 * a + 120.0 * t * k * a) * s2
                + (50.0 * a - 240.0 * t * k * a) * s1)
    )
    denominator = (
        (1.0 + 20.0 * t * k) * c2
        + (26.0 + 40.0 * t * k) * c1
        + 33.0 - 60.0 * t * k
        - 1j * (5.0 * t * a * s2 + 50.0 * t * a * s1)
    )
    gamma = numerator / denominator
    return np.array([tau_diamond_star, gamma.real, gamma.imag], dtype=float)

import numpy as np


def vertical_vms_amplification(order: int, K_star: float, a_star: float, kappa_star: float) -> complex:
    alpha = shifted_rk_matrix(order)
    symbol = vertical_vms_fourier_symbol(K_star, a_star, kappa_star)
    gamma = complex(symbol[1], symbol[2])
    s = int(order)
    stage = np.ones(s, dtype=complex)
    for i in range(2, s + 1):
        stage[i-1] = 1.0 + gamma * np.dot(alpha[i-2, :i-1], stage[:i-1])
    return complex(1.0 + gamma * np.dot(alpha[-1, :], stage))

import numpy as np


def spectral_diagnostics(zeta: complex, K_star: float, a_star: float, kappa_star: float) -> "np.ndarray":
    zeta = complex(zeta)
    K = float(K_star)
    a = float(a_star)
    k = float(kappa_star)
    amplitude = abs(zeta)
    damping_ratio = -np.log(amplitude) / (k * K * K)
    frequency_ratio = np.angle(zeta) / (-a * K)
    return np.array([amplitude, damping_ratio, frequency_ratio], dtype=float)

import numpy as np


def spectral_ordering_mismatch(order: int, K_star: float, a_star: float, kappa_star: float, tau_star: float) -> "np.ndarray":
    zeta_rothe = rothe_vms_amplification(order, K_star, a_star, kappa_star, tau_star)
    zeta_vertical = vertical_vms_amplification(order, K_star, a_star, kappa_star)
    diag_rothe = spectral_diagnostics(zeta_rothe, K_star, a_star, kappa_star)
    diag_vertical = spectral_diagnostics(zeta_vertical, K_star, a_star, kappa_star)
    return np.array([
        diag_rothe[1] - diag_vertical[1],
        diag_rothe[2] - diag_vertical[2],
        diag_rothe[0] - diag_vertical[0],
    ], dtype=float)

import numpy as np


def cumulative_spectral_ordering_path(configurations: "np.ndarray", tau_star: float) -> float:
    configurations = np.asarray(configurations, dtype=float)
    mismatch_vectors = []
    for row in configurations:
        mismatch_vectors.append(
            spectral_ordering_mismatch(
                int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(tau_star)
            )
        )
    mismatch_vectors = np.asarray(mismatch_vectors, dtype=float)
    increments = np.linalg.norm(np.diff(mismatch_vectors, axis=0), axis=1)
    return float(np.sum(increments))
SCICODE_GOLD_EOF
