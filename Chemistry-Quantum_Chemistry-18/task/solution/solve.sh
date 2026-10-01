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


def pair_partial_wave(r1: "np.ndarray", r2: "np.ndarray", l: int, c: float) -> "np.ndarray":
    """Reference implementation."""
    l = int(l)
    if l < 0:
        raise ValueError("the Legendre order must be non-negative")
    a, b = np.broadcast_arrays(np.asarray(r1, dtype=float), np.asarray(r2, dtype=float))
    small = np.minimum(a, b)
    large = np.maximum(a, b)
    safe = np.where(large > 0.0, large, 1.0)
    if l == 0:
        linear = large + small ** 2 / (3.0 * safe)
        value = 1.0 + 0.5 * linear + c * (a ** 2 + b ** 2)
    else:
        ratio = small / safe
        linear = small * ratio ** (l + 1) / (2 * l + 3) - large * ratio ** l / (2 * l - 1)
        value = 0.5 * linear
        if l == 1:
            value = value - 2.0 * c * a * b
    return np.where(large > 0.0, value, 1.0 if l == 0 else 0.0)

import numpy as np
from scipy.integrate import quad


def pair_normalization(omega: float, c: float) -> float:
    """Reference implementation."""
    omega = float(omega)
    c = float(c)
    if not omega > 0.0:
        raise ValueError("omega must be positive")
    if c < 0.0:
        raise ValueError("c must be non-negative")
    centre = (np.pi / (2.0 * omega)) ** 1.5
    relative = quad(lambda s: (1.0 + 0.5 * s + c * s * s) ** 2 * np.exp(-0.5 * omega * s * s) * 4.0 * np.pi * s * s,
                    0.0, np.inf, epsabs=0.0, epsrel=1e-13, limit=200)[0]
    return float(1.0 / np.sqrt(centre * relative))

import numpy as np


def weak_orbital_asymptotics(omega: float, c: float) -> "np.ndarray":
    trap = float(omega)
    quad = float(c)
    if not trap > 0.0:
        raise ValueError("omega must be positive")
    if quad < 0.0:
        raise ValueError("c must be non-negative")
    norm = pair_normalization(trap, quad)
    volume_integral = norm ** 0.75 * (4.0 * np.pi / (3.0 * trap)) ** 1.5
    radial_integral = norm ** 0.25 * (np.pi / trap) ** 0.5
    amplitude = (np.sqrt(2.0) * volume_integral / (3.0 * np.pi ** 1.25)) ** (4.0 / 3.0)
    shell_count = (6.0 * radial_integral ** 3 / (np.pi * volume_integral)) ** (1.0 / 3.0)
    return np.array([volume_integral, radial_integral, amplitude, shell_count], dtype=float)

import numpy as np
from scipy.integrate import quad


def _rho1_integrand(u, omega, c, r):
    """Radial integrand of rho1(r) after the closed-form angular integration, without the prefactor 2 C^2."""
    x = 2.0 * omega * r * u
    if x < 1e-6:
        gauss = np.exp(-omega * (u * u + 2.0 * r * r)) * (1.0 + x * x / 6.0)
    else:
        gauss = np.exp(-omega * r * r) * (np.exp(-omega * (u - r) ** 2) - np.exp(-omega * (u + r) ** 2)) / (2.0 * x)
    return (1.0 + 0.5 * u + c * u * u) ** 2 * gauss * 4.0 * np.pi * u * u


def radial_densities(omega: float, c: float, r: float) -> "np.ndarray":
    """Reference implementation."""
    r = float(r)
    if r < 0.0:
        raise ValueError("r must be non-negative")
    norm = pair_normalization(omega, c)
    omega = float(omega)
    c = float(c)
    ontop = norm ** 2 * np.exp(-2.0 * omega * r * r)

    radial = quad(_rho1_integrand, 0.0, np.inf, args=(omega, c, r), epsabs=0.0, epsrel=1e-12, limit=400)[0]
    density = 2.0 * norm ** 2 * radial
    return np.array([ontop, density, 4.0 * ontop / density ** 2])

import numpy as np


def _radial_grid(omega):
    """Gauss-Legendre nodes and weights on [0, R] with the Gaussian envelope negligible at R."""
    x, w = np.polynomial.legendre.leggauss(1000)
    r_max = np.sqrt(100.0 / omega)
    return 0.5 * r_max * (x + 1.0), 0.5 * r_max * w


def _channel_spectrum(omega, c, l):
    """Signed eigenvalues and radial eigenvectors of the l-channel kernel, ordered by decreasing magnitude."""
    r, w = _radial_grid(omega)
    norm = pair_normalization(omega, c)
    envelope = np.exp(-0.5 * omega * r * r)
    kernel = (4.0 * np.pi / (2 * l + 1)) * norm * pair_partial_wave(r[:, None], r[None, :], l, c)
    kernel = kernel * envelope[:, None] * envelope[None, :]
    scale = np.sqrt(w) * r
    values, vectors = np.linalg.eigh(scale[:, None] * kernel * scale[None, :])
    order = np.argsort(-np.abs(values))
    return values[order], vectors[:, order], r, w, scale


def natural_amplitudes(omega: float, c: float, l: int, n_keep: int) -> "np.ndarray":
    """Reference implementation."""
    omega = float(omega)
    c = float(c)
    if not omega > 0.0:
        raise ValueError("omega must be positive")
    if c < 0.0:
        raise ValueError("c must be non-negative")
    if not 0 <= int(l) <= 12 or not 1 <= int(n_keep) <= 80:
        raise ValueError("l or n_keep out of range")
    values = _channel_spectrum(omega, c, int(l))[0]
    return values[: int(n_keep)].copy()

import numpy as np


def shell_orbital_densities(omega: float, c: float, l: int, n_keep: int, r: float) -> "np.ndarray":
    """Reference implementation."""
    r = float(r)
    if r < 0.0:
        raise ValueError("r must be non-negative")
    l = int(l)
    n_keep = int(n_keep)
    amplitudes = natural_amplitudes(omega, c, l, n_keep)
    values, vectors, grid, weights, scale = _channel_spectrum(float(omega), float(c), l)
    norm = pair_normalization(omega, c)
    radial = vectors[:, :n_keep] / scale[:, None]
    row = (4.0 * np.pi / (2 * l + 1)) * norm * pair_partial_wave(np.full_like(grid, r), grid, l, c)
    row = row * np.exp(-0.5 * float(omega) * (r * r + grid * grid))
    at_r = np.einsum("i,ik->k", row * grid * grid * weights, radial) / values[:n_keep]
    out = np.empty((n_keep, 2))
    out[:, 0] = amplitudes
    out[:, 1] = (2 * l + 1) * at_r ** 2 / (4.0 * np.pi)
    return out

import numpy as np


def truncated_reduced_ontop(omega: float, c: float, r: float, occ_min: float) -> "np.ndarray":
    """Reference implementation."""
    occ_min = float(occ_min)
    if not 1e-9 <= occ_min <= 0.1:
        raise ValueError("occ_min out of range")
    first = second = norm = 0.0
    count = 0
    s_count = 0
    for l in range(13):
        table = shell_orbital_densities(omega, c, l, 80, r)
        keep = table[:, 0] ** 2 >= occ_min
        if not keep.any():
            break
        first += np.sum(table[keep, 0] * table[keep, 1])
        second += np.sum(table[keep, 0] ** 2 * table[keep, 1])
        norm += (2 * l + 1) * np.sum(table[keep, 0] ** 2)
        count += (2 * l + 1) * int(keep.sum())
        if l == 0:
            s_count = int(keep.sum())
    return np.array([(first / second) ** 2 * norm, float(count), float(s_count)])

import numpy as np
from scipy.optimize import minimize_scalar


def overestimation_at_density_maximum(omega: float, c: float, occ_min: float) -> "np.ndarray":
    """Reference implementation."""
    if not 1e-9 <= float(occ_min) <= 0.1:
        raise ValueError("occ_min out of range")
    omega = float(omega)
    upper = 5.0 / np.sqrt(omega)
    radial = lambda x: -x * x * radial_densities(omega, c, x)[1]
    grid = np.linspace(0.05 * upper, 0.95 * upper, 37)
    best = grid[int(np.argmin([radial(x) for x in grid]))]
    step = grid[1] - grid[0]
    found = minimize_scalar(radial, bounds=(best - step, best + step), method="bounded", options={"xatol": 1e-9})
    r_star = float(found.x)
    estimate = truncated_reduced_ontop(omega, c, r_star, occ_min)[0]
    densities = radial_densities(omega, c, r_star)
    ratio = estimate / densities[2]
    constants = weak_orbital_asymptotics(omega, c)
    volume, amplitude = constants[0], constants[2]
    coefficient = 6.0 * amplitude * densities[0] ** -0.125 / volume
    return np.array([r_star, ratio, coefficient])
SCICODE_GOLD_EOF
