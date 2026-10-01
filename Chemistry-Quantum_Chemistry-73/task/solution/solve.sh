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
from scipy.special import k0, k1, modstruve


def _integral_k0(y: "np.ndarray") -> "np.ndarray":
    """Integral of the modified Bessel function K_0 from 0 to y, elementwise for y >= 0."""
    import numpy as np
    from scipy.special import k0, k1, modstruve
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = y < 1e-2
    large = y > 40.0
    mid = ~(small | large)
    ys = np.where(y[small] > 0.0, y[small], 1.0)
    lg = np.log(ys / 2.0)
    series = ys * (1.0 - np.euler_gamma - lg) + ys ** 3 / 12.0 * (-lg - np.euler_gamma + 4.0 / 3.0)
    out[small] = np.where(y[small] > 0.0, series, 0.0)
    ym = y[mid]
    out[mid] = 0.5 * np.pi * ym * (k0(ym) * modstruve(-1, ym) + k1(ym) * modstruve(0, ym))
    out[large] = 0.5 * np.pi
    return out


def _integral_t_k0(y: "np.ndarray") -> "np.ndarray":
    """Integral of t K_0(t) from 0 to y, elementwise for y >= 0, equal to 1 - y K_1(y)."""
    import numpy as np
    from scipy.special import k1
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = y < 1e-2
    ys = np.where(y[small] > 0.0, y[small], 1.0)
    lg = np.log(ys / 2.0)
    series = -(ys ** 2 / 2.0) * (lg + np.euler_gamma - 0.5) - (ys ** 4 / 16.0) * (lg + np.euler_gamma - 1.25)
    out[small] = np.where(y[small] > 0.0, series, 0.0)
    out[~small] = 1.0 - y[~small] * k1(y[~small])
    return out


def polarized_exchange_per_electron(density: "np.ndarray", softening: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    n = np.asarray(density, dtype=float)
    if not np.all(np.isfinite(n)) or np.any(n < 0.0):
        raise ValueError("densities must be finite and non-negative")
    if not softening > 0.0:
        raise ValueError("softening must be strictly positive")
    b = float(softening)
    # With k_F = pi n for one spin channel, eps_x = -I(pi n b) / (2 pi^2 b^2 n), where
    # I(q) = integral over the real line of sin^2(q t) / (t^2 sqrt(t^2 + 1)) dt = 2 q F(2q) - G(2q),
    # F(y) = integral_0^y K_0 and G(y) = integral_0^y t K_0(t) dt, because I''(q) = 4 K_0(2q) and I(0) = I'(0) = 0.
    q = np.pi * n * b
    integral = 2.0 * q * _integral_k0(2.0 * q) - _integral_t_k0(2.0 * q)
    out = np.zeros_like(n)
    pos = n > 0.0
    out[pos] = -integral[pos] / (2.0 * np.pi ** 2 * b ** 2 * n[pos])
    return out

import numpy as np
from scipy.special import k0, k1, modstruve


def _integral_k0(y: "np.ndarray") -> "np.ndarray":
    """Integral of the modified Bessel function K_0 from 0 to y, elementwise for y >= 0."""
    import numpy as np
    from scipy.special import k0, k1, modstruve
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    small = y < 1e-2
    large = y > 40.0
    mid = ~(small | large)
    ys = np.where(y[small] > 0.0, y[small], 1.0)
    lg = np.log(ys / 2.0)
    series = ys * (1.0 - np.euler_gamma - lg) + ys ** 3 / 12.0 * (-lg - np.euler_gamma + 4.0 / 3.0)
    out[small] = np.where(y[small] > 0.0, series, 0.0)
    ym = y[mid]
    out[mid] = 0.5 * np.pi * ym * (k0(ym) * modstruve(-1, ym) + k1(ym) * modstruve(0, ym))
    out[large] = 0.5 * np.pi
    return out


def polarized_exchange_potential(density: "np.ndarray", softening: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    n = np.asarray(density, dtype=float)
    if not np.all(np.isfinite(n)) or np.any(n < 0.0):
        raise ValueError("densities must be finite and non-negative")
    if not softening > 0.0:
        raise ValueError("softening must be strictly positive")
    b = float(softening)
    # d[n eps_x]/dn = -F(2 pi n b) / (pi b) with F(y) = integral_0^y K_0
    return -_integral_k0(2.0 * np.pi * n * b) / (np.pi * b)

import numpy as np
from scipy.linalg import eigh_tridiagonal


def soft_coulomb_states(nuclei: "np.ndarray", softening: float, spacing: float, half_width: float, n_states: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import eigh_tridiagonal
    centers = np.asarray(nuclei, dtype=float).ravel()
    if centers.size == 0:
        raise ValueError("at least one nucleus is required")
    if not (softening > 0.0 and spacing > 0.0 and half_width > 0.0):
        raise ValueError("softening, spacing and half-width must be positive")
    ratio = 2.0 * half_width / spacing
    m = int(round(ratio))
    if abs(ratio - m) > 1e-9 * max(1.0, ratio) or m < 1:
        raise ValueError("2 * half_width / spacing must be an integer")
    if not 1 <= int(n_states) <= m + 1:
        raise ValueError("n_states must lie between 1 and M + 1")
    x = -half_width + spacing * np.arange(m + 1)
    v = np.zeros_like(x)
    for c in centers:
        v -= 1.0 / np.sqrt((x - c) ** 2 + softening ** 2)
    diag = 1.0 / spacing ** 2 + v
    off = np.full(m, -0.5 / spacing ** 2)
    k = int(n_states)
    w, u = eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1))
    phi = u / np.sqrt(spacing)
    for i in range(k):
        col = phi[:, i]
        first = int(np.argmax(np.abs(col) >= 1e-3 * np.max(np.abs(col))))
        if col[first] < 0.0:
            phi[:, i] = -col
    out = np.empty((k, m + 2))
    out[:, 0] = w
    out[:, 1:] = phi.T
    return out

import numpy as np


def hartree_exchange_energy(density_up: "np.ndarray", density_down: "np.ndarray", spacing: float, softening: float) -> float:
    """Reference implementation."""
    import numpy as np
    nu = np.asarray(density_up, dtype=float)
    nd = np.asarray(density_down, dtype=float)
    if nu.ndim != 1 or nd.ndim != 1 or nu.size != nd.size:
        raise ValueError("densities must be one-dimensional arrays of equal length")
    if not (spacing > 0.0 and softening > 0.0):
        raise ValueError("spacing and softening must be positive")
    if not (np.all(np.isfinite(nu)) and np.all(np.isfinite(nd))) or np.any(nu < 0.0) or np.any(nd < 0.0):
        raise ValueError("densities must be finite and non-negative")
    idx = np.arange(nu.size)
    kernel = 1.0 / np.sqrt(((idx[:, None] - idx[None, :]) * spacing) ** 2 + softening ** 2)
    n = nu + nd
    hartree = 0.5 * spacing ** 2 * float(n @ (kernel @ n))
    exchange = spacing * float(np.sum(nu * polarized_exchange_per_electron(nu, softening))
                               + np.sum(nd * polarized_exchange_per_electron(nd, softening)))
    return float(hartree + exchange)

import numpy as np
from scipy.linalg import eigh_tridiagonal


def lsda_ground_state_energy(nuclei: "np.ndarray", softening: float, spacing: float, half_width: float) -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import eigh_tridiagonal
    centers = np.sort(np.asarray(nuclei, dtype=float).ravel())
    if centers.size == 0 or not np.allclose(centers, -centers[::-1], rtol=0.0, atol=1e-9):
        raise ValueError("the nuclear configuration must be symmetric under x -> -x")
    guess = soft_coulomb_states(centers, softening, spacing, half_width, 1)
    phi = guess[0, 1:]
    m = phi.size - 1
    x = -half_width + spacing * np.arange(m + 1)
    v = np.zeros_like(x)
    for c in centers:
        v -= 1.0 / np.sqrt((x - c) ** 2 + softening ** 2)
    idx = np.arange(m + 1)
    kernel = 1.0 / np.sqrt(((idx[:, None] - idx[None, :]) * spacing) ** 2 + softening ** 2)
    off = np.full(m, -0.5 / spacing ** 2)
    n = phi ** 2
    hist = []
    for _ in range(5000):
        veff = v + spacing * (kernel @ n) + polarized_exchange_potential(n, softening)
        veff = 0.5 * (veff + veff[::-1])
        _, u = eigh_tridiagonal(1.0 / spacing ** 2 + veff, off, select="i", select_range=(0, 0))
        phi = u[:, 0] / np.sqrt(spacing)
        new = phi ** 2
        res = new - n
        if spacing * np.abs(res).sum() < 1e-12:
            break
        hist.append((n.copy(), res.copy()))
        hist = hist[-6:]
        if len(hist) >= 2:
            d_res = np.array([hist[i + 1][1] - hist[i][1] for i in range(len(hist) - 1)]).T
            d_n = np.array([hist[i + 1][0] - hist[i][0] for i in range(len(hist) - 1)]).T
            coef = np.linalg.lstsq(d_res, res, rcond=None)[0]
            n = n + 0.3 * res - (d_n + 0.3 * d_res) @ coef
            n = np.maximum(n, 0.0)
            n = n / (spacing * n.sum())
        else:
            n = n + 0.3 * res
    else:
        raise RuntimeError("self-consistent field did not converge")
    padded = np.concatenate(([0.0], phi, [0.0]))
    kinetic = -0.5 * spacing * float(np.sum(phi * (padded[2:] - 2.0 * phi + padded[:-2]))) / spacing ** 2
    potential = spacing * float(np.sum(v * phi ** 2))
    zero = np.zeros_like(phi)
    return float(kinetic + potential + hartree_exchange_energy(phi ** 2, zero, spacing, softening))

import numpy as np


def maximally_localized_orbital(orbital_one: "np.ndarray", orbital_two: "np.ndarray", spacing: float, half_width: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    a = np.asarray(orbital_one, dtype=float).ravel()
    b = np.asarray(orbital_two, dtype=float).ravel()
    m = int(round(2.0 * half_width / spacing))
    if a.size != m + 1 or b.size != m + 1:
        raise ValueError("orbitals must have length 2L / h_x + 1")
    gram = spacing * np.array([[a @ a, a @ b], [a @ b, b @ b]])
    if np.max(np.abs(gram - np.eye(2))) > 1e-8:
        raise ValueError("orbitals must be orthonormal")
    x = -half_width + spacing * np.arange(m + 1)
    weight = np.where(x < -1e-12 * half_width, 1.0, np.where(x > 1e-12 * half_width, 0.0, 0.5))
    p = spacing * np.array([[a @ (weight * a), a @ (weight * b)], [a @ (weight * b), b @ (weight * b)]])
    w, vecs = np.linalg.eigh(p)
    if w[1] - w[0] < 1e-12:
        raise ValueError("the maximally localized combination is not unique")
    c = vecs[:, 1]
    phi = c[0] * a + c[1] * b
    if spacing * float(weight @ phi) < 0.0:
        phi = -phi
    return phi

import numpy as np


def _one_electron_expectation(phi: "np.ndarray", centers: "np.ndarray", softening: float, spacing: float, half_width: float) -> float:
    """Finite-difference expectation value of T + v for a normalized grid orbital."""
    import numpy as np
    x = -half_width + spacing * np.arange(phi.size)
    v = np.zeros_like(x)
    for c in centers:
        v -= 1.0 / np.sqrt((x - c) ** 2 + softening ** 2)
    padded = np.concatenate(([0.0], phi, [0.0]))
    kinetic = -0.5 * spacing * float(np.sum(phi * (padded[2:] - 2.0 * phi + padded[:-2]))) / spacing ** 2
    return kinetic + spacing * float(np.sum(v * phi ** 2))


def dissociation_errors(separation: float, softening: float, spacing: float, half_width: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if not separation > 0.0:
        raise ValueError("separation must be positive")
    atom = np.array([0.0])
    ion = np.array([-0.5 * separation, 0.5 * separation])
    s_atom = soft_coulomb_states(atom, softening, spacing, half_width, 1)
    s_ion = soft_coulomb_states(ion, softening, spacing, half_width, 2)
    e_h, phi_h = s_atom[0, 0], s_atom[0, 1:]
    e_g, e_u = s_ion[0, 0], s_ion[1, 0]
    psi_g, psi_u = s_ion[0, 1:], s_ion[1, 1:]
    zero = np.zeros_like(phi_h)
    exact = e_g - e_h
    scf = lsda_ground_state_energy(ion, softening, spacing, half_width) - lsda_ground_state_energy(atom, softening, spacing, half_width)
    atom_dc = e_h + hartree_exchange_energy(phi_h ** 2, zero, spacing, softening)
    dc_exact = e_g + hartree_exchange_energy(psi_g ** 2, zero, spacing, softening) - atom_dc
    phi_loc = maximally_localized_orbital(psi_g, psi_u, spacing, half_width)
    loc_energy = _one_electron_expectation(phi_loc, ion, softening, spacing, half_width)
    dc_loc = loc_energy + hartree_exchange_energy(phi_loc ** 2, zero, spacing, softening) - atom_dc
    return np.array([scf - exact, dc_exact - exact, dc_loc - exact, e_u - e_g])

import numpy as np
from scipy.optimize import brentq


def _crossover_gap(separation: float, softening: float, spacing: float, half_width: float) -> float:
    """Localized-orbital error magnitude minus self-consistent error magnitude at one separation."""
    err = dissociation_errors(separation, softening, spacing, half_width)
    return abs(float(err[2])) - abs(float(err[0]))


def crossover_separation(softening: float, spacing: float, half_width: float, r_min: float, r_max: float, n_scan: int) -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    if not (0.0 < r_min < r_max) or int(n_scan) < 2:
        raise ValueError("require 0 < r_min < r_max and n_scan >= 2")
    radii = np.linspace(r_min, r_max, int(n_scan))
    if _crossover_gap(radii[0], softening, spacing, half_width) <= 0.0:
        raise ValueError("the scan starts at or beyond the crossover")
    for k in range(1, radii.size):
        current = _crossover_gap(radii[k], softening, spacing, half_width)
        if current <= 0.0:
            return float(brentq(lambda r: _crossover_gap(r, softening, spacing, half_width), radii[k - 1], radii[k], xtol=1e-10))
    raise ValueError("no crossover within the scanned separations")
SCICODE_GOLD_EOF
