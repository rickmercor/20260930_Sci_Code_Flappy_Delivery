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
def fit_trigonometric_well(
    barrier_kcal_per_mol: float,
    minimum_offset_angstrom: float,
    donor_acceptor_distance_angstrom: float,
) -> "np.ndarray":
    """Reference implementation: bracketed root search for each integer order next to the trial one."""
    import math
    import numpy as np
    from scipy.optimize import brentq

    hbar, proton_mass, _, avogadro, kcal = _well_constants()
    inputs = (barrier_kcal_per_mol, minimum_offset_angstrom, donor_acceptor_distance_angstrom)
    for value in inputs:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("scan inputs must be real numbers")
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError("scan inputs must be finite and positive")
    offset = float(minimum_offset_angstrom)
    target = 0.5 * float(donor_acceptor_distance_angstrom)
    if offset >= target:
        raise ValueError("the minimum offset must be smaller than half the distance")
    barrier = float(barrier_kcal_per_mol) * kcal / avogadro

    def _reduced_well(width):
        # Barrier height B, cos^2 of the minimum position and the real order m at half-width `width`.
        unit = hbar ** 2 * math.pi ** 2 / (8.0 * proton_mass * (width * 1.0e-10) ** 2)
        height = barrier / unit
        cos2 = math.cos(0.5 * math.pi * offset / width) ** 2
        return height, cos2, math.sqrt(height * cos2 * cos2 / (1.0 - cos2) ** 2 + 0.25)

    trial = _reduced_well(target)[2]
    best = None
    for order in sorted({math.floor(trial), math.ceil(trial)}):
        if order < 1:
            continue
        low, high = offset * (1.0 + 1.0e-12), max(target, 2.0 * offset)
        while _reduced_well(high)[2] <= order:
            high *= 2.0
        width = brentq(lambda w: _reduced_well(w)[2] - order, low, high,
                       xtol=1.0e-15, rtol=4.0 * np.finfo(float).eps, maxiter=500)
        key = (abs(width - target), width)
        if best is None or key < best[0]:
            best = (key, width, order)
    _, width, order = best
    height, cos2, _ = _reduced_well(width)
    return np.array([width, float(order), math.sqrt(height) / (1.0 - cos2)])


def _well_constants() -> tuple:
    """CODATA 2018 hbar (J s), proton mass (kg), Boltzmann constant (J/K), Avogadro constant (1/mol) and kcal (J)."""
    return 1.054571817e-34, 1.67262192369e-27, 1.380649e-23, 6.02214076e23, 4184.0

import numpy as np
def compute_well_energy_levels(m: int, p: float, n_levels: int) -> "np.ndarray":
    """Reference implementation: spheroidal eigenproblem in a normalized associated Legendre basis."""
    import math
    import numpy as np

    for name, value in (("m", m), ("n_levels", n_levels)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError("p must be a real number")
    if not (math.isfinite(p) and p > 0.0):
        raise ValueError("p must be finite and positive")
    return _well_spectral_problem(int(m), float(p), int(n_levels))[0]


def _well_spectral_problem(m: int, p: float, n_levels: int) -> tuple:
    """Levels, basis coefficient columns and basis degrees of the lowest n_levels states.

    With eta = sin(x) and psi = sqrt(cos x) S(eta), S solves the oblate angular
    spheroidal equation with eigenvalue eps + m^2 - 1/2 and parameter p. In the
    normalized associated Legendre functions of degrees n = m, m + 1, ... the
    operator is n(n + 1) on the diagonal minus p^2 times the matrix of eta^2,
    which couples only degrees of equal parity, so each parity block is tridiagonal.
    """
    import numpy as np
    from scipy.linalg import eigh_tridiagonal

    size = 2 * n_levels + int(2.0 * p) + 60
    degrees = m + np.arange(size, dtype=float)
    # eta P_n = a_n P_{n+1} + a_{n-1} P_{n-1} for the normalized functions.
    a = np.sqrt(((degrees + 1.0) ** 2 - m * m) / ((2.0 * degrees + 1.0) * (2.0 * degrees + 3.0)))
    a_below = np.concatenate(([0.0], a[:-1]))
    diagonal = degrees * (degrees + 1.0) - p * p * (a * a + a_below * a_below)
    coupling = -p * p * a[:-2] * a[1:-1]
    values, columns = [], []
    for parity in (0, 1):
        index = np.arange(parity, size, 2)
        block_values, block_vectors = eigh_tridiagonal(diagonal[index], coupling[index[:-1]])
        full = np.zeros((size, block_values.size))
        full[index, :] = block_vectors
        values.append(block_values)
        columns.append(full)
    values = np.concatenate(values)
    columns = np.hstack(columns)
    order = np.argsort(values, kind="stable")[:n_levels]
    return values[order] + 0.5 - m * m, columns[:, order], degrees

import numpy as np
def evaluate_well_eigenfunction(m: int, p: float, q: int, x: "np.ndarray") -> "np.ndarray":
    """Reference implementation: psi_q = sqrt(cos x) S(sin x) from the Legendre-basis coefficients."""
    import math
    import numpy as np

    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer")
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError("q must be a nonnegative integer")
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError("p must be a real number")
    if not (math.isfinite(p) and p * p > m * m - 0.25):
        raise ValueError("p must satisfy p^2 > m^2 - 1/4")
    points = np.asarray(x, dtype=float)
    if points.ndim != 1 or points.size == 0 or not np.all(np.abs(points) < 0.5 * math.pi):
        raise ValueError("x must be a nonempty 1-D array strictly inside (-pi/2, pi/2)")
    m, p, q = int(m), float(p), int(q)
    _, columns, degrees = _well_spectral_problem(m, p, q + 1)
    x_min = math.acos(((m * m - 0.25) / (p * p)) ** 0.25)
    grid = np.append(points, x_min)
    legendre, slope = _normalized_legendre(m, degrees.size, np.sin(grid))
    angular = columns[:, q] @ legendre
    angular_slope = columns[:, q] @ slope
    cosine = np.cos(grid)
    psi = np.sqrt(cosine) * angular
    dpsi = -np.sin(grid) * angular / (2.0 * np.sqrt(cosine)) + cosine ** 1.5 * angular_slope
    sign = 1.0 if psi[-1] > 0.0 else -1.0
    return sign * np.vstack([psi[:-1], dpsi[:-1]])


def _normalized_legendre(m: int, size: int, eta: "np.ndarray") -> tuple:
    """Unit-norm associated Legendre functions of degrees m .. m + size - 1 and their eta-derivatives."""
    import math
    import numpy as np

    one_minus = 1.0 - eta * eta
    values = np.zeros((size, eta.size))
    log_first = (0.5 * math.log(m + 0.5) + 0.5 * math.lgamma(2 * m + 1)
                 - m * math.log(2.0) - math.lgamma(m + 1))
    values[0] = np.exp(log_first + 0.5 * m * np.log(one_minus))
    if size > 1:
        values[1] = eta * math.sqrt(2 * m + 3.0) * values[0]
    for k in range(2, size):
        n = m + k - 1
        up = math.sqrt(((n + 1.0) ** 2 - m * m) / ((2 * n + 1.0) * (2 * n + 3.0)))
        down = math.sqrt((n * n - m * m) / ((2 * n - 1.0) * (2 * n + 1.0)))
        values[k] = (eta * values[k - 1] - down * values[k - 2]) / up
    slopes = np.empty_like(values)
    for k in range(size):
        n = m + k
        slopes[k] = -n * eta * values[k]
        if k > 0:
            ratio = math.sqrt((2 * n + 1.0) * (n - m) / ((2 * n - 1.0) * (n + m)))
            slopes[k] += (n + m) * ratio * values[k - 1]
    return values, slopes / one_minus

import numpy as np
def compute_right_moving_flux(m: int, p: float, q: int) -> "np.ndarray":
    """Reference implementation: J = (1/4) psi^2 d tan(chi)/dx at x_min with psi'' = (U - eps) psi."""
    import math
    import numpy as np

    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError("q must be a nonnegative integer")
    energy = float(compute_well_energy_levels(m, p, int(q) + 1)[q])
    if not energy < 0.0:
        raise ValueError("level q does not lie below the barrier top")
    x_min = math.acos(((m * m - 0.25) / (p * p)) ** 0.25)
    psi, dpsi = evaluate_well_eigenfunction(m, p, q, np.array([x_min]))[:, 0]
    kinetic = energy - _well_potential(x_min, m, p)
    root2 = math.sqrt(2.0)
    roots = np.roots([2.0 * kinetic, -3.0 * root2 * kinetic * psi,
                      2.0 * dpsi * dpsi, -4.0 * root2 * dpsi * dpsi * psi])
    admissible = [r.real for r in roots
                  if abs(r.imag) <= 1.0e-9 * abs(r) and r.real > abs(psi) * (1.0 + 1.0e-12)]
    if not admissible:
        raise ValueError("no admissible root for the flux regularizer")
    s = max(admissible)
    mu2 = s * s - psi * psi
    dalpha = root2 * abs(dpsi) / s  # alpha(x_min) = 1
    # With D = psi^2 + mu^2: psi^2 (tan chi)' = [alpha psi'^2 - psi psi' (alpha' - 2 psi psi' alpha / D)
    # + alpha psi^2 (eps - U)] / D.
    bracket = dpsi * dpsi - psi * dpsi * (dalpha - 2.0 * psi * dpsi / (s * s)) + psi * psi * kinetic
    return np.array([mu2, bracket / (4.0 * s * s)])


def _well_potential(x: float, m: int, p: float) -> float:
    """Trigonometric double-well potential U(x) = (m^2 - 1/4) tan^2 x - p^2 sin^2 x."""
    import math

    return (m * m - 0.25) * math.tan(x) ** 2 - p * p * math.sin(x) ** 2

import numpy as np
def compute_transmission_probability(m: int, p: float, q: int, mu2: float) -> float:
    """Reference implementation: closed forms of the matched amplitude in terms of tan(chi) and its slope."""
    import math
    import numpy as np
    from scipy.optimize import brentq

    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError("q must be a nonnegative integer")
    if isinstance(mu2, bool) or not isinstance(mu2, (int, float, np.integer, np.floating)):
        raise ValueError("mu2 must be a real number")
    if not (math.isfinite(mu2) and mu2 > 0.0):
        raise ValueError("mu2 must be finite and positive")
    energy = float(compute_well_energy_levels(m, p, int(q) + 1)[q])
    if not energy < 0.0:
        raise ValueError("level q does not lie below the barrier top")
    x_min = math.acos(((m * m - 0.25) / (p * p)) ** 0.25)
    turning = brentq(lambda x: _well_potential(x, m, p) - energy, 0.0, x_min,
                     xtol=1.0e-15, rtol=4.0 * np.finfo(float).eps, maxiter=500)
    values = evaluate_well_eigenfunction(m, p, q, np.array([x_min, turning]))
    (psi_min, psi), (dpsi_min, dpsi) = values
    rate = math.sqrt(2.0) * abs(dpsi_min) / math.sqrt(psi_min * psi_min + mu2)
    alpha = math.exp(rate * (turning - x_min))
    d2psi = (_well_potential(turning, m, p) - energy) * psi  # zero at a turning point
    numerator, d_numerator = dpsi * alpha, d2psi * alpha + dpsi * rate * alpha
    denominator = psi * (psi * psi + mu2)
    d_denominator = dpsi * (3.0 * psi * psi + mu2)
    tan_chi = -numerator / denominator
    d_tan_chi = -(d_numerator * denominator - numerator * d_denominator) / denominator ** 2
    if q % 2 == 1:
        return float(1.0 / (1.0 + tan_chi * tan_chi))
    lead = dpsi * (1.0 + tan_chi * tan_chi) + psi * tan_chi * d_tan_chi
    return float(dpsi * dpsi * (1.0 + tan_chi * tan_chi)
                 / (lead * lead + psi * psi * d_tan_chi * d_tan_chi))

import numpy as np
def compute_reduced_rate_constant(m: int, p: float, beta: float) -> float:
    """Reference implementation: doubling level count until the Boltzmann tail is negligible."""
    import math
    import numpy as np

    if isinstance(beta, bool) or not isinstance(beta, (int, float, np.integer, np.floating)):
        raise ValueError("beta must be a real number")
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError("beta must be finite and positive")
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError("p must be a real number")
    if not (math.isfinite(p) and p * p > m * m - 0.25):
        raise ValueError("p must satisfy p^2 > m^2 - 1/4")
    count = 32
    while True:
        levels = compute_well_energy_levels(m, p, count)
        if beta * (levels[-1] - levels[0]) > 45.0:
            break
        count *= 2
        if count > 1024:
            raise ValueError("Boltzmann sums do not converge within 1024 levels")
    boltzmann = np.exp(-beta * (levels - levels[0]))
    weights = np.ones(count)
    for q in np.flatnonzero(levels < 0.0):
        mu2, flux = compute_right_moving_flux(m, p, int(q))
        weights[q] = flux * compute_transmission_probability(m, p, int(q), float(mu2))
    return float(np.sum(boltzmann * weights) / np.sum(boltzmann))

import numpy as np
def compute_proton_transfer_rate(
    barrier_kcal_per_mol: float = 5.0,
    minimum_offset_angstrom: float = 0.55,
    donor_acceptor_distance_angstrom: float = 2.78,
    temperature_kelvin: float = 400.0,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import math
    import numpy as np

    if isinstance(temperature_kelvin, bool) or not isinstance(
            temperature_kelvin, (int, float, np.integer, np.floating)):
        raise ValueError("temperature must be a real number")
    if not (math.isfinite(temperature_kelvin) and temperature_kelvin > 0.0):
        raise ValueError("temperature must be finite and positive")
    width, order, p = fit_trigonometric_well(
        barrier_kcal_per_mol, minimum_offset_angstrom, donor_acceptor_distance_angstrom)
    hbar, proton_mass, boltzmann, _, _ = _well_constants()
    width_m = float(width) * 1.0e-10
    beta = hbar ** 2 * math.pi ** 2 / (8.0 * proton_mass * width_m ** 2 * boltzmann * temperature_kelvin)
    reduced_rate = compute_reduced_rate_constant(int(round(order)), float(p), beta)
    return float(reduced_rate * hbar / (proton_mass * width_m ** 2) / 1.0e12)
SCICODE_GOLD_EOF
