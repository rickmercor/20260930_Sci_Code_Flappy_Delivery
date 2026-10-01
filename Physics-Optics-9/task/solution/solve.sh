#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math
import numpy as np
from scipy.special import spherical_jn


def riccati_wave_data(ell: int, x: "np.ndarray") -> "np.ndarray":
    if ell not in (1, 2, 3, 4):
        raise ValueError("ell must be 1, 2, 3 or 4")
    x = np.asarray(x, dtype=complex)
    if np.any(x == 0):
        raise ValueError("zero argument is outside this radial-wave contract")
    psi = x * spherical_jn(ell, x)
    dpsi = spherical_jn(ell, x) + x * spherical_jn(ell, x, derivative=True)
    ode = ell * (ell + 1) / x**2 - 1
    data = [psi, dpsi, ode * psi]
    powers = np.arange(ell + 1)
    for sign in (1, -1):
        coeff = np.array(
            [
                math.factorial(ell + j)
                / (math.factorial(j) * math.factorial(ell - j))
                * (sign * 0.5j) ** j
                for j in range(ell + 1)
            ]
        )
        poly = np.sum(coeff * x[..., None] ** (-powers), axis=-1)
        dpoly = np.sum(
            -powers * coeff * x[..., None] ** (-powers - 1), axis=-1
        )
        phase = (-sign * 1j) ** (ell + 1) * np.exp(sign * 1j * x)
        xi = phase * poly
        dxi = phase * (sign * 1j * poly + dpoly)
        data.extend([xi, dxi, ode * xi])
    return np.stack(data, axis=-1)

import math

import numpy as np


def _tm_small_argument_derivative(
    ell: int, n: float, x: "np.ndarray"
) -> "np.ndarray":
    """Evaluate D' from its regular series when abs(x) < 0.1."""
    # D = (-i)^(ell+1) n^ell exp(ix) P(x). Collect nonnegative
    # powers before differentiating the Bessel/Hankel product.
    coefficients = np.zeros(ell + 26, dtype=complex)
    regular = 1.0 / math.prod(range(1, 2 * ell + 2, 2))
    for k in range(12):
        for j in range(ell + 1):
            hankel = (
                math.factorial(ell + j)
                / (math.factorial(j) * math.factorial(ell - j))
                * (0.5j) ** j
            )
            power = ell + 2 * k - j
            coefficients[power] += (
                regular * hankel * (ell + 1 + 2 * k + n * n * j)
            )
            coefficients[power + 1] -= 1j * n * n * regular * hankel
        regular *= -n * n / (2 * (k + 1) * (2 * ell + 2 * k + 3))
    derivative = 1j * coefficients
    derivative[:-1] += np.arange(1, len(coefficients)) * coefficients[1:]
    return (
        (-1j) ** (ell + 1)
        * n**ell
        * np.exp(1j * x)
        * np.polynomial.polynomial.polyval(x, derivative)
    )


def tm_boundary_data(
    ell: int, n: float, x: "np.ndarray"
) -> "np.ndarray":
    if not 2.0 <= n <= 3.0:
        raise ValueError("n must lie in [2, 3]")
    x = np.asarray(x, dtype=complex)
    interior = riccati_wave_data(ell, n * x)
    outer = riccati_wave_data(ell, x)
    p, p1, p2 = interior[..., 0], interior[..., 1], interior[..., 2]
    xp, xp1, xp2 = outer[..., 3], outer[..., 4], outer[..., 5]
    xm, xm1 = outer[..., 6], outer[..., 7]
    numerator = -p1 * xm + n * p * xm1
    denominator = p1 * xp - n * p * xp1
    derivative = n * p2 * xp + (1 - n**2) * p1 * xp1 - n * p * xp2
    small = np.abs(x) < 0.1
    if np.any(small):
        derivative = np.array(derivative, copy=True)
        derivative[small] = _tm_small_argument_derivative(ell, n, x[small])
    return np.stack([numerator, denominator, derivative], axis=-1)

import numpy as np


def physical_tm_poles(
    ell: int, n: float, x_cut: float
) -> "np.ndarray":
    if not 8.0 <= x_cut <= 22.0:
        raise ValueError("x_cut must lie in [8, 22]")
    real = np.linspace(-x_cut, x_cut, int(2.6 * n * x_cut) + 1)
    imag = np.linspace(-4.0, -0.02, 7)
    roots = (real[:, None] + 1j * imag).ravel()
    active = np.ones(roots.size, dtype=bool)
    for _ in range(80):
        if not np.any(active):
            break
        values = tm_boundary_data(ell, n, roots[active])
        correction = values[:, 1] / values[:, 2]
        correction /= np.maximum(1.0, np.abs(correction))
        indices = np.flatnonzero(active)
        roots[active] -= correction
        roots = np.clip(roots.real, -x_cut - 2, x_cut + 2) + 1j * np.clip(
            roots.imag, -6.0, -1e-7
        )
        active[indices[np.abs(correction) < 1e-12]] = False
    values = tm_boundary_data(ell, n, roots)
    valid = (
        (np.abs(values[:, 1] / values[:, 2]) < 1e-10)
        & (np.abs(roots.real) < x_cut)
        & (roots.imag > -4.0)
        & (roots.imag < 0.0)
    )
    unique = []
    for root in roots[valid]:
        if abs(root.real) < 1e-9:
            root = complex(0.0, root.imag)
        if not any(abs(root - previous) < 1e-7 for previous in unique):
            unique.append(root)
    poles = np.array(
        sorted(unique, key=lambda z: (z.real, z.imag)), dtype=complex
    )
    corners = [-x_cut - 4j, x_cut - 4j, x_cut - 1e-8j, -x_cut - 1e-8j]
    contour = np.concatenate(
        [
            np.linspace(corners[j], corners[(j + 1) % 4], 1024, endpoint=False)
            for j in range(4)
        ]
    )
    determinant = tm_boundary_data(ell, n, contour)[:, 1]
    increments = np.angle(np.roll(determinant, -1) / determinant)
    count = np.sum(increments) / (2 * np.pi)
    if abs(count - round(count)) > 1e-6 or len(poles) != round(count):
        raise ValueError("complete simple-pole set was not resolved")
    return poles

import math
import numpy as np


def tm_channel_data(ell: int, n: float) -> "np.ndarray":
    if ell not in (1, 2, 3, 4) or not 2.0 <= n <= 3.0:
        raise ValueError("unsupported angular momentum or refractive index")
    polynomial = np.poly1d(
        [
            math.factorial(ell + j)
            / (math.factorial(j) * math.factorial(ell - j))
            * (-0.5j) ** j
            for j in range(ell + 1)
        ]
    )
    equation = (
        np.poly1d([1, 0]) * np.polyder(polynomial)
        - np.poly1d([1j, ell]) * polynomial
    )
    poles = np.roots(equation)
    poles.real[np.abs(poles.real) < 1e-9] = 0.0
    poles = poles[np.lexsort((poles.imag, poles.real))]
    radial = riccati_wave_data(ell, poles)
    boundary = tm_boundary_data(ell, n, poles)
    residues = boundary[:, 0] / boundary[:, 1] * radial[:, 4] / radial[:, 8]
    return np.column_stack([poles, residues])

import numpy as np


def tm_physical_residues(
    ell: int, n: float, poles: "np.ndarray"
) -> "np.ndarray":
    poles = np.asarray(poles, dtype=complex)
    boundary = tm_boundary_data(ell, n, poles)
    radial = riccati_wave_data(ell, poles)
    return boundary[:, 0] / boundary[:, 2] * radial[:, 4] / radial[:, 7]

import numpy as np


def subtracted_pole_sum(
    x: "np.ndarray",
    poles: "np.ndarray",
    residues: "np.ndarray",
    at_zero: complex,
) -> "np.ndarray":
    x = np.asarray(x, dtype=complex)
    poles = np.asarray(poles, dtype=complex)
    residues = np.asarray(residues, dtype=complex)
    if poles.shape != residues.shape or np.any(poles == 0):
        raise ValueError(
            "poles and residues must match and poles must be nonzero"
        )
    terms = residues * x[..., None] / (poles * (x[..., None] - poles))
    return np.asarray(at_zero + np.sum(terms, axis=-1), dtype=complex)

import numpy as np
from scipy.special import roots_legendre


def tm_spectral_errors(
    ell: int,
    n: float,
    poles: "np.ndarray",
    residues: "np.ndarray",
    channels: "np.ndarray",
    left: float,
    right: float,
    order: int,
) -> "np.ndarray":
    if (
        not 0.1 <= left < right <= 8.0
        or not 16 <= order <= 800
        or int(order) != order
    ):
        raise ValueError("unsupported band or quadrature order")
    poles = np.asarray(poles, dtype=complex)
    residues = np.asarray(residues, dtype=complex)
    channels = np.asarray(channels, dtype=complex)
    nodes, weights = roots_legendre(order)
    x = left + 0.5 * (right - left) * (nodes + 1)
    weights = 0.5 * (right - left) * weights
    radial = riccati_wave_data(ell, x)
    boundary = tm_boundary_data(ell, n, x)
    exact = radial[:, 4] / radial[:, 7] * boundary[:, 0] / boundary[:, 1]
    physical = subtracted_pole_sum(x, poles, residues, -1.0)
    complete = subtracted_pole_sum(
        x,
        np.concatenate([poles, channels[:, 0]]),
        np.concatenate([residues, channels[:, 1]]),
        -1.0,
    )
    weighted = weights * x**2
    return np.array(
        [
            np.sum(weighted * np.abs(exact - physical) ** 2),
            np.sum(weighted * np.abs(exact - complete) ** 2),
        ],
        dtype=float,
    )

import numpy as np


def scattering_pole_audit(
    n: float, ell: int, x_cut: float
) -> "np.ndarray":
    poles = physical_tm_poles(ell, n, x_cut)
    residues = tm_physical_residues(ell, n, poles)
    channels = tm_channel_data(ell, n)
    errors = tm_spectral_errors(
        ell, n, poles, residues, channels, 0.35, 7.4, 400
    )
    return np.array(
        [
            np.log10(errors[0] / errors[1]),
            errors[0],
            errors[1],
            len(poles),
            len(channels),
        ],
        dtype=float,
    )
SCICODE_GOLD_EOF
