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


def spectral_operators_and_weights(nx: int, nz: int, d: float, h: float, alpha: float) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not isinstance(nz, (int, np.integer)) or isinstance(nz, (bool, np.bool_)) or nz < 2:
        raise ValueError("nz must be an integer at least 2")
    if not np.isfinite(d) or d <= 0.0 or not np.isfinite(h) or h <= 0.0:
        raise ValueError("d and h must be positive and finite")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")

    x = np.arange(nx, dtype=float) * (float(d) / nx)
    p = np.arange(-nx // 2, nx // 2, dtype=int)
    alpha_p = float(alpha) + 2.0 * np.pi * p / float(d)
    qmat = np.exp(-1j * alpha_p[:, None] * x[None, :]) / nx
    qinv = np.exp(1j * x[:, None] * alpha_p[None, :])
    dx_1d = qinv @ np.diag(1j * alpha_p) @ qmat

    r = np.arange(nz + 1, dtype=int)
    zeta = np.cos(np.pi * r / nz)
    c = np.ones(nz + 1, dtype=float)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** r
    differences = zeta[:, None] - zeta[None, :]
    d_cheb = (c[:, None] / c[None, :]) / (differences + np.eye(nz + 1))
    d_cheb -= np.diag(np.sum(d_cheb, axis=1))
    dz_1d = d_cheb / float(h)

    identity_x = np.eye(nx, dtype=complex)
    identity_z = np.eye(nz + 1, dtype=complex)
    dx = np.kron(dx_1d, identity_z)
    dz = np.kron(identity_x, dz_1d)

    cosine_moments = np.cos(
        np.pi * np.arange(nz + 1)[:, None] * np.arange(nz + 1)[None, :] / nz
    )
    moments = np.zeros(nz + 1, dtype=float)
    for order in range(0, nz + 1, 2):
        moments[order] = 2.0 * float(h) / (1.0 - order * order)
    weights_z = np.linalg.solve(cosine_moments, moments)
    normalized_weights = np.tile(weights_z / (2.0 * float(h) * nx), nx)
    weights = np.diag(normalized_weights.astype(complex))
    return np.stack((dx, dz, weights)).astype(np.complex128)

import numpy as np


def radiation_and_bohren_data(nx: int, d: float, h: float, wavelength: float, theta: float) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not np.isfinite(d) or d <= 0.0 or not np.isfinite(h) or h <= 0.0:
        raise ValueError("d and h must be positive and finite")
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength must be positive and finite")
    if not np.isfinite(theta) or abs(float(theta)) >= 0.5 * np.pi:
        raise ValueError("theta must be finite with abs(theta) < pi/2")

    k0 = 2.0 * np.pi / float(wavelength)
    alpha = k0 * np.sin(float(theta))
    p = np.arange(-nx // 2, nx // 2, dtype=int)
    alpha_p = alpha + 2.0 * np.pi * p / float(d)
    radicand = k0 * k0 - alpha_p * alpha_p
    wood_tol = 64.0 * np.finfo(float).eps * k0 * k0
    if np.any(np.abs(radicand) <= wood_tol):
        raise ValueError("a retained Fourier order lies at a Wood anomaly")
    gamma_p = np.where(
        radicand > 0.0,
        np.sqrt(np.maximum(radicand, 0.0)).astype(complex),
        1j * np.sqrt(np.maximum(-radicand, 0.0)),
    )
    data = np.zeros((nx, 5), dtype=np.complex128)
    data[:, 0] = alpha_p
    data[:, 1] = gamma_p
    data[:, 2] = -1j * gamma_p
    zero_slot = nx // 2
    phase = np.exp(-1j * gamma_p[zero_slot] * float(h))
    data[zero_slot, 3] = 2j * gamma_p[zero_slot] * phase * 0.5
    data[zero_slot, 4] = 2j * gamma_p[zero_slot] * phase * (0.5j)
    return data

import numpy as np


def centered_channel_profiles(sign: int, nx: int, nz: int, d: float, h: float, t: float, sharpness: float, wavelength: float, chi_bar: float, chi_amplitude: float, lateral_scale: float, delta_center: float) -> "np.ndarray":
    """Reference implementation."""
    if sign not in (-1, 1) or isinstance(sign, (bool, np.bool_)):
        raise ValueError("sign must equal +1 or -1")
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not isinstance(nz, (int, np.integer)) or isinstance(nz, (bool, np.bool_)) or nz < 2:
        raise ValueError("nz must be an integer at least 2")
    if not np.isfinite(d) or d <= 0.0 or not np.isfinite(h) or h <= 0.0:
        raise ValueError("d and h must be positive and finite")
    if not np.isfinite(t) or t <= 0.0 or t >= h:
        raise ValueError("t must satisfy 0 < t < h")
    if not np.isfinite(sharpness) or sharpness <= 0.0 or not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("sharpness and wavelength must be positive and finite")
    scalars = (chi_bar, chi_amplitude, lateral_scale, delta_center)
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("chirality and deformation scalars must be finite")

    x = np.arange(nx, dtype=float)[:, None] * (float(d) / nx)
    z = (float(h) * np.cos(np.pi * np.arange(nz + 1) / nz))[None, :]
    phi = 0.5 * (
        np.tanh(float(sharpness) * (z + float(t)))
        - np.tanh(float(sharpness) * (z - float(t)))
    )
    harmonics = (
        0.23 * np.cos(2.0 * np.pi * x / float(d))
        - 0.19 * np.sin(4.0 * np.pi * x / float(d))
        + 0.13 * np.cos(6.0 * np.pi * x / float(d))
        + 0.07 * np.sin(8.0 * np.pi * x / float(d))
    )
    lateral = 1.0 + float(lateral_scale) * harmonics
    envelope = float(chi_amplitude) * phi * (1.0 + 0.14 * z / float(h)) * lateral
    envelope = envelope.reshape(-1).astype(float)
    k0 = 2.0 * np.pi / float(wavelength)
    centered_chi = float(chi_bar) - float(delta_center) * envelope
    if np.max(np.abs(k0 * centered_chi)) >= 1.0:
        raise ValueError("the centered chirality violates abs(k0*chi) < 1")
    rho_center = 1.0 - sign * k0 * float(chi_bar) + sign * k0 * float(delta_center) * envelope
    rho_one = sign * k0 * envelope
    return np.stack((envelope, rho_center, rho_one)).astype(np.float64)

import numpy as np


def assemble_transmission_operator(profiles: "np.ndarray", operators: "np.ndarray", radiation: "np.ndarray", nx: int, nz: int) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not isinstance(nz, (int, np.integer)) or isinstance(nz, (bool, np.bool_)) or nz < 2:
        raise ValueError("nz must be an integer at least 2")
    n_total = nx * (nz + 1)
    profiles_array = np.asarray(profiles)
    operators_array = np.asarray(operators)
    radiation_array = np.asarray(radiation)
    if profiles_array.shape != (3, n_total):
        raise ValueError("profiles must have shape (3, nx*(nz+1))")
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")
    if radiation_array.shape != (nx, 5):
        raise ValueError("radiation must have shape (nx, 5)")

    rho_center = np.asarray(profiles_array[1], dtype=float)
    dx = np.asarray(operators_array[0], dtype=complex)
    dz = np.asarray(operators_array[1], dtype=complex)
    alpha_p = np.asarray(radiation_array[:, 0], dtype=complex).real
    dtn_multiplier = np.asarray(radiation_array[:, 2], dtype=complex)
    spacing = alpha_p[1] - alpha_p[0]
    d = 2.0 * np.pi / spacing
    alpha = alpha_p[nx // 2]
    x = np.arange(nx, dtype=float) * d / nx
    qmat = np.exp(-1j * alpha_p[:, None] * x[None, :]) / nx
    qinv = np.exp(1j * x[:, None] * alpha_p[None, :])
    dtn = qinv @ np.diag(dtn_multiplier) @ qmat

    k0_squared = alpha * alpha + radiation_array[nx // 2, 1].real ** 2
    rho_diag = np.diag(rho_center.astype(complex))
    matrix = rho_diag @ (dx @ rho_diag @ dx + dz @ rho_diag @ dz)
    matrix += k0_squared * np.eye(n_total, dtype=complex)
    top_columns = np.arange(nx) * (nz + 1)
    bottom_columns = top_columns + nz
    for j in range(nx):
        top_row = j * (nz + 1)
        matrix[top_row, :] = -rho_center[top_row] * dz[top_row, :]
        matrix[top_row, top_columns] -= dtn[j, :]
        bottom_row = top_row + nz
        matrix[bottom_row, :] = rho_center[bottom_row] * dz[bottom_row, :]
        matrix[bottom_row, bottom_columns] -= dtn[j, :]
    return matrix.astype(np.complex128)

import numpy as np


def assemble_transmission_source(previous: "np.ndarray", previous_two: "np.ndarray", profiles: "np.ndarray", operators: "np.ndarray", nx: int, nz: int) -> "np.ndarray":
    """Reference implementation."""
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 4 or nx % 2:
        raise ValueError("nx must be an even integer at least 4")
    if not isinstance(nz, (int, np.integer)) or isinstance(nz, (bool, np.bool_)) or nz < 2:
        raise ValueError("nz must be an integer at least 2")
    n_total = nx * (nz + 1)
    previous_array = np.asarray(previous)
    previous_two_array = np.asarray(previous_two)
    profiles_array = np.asarray(profiles)
    operators_array = np.asarray(operators)
    if previous_array.shape != (n_total,) or previous_two_array.shape != (n_total,):
        raise ValueError("previous fields must each have shape (N,)")
    if profiles_array.shape != (3, n_total):
        raise ValueError("profiles must have shape (3, N)")
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")

    v_one = np.asarray(previous_array, dtype=complex)
    v_two = np.asarray(previous_two_array, dtype=complex)
    rho_center = np.asarray(profiles_array[1], dtype=float)
    rho_one = np.asarray(profiles_array[2], dtype=float)
    dx = np.asarray(operators_array[0], dtype=complex)
    dz = np.asarray(operators_array[1], dtype=complex)

    div_one_previous = dx @ (rho_one * (dx @ v_one)) + dz @ (rho_one * (dz @ v_one))
    div_center_previous = dx @ (rho_center * (dx @ v_one)) + dz @ (rho_center * (dz @ v_one))
    div_one_previous_two = dx @ (rho_one * (dx @ v_two)) + dz @ (rho_one * (dz @ v_two))
    source = -rho_center * div_one_previous
    source -= rho_one * div_center_previous
    source -= rho_one * div_one_previous_two
    top = np.arange(nx) * (nz + 1)
    bottom = top + nz
    normal_derivative = dz @ v_one
    source[top] = rho_one[top] * normal_derivative[top]
    source[bottom] = -rho_one[bottom] * normal_derivative[bottom]
    return source.astype(np.complex128)

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def solve_transmission_channel_series(sign: int, matrix: "np.ndarray", profiles: "np.ndarray", operators: "np.ndarray", radiation: "np.ndarray", max_order: int) -> "np.ndarray":
    """Reference implementation."""
    if sign not in (-1, 1) or isinstance(sign, (bool, np.bool_)):
        raise ValueError("sign must equal +1 or -1")
    if not isinstance(max_order, (int, np.integer)) or isinstance(max_order, (bool, np.bool_)) or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer")
    radiation_array = np.asarray(radiation)
    if radiation_array.ndim != 2 or radiation_array.shape[1] != 5:
        raise ValueError("radiation must have shape (nx, 5)")
    nx = radiation_array.shape[0]
    profiles_array = np.asarray(profiles)
    if profiles_array.ndim != 2 or profiles_array.shape[0] != 3:
        raise ValueError("profiles must have shape (3, N)")
    n_total = profiles_array.shape[1]
    if nx < 4 or nx % 2 or n_total % nx:
        raise ValueError("array dimensions do not define a valid even Fourier grid")
    nz = n_total // nx - 1
    if nz < 2:
        raise ValueError("the inferred Chebyshev degree must be at least 2")
    matrix_array = np.asarray(matrix)
    operators_array = np.asarray(operators)
    if matrix_array.shape != (n_total, n_total):
        raise ValueError("matrix must have shape (N, N)")
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")

    alpha_p = np.asarray(radiation_array[:, 0], dtype=complex).real
    spacing = alpha_p[1] - alpha_p[0]
    d = 2.0 * np.pi / spacing
    x = np.arange(nx, dtype=float) * d / nx
    qinv = np.exp(1j * x[:, None] * alpha_p[None, :])
    forcing_column = 3 if sign == 1 else 4
    top_forcing = qinv @ np.asarray(radiation_array[:, forcing_column], dtype=complex)
    top_indices = np.arange(nx) * (nz + 1)
    bottom_indices = top_indices + nz

    factorization = lu_factor(np.asarray(matrix_array, dtype=complex))
    series = np.zeros((max_order + 1, n_total), dtype=np.complex128)
    zero_field = np.zeros(n_total, dtype=complex)
    for order in range(max_order + 1):
        if order == 0:
            right_hand_side = np.zeros(n_total, dtype=complex)
        else:
            previous_two = series[order - 2] if order >= 2 else zero_field
            right_hand_side = assemble_transmission_source(
                series[order - 1], previous_two, profiles_array,
                operators_array, nx, nz
            )
        if order == 0:
            right_hand_side[top_indices] = top_forcing
            right_hand_side[bottom_indices] = 0.0
        series[order] = lu_solve(factorization, right_hand_side)
    return series

import numpy as np


def reconstruct_electric_series(left_series: "np.ndarray", right_series: "np.ndarray", left_profiles: "np.ndarray", right_profiles: "np.ndarray", operators: "np.ndarray", k0: float) -> "np.ndarray":
    """Reference implementation."""
    if not np.isfinite(k0) or k0 <= 0.0:
        raise ValueError("k0 must be positive and finite")
    left = np.asarray(left_series)
    right = np.asarray(right_series)
    if left.ndim != 2 or right.shape != left.shape or left.shape[0] < 1 or left.shape[1] < 1:
        raise ValueError("left and right series must share shape (M+1, N)")
    orders, n_total = left.shape
    left_profiles_array = np.asarray(left_profiles)
    right_profiles_array = np.asarray(right_profiles)
    operators_array = np.asarray(operators)
    if left_profiles_array.shape != (3, n_total) or right_profiles_array.shape != (3, n_total):
        raise ValueError("each channel profile must have shape (3, N)")
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")

    dx = np.asarray(operators_array[0], dtype=complex)
    dz = np.asarray(operators_array[1], dtype=complex)
    left_center = np.asarray(left_profiles_array[1], dtype=float)
    left_one = np.asarray(left_profiles_array[2], dtype=float)
    right_center = np.asarray(right_profiles_array[1], dtype=float)
    right_one = np.asarray(right_profiles_array[2], dtype=float)
    electric = np.empty((orders, n_total, 3), dtype=np.complex128)
    zero_field = np.zeros(n_total, dtype=complex)
    for order in range(orders):
        left_previous = left[order - 1] if order >= 1 else zero_field
        right_previous = right[order - 1] if order >= 1 else zero_field
        left_vector = np.empty((n_total, 3), dtype=complex)
        right_vector = np.empty((n_total, 3), dtype=complex)
        left_vector[:, 0] = -(
            left_center * (dz @ left[order]) + left_one * (dz @ left_previous)
        ) / float(k0)
        left_vector[:, 1] = left[order]
        left_vector[:, 2] = (
            left_center * (dx @ left[order]) + left_one * (dx @ left_previous)
        ) / float(k0)
        right_vector[:, 0] = (
            right_center * (dz @ right[order]) + right_one * (dz @ right_previous)
        ) / float(k0)
        right_vector[:, 1] = right[order]
        right_vector[:, 2] = -(
            right_center * (dx @ right[order]) + right_one * (dx @ right_previous)
        ) / float(k0)
        electric[order] = left_vector - 1j * right_vector
    return electric

import numpy as np


def electric_intensity_coefficients(electric: "np.ndarray", operators: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    electric_array = np.asarray(electric)
    if electric_array.ndim != 3 or electric_array.shape[0] < 1 or electric_array.shape[1] < 1 or electric_array.shape[2] != 3:
        raise ValueError("electric must have shape (M+1, N, 3)")
    orders, n_total, _ = electric_array.shape
    operators_array = np.asarray(operators)
    if operators_array.shape != (3, n_total, n_total):
        raise ValueError("operators must have shape (3, N, N)")
    diagonal = np.diag(np.asarray(operators_array[2], dtype=complex))
    if not np.all(np.isfinite(diagonal)) or np.max(np.abs(diagonal.imag)) > 1e-13:
        raise ValueError("quadrature diagonal must be finite and real")
    weights = diagonal.real
    fields = np.asarray(electric_array, dtype=complex)
    coefficients = np.empty(orders, dtype=float)
    for total_order in range(orders):
        value = 0.0j
        for left_order in range(total_order + 1):
            right_order = total_order - left_order
            value += np.sum(
                weights[:, None]
                * fields[left_order]
                * np.conj(fields[right_order])
            )
        coefficients[total_order] = value.real
    return coefficients

import numpy as np


def pade_second_derivative(coefficients: "np.ndarray", numerator_degree: int, denominator_degree: int, evaluation: float) -> float:
    """Reference implementation."""
    for value, name in ((numerator_degree, "numerator_degree"), (denominator_degree, "denominator_degree")):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    coefficients_array = np.asarray(coefficients)
    required = numerator_degree + denominator_degree + 1
    if coefficients_array.ndim != 1 or coefficients_array.size < required:
        raise ValueError("coefficients must be one-dimensional and long enough")
    used = np.asarray(coefficients_array[:required], dtype=float)
    if not np.all(np.isfinite(used)) or not np.isfinite(evaluation):
        raise ValueError("coefficients and evaluation must be finite")

    if denominator_degree == 0:
        denominator = np.array([1.0], dtype=float)
    else:
        system = np.empty((denominator_degree, denominator_degree), dtype=float)
        right_hand_side = np.empty(denominator_degree, dtype=float)
        for row, order in enumerate(
            range(numerator_degree + 1, numerator_degree + denominator_degree + 1)
        ):
            system[row, :] = [
                used[order - q] if order - q >= 0 else 0.0
                for q in range(1, denominator_degree + 1)
            ]
            right_hand_side[row] = -used[order]
        if np.linalg.matrix_rank(system) < denominator_degree:
            raise ValueError("the Pade denominator system is rank deficient")
        denominator = np.concatenate((
            np.array([1.0]), np.linalg.solve(system, right_hand_side)
        ))

    numerator = np.empty(numerator_degree + 1, dtype=float)
    for order in range(numerator_degree + 1):
        numerator[order] = sum(
            denominator[q] * used[order - q]
            for q in range(min(order, denominator_degree) + 1)
        )

    point = float(evaluation)
    def _polynomial_jet(values):
        value = sum(values[j] * point ** j for j in range(values.size))
        first = sum(j * values[j] * point ** (j - 1) for j in range(1, values.size))
        second = sum(
            j * (j - 1) * values[j] * point ** (j - 2)
            for j in range(2, values.size)
        )
        return value, first, second

    a_value, a_first, a_second = _polynomial_jet(numerator)
    b_value, b_first, b_second = _polynomial_jet(denominator)
    denominator_scale = max(
        1.0,
        sum(abs(denominator[j] * point ** j) for j in range(denominator.size)),
    )
    if abs(b_value) <= 64.0 * np.finfo(float).eps * denominator_scale:
        raise ValueError("the Pade denominator vanishes at the evaluation point")
    curvature = (
        a_second / b_value
        - (a_value * b_second + 2.0 * a_first * b_first) / (b_value * b_value)
        + 2.0 * a_value * b_first * b_first / (b_value ** 3)
    )
    return float(curvature)

import numpy as np


def chiral_field_intensity_curvature(d: float, h: float, t: float, sharpness: float, wavelength: float, theta: float, chi_bar: float, chi_amplitude: float, lateral_scale: float, delta_center: float, delta_target: float, nx: int, nz: int, max_order: int, pade_numerator: int, pade_denominator: int) -> float:
    """Reference implementation composed exclusively from earlier oracles."""
    if not isinstance(max_order, (int, np.integer)) or isinstance(max_order, (bool, np.bool_)) or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer")
    for value, name in ((pade_numerator, "pade_numerator"), (pade_denominator, "pade_denominator")):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if pade_numerator + pade_denominator > max_order:
        raise ValueError("Pade degrees must sum to at most max_order")
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength must be a positive finite scalar")
    if not np.isfinite(delta_target):
        raise ValueError("delta_target must be finite")

    k0 = 2.0 * np.pi / float(wavelength)
    alpha = k0 * np.sin(float(theta))
    operators = spectral_operators_and_weights(nx, nz, d, h, alpha)
    radiation = radiation_and_bohren_data(nx, d, h, wavelength, theta)
    left_profiles = centered_channel_profiles(
        1, nx, nz, d, h, t, sharpness, wavelength, chi_bar,
        chi_amplitude, lateral_scale, delta_center
    )
    right_profiles = centered_channel_profiles(
        -1, nx, nz, d, h, t, sharpness, wavelength, chi_bar,
        chi_amplitude, lateral_scale, delta_center
    )
    envelope = left_profiles[0]
    target_chi = float(chi_bar) - float(delta_target) * envelope
    if np.max(np.abs(k0 * target_chi)) >= 1.0:
        raise ValueError("the target chirality violates abs(k0*chi) < 1")

    left_matrix = assemble_transmission_operator(
        left_profiles, operators, radiation, nx, nz
    )
    right_matrix = assemble_transmission_operator(
        right_profiles, operators, radiation, nx, nz
    )
    left_series = solve_transmission_channel_series(
        1, left_matrix, left_profiles, operators, radiation, max_order
    )
    right_series = solve_transmission_channel_series(
        -1, right_matrix, right_profiles, operators, radiation, max_order
    )
    electric = reconstruct_electric_series(
        left_series, right_series, left_profiles, right_profiles, operators, k0
    )
    coefficients = electric_intensity_coefficients(electric, operators)
    return pade_second_derivative(
        coefficients, pade_numerator, pade_denominator,
        float(delta_target) - float(delta_center)
    )
SCICODE_GOLD_EOF
