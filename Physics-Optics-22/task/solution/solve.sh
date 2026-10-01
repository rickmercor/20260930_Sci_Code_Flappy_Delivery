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


def _finite_scalar(value: float, name: str, positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        raise ValueError(f"{name} violates its finite nonnegative/positive contract")
    return result


def compute_thermal_curvature(
    heat: float,
    alpha: float,
    length: float,
    pump_radius: float,
    conductivity: float,
    index: float,
    thermooptic: float,
) -> "np.ndarray":
    heat = _finite_scalar(heat, "heat")
    alpha = _finite_scalar(alpha, "alpha")
    length = _finite_scalar(length, "length", True)
    pump_radius = _finite_scalar(pump_radius, "pump_radius", True)
    conductivity = _finite_scalar(conductivity, "conductivity", True)
    index = _finite_scalar(index, "index", True)
    thermooptic = _finite_scalar(thermooptic, "thermooptic", True)
    x = alpha * length
    if x < 1e-4:
        factor = (1.0 + x / 2.0 + x**2 / 12.0 - x**4 / 720.0) / length
        derivative = 0.5 + x / 6.0 - x**3 / 180.0
    else:
        denominator = -np.expm1(-x)
        factor = alpha / denominator
        derivative = (denominator - x * np.exp(-x)) / denominator**2
    scale = thermooptic / (2.0 * conductivity * index * np.pi * pump_radius**2)
    heat_derivative = scale * factor
    return np.array(
        [heat * heat_derivative, heat_derivative, heat * scale * derivative]
    )

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, yv


def compute_ray_matrix(g: float, alpha: float, distance: float) -> "np.ndarray":
    g = _finite_scalar(g, "g")
    alpha = _finite_scalar(alpha, "alpha")
    distance = _finite_scalar(distance, "distance")
    if g == 0.0 or distance == 0.0:
        return np.array([[1.0, distance], [0.0, 1.0]])
    frequency = np.sqrt(g)
    if alpha == 0.0:
        phase = frequency * distance
        cosine, sine = np.cos(phase), np.sin(phase)
        return np.array([[cosine, sine / frequency], [-frequency * sine, cosine]])
    if alpha * distance < 1e-7:
        # Integrate the same continuous equation when the basis loses precision.
        def _rhs(z, state):
            matrix = state.reshape(2, 2)
            return (
                np.array([[0.0, 1.0], [-g * np.exp(-alpha * z), 0.0]]) @ matrix
            ).ravel()

        solution = solve_ivp(
            _rhs,
            (0.0, distance),
            np.eye(2).ravel(),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        if not solution.success:
            raise ValueError("continuous ray integration did not converge")
        return solution.y[:, -1].reshape(2, 2)
    entrance = 2.0 * frequency / alpha
    attenuation = np.exp(-alpha * distance / 2.0)
    exit_value = entrance * attenuation
    initial = np.array(
        [
            [jv(0, entrance), yv(0, entrance)],
            [frequency * jv(1, entrance), frequency * yv(1, entrance)],
        ]
    )
    final = np.array(
        [
            [jv(0, exit_value), yv(0, exit_value)],
            [
                frequency * attenuation * jv(1, exit_value),
                frequency * attenuation * yv(1, exit_value),
            ],
        ]
    )
    matrix = np.linalg.solve(initial.T, final.T).T
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite continuous ray transfer")
    return matrix

import numpy as np
from numpy.polynomial.legendre import leggauss


def _quadrature_order(order: int) -> int:
    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, (int, np.integer))
        or order < 16
    ):
        raise ValueError("order must be an integer at least 16")
    return int(order)


def compute_ray_response(
    g: float,
    alpha: float,
    length: float,
    g_direction: float,
    alpha_direction: float,
    order: int = 64,
) -> "np.ndarray":
    g = _finite_scalar(g, "g")
    alpha = _finite_scalar(alpha, "alpha")
    length = _finite_scalar(length, "length")
    if not np.isfinite([g_direction, alpha_direction]).all():
        raise ValueError("parameter directions must be finite")
    nodes, weights = leggauss(_quadrature_order(order))
    response_integral = np.zeros((2, 2))
    for node, weight in zip(nodes, weights):
        z = length * (node + 1.0) / 2.0
        matrix = compute_ray_matrix(g, alpha, z)
        perturbation = np.zeros((2, 2))
        perturbation[1, 0] = -np.exp(-alpha * z) * (
            g_direction - g * z * alpha_direction
        )
        response_integral += weight * np.linalg.solve(matrix, perturbation @ matrix)
    final = compute_ray_matrix(g, alpha, length)
    return final @ response_integral * (length / 2.0)

import numpy as np
from scipy.special import i0e


def _beam_parameters(wave_number: float, waist: float, cone_angle: float) -> tuple:
    wave_number = _finite_scalar(wave_number, "wave_number", True)
    waist = _finite_scalar(waist, "waist", True)
    cone_angle = _finite_scalar(cone_angle, "cone_angle")
    if cone_angle >= np.pi / 2.0:
        raise ValueError("cone_angle must be less than pi/2")
    return wave_number, waist, wave_number * np.sin(cone_angle)


def compute_input_power(
    wave_number: float, waist: float, cone_angle: float
) -> float:
    _, waist, transverse = _beam_parameters(wave_number, waist, cone_angle)
    argument = transverse**2 * waist**2 / 4.0
    return float(np.pi * waist**2 * i0e(argument) / 2.0)

import numpy as np
from scipy.special import jve


def _real_array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} has invalid shape or entries")
    return array


def compute_field_response(
    radii: "np.ndarray",
    matrix: "np.ndarray",
    response: "np.ndarray",
    wave_number: float,
    waist: float,
    cone_angle: float,
) -> "np.ndarray":
    if np.iscomplexobj(radii):
        raise ValueError("radii must be real")
    radii = np.asarray(radii, dtype=float)
    if (
        radii.ndim != 1
        or radii.size == 0
        or not np.isfinite(radii).all()
        or np.any(radii < 0.0)
    ):
        raise ValueError("radii must be a nonempty nonnegative finite vector")
    matrix = _real_array(matrix, (2, 2), "matrix")
    response = _real_array(response, (2, 2), "response")
    a, b, c, d = matrix.ravel()
    da, db, dc, dd = response.ravel()
    if abs(a * d - b * c - 1.0) > 1e-8:
        raise ValueError("matrix must have determinant one")
    if abs(d * da + a * dd - c * db - b * dc) > 1e-7:
        raise ValueError("response must preserve the determinant to first order")
    wave_number, waist, transverse = _beam_parameters(wave_number, waist, cone_angle)
    q = a + 2j * b / (wave_number * waist**2)
    dq = da + 2j * db / (wave_number * waist**2)
    h = d / waist**2 - 0.5j * wave_number * c
    dh = dd / waist**2 - 0.5j * wave_number * dc
    argument = transverse * radii / q
    argument_response = -argument * dq / q
    exponent = -h * radii**2 / q - 0.5j * b * transverse**2 / (wave_number * q)
    log_response = (
        -dq / q
        - radii**2 * (dh / q - h * dq / q**2)
        - 0.5j * transverse**2 / wave_number * (db / q - b * dq / q**2)
    )
    # Both orders share the same real scaling, so no logarithmic Bessel ratio is needed.
    prefactor = np.exp(exponent + np.abs(argument.imag)) / q
    zero = jve(0, argument)
    first = jve(1, argument)
    field = prefactor * zero
    derivative = prefactor * (zero * log_response - first * argument_response)
    result = np.array([field, derivative])
    if not np.isfinite(result).all():
        raise ValueError("field evaluation is nonfinite")
    return result

import numpy as np
from numpy.polynomial.legendre import leggauss


def compute_aperture_response(
    matrix: "np.ndarray",
    response: "np.ndarray",
    wave_number: float,
    waist: float,
    cone_angle: float,
    aperture: float,
    order: int = 96,
) -> "np.ndarray":
    aperture = _finite_scalar(aperture, "aperture")
    nodes, weights = leggauss(_quadrature_order(order))
    radii = aperture * (nodes + 1.0) / 2.0
    field, derivative = compute_field_response(
        radii, matrix, response, wave_number, waist, cone_angle
    )
    power = compute_input_power(wave_number, waist, cone_angle)
    measure = np.pi * aperture * weights * radii / power
    fraction = np.dot(measure, np.abs(field) ** 2)
    variation = np.dot(measure, 2.0 * np.real(np.conj(field) * derivative))
    return np.array([fraction, variation])

import numpy as np
from scipy.optimize import brentq


def _capture_inputs(thermal, beam, alpha, aperture, target, heat_interval, order):
    thermal = _real_array(thermal, (5,), "thermal")
    beam = _real_array(beam, (3,), "beam")
    heat_interval = _real_array(heat_interval, (2,), "heat_interval")
    if np.any(thermal <= 0.0) or np.any(beam[:2] <= 0.0):
        raise ValueError("thermal parameters, wavelength, and waist must be positive")
    alpha = _finite_scalar(alpha, "alpha")
    aperture = _finite_scalar(aperture, "aperture", True)
    if not np.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between zero and one")
    if heat_interval[0] <= 0.0 or heat_interval[1] <= heat_interval[0]:
        raise ValueError("heat_interval must have increasing positive endpoints")
    order = _quadrature_order(order)
    wave_number = 2.0 * np.pi * thermal[3] / beam[0]
    _beam_parameters(wave_number, beam[1], beam[2])
    return (
        thermal,
        beam,
        alpha,
        aperture,
        float(target),
        heat_interval,
        order,
        wave_number,
    )


def solve_capture_heat(
    thermal: "np.ndarray",
    beam: "np.ndarray",
    alpha: float,
    aperture: float,
    target: float,
    heat_interval: "np.ndarray",
    order: int = 96,
) -> float:
    thermal, beam, alpha, aperture, target, heat_interval, order, wave_number = (
        _capture_inputs(thermal, beam, alpha, aperture, target, heat_interval, order)
    )

    def _residual(heat):
        g = compute_thermal_curvature(heat, alpha, *thermal)[0]
        matrix = compute_ray_matrix(g, alpha, thermal[0])
        fraction = compute_aperture_response(
            matrix, np.zeros((2, 2)), wave_number, beam[1], beam[2], aperture, order
        )[0]
        return fraction - target

    return float(brentq(_residual, *heat_interval, xtol=1e-10, rtol=2e-13))

import numpy as np
from numpy.polynomial.legendre import leggauss


def compute_midpoint_transfer_error(
    g: float, alpha: float, length: float, order: int = 96
) -> "np.ndarray":
    g = _finite_scalar(g, "g")
    alpha = _finite_scalar(alpha, "alpha")
    length = _finite_scalar(length, "length")
    nodes, weights = leggauss(_quadrature_order(order))
    if g == 0.0 or alpha == 0.0 or length == 0.0:
        return np.zeros((2, 2))

    full = compute_ray_matrix(g, alpha, length)
    integral = np.zeros((2, 2))
    for node, weight in zip(nodes, weights):
        z = length * (node + 1.0) / 2.0
        curvature = g * np.exp(-alpha * z)
        generator = np.array(
            [
                [alpha * curvature / 12.0, 0.0],
                [alpha**2 * curvature / 24.0, -alpha * curvature / 12.0],
            ]
        )
        entrance_to_z = compute_ray_matrix(g, alpha, z)
        z_to_exit = np.linalg.solve(entrance_to_z.T, full.T).T
        integral += weight * (z_to_exit @ generator @ entrance_to_z)
    error = 0.5 * length**3 * integral
    if not np.all(np.isfinite(error)):
        raise ValueError("midpoint transfer error is nonfinite")
    return error

import numpy as np


def compute_midpoint_heat_bias(
    thermal: "np.ndarray",
    beam: "np.ndarray",
    alpha: float,
    aperture: float,
    target: float,
    heat_interval: "np.ndarray",
    order: int = 96,
) -> float:
    thermal, beam, alpha, aperture, target, heat_interval, order, wave_number = (
        _capture_inputs(thermal, beam, alpha, aperture, target, heat_interval, order)
    )
    heat = solve_capture_heat(
        thermal, beam, alpha, aperture, target, heat_interval, order
    )
    g, g_heat, _ = compute_thermal_curvature(heat, alpha, *thermal)
    matrix = compute_ray_matrix(g, alpha, thermal[0])
    sliced_direction = compute_midpoint_transfer_error(
        g, alpha, thermal[0], order
    )
    heat_direction = compute_ray_response(
        g, alpha, thermal[0], g_heat, 0.0, order
    )
    sliced_capture = compute_aperture_response(
        matrix, sliced_direction, wave_number, beam[1], beam[2], aperture, order
    )[1]
    heat_capture = compute_aperture_response(
        matrix, heat_direction, wave_number, beam[1], beam[2], aperture, order
    )[1]
    if not np.isfinite(heat_capture) or abs(heat_capture) < 1e-14:
        raise ValueError("the operating heat root is not simple")
    bias = -sliced_capture / (heat * heat_capture)
    if not np.isfinite(bias):
        raise ValueError("midpoint heat bias is nonfinite")
    return float(bias)
SCICODE_GOLD_EOF
