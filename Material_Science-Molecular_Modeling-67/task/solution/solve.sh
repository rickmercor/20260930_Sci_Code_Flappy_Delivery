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


def _finite_array(value, name, ndim):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numerical") from error
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} has invalid dimensions or nonfinite values")
    return result


def _finite_scalar(value, name):
    result = _finite_array(value, name, 0)
    return float(result)


def integrate_surface_tension(
    z: np.ndarray, pressure: np.ndarray
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    z = _finite_array(z, "z", 1)
    pressure = _finite_array(pressure, "pressure", 3)
    if z.size < 2 or np.any(np.diff(z) <= 0):
        raise ValueError("z must be strictly increasing with at least two points")
    if pressure.shape != (pressure.shape[0], z.size, 3) or pressure.shape[0] < 1:
        raise ValueError("pressure must have shape (m, n, 3), m >= 1")
    anisotropy = pressure[:, :, 2] - 0.5 * (pressure[:, :, 0] + pressure[:, :, 1])
    return 0.05 * np.sum(
        0.5 * (anisotropy[:, 1:] + anisotropy[:, :-1]) * np.diff(z), axis=1
    )

import numpy as np


def compute_film_thickness(
    counts: np.ndarray,
    area: float,
    box_length: float,
    vapor_density: np.ndarray,
    liquid_density: np.ndarray,
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    counts = _finite_array(counts, "counts", 1)
    vapor_density = _finite_array(vapor_density, "vapor_density", 1)
    liquid_density = _finite_array(liquid_density, "liquid_density", 1)
    area = _finite_scalar(area, "area")
    box_length = _finite_scalar(box_length, "box_length")
    if (
        counts.size < 1
        or vapor_density.shape != counts.shape
        or liquid_density.shape != counts.shape
    ):
        raise ValueError(
            "inventory and density vectors must have matching nonempty shapes"
        )
    if area <= 0 or box_length <= 0 or np.any(counts <= 0) or np.any(vapor_density < 0):
        raise ValueError("invalid geometry, inventory, or vapor density")
    if np.any(liquid_density <= vapor_density):
        raise ValueError("liquid density must exceed vapor density")
    excess = counts / area - vapor_density * box_length
    thickness = excess / (liquid_density - vapor_density)
    if np.any(excess <= 0) or np.any(thickness >= box_length):
        raise ValueError(
            "the film must have positive thickness strictly below box length"
        )
    return thickness

import numpy as np
from scipy.optimize import brentq


def _fit_inputs(thickness, tension, bounds):
    thickness = _finite_array(thickness, "thickness", 1)
    tension = _finite_array(tension, "tension", 1)
    bounds = _finite_array(bounds, "bounds", 1)
    if thickness.size < 4 or tension.shape != thickness.shape:
        raise ValueError("at least four aligned observations are required")
    if np.any(thickness <= 0) or np.unique(thickness).size != thickness.size:
        raise ValueError("thicknesses must be positive and distinct")
    if np.ptp(tension) == 0:
        raise ValueError("constant tensions do not identify the decay rate")
    if bounds.shape != (2,) or not bounds[0] < bounds[1] < 0:
        raise ValueError("bounds must be two strictly ordered negative values")
    return thickness, tension, bounds


def _profile_fit(b, thickness, tension):
    exponential = np.exp(b * thickness)
    centered = exponential - exponential.mean()
    denominator = centered @ centered
    if denominator <= np.finfo(float).tiny:
        raise ValueError("the exponential design is numerically rank deficient")
    a = (centered @ (tension - tension.mean())) / denominator
    c = tension.mean() - a * exponential.mean()
    residual = a * exponential + c - tension
    slope = residual @ (a * thickness * exponential)
    return float(a), float(c), float(residual @ residual), float(slope)


def fit_interfacial_curve(
    thickness: np.ndarray, tension: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    thickness, tension, bounds = _fit_inputs(thickness, tension, bounds)
    grid = np.linspace(bounds[0], bounds[1], 129)
    candidates = [float(bounds[0]), float(bounds[1])]
    gradients = [_profile_fit(float(b), thickness, tension)[3] for b in grid]
    for index in range(grid.size - 1):
        left, right = float(grid[index]), float(grid[index + 1])
        if gradients[index] == 0:
            candidates.append(left)
        if gradients[index] * gradients[index + 1] < 0:
            candidates.append(
                brentq(
                    lambda b: _profile_fit(b, thickness, tension)[3],
                    left,
                    right,
                    xtol=1e-14,
                    rtol=1e-14,
                )
            )
    best = min(candidates, key=lambda b: (_profile_fit(b, thickness, tension)[2], b))
    a, c, _, _ = _profile_fit(best, thickness, tension)
    return np.array([a, best, c])

import numpy as np


def differentiate_disjoining_pressure(
    thickness: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    thickness = _finite_array(thickness, "thickness", 1)
    parameters = _finite_array(parameters, "parameters", 1)
    if thickness.size < 1 or np.any(thickness <= 0):
        raise ValueError("thicknesses must be positive and nonempty")
    if parameters.shape != (3,) or parameters[1] >= 0:
        raise ValueError("parameters must be (a, b, c) with b < 0")
    a, b, _ = parameters
    exponential = np.exp(b * thickness)
    pressure = -20.0 * a * b * exponential
    return np.column_stack(
        (
            pressure,
            b * pressure,
            -20.0 * b * exponential,
            -20.0 * a * exponential * (1.0 + b * thickness),
            np.zeros_like(thickness),
        )
    )

import numpy as np


def evaluate_bulk_response(
    liquid_pressure: np.ndarray, coefficients: np.ndarray, coupling: float
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    liquid_pressure = _finite_array(liquid_pressure, "liquid_pressure", 1)
    coefficients = _finite_array(coefficients, "coefficients", 1)
    coupling = _finite_scalar(coupling, "coupling")
    if liquid_pressure.size < 1 or np.any(
        (liquid_pressure < 0) | (liquid_pressure > 200)
    ):
        raise ValueError("liquid pressure must lie in [0, 200] MPa")
    if coefficients.shape != (4,) or coefficients[0] <= 0 or coupling < 0:
        raise ValueError("invalid equation of state or coupling")
    c0, c1, c2, c3 = coefficients
    correction = liquid_pressure * (c1 + liquid_pressure * (c2 + liquid_pressure * c3))
    density = c0 + coupling * correction
    pressure_derivative = coupling * (
        c1 + liquid_pressure * (2 * c2 + 3 * liquid_pressure * c3)
    )
    if np.any(density <= 0) or np.any(pressure_derivative < 0):
        raise ValueError(
            "the liquid response must have positive density and nonnegative slope"
        )
    return np.column_stack((density, pressure_derivative, correction))

import numpy as np


def solve_coupled_thickness(
    counts: np.ndarray,
    area: float,
    box_length: float,
    vapor_density: np.ndarray,
    gas_pressure: np.ndarray,
    tension: np.ndarray,
    coefficients: np.ndarray,
    coupling: float,
    bounds: np.ndarray,
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    gas_pressure = _finite_array(gas_pressure, "gas_pressure", 1)
    counts = _finite_array(counts, "counts", 1)
    coefficients = _finite_array(coefficients, "coefficients", 1)
    if gas_pressure.shape != counts.shape or np.any(gas_pressure < 0):
        raise ValueError("gas pressure must be aligned and nonnegative")
    if coefficients.shape != (4,) or coefficients[0] <= 0:
        raise ValueError(
            "four coefficients with positive baseline density are required"
        )
    thickness = compute_film_thickness(
        counts, area, box_length, vapor_density, np.full(counts.size, coefficients[0])
    )
    for _ in range(500):
        parameters = fit_interfacial_curve(thickness, tension, bounds)
        pressure = differentiate_disjoining_pressure(thickness, parameters)[
            :, 0
        ]
        response = evaluate_bulk_response(
            gas_pressure - pressure, coefficients, coupling
        )
        updated = compute_film_thickness(
            counts, area, box_length, vapor_density, response[:, 0]
        )
        residual = np.max(np.abs(updated - thickness) / thickness)
        if residual <= 1e-12:
            return thickness
        thickness = updated
    raise RuntimeError("the thickness iteration did not converge in 500 updates")

import numpy as np


def _validated_inventory_direction(inventory_direction, size, area):
    direction = _finite_array(inventory_direction, "inventory_direction", 1)
    area = _finite_scalar(area, "area")
    if direction.shape != (size,) or area <= 0:
        raise ValueError(
            "the population direction must align with films and area must be positive"
        )
    if abs(direction.sum()) > 1e-12 * max(1.0, np.abs(direction).sum()):
        raise ValueError(
            "the population direction must conserve total mean molecule count"
        )
    return direction, area


def _scaled_stationary_solve(matrix, rhs):
    row_scale = np.max(np.abs(matrix), axis=1)
    if np.any(row_scale == 0):
        raise ValueError("the coupled stationary system is singular")
    row_scaled = matrix / row_scale[:, None]
    column_scale = np.max(np.abs(row_scaled), axis=0)
    if np.any(column_scale == 0):
        raise ValueError("the coupled stationary system is singular")
    scaled = row_scaled / column_scale[None, :]
    if np.linalg.cond(scaled) > 1e12:
        raise ValueError("the scaled coupled stationary system is numerically singular")
    if rhs.ndim == 1:
        return np.linalg.solve(scaled, rhs / row_scale) / column_scale
    return np.linalg.solve(scaled, rhs / row_scale[:, None]) / column_scale[:, None]


def _directional_interfacial_terms(
    direction, thickness, a, b, exponential, pressure_gradient
):
    size = thickness.size
    dh = direction[:size]
    da, db, dc = direction[size:]
    exponent_derivative = b * dh + thickness * db
    de = exponential * exponent_derivative
    df = da * exponential + a * de + dc
    dp = pressure_gradient @ direction
    dj = np.column_stack(
        (
            de,
            (da * thickness + a * dh) * exponential + a * thickness * de,
            np.zeros(size),
        )
    )
    return dh, da, db, exponent_derivative, de, df, dp, dj


def compute_coupled_sensitivity(
    thickness: np.ndarray,
    tension: np.ndarray,
    vapor_density: np.ndarray,
    gas_pressure: np.ndarray,
    coefficients: np.ndarray,
    coupling: float,
    bounds: np.ndarray,
    inventory_direction: np.ndarray,
    area: float,
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    thickness, tension, bounds = _fit_inputs(thickness, tension, bounds)
    size = thickness.size
    vapor_density = _finite_array(vapor_density, "vapor_density", 1)
    gas_pressure = _finite_array(gas_pressure, "gas_pressure", 1)
    coefficients = _finite_array(coefficients, "coefficients", 1)
    direction, area = _validated_inventory_direction(inventory_direction, size, area)
    if vapor_density.shape != (size,) or gas_pressure.shape != (size,):
        raise ValueError("phase data must align with the films")
    if np.any(vapor_density < 0) or np.any(gas_pressure < 0):
        raise ValueError("gas properties must be nonnegative")
    parameters = fit_interfacial_curve(thickness, tension, bounds)
    a, b, c = parameters
    if min(b - bounds[0], bounds[1] - b) <= 1e-9:
        raise ValueError("the fit must be a regular interior optimum")
    partials = differentiate_disjoining_pressure(thickness, parameters)
    liquid_pressure = gas_pressure - partials[:, 0]
    response = evaluate_bulk_response(liquid_pressure, coefficients, coupling)
    if np.any(response[:, 0] <= vapor_density):
        raise ValueError("liquid density must exceed vapor density")
    exponential = np.exp(b * thickness)
    residual = a * exponential + c - tension
    jacobian = np.column_stack(
        (exponential, a * thickness * exponential, np.ones(size))
    )
    hessian = jacobian.T @ jacobian
    mixed = residual @ (thickness * exponential)
    hessian[0, 1] += mixed
    hessian[1, 0] += mixed
    hessian[1, 1] += residual @ (a * thickness**2 * exponential)
    cross = jacobian.T * (a * b * exponential)
    cross[0] += residual * b * exponential
    cross[1] += residual * a * exponential * (1.0 + b * thickness)
    pressure_gradient = np.column_stack((np.diag(partials[:, 1]), partials[:, 2:]))
    stationary_matrix = np.zeros((size + 3, size + 3))
    stationary_matrix[:size, :size] = np.diag(response[:, 0] - vapor_density)
    stationary_matrix[:size] -= (thickness * response[:, 1])[
        :, None
    ] * pressure_gradient
    stationary_matrix[size:, :size] = cross
    stationary_matrix[size:, size:] = hessian
    forcing = np.zeros((size + 3, 2))
    forcing[:size, 0] = -thickness * response[:, 2]
    forcing[:size, 1] = direction / area
    first = _scaled_stationary_solve(stationary_matrix, forcing)
    first_pressure = pressure_gradient @ first
    c1, c2, c3 = coefficients[1:]
    pressure = liquid_pressure
    correction_slope = c1 + 2 * c2 * pressure + 3 * c3 * pressure**2
    correction_curvature = 2 * c2 + 6 * c3 * pressure

    def _second_pressure(left, right, left_coupling, right_coupling):
        hp, ap, bp, zp, ep, fp, qp, jp = _directional_interfacial_terms(
            left, thickness, a, b, exponential, pressure_gradient
        )
        hq, aq, bq, zq, eq, fq, qq, jq = _directional_interfacial_terms(
            right, thickness, a, b, exponential, pressure_gradient
        )
        epq = exponential * (zp * zq + bp * hq + bq * hp)
        fpq = ap * eq + aq * ep + a * epq
        qpq = -20.0 * (
            (ap * bq + aq * bp) * exponential
            + (ap * b + a * bp) * eq
            + (aq * b + a * bq) * ep
            + a * b * epq
        )
        jpq = np.column_stack(
            (
                epq,
                (ap * hq + aq * hp) * exponential
                + (ap * thickness + a * hp) * eq
                + (aq * thickness + a * hq) * ep
                + a * thickness * epq,
                np.zeros(size),
            )
        )
        density_p = left_coupling * response[:, 2] - response[:, 1] * qp
        density_q = right_coupling * response[:, 2] - response[:, 1] * qq
        density_pq = (
            -left_coupling * correction_slope * qq
            - right_coupling * correction_slope * qp
            + coupling * (correction_curvature * qp * qq - correction_slope * qpq)
        )
        balance_second = hp * density_q + hq * density_p + thickness * density_pq
        fit_second = fpq @ jacobian + fp @ jq + fq @ jp + residual @ jpq
        second_state = _scaled_stationary_solve(
            stationary_matrix, -np.concatenate((balance_second, fit_second))
        )
        return qpq + pressure_gradient @ second_state

    lambda_lambda = _second_pressure(first[:, 0], first[:, 0], 1.0, 1.0)
    lambda_eta = _second_pressure(first[:, 0], first[:, 1], 1.0, 0.0)
    eta_eta = _second_pressure(first[:, 1], first[:, 1], 0.0, 0.0)
    return np.column_stack(
        (partials[:, 0], first_pressure, lambda_lambda, lambda_eta, eta_eta)
    )

import numpy as np
from scipy.optimize import brentq


def compute_nanofilm_response(
    z: np.ndarray,
    pressure: np.ndarray,
    counts: np.ndarray,
    area: float,
    box_length: float,
    vapor_density: np.ndarray,
    gas_pressure: np.ndarray,
    coefficients: np.ndarray,
    inventory_direction: np.ndarray,
    coupling_interval: np.ndarray,
    bounds: np.ndarray,
) -> float:
    """Evaluate the reference numerical map."""
    counts = _finite_array(counts, "counts", 1)
    interval = _finite_array(coupling_interval, "coupling_interval", 1)
    direction, area = _validated_inventory_direction(
        inventory_direction, counts.size, area
    )
    if interval.shape != (2,) or not 0 <= interval[0] < interval[1]:
        raise ValueError(
            "the coupling interval must have ordered nonnegative endpoints"
        )
    tension = integrate_surface_tension(z, pressure)

    def _response_at(coupling):
        thickness = solve_coupled_thickness(
            counts,
            area,
            box_length,
            vapor_density,
            gas_pressure,
            tension,
            coefficients,
            coupling,
            bounds,
        )
        return compute_coupled_sensitivity(
            thickness,
            tension,
            vapor_density,
            gas_pressure,
            coefficients,
            coupling,
            bounds,
            direction,
            area,
        )[0]

    left = _response_at(float(interval[0]))
    right = _response_at(float(interval[1]))
    if left[1] <= 0 or right[1] >= 0:
        raise ValueError("the interval must bracket a regular pressure maximum")
    peak = brentq(
        lambda coupling: _response_at(coupling)[1],
        float(interval[0]),
        float(interval[1]),
        xtol=1e-10,
        rtol=1e-12,
        maxiter=100,
    )
    response = _response_at(peak)
    if abs(response[1]) > 1e-8:
        raise RuntimeError("the maximizing coupling did not satisfy stationarity")
    if response[3] >= 0:
        raise ValueError("the stationary pressure must be a regular maximum")
    return float(response[5] - response[4] ** 2 / response[3])
SCICODE_GOLD_EOF
