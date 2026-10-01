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
from scipy.special import gammaln


def compute_photon_number_bounds(intensity: float, delta_max: float, n_cut: int) -> "tuple[np.ndarray, np.ndarray, float]":
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float, np.integer, np.floating)):
        raise ValueError("intensity must be a real scalar")
    if isinstance(delta_max, bool) or not isinstance(delta_max, (int, float, np.integer, np.floating)):
        raise ValueError("delta_max must be a real scalar")
    if isinstance(n_cut, bool) or not isinstance(n_cut, (int, np.integer)):
        raise ValueError("n_cut must be an integer")
    intensity = float(intensity)
    delta_max = float(delta_max)
    n_cut = int(n_cut)
    if not np.isfinite(intensity) or intensity <= 0.0:
        raise ValueError("intensity must be strictly positive")
    if not np.isfinite(delta_max) or delta_max < 0.0 or delta_max >= 1.0:
        raise ValueError("delta_max must lie in [0, 1)")
    if n_cut < 0:
        raise ValueError("n_cut must be non-negative")

    a_lo = intensity * (1.0 - delta_max)
    a_hi = intensity * (1.0 + delta_max)
    if a_hi > 1.0:
        raise ValueError("intensity * (1 + delta_max) must not exceed one")

    n = np.arange(n_cut + 1, dtype=float)
    log_factorial = gammaln(n + 1.0)

    # e^{-x} x^n / n! increases with x throughout (0, 1) for every n >= 1, so the
    # extremes of each n >= 1 probability sit at the ends of the interval; the
    # vacuum probability e^{-x} decreases with x and its extremes are exchanged.
    lower = np.exp(-a_lo + n * np.log(a_lo) - log_factorial)
    upper = np.exp(-a_hi + n * np.log(a_hi) - log_factorial)
    lower[0] = np.exp(-a_hi)
    upper[0] = np.exp(-a_lo)

    truncated = np.exp(-a_hi + n * np.log(a_hi) - log_factorial)
    cut_mass = float(max(0.0, 1.0 - float(np.sum(truncated))))
    return lower, upper, cut_mass

import numpy as np


def _check_intensity_family(intensities: "np.ndarray", probabilities: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Validate and normalise a family of nominal settings and their weights."""
    intensities = np.asarray(intensities, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if intensities.ndim != 1 or intensities.size < 1:
        raise ValueError("intensities must be a non-empty 1D array")
    if not np.all(np.isfinite(intensities)) or np.any(intensities <= 0.0):
        raise ValueError("intensities must all be strictly positive")
    if probabilities.shape != intensities.shape:
        raise ValueError("probabilities must have the same shape as intensities")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if abs(float(np.sum(probabilities)) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to one")
    return intensities, probabilities


def compute_cs_overlap_parameters(intensity_a: float, intensity_b: float, intensities: "np.ndarray", probabilities: "np.ndarray", delta_max: float, correlation_range: int, n_cut: int) -> "np.ndarray":
    for value in (intensity_a, intensity_b, delta_max):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("intensity_a, intensity_b and delta_max must be real scalars")
    for value in (correlation_range, n_cut):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("correlation_range and n_cut must be integers")
    intensity_a = float(intensity_a)
    intensity_b = float(intensity_b)
    delta_max = float(delta_max)
    correlation_range = int(correlation_range)
    n_cut = int(n_cut)
    if not np.isfinite(intensity_a) or intensity_a <= 0.0:
        raise ValueError("intensity_a must be strictly positive")
    if not np.isfinite(intensity_b) or intensity_b <= 0.0:
        raise ValueError("intensity_b must be strictly positive")
    if not np.isfinite(delta_max) or delta_max < 0.0 or delta_max >= 1.0:
        raise ValueError("delta_max must lie in [0, 1)")
    if correlation_range < 1:
        raise ValueError("correlation_range must be at least one")
    if n_cut < 0:
        raise ValueError("n_cut must be non-negative")
    intensities, probabilities = _check_intensity_family(intensities, probabilities)

    a_lo, a_hi = intensity_a * (1.0 - delta_max), intensity_a * (1.0 + delta_max)
    b_lo, b_hi = intensity_b * (1.0 - delta_max), intensity_b * (1.0 + delta_max)

    # Weight lost to the spread of every selectable setting's vacuum probability,
    # compounded over the rounds on both sides of the memory span.
    spread = float(np.sum(probabilities * (np.exp(-intensities * (1.0 - delta_max))
                                           - np.exp(-intensities * (1.0 + delta_max)))))
    memory = (1.0 - spread) ** (2 * correlation_range)

    n = np.arange(n_cut + 1, dtype=float)
    overlaps = (np.exp(a_hi + b_hi - (a_lo + b_lo))
                * ((a_lo * b_lo) / (a_hi * b_hi)) ** n * memory)
    overlaps[0] = np.exp(a_lo + b_lo - (a_hi + b_hi)) * memory
    return overlaps

import numpy as np


def evaluate_cs_boundaries(parameter_values: "np.ndarray", overlaps: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    values = np.asarray(parameter_values, dtype=float)
    weights = np.asarray(overlaps, dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("inputs must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("parameter_values must lie in [0, 1]")
    if np.any(weights <= 0.0) or np.any(weights > 1.0):
        raise ValueError("overlaps must lie in (0, 1]")
    try:
        values, weights = np.broadcast_arrays(values, weights)
    except ValueError as exc:
        raise ValueError("parameter_values and overlaps must broadcast") from exc

    slack = 1.0 - weights
    centre = values + slack * (1.0 - 2.0 * values)
    radius = 2.0 * np.sqrt(np.clip(weights * slack * values * (1.0 - values), 0.0, None))

    lower = np.where(values > slack, centre - radius, 0.0)
    upper = np.where(values < weights, centre + radius, 1.0)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

import numpy as np


def build_cs_tangent_coefficients(reference_values: "np.ndarray", overlaps: "np.ndarray", tangent_floor: float = 1e-12) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    if isinstance(tangent_floor, bool) or not isinstance(tangent_floor, (int, float, np.integer, np.floating)):
        raise ValueError("tangent_floor must be a real scalar")
    tangent_floor = float(tangent_floor)
    if not np.isfinite(tangent_floor) or tangent_floor <= 0.0 or tangent_floor >= 0.5:
        raise ValueError("tangent_floor must lie in (0, 0.5)")

    values = np.asarray(reference_values, dtype=float)
    weights = np.asarray(overlaps, dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("inputs must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("reference_values must lie in [0, 1]")
    if np.any(weights <= 0.0) or np.any(weights > 1.0):
        raise ValueError("overlaps must lie in (0, 1]")
    try:
        values, weights = np.broadcast_arrays(values, weights)
    except ValueError as exc:
        raise ValueError("reference_values and overlaps must broadcast") from exc

    anchor = np.clip(values, tangent_floor, 1.0 - tangent_floor)
    slack = 1.0 - weights
    lower_value, upper_value = evaluate_cs_boundaries(anchor, weights)

    curvature = np.sqrt(weights * slack / (anchor * (1.0 - anchor)))
    tilt = (1.0 - 2.0 * anchor) * curvature
    lower_slope = np.where(anchor > slack, -1.0 + 2.0 * weights - tilt, 0.0)
    upper_slope = np.where(anchor < weights, -1.0 + 2.0 * weights + tilt, 0.0)

    lower_offset = lower_value - lower_slope * anchor
    upper_offset = upper_value - upper_slope * anchor
    return (np.asarray(lower_slope, dtype=float), np.asarray(lower_offset, dtype=float),
            np.asarray(upper_slope, dtype=float), np.asarray(upper_offset, dtype=float))

import numpy as np


def compute_channel_observables(intensities: "np.ndarray", transmittance: float, misalignment: float, dark_count: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    for value in (transmittance, misalignment, dark_count):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("transmittance, misalignment and dark_count must be real scalars")
    transmittance = float(transmittance)
    misalignment = float(misalignment)
    dark_count = float(dark_count)
    if not np.isfinite(transmittance) or transmittance < 0.0 or transmittance > 1.0:
        raise ValueError("transmittance must lie in [0, 1]")
    if not np.isfinite(misalignment) or misalignment < 0.0 or misalignment > 0.25 * np.pi:
        raise ValueError("misalignment must lie in [0, pi / 4]")
    if not np.isfinite(dark_count) or dark_count < 0.0 or dark_count >= 1.0:
        raise ValueError("dark_count must lie in [0, 1)")
    settings = np.asarray(intensities, dtype=float)
    if settings.ndim != 1 or settings.size < 1:
        raise ValueError("intensities must be a non-empty 1D array")
    if not np.all(np.isfinite(settings)) or np.any(settings <= 0.0):
        raise ValueError("intensities must all be strictly positive")

    no_dark = (1.0 - dark_count) ** 2
    detected = 1.0 - no_dark * np.exp(-transmittance * settings)

    aligned = np.exp(-transmittance * settings * np.cos(misalignment) ** 2)
    crossed = np.exp(-transmittance * settings * np.sin(misalignment) ** 2)
    imbalance = 0.5 * (aligned - crossed)

    errors = (0.5 * dark_count ** 2
              + dark_count * (1.0 - dark_count) * (1.0 + imbalance)
              + no_dark * (0.5 + imbalance - 0.5 * np.exp(-transmittance * settings)))
    return detected, np.array(detected, dtype=float, copy=True), errors

import numpy as np


def compute_fock_reference_points(n_cut: int, transmittance: float, misalignment: float, dark_count: float) -> "tuple[np.ndarray, np.ndarray]":
    if isinstance(n_cut, bool) or not isinstance(n_cut, (int, np.integer)):
        raise ValueError("n_cut must be an integer")
    for value in (transmittance, misalignment, dark_count):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("transmittance, misalignment and dark_count must be real scalars")
    n_cut = int(n_cut)
    transmittance = float(transmittance)
    misalignment = float(misalignment)
    dark_count = float(dark_count)
    if n_cut < 0:
        raise ValueError("n_cut must be non-negative")
    if not np.isfinite(transmittance) or transmittance < 0.0 or transmittance > 1.0:
        raise ValueError("transmittance must lie in [0, 1]")
    if not np.isfinite(misalignment) or misalignment < 0.0 or misalignment > 0.25 * np.pi:
        raise ValueError("misalignment must lie in [0, pi / 4]")
    if not np.isfinite(dark_count) or dark_count < 0.0 or dark_count >= 1.0:
        raise ValueError("dark_count must lie in [0, 1)")

    n = np.arange(n_cut + 1, dtype=float)
    dark = 1.0 - transmittance
    silent = dark ** n
    aligned = (transmittance * np.cos(misalignment) ** 2 + dark) ** n - silent
    crossed = (transmittance * np.sin(misalignment) ** 2 + dark) ** n - silent
    both = 1.0 - silent - aligned - crossed

    clean = crossed + 0.5 * both
    stray_correct = 0.5 * (crossed + both)
    stray_wrong = silent + crossed + 0.5 * (aligned + both)

    no_dark = (1.0 - dark_count) ** 2
    error_reference = (no_dark * clean
                       + dark_count * (1.0 - dark_count) * (stray_correct + stray_wrong)
                       + 0.5 * dark_count ** 2)
    yield_reference = 1.0 - no_dark * silent
    return np.asarray(yield_reference, dtype=float), np.asarray(error_reference, dtype=float)

import numpy as np
from scipy.optimize import linprog


def solve_outer_linearised_program(
    observed_rates: "np.ndarray",
    reference_points: "np.ndarray",
    intensities: "np.ndarray",
    probabilities: "np.ndarray",
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    maximise: bool,
) -> "tuple[float, np.ndarray]":
    if isinstance(n_cut, bool) or not isinstance(n_cut, (int, np.integer)):
        raise ValueError("n_cut must be an integer")
    n_cut = int(n_cut)
    if n_cut < 1:
        raise ValueError("n_cut must be at least one")
    if not isinstance(maximise, (bool, np.bool_)):
        raise ValueError("maximise must be a boolean")

    settings, weights = _check_intensity_family(
        intensities, probabilities
    )
    n_set = settings.size
    width = n_cut + 1
    rates = np.asarray(observed_rates, dtype=float)
    if rates.shape != (n_set,):
        raise ValueError("observed_rates must have shape (A,)")
    if (
        not np.all(np.isfinite(rates))
        or np.any(rates < 0.0)
        or np.any(rates > 1.0)
    ):
        raise ValueError("observed_rates must lie in [0, 1]")

    anchors = np.asarray(reference_points, dtype=float)
    if (
        anchors.ndim != 3
        or anchors.shape[1:] != (n_set, width)
        or anchors.shape[0] < 1
    ):
        raise ValueError(
            "reference_points must have shape (R, A, n_cut + 1), R >= 1"
        )
    if (
        not np.all(np.isfinite(anchors))
        or np.any(anchors < 0.0)
        or np.any(anchors > 1.0)
    ):
        raise ValueError("reference_points must lie in [0, 1]")

    rows, limits = [], []
    for index, setting in enumerate(settings):
        lower, upper, cut_mass = compute_photon_number_bounds(
            float(setting), delta_max, n_cut
        )
        block = np.zeros((n_set, width))
        block[index] = lower
        rows.append(block.ravel())
        limits.append(float(rates[index]))
        block = np.zeros((n_set, width))
        block[index] = -upper
        rows.append(block.ravel())
        limits.append(float(cut_mass - rates[index]))

    for first in range(n_set):
        for second in range(n_set):
            if first == second:
                continue
            overlaps = compute_cs_overlap_parameters(
                float(settings[first]), float(settings[second]),
                settings, weights, delta_max, correlation_range, n_cut,
            )
            for point in anchors:
                coefficients = build_cs_tangent_coefficients(
                    point[first], overlaps
                )
                low_slope, low_offset, high_slope, high_offset = coefficients
                for photons in range(width):
                    block = np.zeros((n_set, width))
                    block[first, photons] = low_slope[photons]
                    block[second, photons] -= 1.0
                    rows.append(block.ravel())
                    limits.append(float(-low_offset[photons]))
                    block = np.zeros((n_set, width))
                    block[second, photons] += 1.0
                    block[first, photons] -= high_slope[photons]
                    rows.append(block.ravel())
                    limits.append(float(high_offset[photons]))

    cost = np.zeros(n_set * width)
    cost[1] = -1.0 if maximise else 1.0
    # Common positive scaling preserves the program and resolves tiny
    # affine residuals more accurately than default solver tolerances.
    solution = linprog(
        1000.0 * cost,
        A_ub=1000.0 * np.asarray(rows),
        b_ub=1000.0 * np.asarray(limits),
        bounds=[(0.0, 1.0)] * (n_set * width),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if not solution.success and solution.status == 4:
        solution = linprog(
            1000.0 * cost,
            A_ub=1000.0 * np.asarray(rows),
            b_ub=1000.0 * np.asarray(limits),
            bounds=[(0.0, 1.0)] * (n_set * width),
            method="highs-ipm",
            options={
                "primal_feasibility_tolerance": 1e-10,
                "dual_feasibility_tolerance": 1e-10,
            },
        )
    if not solution.success or solution.x is None:
        raise ValueError(
            "linear program not solved to optimality: " + solution.message
        )

    objective = float(cost @ solution.x)
    objective = -objective if maximise else objective
    return objective, np.asarray(solution.x, dtype=float).reshape(
        n_set, width
    )

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def compute_certified_parameter_bounds(
    key_basis_rates: "np.ndarray",
    check_basis_rates: "np.ndarray",
    check_basis_error_rates: "np.ndarray",
    yield_reference: "np.ndarray",
    error_reference: "np.ndarray",
    intensities: "np.ndarray",
    probabilities: "np.ndarray",
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    max_iterations: int = 40,
    objective_rtol: float = 1e-12,
) -> "tuple[float, float, float, float, float, float]":
    if isinstance(max_iterations, bool) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be an integer")
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")
    if isinstance(objective_rtol, bool) or not isinstance(
        objective_rtol, (int, float, np.integer, np.floating)
    ):
        raise ValueError("objective_rtol must be a real scalar")
    objective_rtol = float(objective_rtol)
    if not np.isfinite(objective_rtol) or objective_rtol <= 0.0:
        raise ValueError("objective_rtol must be strictly positive")

    settings, weights = _check_intensity_family(
        intensities, probabilities
    )
    n_set = settings.size
    if isinstance(n_cut, bool) or not isinstance(n_cut, (int, np.integer)):
        raise ValueError("n_cut must be an integer")
    n_cut = int(n_cut)
    if n_cut < 1:
        raise ValueError("n_cut must be at least one")

    references = []
    for supplied in (yield_reference, error_reference):
        start = np.asarray(supplied, dtype=float)
        if start.shape != (n_cut + 1,):
            raise ValueError("reference shape must be (n_cut + 1,)")
        if (
            not np.all(np.isfinite(start))
            or np.any(start < 0.0)
            or np.any(start > 1.0)
        ):
            raise ValueError("reference arrays must lie in [0, 1]")
        references.append(np.tile(start, (n_set, 1)))
    yield_start, error_start = references

    poisson = [
        compute_photon_number_bounds(
            float(setting), delta_max, n_cut
        )
        for setting in settings
    ]
    pairs = [
        (
            first,
            second,
            compute_cs_overlap_parameters(
                float(settings[first]), float(settings[second]),
                settings, weights, delta_max, correlation_range, n_cut,
            ),
        )
        for first in range(n_set)
        for second in range(n_set)
        if first != second
    ]

    def _violation(rates, table):
        worst = 0.0
        for index, (lower, upper, tail) in enumerate(poisson):
            worst = max(
                worst,
                float(lower @ table[index] - rates[index]),
                float(rates[index] - tail - upper @ table[index]),
            )
        for first, second, overlaps in pairs:
            lower, upper = evaluate_cs_boundaries(
                table[first], overlaps
            )
            worst = max(
                worst,
                float(np.max(lower - table[second])),
                float(np.max(table[second] - upper)),
            )
        return worst

    def _polish_exact_candidate(rates, table, value, maximise):
        # Y = sin(theta)^2 makes each exact CS pair equivalent to
        # |theta_i - theta_j| <= acos(sqrt(tau)).
        width = n_cut + 1
        cost = np.zeros(n_set * width)
        cost[1] = -1.0 if maximise else 1.0
        matrix = np.zeros((2 * n_set + 1, n_set * width))
        rhs = np.empty(2 * n_set + 1)
        for i, (lower, upper, tail) in enumerate(poisson):
            matrix[2 * i, i * width:(i + 1) * width] = lower
            matrix[2 * i + 1, i * width:(i + 1) * width] = -upper
            rhs[2 * i] = rates[i]
            rhs[2 * i + 1] = tail - rates[i]
        # Search near the accumulated outer bound, not an arbitrary
        # feasible point with a different objective.
        matrix[-1] = cost
        rhs[-1] = (-value if maximise else value) + 5e-9
        rows, caps = [], []
        for i, j, tau in pairs:
            if i >= j:
                continue
            angles = np.arccos(np.sqrt(tau))
            for n in range(width):
                row = np.zeros(n_set * width)
                row[i * width + n] = 1.0
                row[j * width + n] = -1.0
                rows.extend((row, -row))
                caps.extend((angles[n], angles[n]))
        linear = LinearConstraint(
            np.asarray(rows), -np.inf, np.asarray(caps)
        )

        def _rate_constraints(theta):
            return rhs - matrix @ np.sin(theta) ** 2

        def _rate_jacobian(theta):
            return -matrix * np.sin(2 * theta)[None, :]

        point = np.arcsin(np.sqrt(np.clip(table.ravel(), 0.0, 1.0)))
        fit = minimize(
            lambda theta: float(cost @ theta),
            point,
            jac=lambda theta: cost,
            bounds=[(0.0, np.pi / 2)] * len(point),
            method="SLSQP",
            constraints=[
                linear,
                {
                    "type": "ineq",
                    "fun": _rate_constraints,
                    "jac": _rate_jacobian,
                },
            ],
            options={"maxiter": 200, "ftol": 1e-13},
        )
        # The original inequalities and the final objective gap below
        # decide acceptance; optimizer status alone is not a certificate.
        return np.sin(fit.x).reshape(n_set, width) ** 2

    def _certify(rates, start, maximise):
        points = [np.clip(start, 0.0, 1.0)]
        value, table = solve_outer_linearised_program(
            rates, np.asarray(points), settings, weights, delta_max,
            correlation_range, n_cut, maximise,
        )
        if max_iterations == 1:
            return float(value), abs(float(value) - float(start[0, 1]))
        for _ in range(1, max_iterations):
            points.append(np.clip(table, 0.0, 1.0))
            nxt, nxt_table = solve_outer_linearised_program(
                rates, np.asarray(points), settings, weights, delta_max,
                correlation_range, n_cut, maximise,
            )
            relative_scale = max(
                abs(nxt), abs(value), np.finfo(float).tiny
            )
            settled = abs(nxt - value) <= objective_rtol * relative_scale
            value, table = nxt, np.clip(nxt_table, 0.0, 1.0)
            if not settled:
                continue
            if delta_max > 0.0:
                table = _polish_exact_candidate(
                    rates, table, value, maximise
                )
            if not np.all(np.isfinite(table)):
                continue
            if _violation(rates, table) > 1e-9:
                continue
            certified, _ = solve_outer_linearised_program(
                rates, table[None, :, :], settings, weights, delta_max,
                correlation_range, n_cut, maximise,
            )
            residual = abs(float(table[0, 1]) - float(certified))
            if residual <= 1e-8:
                return float(certified), residual
        raise ValueError(
            "exact-program convergence and certification checks "
            "were not satisfied"
        )

    key_yield_bound, key_yield_residual = _certify(
        key_basis_rates, yield_start, False
    )
    if np.array_equal(key_basis_rates, check_basis_rates):
        check_yield_bound = key_yield_bound
        check_yield_residual = key_yield_residual
    else:
        check_yield_bound, check_yield_residual = _certify(
            check_basis_rates, yield_start, False
        )
    error_bound, error_residual = _certify(
        check_basis_error_rates, error_start, True
    )
    return (
        key_yield_bound, check_yield_bound, error_bound,
        key_yield_residual, check_yield_residual, error_residual,
    )

import numpy as np


def _binary_entropy(fraction: float) -> float:
    """Shannon entropy of a biased bit, in bits."""
    value = float(fraction)
    if value <= 0.0 or value >= 1.0:
        return 0.0
    return float(-value * np.log2(value) - (1.0 - value) * np.log2(1.0 - value))


def compute_certified_key_rate(intensities: "np.ndarray", probabilities: "np.ndarray", key_basis_probability: float, distance_km: float, attenuation_db_per_km: float, detector_efficiency: float, dark_count: float, misalignment: float, delta_max: float, correlation_range: int, n_cut: int, error_correction_efficiency: float, tolerated_error_rate: float, max_iterations: int = 40, objective_rtol: float = 1e-12) -> float:
    scalars = (key_basis_probability, distance_km, attenuation_db_per_km,
               detector_efficiency, error_correction_efficiency, tolerated_error_rate)
    for value in scalars:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("scalar arguments must be real")
    key_basis_probability = float(key_basis_probability)
    distance_km = float(distance_km)
    attenuation_db_per_km = float(attenuation_db_per_km)
    detector_efficiency = float(detector_efficiency)
    error_correction_efficiency = float(error_correction_efficiency)
    tolerated_error_rate = float(tolerated_error_rate)
    if not np.isfinite(key_basis_probability) or key_basis_probability < 0.0 or key_basis_probability > 1.0:
        raise ValueError("key_basis_probability must lie in [0, 1]")
    if not np.isfinite(distance_km) or distance_km < 0.0:
        raise ValueError("distance_km must be non-negative")
    if not np.isfinite(attenuation_db_per_km) or attenuation_db_per_km < 0.0:
        raise ValueError("attenuation_db_per_km must be non-negative")
    if not np.isfinite(detector_efficiency) or detector_efficiency < 0.0 or detector_efficiency > 1.0:
        raise ValueError("detector_efficiency must lie in [0, 1]")
    if not np.isfinite(error_correction_efficiency) or error_correction_efficiency < 1.0:
        raise ValueError("error_correction_efficiency must be at least one")
    if not np.isfinite(tolerated_error_rate) or tolerated_error_rate < 0.0 or tolerated_error_rate > 0.5:
        raise ValueError("tolerated_error_rate must lie in [0, 0.5]")

    settings, weights = _check_intensity_family(intensities, probabilities)
    transmittance = 10.0 ** (-attenuation_db_per_km * distance_km / 10.0) * detector_efficiency

    key_rates, check_rates, error_rates = compute_channel_observables(
        settings, transmittance, misalignment, dark_count)
    yield_reference, error_reference = compute_fock_reference_points(
        n_cut, transmittance, misalignment, dark_count)
    key_yield, check_yield, error_bound, _, _, _ = compute_certified_parameter_bounds(
        key_rates, check_rates, error_rates, yield_reference, error_reference,
        settings, weights, delta_max, correlation_range, n_cut,
        max_iterations, objective_rtol)

    emission_lower, _, _ = compute_photon_number_bounds(
        float(settings[0]), delta_max, n_cut)
    sifting = key_basis_probability ** 2 * float(weights[0])

    if key_yield <= 0.0 or check_yield <= 0.0:
        privacy = 0.0
    else:
        phase_error = min(error_bound / check_yield, 0.5)
        privacy = (sifting * float(emission_lower[1]) * key_yield
                   * (1.0 - _binary_entropy(phase_error)))

    reconciliation = (error_correction_efficiency * sifting * float(key_rates[0])
                      * _binary_entropy(tolerated_error_rate))
    return float(privacy - reconciliation)
SCICODE_GOLD_EOF
