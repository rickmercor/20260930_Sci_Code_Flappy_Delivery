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


def reduce_neo_hessian(
    extended_hessian: 'np.ndarray',
    quantum_dim: int = 3,
) -> 'np.ndarray':
    if (
        not isinstance(quantum_dim, (int, np.integer))
        or isinstance(quantum_dim, (bool, np.bool_))
        or int(quantum_dim) < 1
    ):
        raise ValueError("quantum_dim must be a positive integer")
    matrix = np.asarray(extended_hessian)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] <= int(quantum_dim)
        or not np.issubdtype(matrix.dtype, np.number)
        or not np.isrealobj(matrix)
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError("extended_hessian must be a finite real square matrix")
    matrix = np.asarray(matrix, dtype=float)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("extended_hessian must be symmetric")
    split = matrix.shape[0] - int(quantum_dim)
    h_cc = matrix[:split, :split]
    h_cq = matrix[:split, split:]
    h_qc = matrix[split:, :split]
    h_qq = matrix[split:, split:]
    try:
        np.linalg.cholesky(h_qq)
        reduced = h_cc - h_cq @ np.linalg.solve(h_qq, h_qc)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the quantum-coordinate block must be positive definite") from exc
    return 0.5 * (reduced + reduced.T)

import numpy as np


def compute_crossing_descriptor(
    hessian_a: 'np.ndarray',
    hessian_b: 'np.ndarray',
    gradient_a: 'np.ndarray',
    gradient_b: 'np.ndarray',
    masses: 'np.ndarray',
    peaked: bool = True,
) -> 'np.ndarray':
    raw = [
        np.asarray(hessian_a),
        np.asarray(hessian_b),
        np.asarray(gradient_a),
        np.asarray(gradient_b),
        np.asarray(masses),
    ]
    if any(
        not np.issubdtype(value.dtype, np.number)
        or not np.isrealobj(value)
        or np.any(~np.isfinite(value))
        for value in raw
    ):
        raise ValueError("all arrays must be finite, real, and numeric")
    ha, hb, ga, gb, mass = [np.asarray(value, dtype=float) for value in raw]
    if (
        ha.ndim != 2
        or hb.shape != ha.shape
        or ha.shape[0] != ha.shape[1]
        or ha.shape[0] < 1
        or ga.shape != (ha.shape[0],)
        or gb.shape != ga.shape
        or mass.shape != ga.shape
    ):
        raise ValueError("Hessians, gradients, and masses have incompatible shapes")
    if (
        not np.allclose(ha, ha.T, rtol=0.0, atol=1e-12)
        or not np.allclose(hb, hb.T, rtol=0.0, atol=1e-12)
        or np.any(mass <= 0.0)
        or not isinstance(peaked, (bool, np.bool_))
    ):
        raise ValueError("Hessians must be symmetric, masses positive, and peaked boolean")
    norm_a = float(np.linalg.norm(ga))
    norm_b = float(np.linalg.norm(gb))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("both diabatic gradients must be nonzero")
    branch = 1.0 if bool(peaked) else -1.0
    denominator = norm_b + branch * norm_a
    if abs(denominator) <= 1e-12:
        raise ValueError("the selected effective-Hessian branch is singular")
    effective = (norm_b * ha + branch * norm_a * hb) / denominator
    root_mass = np.sqrt(mass)
    mass_weighted_hessian = effective / np.outer(root_mass, root_mass)
    difference = ga / root_mass - gb / root_mass
    difference_norm = float(np.linalg.norm(difference))
    if difference_norm == 0.0:
        raise ValueError("the mass-weighted gradient difference must be nonzero")
    direction = difference / difference_norm
    curvature = float(direction @ mass_weighted_hessian @ direction)
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("the projected effective curvature must be positive")
    unweighted_direction = direction / root_mass
    reduced_mass = 1.0 / float(unweighted_direction @ unweighted_direction)
    gradient_gap = float(np.linalg.norm(ga - gb))
    gradient_product_root = float(np.sqrt(norm_a * norm_b))
    return np.array(
        [reduced_mass, gradient_gap, gradient_product_root, curvature],
        dtype=float,
    )

import numpy as np


def _step3_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _step3_numeric_vector(value: np.ndarray, length: int | None, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.size == 0
        or (length is not None and array.size != length)
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or np.any(~np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real vector")
    return np.asarray(array, dtype=float)


def _step3_polynomial(coefficients: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return (
        coefficients[0]
        + coefficients[1] * offset
        + coefficients[2] * offset**2
        + coefficients[3] * offset**3
        + coefficients[4] * offset**4
    )


def evaluate_distance_profiles(
    distance: 'np.ndarray',
    reactant_coefficients: 'np.ndarray',
    mecp_coefficients: 'np.ndarray',
    product_coefficients: 'np.ndarray',
    coupling_parameters: 'np.ndarray',
    reference_distance: float,
    critical_distance: float,
) -> 'np.ndarray':
    r = _step3_numeric_vector(distance, None, "distance")
    reactant = _step3_numeric_vector(reactant_coefficients, 5, "reactant_coefficients")
    mecp = _step3_numeric_vector(mecp_coefficients, 5, "mecp_coefficients")
    product = _step3_numeric_vector(product_coefficients, 5, "product_coefficients")
    coupling = _step3_numeric_vector(coupling_parameters, 3, "coupling_parameters")
    if r.size > 1 and np.any(np.diff(r) <= 0.0):
        raise ValueError("distance must be strictly increasing")
    reference = _step3_real_scalar(reference_distance, "reference_distance")
    critical = _step3_real_scalar(critical_distance, "critical_distance")
    if coupling[0] < 0.0:
        raise ValueError("V0 must be nonnegative")
    y = r - reference
    reactant_energy = _step3_polynomial(reactant, y)
    mecp_energy = _step3_polynomial(mecp, y)
    product_energy = _step3_polynomial(product, y)
    vibronic_coupling = coupling[0] * np.exp(coupling[1] * y + coupling[2] * y**2)
    vibronic_coupling = np.where(r < critical, 0.0, vibronic_coupling)
    result = np.column_stack(
        [reactant_energy, mecp_energy, product_energy, vibronic_coupling]
    )
    if np.any(~np.isfinite(result)):
        raise ValueError("profile evaluation produced a non-finite value")
    return result

import numpy as np


def _step4_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def compute_holstein_transmission(
    velocity: 'np.ndarray',
    coupling: float,
    gradient_gap: float,
) -> 'np.ndarray':
    raw_velocity = np.asarray(velocity)
    if (
        raw_velocity.ndim != 1
        or raw_velocity.size == 0
        or not np.issubdtype(raw_velocity.dtype, np.number)
        or not np.isrealobj(raw_velocity)
        or np.any(~np.isfinite(raw_velocity))
    ):
        raise ValueError("velocity must be a finite real vector")
    velocities = np.asarray(raw_velocity, dtype=float)
    if np.any(velocities < 0.0):
        raise ValueError("velocity must be nonnegative")
    coupling_value = _step4_real_scalar(coupling, "coupling")
    gradient_gap_value = _step4_real_scalar(gradient_gap, "gradient_gap")
    if coupling_value < 0.0 or gradient_gap_value <= 0.0:
        raise ValueError("coupling must be nonnegative and gradient_gap positive")
    if coupling_value == 0.0:
        return np.zeros_like(velocities)
    single_crossing = np.ones_like(velocities)
    positive = velocities > 0.0
    with np.errstate(over="ignore"):
        exponent = (
            -2.0
            * np.pi
            * np.square(np.float64(coupling_value))
            / (gradient_gap_value * velocities[positive])
        )
    single_crossing[positive] = -np.expm1(exponent)
    return 2.0 * single_crossing / (1.0 + single_crossing)

import numpy as np


def _step5_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def integrate_lz_coefficient(
    coupling: float,
    gradient_gap: float,
    reduced_mass: float,
    beta: float,
    quadrature_order: int,
) -> float:
    coupling_value = _step5_real_scalar(coupling, "coupling")
    gradient_gap_value = _step5_real_scalar(gradient_gap, "gradient_gap")
    reduced_mass_value = _step5_real_scalar(reduced_mass, "reduced_mass")
    beta_value = _step5_real_scalar(beta, "beta")
    if (
        coupling_value < 0.0
        or gradient_gap_value <= 0.0
        or reduced_mass_value <= 0.0
        or beta_value <= 0.0
        or not isinstance(quadrature_order, (int, np.integer))
        or isinstance(quadrature_order, (bool, np.bool_))
        or int(quadrature_order) < 8
        or int(quadrature_order) > 100
    ):
        raise ValueError("physical scalars or quadrature_order are outside the domain")
    if coupling_value == 0.0:
        return 0.0
    nodes, weights = np.polynomial.laguerre.laggauss(int(quadrature_order))
    if (
        np.any(~np.isfinite(nodes))
        or np.any(~np.isfinite(weights))
        or np.any(nodes <= 0.0)
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Gauss-Laguerre rule is non-finite")
    velocities = np.sqrt(2.0 * nodes / (beta_value * reduced_mass_value))
    transmission = compute_holstein_transmission(
        velocities,
        coupling_value,
        gradient_gap_value,
    )
    coefficient = float(np.dot(weights, transmission) / beta_value)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("the LZ/Holstein integral is non-finite")
    return coefficient

import numpy as np
from scipy.special import airy


def _step6_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def integrate_wc_coefficient(
    coupling: float,
    gradient_gap: float,
    gradient_product_root: float,
    reduced_mass: float,
    beta: float,
    energy_floor: float,
    quadrature_order: int,
    cap_probability: bool = True,
) -> float:
    coupling_value = _step6_real_scalar(coupling, "coupling")
    gradient_gap_value = _step6_real_scalar(gradient_gap, "gradient_gap")
    gradient_product_root_value = _step6_real_scalar(
        gradient_product_root, "gradient_product_root"
    )
    reduced_mass_value = _step6_real_scalar(reduced_mass, "reduced_mass")
    beta_value = _step6_real_scalar(beta, "beta")
    energy_floor_value = _step6_real_scalar(energy_floor, "energy_floor")
    if (
        coupling_value < 0.0
        or gradient_gap_value <= 0.0
        or gradient_product_root_value <= 0.0
        or reduced_mass_value <= 0.0
        or beta_value <= 0.0
        or -beta_value * energy_floor_value >= 600.0
        or not isinstance(quadrature_order, (int, np.integer))
        or isinstance(quadrature_order, (bool, np.bool_))
        or int(quadrature_order) < 8
        or int(quadrature_order) > 100
        or not isinstance(cap_probability, (bool, np.bool_))
    ):
        raise ValueError("physical scalars or control arguments are outside the domain")
    if coupling_value == 0.0:
        return 0.0
    nodes, weights = np.polynomial.laguerre.laggauss(int(quadrature_order))
    if (
        np.any(~np.isfinite(nodes))
        or np.any(~np.isfinite(weights))
        or np.any(nodes <= 0.0)
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Gauss-Laguerre rule is non-finite")
    energies = energy_floor_value + nodes / beta_value
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        airy_scale = np.cbrt(
            16.0
            * reduced_mass_value
            / (gradient_product_root_value * gradient_gap_value)
        )
        airy_argument = -(
            energies
            * gradient_gap_value
            * airy_scale
            / (2.0 * gradient_product_root_value)
        )
        airy_value = airy(airy_argument)[0]
        probability = (
            np.pi**2
            * airy_scale**2
            * np.square(np.float64(coupling_value))
            * airy_value**2
        )
    if np.any(np.isnan(probability)) or np.any(probability < 0.0):
        raise ValueError("the weak-coupling probability is non-finite")
    if bool(cap_probability):
        probability = np.minimum(probability, 1.0)
    elif np.any(~np.isfinite(probability)):
        raise ValueError("the uncapped weak-coupling probability is non-finite")
    coefficient = (
        np.exp(-beta_value * energy_floor_value)
        * np.dot(weights, probability)
        / beta_value
    )
    if not np.isfinite(coefficient):
        raise ValueError("the weak-coupling integral is non-finite")
    return float(coefficient)

import numpy as np


def _step7_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _step7_trapezoid(values: np.ndarray, grid: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))


def compute_thermal_isotope_rates(
    distance: "np.ndarray",
    profile_values: "np.ndarray",
    beta: float,
    crossing_descriptor: "np.ndarray",
    lz_order: int,
    wc_order: int,
) -> "np.ndarray":
    raw_distance = np.asarray(distance)
    raw_profiles = np.asarray(profile_values)
    raw_descriptor = np.asarray(crossing_descriptor)
    arrays = [raw_distance, raw_profiles, raw_descriptor]
    if any(
        not np.issubdtype(value.dtype, np.number)
        or not np.isrealobj(value)
        or np.any(~np.isfinite(value))
        for value in arrays
    ):
        raise ValueError("arrays must be finite, real, and numeric")
    r = np.asarray(raw_distance, dtype=float)
    profiles = np.asarray(raw_profiles, dtype=float)
    descriptor = np.asarray(raw_descriptor, dtype=float)
    beta_value = _step7_real_scalar(beta, "beta")
    if (
        r.ndim != 1
        or r.size < 2
        or profiles.shape != (r.size, 4)
        or descriptor.shape != (4,)
        or np.any(np.diff(r) <= 0.0)
        or np.any(profiles[:, 3] < 0.0)
        or beta_value <= 0.0
        or not isinstance(lz_order, (int, np.integer))
        or isinstance(lz_order, (bool, np.bool_))
        or not isinstance(wc_order, (int, np.integer))
        or isinstance(wc_order, (bool, np.bool_))
        or not 8 <= int(lz_order) <= 100
        or not 8 <= int(wc_order) <= 100
    ):
        raise ValueError("distance, profiles, beta, or descriptor is invalid")
    reduced_mass, gradient_gap, gradient_product_root, curvature = descriptor
    if (
        reduced_mass <= 0.0
        or gradient_gap <= 0.0
        or gradient_product_root <= 0.0
        or curvature <= 0.0
    ):
        raise ValueError("crossing_descriptor must contain four positive values")
    reactant, mecp, product, coupling = profiles.T
    shifted_boltzmann = np.exp(-beta_value * (reactant - np.min(reactant)))
    partition = _step7_trapezoid(shifted_boltzmann, r)
    if not np.isfinite(partition) or partition <= 0.0:
        raise ValueError("the distance partition must be positive and finite")
    distance_probability = shifted_boltzmann / partition
    activation_argument = -beta_value * (mecp - reactant)
    if np.any(np.abs(activation_argument) >= 600.0):
        raise ValueError("activation exponent is outside the supported finite domain")
    lz_integrand = np.empty(r.size, dtype=float)
    wc_integrand = np.empty(r.size, dtype=float)
    for index in range(r.size):
        lz_coefficient = integrate_lz_coefficient(
            float(coupling[index]),
            float(gradient_gap),
            float(reduced_mass),
            beta_value,
            lz_order,
        )
        energy_floor = -(
            float(mecp[index]) - max(float(reactant[index]), float(product[index]))
        )
        wc_coefficient = integrate_wc_coefficient(
            float(coupling[index]),
            float(gradient_gap),
            float(gradient_product_root),
            float(reduced_mass),
            beta_value,
            energy_floor,
            wc_order,
            True,
        )
        activation = float(np.exp(activation_argument[index]))
        lz_integrand[index] = distance_probability[index] * activation * lz_coefficient
        wc_integrand[index] = distance_probability[index] * activation * wc_coefficient
    lz_rate = _step7_trapezoid(lz_integrand, r)
    wc_rate = _step7_trapezoid(wc_integrand, r)
    mean_distance = _step7_trapezoid(r * distance_probability, r)
    result = np.array([lz_rate, wc_rate, partition, mean_distance], dtype=float)
    if np.any(~np.isfinite(result)) or lz_rate < 0.0 or wc_rate < 0.0:
        raise ValueError("the thermal rate calculation must be finite and nonnegative")
    return result

import numpy as np


def _step8_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _step8_make_snapshot(
    seed: int,
    isotope_index: int,
    n_distance: int,
) -> tuple:
    rng = np.random.default_rng(int(seed) + 7919 * int(isotope_index))
    classical_dim = 6
    quantum_dim = 3
    matrices = []
    for surface in range(2):
        factor = rng.normal(
            scale=0.055 + 0.004 * surface,
            size=(classical_dim, classical_dim),
        )
        h_cc = factor.T @ factor + np.diag(
            np.linspace(0.075, 0.145, classical_dim)
        )
        h_qc = rng.normal(scale=0.0055, size=(quantum_dim, classical_dim))
        quantum_factor = rng.normal(scale=0.035, size=(quantum_dim, quantum_dim))
        h_qq = (
            quantum_factor.T @ quantum_factor
            + (0.27 + 0.02 * surface) * np.eye(quantum_dim)
        )
        matrices.append(np.block([[h_cc, h_qc.T], [h_qc, h_qq]]))
    base_a = np.array([0.031, -0.025, 0.019, -0.014, 0.012, -0.009])
    base_b = np.array([-0.027, 0.021, -0.016, 0.017, -0.010, 0.011])
    gradient_a = (
        (1.0 + 0.035 * isotope_index) * base_a
        + rng.normal(scale=0.0018, size=classical_dim)
    )
    gradient_b = (
        (1.0 - 0.025 * isotope_index) * base_b
        + rng.normal(scale=0.0018, size=classical_dim)
    )
    masses = 1822.888486217313 * np.array(
        [12.0, 12.0, 15.99491462, 15.99491462, 14.003074, 14.003074]
    )
    reference_distance = (
        4.94 + 0.022 * isotope_index + float(rng.normal(scale=0.004))
    )
    distance = np.linspace(4.42, 5.50, int(n_distance))
    reactant = np.array(
        [
            0.0,
            0.00015 * (-1.0 if isotope_index == 0 else 1.0),
            0.0125 + 0.0012 * isotope_index,
            -0.0018 + 0.0002 * isotope_index,
            0.0021,
        ]
    )
    mecp = np.array(
        [
            0.00485 + 0.00058 * isotope_index,
            -0.0016 + 0.00018 * isotope_index,
            0.0088 + 0.0007 * isotope_index,
            0.0011,
            0.0015,
        ]
    )
    product = np.array(
        [
            0.00055 + 0.00016 * isotope_index,
            -0.00035,
            0.0112 + 0.0008 * isotope_index,
            0.0014,
            0.0018,
        ]
    )
    coupling = np.array(
        [
            (0.00048 if isotope_index == 0 else 0.000215)
            * (1.0 + float(rng.normal(scale=0.025))),
            -2.20 - 0.16 * isotope_index,
            -0.55 - 0.06 * isotope_index,
        ]
    )
    critical_distance = 4.505 + 0.018 * isotope_index
    return (
        matrices,
        gradient_a,
        gradient_b,
        masses,
        distance,
        reactant,
        mecp,
        product,
        coupling,
        reference_distance,
        critical_distance,
    )


def compute_neogrt_kie_shift(
    seed: int,
    temperature: float,
    n_distance: int = 61,
    lz_order: int = 48,
    wc_order: int = 64,
) -> float:
    integer_inputs = [seed, n_distance, lz_order, wc_order]
    if any(
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        for value in integer_inputs
    ):
        raise ValueError("seed, n_distance, and quadrature orders must be integers")
    temperature_value = _step8_real_scalar(temperature, "temperature")
    if (
        temperature_value <= 0.0
        or int(n_distance) < 9
        or int(n_distance) % 2 == 0
        or int(lz_order) < 8
        or int(wc_order) < 8
        or int(lz_order) > 100
        or int(wc_order) > 100
    ):
        raise ValueError("temperature, distance count, or quadrature order is invalid")
    beta = 1.0 / (3.1668114e-6 * temperature_value)
    isotope_rates = []
    for isotope_index in (0, 1):
        snapshot = _step8_make_snapshot(int(seed), isotope_index, int(n_distance))
        (
            matrices,
            gradient_a,
            gradient_b,
            masses,
            distance,
            reactant,
            mecp,
            product,
            coupling,
            reference_distance,
            critical_distance,
        ) = snapshot
        hessian_a = reduce_neo_hessian(matrices[0], 3)
        hessian_b = reduce_neo_hessian(matrices[1], 3)
        descriptor = compute_crossing_descriptor(
            hessian_a,
            hessian_b,
            gradient_a,
            gradient_b,
            masses,
            True,
        )
        profile_values = evaluate_distance_profiles(
            distance,
            reactant,
            mecp,
            product,
            coupling,
            reference_distance,
            critical_distance,
        )
        rates = compute_thermal_isotope_rates(
            distance,
            profile_values,
            beta,
            descriptor,
            int(lz_order),
            int(wc_order),
        )
        isotope_rates.append(rates)
    if any(
        not np.isfinite(rates[0])
        or not np.isfinite(rates[1])
        or rates[0] <= 0.0
        or rates[1] <= 0.0
        for rates in isotope_rates
    ):
        raise ValueError("both isotope rates must be finite and strictly positive")
    kie_lz = float(isotope_rates[0][0] / isotope_rates[1][0])
    kie_wc = float(isotope_rates[0][1] / isotope_rates[1][1])
    answer = 100.0 * (kie_wc / kie_lz - 1.0)
    if not np.isfinite(answer):
        raise ValueError("the KIE comparison must be finite")
    return float(answer)
SCICODE_GOLD_EOF
