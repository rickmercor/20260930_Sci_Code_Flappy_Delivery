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


def compute_deep_lpm_rate(
    z: float,
    momentum: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> float:
    zbar = 1.0 - z
    prefactor = alpha_s * c_a / (2.0 * np.pi * z * zbar)
    scale = np.sqrt(c_a * qhat / (z * zbar * momentum))
    return float(prefactor * scale)

import numpy as np


def _bose_occupation(energy: float, temperature: float) -> float:
    return float(1.0 / np.expm1(energy / temperature))


def compute_thermal_channel_rates(
    z: float,
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> "np.ndarray":
    zbar = 1.0 - z

    base_rate = compute_deep_lpm_rate(
        z,
        momentum,
        alpha_s,
        c_a,
        qhat,
    )

    n_first = _bose_occupation(z * momentum, temperature)
    n_second = _bose_occupation(zbar * momentum, temperature)

    gamma_1 = 0.5 * base_rate * (1.0 + n_first + n_second)

    transformed_momentum = momentum / zbar
    transformed_rate = compute_deep_lpm_rate(
        z,
        transformed_momentum,
        alpha_s,
        c_a,
        qhat,
    )

    n_incoming = _bose_occupation(z * momentum / zbar, temperature)
    n_parent = _bose_occupation(momentum / zbar, temperature)

    gamma_2 = transformed_rate * (n_incoming - n_parent) / zbar**3

    return np.array([gamma_1, gamma_2], dtype=float)

import numpy as np


def integrate_inelastic_hazard(
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
) -> "np.ndarray":
    z_nodes = np.asarray(z_nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    split_hazard = 0.0
    merge_hazard = 0.0

    for z, weight in zip(z_nodes, weights):
        rates = compute_thermal_channel_rates(
            float(z),
            momentum,
            temperature,
            alpha_s,
            c_a,
            qhat,
        )
        split_hazard += float(weight) * float(rates[0])
        merge_hazard += float(weight) * float(rates[1])

    total_hazard = split_hazard + merge_hazard

    return np.array(
        [split_hazard, merge_hazard, total_hazard],
        dtype=float,
    )

def compute_designated_split_rate(
    z_star: float,
    weight_star: float,
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> float:
    rates = compute_thermal_channel_rates(
        z_star,
        momentum,
        temperature,
        alpha_s,
        c_a,
        qhat,
    )
    return float(weight_star * rates[0])

import numpy as np


def compute_daughter_shower_state(
    z_star: float,
    parent_momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
) -> "np.ndarray":
    p1 = z_star * parent_momentum
    p2 = (1.0 - z_star) * parent_momentum

    hazard_1 = integrate_inelastic_hazard(
        p1,
        temperature,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
    )[2]

    hazard_2 = integrate_inelastic_hazard(
        p2,
        temperature,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
    )[2]

    return np.array(
        [p1, p2, hazard_1, hazard_2, hazard_1 + hazard_2],
        dtype=float,
    )

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def infer_temperature_branches(
    momentum: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
    balance_observed: float,
    survival_observed: float,
    calibration_length: float,
    temperature_bounds: "np.ndarray",
) -> "np.ndarray":
    z_nodes = np.asarray(z_nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    temperature_bounds = np.asarray(temperature_bounds, dtype=float)

    t_min = float(temperature_bounds[0])
    t_max = float(temperature_bounds[1])

    def _channel_data(temperature):
        hazards = integrate_inelastic_hazard(
            momentum,
            float(temperature),
            alpha_s,
            c_a,
            qhat,
            z_nodes,
            weights,
        )
        balance = float(hazards[0] - hazards[1])
        total_hazard = float(hazards[2])
        return balance, total_hazard

    peak_result = minimize_scalar(
        lambda temperature: -_channel_data(temperature)[0],
        bounds=(t_min, t_max),
        method="bounded",
        options={"xatol": 1e-14, "maxiter": 1000},
    )
    t_peak = float(peak_result.x)

    def _residual(temperature):
        return _channel_data(temperature)[0] - balance_observed

    t_low = brentq(
        _residual,
        t_min,
        t_peak,
        xtol=5e-15,
        rtol=1e-14,
        maxiter=500,
    )

    t_high = brentq(
        _residual,
        t_peak,
        t_max,
        xtol=5e-15,
        rtol=1e-14,
        maxiter=500,
    )

    roots = np.array([t_low, t_high], dtype=float)

    survivals = np.empty(2, dtype=float)
    for i, temperature in enumerate(roots):
        total_hazard = _channel_data(float(temperature))[1]
        survivals[i] = np.exp(-total_hazard * calibration_length)

    selected_index = int(
        np.argmin(np.abs(survivals - survival_observed))
    )
    selected_temperature = float(roots[selected_index])

    return np.array(
        [roots[0], roots[1], selected_temperature],
        dtype=float,
    )

import numpy as np
from scipy.linalg import expm


def integrate_exclusive_history_probability(
    segment_hazards: "np.ndarray",
    split_rates: "np.ndarray",
    length: float,
) -> float:
    segment_hazards = np.asarray(segment_hazards, dtype=float)
    split_rates = np.asarray(split_rates, dtype=float)

    n_emissions = split_rates.size
    generator = -np.diag(segment_hazards)
    if n_emissions > 0:
        generator[
            np.arange(n_emissions),
            np.arange(1, n_emissions + 1),
        ] = split_rates

    # Incorporating transition rates avoids a separately overflowing product.
    return float(expm(generator * float(length))[0, n_emissions])

import numpy as np


def solve_calibrated_thermal_shower_case(
    p0: float,
    length: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
    split_index: int,
    balance_observed: float,
    survival_observed: float,
    calibration_length: float,
    temperature_bounds: "np.ndarray",
) -> float:
    temperature_branches = infer_temperature_branches(
        p0,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
        balance_observed,
        survival_observed,
        calibration_length,
        temperature_bounds,
    )

    temperature = float(temperature_branches[2])

    parent_hazards = integrate_inelastic_hazard(
        p0,
        temperature,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
    )
    parent_hazard = float(parent_hazards[2])

    z_star = float(z_nodes[split_index])
    weight_star = float(weights[split_index])

    split_rate = compute_designated_split_rate(
        z_star,
        weight_star,
        p0,
        temperature,
        alpha_s,
        c_a,
        qhat,
    )

    daughter_state = compute_daughter_shower_state(
        z_star,
        p0,
        temperature,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
    )

    daughter_hazard_sum = float(daughter_state[4])

    segment_hazards = np.array(
        [parent_hazard, daughter_hazard_sum],
        dtype=float,
    )
    split_rates = np.array(
        [split_rate],
        dtype=float,
    )

    return integrate_exclusive_history_probability(
        segment_hazards,
        split_rates,
        length,
    )
SCICODE_GOLD_EOF
