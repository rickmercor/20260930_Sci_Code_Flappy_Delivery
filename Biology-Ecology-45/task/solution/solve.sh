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
 
def environmental_degradation_rate(
    temperature_c: "np.ndarray",
    moisture: "np.ndarray",
    field_capacity: "np.ndarray",
    k_ref: float,
    activation_energy: float,
    gas_constant: float,
    reference_temperature_k: float,
    moisture_exponent: float,
) -> "np.ndarray":
    t = np.asarray(temperature_c, dtype=float)
    theta = np.asarray(moisture, dtype=float)
    fc = np.asarray(field_capacity, dtype=float)
    if t.shape != theta.shape or t.shape != fc.shape or t.ndim != 1 or t.size == 0:
        raise ValueError("temperature, moisture, and field_capacity must share a nonempty 1D shape")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(theta)) or not np.all(np.isfinite(fc)):
        raise ValueError("environmental arrays must be finite")
    if np.any(theta < 0.0) or np.any(fc <= 0.0):
        raise ValueError("moisture must be nonnegative and field_capacity positive")
    scalars = [k_ref, activation_energy, gas_constant, reference_temperature_k, moisture_exponent]
    if not all(np.isfinite(x) for x in scalars):
        raise ValueError("kinetic parameters must be finite")
    if k_ref < 0.0 or activation_energy < 0.0 or gas_constant <= 0.0 or reference_temperature_k <= 0.0 or moisture_exponent < 0.0:
        raise ValueError("kinetic parameters are outside their allowed ranges")
    out = np.zeros_like(t, dtype=float)
    mask = t > 0.0
    if np.any(mask):
        tk = t[mask] + 273.15
        walker = np.minimum(theta[mask] / fc[mask], 1.0) ** moisture_exponent
        arrhenius = np.exp(
            activation_energy * (tk - reference_temperature_k)
            / (gas_constant * tk * reference_temperature_k)
        )
        out[mask] = k_ref * arrhenius * walker
    return out

import math
import numpy as np
 
def freundlich_aqueous_concentration(
    total_mass: float,
    water_volume: float,
    soil_mass: float,
    freundlich_coefficient: float,
    freundlich_exponent: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    values = [total_mass, water_volume, soil_mass, freundlich_coefficient, freundlich_exponent, tolerance]
    if not all(np.isfinite(x) for x in values):
        raise ValueError("all numeric inputs must be finite")
    if total_mass < 0.0 or water_volume <= 0.0 or soil_mass < 0.0 or freundlich_coefficient < 0.0 or freundlich_exponent <= 0.0 or tolerance <= 0.0:
        raise ValueError("mass and partition parameters are outside their allowed ranges")
    if not isinstance(max_iterations, (int, np.integer)) or isinstance(max_iterations, (bool, np.bool_)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if total_mass == 0.0:
        return 0.0
    lo = 0.0
    hi = total_mass / water_volume
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        reconstructed = water_volume * mid + soil_mass * freundlich_coefficient * mid ** freundlich_exponent
        if reconstructed < total_mass:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance * max(1.0, hi):
            return float(0.5 * (lo + hi))
    raise RuntimeError("Freundlich solve did not converge")

import math
import numpy as np
 
def canopy_washoff(
    canopy_mass: float,
    rainfall: float,
    leaf_area_index: float,
    ground_cover_fraction: float,
    solubility_g_l: float,
    foliar_half_life_days: float,
    interception_alpha: float,
) -> "np.ndarray":
    values = [canopy_mass, rainfall, leaf_area_index, ground_cover_fraction, solubility_g_l, foliar_half_life_days, interception_alpha]
    if not all(np.isfinite(x) for x in values):
        raise ValueError("all canopy inputs must be finite")
    if canopy_mass < 0.0 or rainfall < 0.0 or leaf_area_index < 0.0 or solubility_g_l <= 0.0 or foliar_half_life_days <= 0.0 or interception_alpha <= 0.0:
        raise ValueError("canopy inputs are outside their allowed ranges")
    if not 0.0 <= ground_cover_fraction <= 1.0:
        raise ValueError("ground_cover_fraction must lie in [0, 1]")
    if rainfall == 0.0 or leaf_area_index == 0.0 or ground_cover_fraction == 0.0:
        intercepted = 0.0
    else:
        x = ground_cover_fraction * rainfall / (interception_alpha * leaf_area_index)
        intercepted = interception_alpha * leaf_area_index * (1.0 - 1.0 / (1.0 + x))
    effective = max(rainfall - intercepted, 0.0)
    extraction = 0.0160 * solubility_g_l ** 0.3832
    post_wash = canopy_mass * math.exp(-extraction * effective)
    transferred = canopy_mass - post_wash
    remaining = post_wash * math.exp(-math.log(2.0) / foliar_half_life_days)
    dissipated = post_wash - remaining
    return np.array([remaining, transferred, dissipated, intercepted], dtype=float)

import math
import numpy as np
 
def preferential_bypass(
    aqueous_concentration: float,
    available_dissolved_mass: float,
    rainfall: float,
    retention_current: float,
    retention_maximum: float,
    movement_fraction: float,
    adsorption_fraction: float,
    layer_thickness: "np.ndarray",
) -> "np.ndarray":
    thickness = np.asarray(layer_thickness, dtype=float)
    values = [aqueous_concentration, available_dissolved_mass, rainfall, retention_current, retention_maximum, movement_fraction, adsorption_fraction]
    if thickness.ndim != 1 or thickness.size == 0 or not np.all(np.isfinite(thickness)) or np.any(thickness <= 0.0):
        raise ValueError("layer_thickness must be a nonempty positive finite 1D array")
    if not all(np.isfinite(x) for x in values):
        raise ValueError("preferential-flow inputs must be finite")
    if aqueous_concentration < 0.0 or available_dissolved_mass < 0.0 or rainfall < 0.0 or retention_current < 0.0 or retention_maximum <= 0.0:
        raise ValueError("preferential-flow magnitudes are outside their allowed ranges")
    if retention_current > retention_maximum:
        raise ValueError("retention_current cannot exceed retention_maximum")
    if not 0.0 <= movement_fraction <= 1.0 or not 0.0 <= adsorption_fraction <= 1.0:
        raise ValueError("movement and adsorption fractions must lie in [0, 1]")
    wcrk = movement_fraction * rainfall * retention_current / retention_maximum
    bypass = min(wcrk * aqueous_concentration, available_dissolved_mass)
    deposited = bypass * adsorption_fraction * thickness / np.sum(thickness)
    exported = bypass - float(np.sum(deposited))
    return np.concatenate(([wcrk, bypass], deposited, [exported])).astype(float)

import math
import numpy as np
 
def daily_glyphosate_ampa_update(
    state: "np.ndarray",
    temperature_c: "np.ndarray",
    moisture: "np.ndarray",
    rainfall: float,
    leaf_area_index: float,
    ground_cover_fraction: float,
    retention_current: float,
    retention_maximum: float,
    profile: dict,
    parameters: dict,
) -> "np.ndarray":
    thickness = np.asarray(profile["layer_thickness"], dtype=float)
    layers = thickness.size
    y = np.asarray(state, dtype=float).copy()
    if y.shape != (2 * layers + 3,) or not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("state has an invalid shape or value")
    soil_mass = np.asarray(profile["soil_mass"], dtype=float)
    field_capacity = np.asarray(profile["field_capacity"], dtype=float)
    gly_kf = np.asarray(profile["glyphosate_kf"], dtype=float) * float(parameters["glyphosate_kf_scale"])
    gly_n = np.asarray(profile["glyphosate_exponent"], dtype=float) * float(parameters["glyphosate_exponent_scale"])
    ampa_kf = np.asarray(profile["ampa_kf"], dtype=float) * float(parameters["ampa_kf_scale"])
    ampa_n = np.asarray(profile["ampa_exponent"], dtype=float) * float(parameters["ampa_exponent_scale"])
    for arr in [soil_mass, field_capacity, gly_kf, gly_n, ampa_kf, ampa_n, np.asarray(temperature_c), np.asarray(moisture)]:
        if np.asarray(arr).shape != (layers,):
            raise ValueError("all profile and daily layer arrays must have shape (L,)")
    canopy = y[0]
    gly = y[1 : 1 + layers].copy()
    ampa = y[1 + layers : 1 + 2 * layers].copy()
    export_gly, export_ampa = y[-2:]
 
    canopy_result = canopy_washoff(
        canopy,
        rainfall,
        leaf_area_index,
        ground_cover_fraction,
        parameters["solubility_g_l"],
        parameters["foliar_half_life_days"],
        parameters["interception_alpha"],
    )
    canopy = canopy_result[0]
    gly[0] += canopy_result[1]
 
    rate = environmental_degradation_rate(
        temperature_c,
        moisture,
        field_capacity,
        parameters["k_ref"],
        parameters["activation_energy"],
        parameters["gas_constant"],
        parameters["reference_temperature_k"],
        parameters["moisture_exponent"],
    )
    gly_after = gly * np.exp(-rate)
    gly_degraded = gly - gly_after
    ampa_after = ampa * np.exp(-rate) + parameters["transformation_fraction"] * gly_degraded
    gly = gly_after
    ampa = ampa_after
 
    water_volume = np.asarray(moisture, dtype=float) * thickness * 10.0
    if np.any(water_volume <= 0.0):
        raise ValueError("daily moisture must imply positive layer water volumes")
    for solute, kf, exponent, export_name in [
        (gly, gly_kf, gly_n, "gly"),
        (ampa, ampa_kf, ampa_n, "ampa"),
    ]:
        aqueous = np.array([
            freundlich_aqueous_concentration(
                solute[i], water_volume[i], soil_mass[i], kf[i], exponent[i], 1.0e-13, 300
            )
            for i in range(layers)
        ])
        available = water_volume[0] * aqueous[0]
        routed = preferential_bypass(
            aqueous[0],
            available,
            rainfall,
            retention_current,
            retention_maximum,
            parameters["movement_fraction"],
            parameters["adsorption_fraction"],
            thickness,
        )
        bypass = routed[1]
        deposits = routed[2 : 2 + layers]
        exported = routed[-1]
        solute[0] -= bypass
        solute += deposits
        solute[solute < 0.0] = 0.0
        if export_name == "gly":
            export_gly += exported
        else:
            export_ampa += exported
 
    return np.concatenate(([canopy], gly, ampa, [export_gly, export_ampa])).astype(float)

import math
import numpy as np
 
def simulate_glyphosate_ampa_fate(
    forcing: dict,
    profile: dict,
    parameters: dict,
) -> "np.ndarray":
    """Reference multi-day parent--metabolite simulation."""
    required_forcing = ["temperature_c", "moisture", "rainfall", "leaf_area_index", "ground_cover_fraction", "retention_current", "retention_maximum"]
    required_profile = ["layer_thickness", "soil_mass", "field_capacity", "glyphosate_kf", "glyphosate_exponent", "ampa_kf", "ampa_exponent"]
    required_parameters = ["application_mass", "k_ref", "activation_energy", "gas_constant", "reference_temperature_k", "moisture_exponent", "solubility_g_l", "foliar_half_life_days", "transformation_fraction", "movement_fraction", "adsorption_fraction", "interception_alpha", "glyphosate_kf_scale", "glyphosate_exponent_scale", "ampa_kf_scale", "ampa_exponent_scale"]
    if not all(k in forcing for k in required_forcing) or not all(k in profile for k in required_profile) or not all(k in parameters for k in required_parameters):
        raise ValueError("configuration is missing required keys")
    temp = np.asarray(forcing["temperature_c"], dtype=float)
    moisture = np.asarray(forcing["moisture"], dtype=float)
    if temp.ndim != 2 or temp.shape != moisture.shape or temp.shape[0] < 1 or temp.shape[1] < 1:
        raise ValueError("temperature_c and moisture must share a nonempty (D,L) shape")
    days, layers = temp.shape
    for key in required_forcing[2:]:
        arr = np.asarray(forcing[key], dtype=float)
        if arr.shape != (days,) or not np.all(np.isfinite(arr)):
            raise ValueError("every scalar forcing must be finite with shape (D,)")
    for key in required_profile:
        arr = np.asarray(profile[key], dtype=float)
        if arr.shape != (layers,) or not np.all(np.isfinite(arr)):
            raise ValueError("every profile array must be finite with shape (L,)")
    for key in required_parameters:
        if not np.isfinite(parameters[key]):
            raise ValueError("every required parameter must be finite")
    if parameters["application_mass"] < 0.0:
        raise ValueError("application_mass must be nonnegative")
    cover = np.asarray(forcing["ground_cover_fraction"], dtype=float)
    if np.any((cover < 0.0) | (cover > 1.0)):
        raise ValueError("ground_cover_fraction must lie in [0,1]")
 
    state = np.zeros(2 * layers + 3, dtype=float)
    state[0] = cover[0] * parameters["application_mass"]
    state[1] = (1.0 - cover[0]) * parameters["application_mass"]
    for day in range(days):
        state = daily_glyphosate_ampa_update(
            state, temp[day], moisture[day],
            float(np.asarray(forcing["rainfall"], dtype=float)[day]),
            float(np.asarray(forcing["leaf_area_index"], dtype=float)[day]),
            float(cover[day]),
            float(np.asarray(forcing["retention_current"], dtype=float)[day]),
            float(np.asarray(forcing["retention_maximum"], dtype=float)[day]),
            profile, parameters,
        )
    return state

import math
import numpy as np
 
def pathway_sensitivity_matrix(
    forcing: dict,
    profile: dict,
    parameters: dict,
    sensitive_keys: list[str],
    response_indices: "np.ndarray",
    relative_step: float,
) -> "np.ndarray":
    """Reference standardized central-difference sensitivity matrix."""
    if not np.isfinite(relative_step) or not 0.0 < relative_step < 1.0:
        raise ValueError("relative_step must be finite and lie in (0,1)")
    if not isinstance(sensitive_keys, list) or len(sensitive_keys) == 0 or len(set(sensitive_keys)) != len(sensitive_keys):
        raise ValueError("sensitive_keys must be a nonempty list of unique names")
    raw_indices = np.asarray(response_indices)
    if raw_indices.ndim != 1 or raw_indices.size == 0 or raw_indices.dtype.kind not in "iu":
        raise ValueError("response_indices must be a nonempty integer array")
    baseline = simulate_glyphosate_ampa_fate(forcing, profile, parameters)
    indices = raw_indices.astype(int, copy=False)
    if np.any(indices < 0) or np.any(indices >= baseline.size) or np.unique(indices).size != indices.size:
        raise ValueError("response_indices are out of range or duplicated")
    selected = baseline[indices]
    if np.any(selected <= 0.0):
        raise ValueError("selected baseline responses must be strictly positive")
    matrix = np.empty((indices.size, len(sensitive_keys)), dtype=float)
    for column, key in enumerate(sensitive_keys):
        if key not in parameters or not np.isfinite(parameters[key]) or parameters[key] <= 0.0:
            raise ValueError("every sensitive parameter must exist and be finite and positive")
        minus = dict(parameters)
        plus = dict(parameters)
        minus[key] = parameters[key] * (1.0 - relative_step)
        plus[key] = parameters[key] * (1.0 + relative_step)
        if key in {"movement_fraction", "adsorption_fraction", "transformation_fraction"} and (minus[key] < 0.0 or plus[key] > 1.0):
            raise ValueError("relative perturbation leaves a fraction outside [0,1]")
        y_minus = simulate_glyphosate_ampa_fate(forcing, profile, minus)[indices]
        y_plus = simulate_glyphosate_ampa_fate(forcing, profile, plus)[indices]
        matrix[:, column] = (y_plus - y_minus) / (2.0 * relative_step * selected)
    return matrix

import math
import numpy as np
 
def epic_fate_sensitivity_norm(
    forcing: dict,
    profile: dict,
    parameters: dict,
    sensitive_keys: list[str],
    response_indices: "np.ndarray",
    relative_step: float,
) -> float:
    """Reference end-to-end standardized sensitivity norm."""
    matrix = pathway_sensitivity_matrix(
        forcing, profile, parameters, sensitive_keys, response_indices, relative_step
    )
    return float(np.linalg.norm(matrix, ord="fro"))
SCICODE_GOLD_EOF
