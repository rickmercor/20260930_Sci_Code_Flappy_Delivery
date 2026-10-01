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

def temperature_response_factors(
    temperature: float,
    q10_values: "np.ndarray",
    reference_temperature: float = 20.0,
) -> "np.ndarray":
    temperature = float(temperature)
    q10_values = np.asarray(q10_values, dtype=np.float64)
    reference_temperature = float(reference_temperature)
    if not np.isfinite(temperature) or not np.isfinite(reference_temperature):
        raise ValueError("temperatures must be finite")
    if q10_values.ndim != 1 or q10_values.size == 0:
        raise ValueError("q10_values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must be finite and positive")
    factors = q10_values ** ((temperature - reference_temperature) / 10.0)
    return factors

import numpy as np

def phytoplankton_process_rates(
    temperature: float,
    light: float,
    phytoplankton: float,
    ammonium: float,
    nitrate: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    values = np.asarray(
        [temperature, light, phytoplankton, ammonium, nitrate], dtype=np.float64
    )
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values[1:] < 0.0):
        raise ValueError("temperature must be finite and other scalar inputs nonnegative")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)):
        raise ValueError("parameters must contain 22 finite values")
    if np.any(parameters <= 0.0):
        raise ValueError("parameters must be positive")

    a_n, mu_star, mortality_star, light_star, k_n = parameters[:5]
    respiration_star, respiration_factor, growth_respiration_fraction = parameters[12:15]
    lysis_rate = parameters[21]
    tau = temperature_response_factors(temperature, q10_values)
    light_star_t = light_star * tau[5]
    f_light = 1.0 - np.exp(-light / light_star_t)
    din = ammonium + nitrate
    if din <= 0.0:
        raise ValueError("ammonium plus nitrate must be positive")
    f_nutrient = din / (din + k_n)
    gross_rate = f_light * f_nutrient * tau[0] * mu_star
    if temperature >= 20.0:
        mortality_rate = mortality_star * tau[1]
    elif temperature > 5.0:
        mortality_rate = mortality_star
    else:
        mortality_rate = mortality_star * 0.33
    respiration_rate = (
        growth_respiration_fraction * gross_rate
        + (1.0 - growth_respiration_fraction) * respiration_star * tau[2]
    )
    loss_flux = (mortality_rate + lysis_rate) * phytoplankton
    uptake_total = a_n * (gross_rate - respiration_rate) * phytoplankton
    fraction_ammonium = 1.0 if ammonium >= 0.7 else ammonium / din
    ammonium_uptake = fraction_ammonium * uptake_total
    nitrate_uptake = (1.0 - fraction_ammonium) * uptake_total
    photosynthesis_flux = gross_rate * phytoplankton
    respiration_flux = (
        respiration_rate + respiration_factor * mortality_rate
    ) * phytoplankton
    rates = np.array(
        [
            gross_rate,
            mortality_rate,
            respiration_rate,
            loss_flux,
            ammonium_uptake,
            nitrate_uptake,
            photosynthesis_flux,
            respiration_flux,
        ],
        dtype=np.float64,
    )
    return rates

import numpy as np

def organic_matter_transformations(
    temperature: float,
    poc_labile: float,
    poc_semilabile: float,
    doc: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    pools = np.asarray([poc_labile, poc_semilabile, doc], dtype=np.float64)
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if not np.isfinite(temperature) or not np.all(np.isfinite(pools)) or np.any(pools < 0.0):
        raise ValueError("temperature must be finite and organic pools nonnegative")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("parameters must contain 22 finite positive values")
    tau_min = temperature_response_factors(temperature, q10_values)[3]
    mineralization = parameters[6:9] * tau_min * pools
    d_labile_semilabile = parameters[9] * mineralization[0]
    d_labile_doc = parameters[10] * mineralization[0]
    d_semilabile_doc = parameters[11] * mineralization[1]
    transformations = np.array(
        [
            mineralization[0],
            mineralization[1],
            mineralization[2],
            d_labile_semilabile,
            d_labile_doc,
            d_semilabile_doc,
            np.sum(mineralization),
        ],
        dtype=np.float64,
    )
    return transformations

import numpy as np

def nitrogen_oxygen_fluxes(
    temperature: float,
    oxygen: float,
    ammonium: float,
    nitrate: float,
    total_mineralization: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    values = np.asarray(
        [temperature, oxygen, ammonium, nitrate, total_mineralization],
        dtype=np.float64,
    )
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values[1:] < 0.0):
        raise ValueError("temperature must be finite and pool/flux inputs nonnegative")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("parameters must contain 22 finite positive values")
    tau = temperature_response_factors(temperature, q10_values)
    nitrification = (
        parameters[15]
        * tau[4]
        * ammonium
        / (ammonium + parameters[16])
        * oxygen
        / (oxygen + parameters[17])
    )
    phi_nitrate = (
        nitrate
        / (nitrate + parameters[18])
        * (1.0 - oxygen / (oxygen + parameters[19]))
        * tau[3]
    )
    phi_oxygen = oxygen / (oxygen + parameters[20]) * tau[3]
    denitrification_fraction = phi_nitrate / (phi_nitrate + phi_oxygen)
    denitrification_flux = denitrification_fraction * total_mineralization
    nitrification_oxygen_sink = 2.0 * nitrification
    mineralization_oxygen_sink = (
        1.0 - denitrification_fraction
    ) * total_mineralization
    fluxes = np.array(
        [
            nitrification,
            denitrification_fraction,
            denitrification_flux,
            nitrification_oxygen_sink,
            mineralization_oxygen_sink,
        ],
        dtype=np.float64,
    )
    return fluxes

import numpy as np

def oxypom_reduced_rhs(
    time: float,
    state: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    state = np.asarray(state, dtype=np.float64)
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    forcing_parameters = np.asarray(forcing_parameters, dtype=np.float64)
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if state.shape != (7,) or not np.all(np.isfinite(state)) or np.any(state < 0.0):
        raise ValueError("state must contain seven finite nonnegative values")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("parameters must contain 22 finite positive values")
    if forcing_parameters.shape != (9,) or not np.all(np.isfinite(forcing_parameters)):
        raise ValueError("forcing_parameters must contain nine finite values")
    if forcing_parameters[2] <= 0.0:
        raise ValueError("forcing period must be positive")

    phase = 2.0 * np.pi * time / forcing_parameters[2]
    temperature = forcing_parameters[0] + forcing_parameters[1] * np.sin(phase)
    light = forcing_parameters[3] + forcing_parameters[4] * np.sin(phase - 0.4)
    k_air = forcing_parameters[5] + forcing_parameters[6] * np.cos(phase + 0.2)
    oxygen_saturation = forcing_parameters[7] - forcing_parameters[8] * temperature
    if light < 0.0 or k_air < 0.0 or oxygen_saturation < 0.0:
        raise ValueError("forcing produces a negative light, exchange, or saturation value")

    oxygen, phyto, poc_labile, poc_semilabile, doc, ammonium, nitrate = state
    phyto_rates = phytoplankton_process_rates(
        temperature, light, phyto, ammonium, nitrate, q10_values, parameters
    )
    organic_rates = organic_matter_transformations(
        temperature, poc_labile, poc_semilabile, doc, q10_values, parameters
    )
    nitrogen_rates = nitrogen_oxygen_fluxes(
        temperature,
        oxygen,
        ammonium,
        nitrate,
        organic_rates[6],
        q10_values,
        parameters,
    )
    gross_rate, mortality_rate, respiration_rate = phyto_rates[:3]
    loss_flux, uptake_ammonium, uptake_nitrate = phyto_rates[3:6]
    photo_flux, respiration_flux = phyto_rates[6:8]
    mineral_labile, mineral_semilabile, mineral_doc = organic_rates[:3]
    d_labile_semilabile, d_labile_doc, d_semilabile_doc = organic_rates[3:6]
    nitrification, _, denitrification = nitrogen_rates[:3]
    nitrification_oxygen, mineralization_oxygen = nitrogen_rates[3:5]
    reaeration = k_air * (oxygen_saturation - oxygen)
    release_fraction = parameters[5]
    derivative = np.array(
        [
            reaeration + photo_flux - respiration_flux
            - nitrification_oxygen - mineralization_oxygen,
            phyto * (gross_rate - mortality_rate - respiration_rate - parameters[21]),
            (1.0 - release_fraction) * loss_flux
            - d_labile_semilabile - d_labile_doc - mineral_labile,
            d_labile_semilabile - d_semilabile_doc - mineral_semilabile,
            d_labile_doc + d_semilabile_doc - mineral_doc,
            parameters[0] * release_fraction * loss_flux
            + parameters[0] * organic_rates[6]
            - uptake_ammonium - nitrification,
            nitrification - denitrification - uptake_nitrate,
        ],
        dtype=np.float64,
    )
    diagnostic_fluxes = np.array(
        [photo_flux, respiration_flux, nitrification_oxygen, mineralization_oxygen],
        dtype=np.float64,
    )
    return derivative, diagnostic_fluxes

import numpy as np
 
 
def _critical_ammonium():
    return 0.7
 
 
def _environment(time, forcing_parameters):
    phase = 2.0 * np.pi * time / forcing_parameters[2]
    temperature = forcing_parameters[0] + forcing_parameters[1] * np.sin(phase)
    light = forcing_parameters[3] + forcing_parameters[4] * np.sin(phase - 0.4)
    return temperature, light
 
 
def ammonium_sliding_field(
    time: float,
    state: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
) -> "np.ndarray":
    surface_state = np.array(state, dtype=np.float64)
    if surface_state.shape != (7,):
        raise ValueError("state must contain seven values")
    surface_state[5] = _critical_ammonium()
    forcing_parameters = np.asarray(forcing_parameters, dtype=np.float64)
    # Step 5 validates every contract; at NH4 = c it evaluates the chi = 1 branch.
    f_plus = oxypom_reduced_rhs(
        time, surface_state, q10_values, parameters, forcing_parameters
    )[0]
    temperature, light = _environment(time, forcing_parameters)
    rates = phytoplankton_process_rates(
        temperature,
        light,
        surface_state[1],
        surface_state[5],
        surface_state[6],
        q10_values,
        parameters,
    )
    total_uptake = rates[4] + rates[5]
    chi_minus = _critical_ammonium() / (_critical_ammonium() + surface_state[6])
    f_minus = f_plus.copy()
    f_minus[5] += (1.0 - chi_minus) * total_uptake
    f_minus[6] -= (1.0 - chi_minus) * total_uptake
    gap = f_minus[5] - f_plus[5]
    if gap == 0.0:
        raise ValueError("the one-sided ammonium rates coincide; no sliding combination exists")
    alpha = f_minus[5] / gap
    sliding = (1.0 - alpha) * f_minus + alpha * f_plus
    sliding[5] = 0.0
    chi_eq = (1.0 - alpha) * chi_minus + alpha
    return np.concatenate([[f_minus[5], f_plus[5], alpha, chi_eq], sliding]).astype(np.float64)

import numpy as np
from scipy.optimize import brentq
 
 
def _validate_filippov_cycle_inputs(
    cycle_start,
    q10_values,
    parameters,
    forcing_parameters,
    renewal_state,
    retention_fraction,
    time_step,
):
    cycle_start = np.asarray(cycle_start, dtype=np.float64)
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    forcing_parameters = np.asarray(forcing_parameters, dtype=np.float64)
    renewal_state = np.asarray(renewal_state, dtype=np.float64)
    retention_fraction = float(retention_fraction)
    time_step = float(time_step)
    for name, vector in (("cycle_start", cycle_start), ("renewal_state", renewal_state)):
        if vector.shape != (7,) or not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
            raise ValueError(f"{name} must contain seven finite nonnegative values")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("parameters must contain 22 finite positive values")
    if forcing_parameters.shape != (9,) or not np.all(np.isfinite(forcing_parameters)) or forcing_parameters[2] <= 0.0:
        raise ValueError("forcing_parameters must contain nine finite values with positive period")
    if not np.isfinite(retention_fraction) or not 0.0 < retention_fraction < 1.0:
        raise ValueError("retention_fraction must lie strictly between zero and one")
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be finite and positive")
    number_of_steps = int(round(forcing_parameters[2] / time_step))
    if number_of_steps < 1 or abs(number_of_steps * time_step - forcing_parameters[2]) > 1.0e-12:
        raise ValueError("forcing period must be an integer multiple of time_step")
    return (
        cycle_start,
        q10_values,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
        number_of_steps,
    )
 
 
def _branch_field(time, state, side, q10_values, parameters, forcing_parameters):
    """Smooth one-sided Step-5 branch: side -1 uses chi = NH4/DIN, side +1 uses chi = 1."""
    derivative = oxypom_reduced_rhs(
        time, state, q10_values, parameters, forcing_parameters
    )[0].copy()
    ammonium, nitrate = state[5], state[6]
    chi_step5 = 1.0 if ammonium >= _critical_ammonium() else ammonium / (ammonium + nitrate)
    chi_branch = 1.0 if side > 0 else ammonium / (ammonium + nitrate)
    if chi_branch != chi_step5:
        temperature, light = _environment(time, forcing_parameters)
        rates = phytoplankton_process_rates(
            temperature, light, state[1], ammonium, nitrate, q10_values, parameters
        )
        total_uptake = rates[4] + rates[5]
        derivative[5] -= (chi_branch - chi_step5) * total_uptake
        derivative[6] += (chi_branch - chi_step5) * total_uptake
    return derivative
 
 
def _surface_rates(time, state, q10_values, parameters, forcing_parameters):
    try:
        return ammonium_sliding_field(time, state, q10_values, parameters, forcing_parameters)
    except ValueError as error:
        raise RuntimeError("no sliding field exists on the critical-ammonium surface") from error
 
 
def _rk4_mode_step(time, state, step_size, mode, q10_values, parameters, forcing_parameters):
    if step_size == 0.0:
        return state.copy()
    if mode == 0:
        def _field(t, y):
            return _surface_rates(t, y, q10_values, parameters, forcing_parameters)[4:]
    else:
        def _field(t, y):
            return _branch_field(t, y, mode, q10_values, parameters, forcing_parameters)
    k1 = _field(time, state)
    k2 = _field(time + 0.5 * step_size, state + 0.5 * step_size * k1)
    k3 = _field(time + 0.5 * step_size, state + 0.5 * step_size * k2)
    k4 = _field(time + step_size, state + step_size * k3)
    return state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
 
 
def _event_offset(function, length):
    """First offset in [0, length] at which a scalar event function reaches zero.
 
    The function is negative before the event; a nonnegative value at offset 0
    means the event has already occurred when the sub-interval starts.
    """
    if function(0.0) >= 0.0:
        return 0.0
    if function(length) == 0.0:
        return length
    return brentq(function, 0.0, length, xtol=1.0e-15, rtol=4.0 * np.finfo(float).eps, maxiter=200)
 
 
def integrate_filippov_cycle_rk4(
    cycle_start: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    (
        cycle_start,
        q10_values,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
        number_of_steps,
    ) = _validate_filippov_cycle_inputs(
        cycle_start,
        q10_values,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
    )
    critical = _critical_ammonium()
    args = (q10_values, parameters, forcing_parameters)
    times = np.linspace(0.0, forcing_parameters[2], number_of_steps + 1)
    states = np.empty((number_of_steps + 1, 7), dtype=np.float64)
    fluxes = np.empty((number_of_steps + 1, 4), dtype=np.float64)
    states[0] = cycle_start
    events = []
 
    if cycle_start[5] < critical:
        mode = -1
    elif cycle_start[5] > critical:
        mode = 1
    else:
        f_minus, f_plus = _surface_rates(0.0, cycle_start, *args)[:2]
        if f_minus > 0.0 and f_plus < 0.0:
            mode = 0
            events.append((0.0, 0.0))
        elif f_minus > 0.0 and f_plus > 0.0:
            mode = 1
        elif f_minus < 0.0 and f_plus < 0.0:
            mode = -1
        else:
            raise RuntimeError("the cycle starts on a repelling part of the surface")
 
    for index in range(number_of_steps):
        time, state = times[index], states[index].copy()
        remaining = time_step
        count = 0
        while remaining > 0.0:
            if count > 4:
                raise RuntimeError("more than four events in one grid interval")
            trial = _rk4_mode_step(time, state, remaining, mode, *args)
            if mode == 0:
                end_minus, end_plus = _surface_rates(time + remaining, trial, *args)[:2]
                if end_minus > 0.0 and end_plus < 0.0:
                    state, remaining = trial, 0.0
                    continue
 
                def _normal(offset, which):
                    end = _rk4_mode_step(time, state, offset, 0, *args)
                    return _surface_rates(time + offset, end, *args)[which]
 
                candidates = []
                if end_minus <= 0.0:
                    candidates.append((_event_offset(lambda u: -_normal(u, 0), remaining), -1))
                if end_plus >= 0.0:
                    candidates.append((_event_offset(lambda u: _normal(u, 1), remaining), 1))
                offset, mode = min(candidates)
                state = _rk4_mode_step(time, state, offset, 0, *args)
                state[5] = critical
                time, remaining = time + offset, remaining - offset
                events.append((time, float(mode)))
            else:
                reached = trial[5] >= critical if mode < 0 else trial[5] <= critical
                if not (state[5] != critical and reached):
                    state, remaining = trial, 0.0
                    continue
                offset = _event_offset(
                    lambda u: mode * (critical - _rk4_mode_step(time, state, u, mode, *args)[5]), remaining
                )
                state = _rk4_mode_step(time, state, offset, mode, *args)
                state[5] = critical
                time, remaining = time + offset, remaining - offset
                f_minus, f_plus = _surface_rates(time, state, *args)[:2]
                if mode < 0:
                    if f_plus < 0.0:
                        mode, code = 0, 0.0
                    else:
                        mode, code = 1, 2.0
                else:
                    if f_minus > 0.0:
                        mode, code = 0, 0.0
                    else:
                        mode, code = -1, -2.0
                events.append((time, code))
            count += 1
        if not np.all(np.isfinite(state)) or np.any(state < 0.0):
            raise RuntimeError("integration produced a nonfinite or negative state")
        states[index + 1] = state
 
    for index in range(number_of_steps + 1):
        fluxes[index] = oxypom_reduced_rhs(times[index], states[index], *args)[1]
    next_cycle_start = retention_fraction * states[-1] + (1.0 - retention_fraction) * renewal_state
    event_array = np.array(events, dtype=np.float64).reshape(-1, 2)
    return times, states, fluxes, next_cycle_start, event_array

import numpy as np
 
 
def _cycle_outputs(start, q10_values, parameters, forcing_parameters, renewal_state, retention_fraction, time_step):
    """Renewed state (7) followed by the ten cycle diagnostics (10)."""
    times, states, fluxes, next_start, events = integrate_filippov_cycle_rk4(
        start, q10_values, parameters, forcing_parameters, renewal_state, retention_fraction, time_step
    )
    onset = [time for time, code in events if code == 0.0]
    if not onset:
        raise RuntimeError("the cycle never starts sliding")
    release = [time for time, code in events if abs(code) == 1.0 and time > onset[0]]
    if not release:
        raise RuntimeError("the cycle never leaves the surface after sliding starts")
    shares = np.empty(times.size)
    for index, (time, state) in enumerate(zip(times, states)):
        if state[5] == _critical_ammonium():
            shares[index] = ammonium_sliding_field(
                time, state, q10_values, parameters, forcing_parameters
            )[3]
        elif state[5] > _critical_ammonium():
            shares[index] = 1.0
        else:
            shares[index] = state[5] / (state[5] + state[6])
    series = np.column_stack(
        [states[:, 0], states[:, 2] + states[:, 3], shares, states[:, 6], fluxes]
    )
    weights = np.ones(times.size)
    weights[0] = weights[-1] = 0.5
    means = time_step * (weights @ series) / forcing_parameters[2]
    sliding_fraction = (release[0] - onset[0]) / forcing_parameters[2]
    return np.concatenate([next_start, means, [sliding_fraction, release[0]]])
 
 
def _centered_derivative(function, base, direction, step):
    """Second-order centered derivative of function at base along direction."""
    return (function(base + step * direction) - function(base - step * direction)) / (2.0 * step)
 
 
def filippov_periodic_q10_elasticities(
    initial_cycle_start: "np.ndarray",
    reference_q10: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
    closure_tolerance: float,
    max_cycles: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    closure_tolerance = float(closure_tolerance)
    if not np.isfinite(closure_tolerance) or closure_tolerance <= 0.0:
        raise ValueError("closure_tolerance must be finite and positive")
    if isinstance(max_cycles, (bool, np.bool_)) or not isinstance(max_cycles, (int, np.integer)) or max_cycles < 1:
        raise ValueError("max_cycles must be a positive integer")
    (
        start,
        reference_q10,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
        _,
    ) = _validate_filippov_cycle_inputs(
        initial_cycle_start,
        reference_q10,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
    )
    fixed = (parameters, forcing_parameters, renewal_state, retention_fraction, time_step)
 
    cycle_start = start.copy()
    for _ in range(int(max_cycles)):
        next_start = integrate_filippov_cycle_rk4(cycle_start, reference_q10, *fixed)[3]
        closure = np.max(np.abs(next_start - cycle_start))
        cycle_start = next_start
        if closure <= closure_tolerance:
            break
    else:
        raise RuntimeError("the renewal map did not close within max_cycles")
 
    reference = _cycle_outputs(cycle_start, reference_q10, *fixed)
    diagnostics = reference[7:]
    onset_time = diagnostics[9] - diagnostics[8] * forcing_parameters[2]
    if not onset_time > 0.0:
        raise RuntimeError("the periodic cycle must start sliding at a positive time")
    if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics == 0.0):
        raise RuntimeError("diagnostics must be finite and nonzero")
 
    log_q10 = np.log(reference_q10)
    state_jacobian = np.empty((17, 7))
    log_q10_jacobian = np.empty((17, 6))
    for j in range(7):
        step = 1.0e-5 * max(1.0, abs(cycle_start[j]))
        state_jacobian[:, j] = _centered_derivative(
            lambda y: _cycle_outputs(y, reference_q10, *fixed), cycle_start, np.eye(7)[j], step
        )
    for k in range(6):
        log_q10_jacobian[:, k] = _centered_derivative(
            lambda ell: _cycle_outputs(cycle_start, np.exp(ell), *fixed), log_q10, np.eye(6)[k], 1.0e-5
        )
 
    map_state = state_jacobian[:7]
    try:
        start_sensitivity = np.linalg.solve(np.eye(7) - map_state, log_q10_jacobian[:7])
    except np.linalg.LinAlgError as error:
        raise RuntimeError("I - dM/dy is singular") from error
    total = state_jacobian[7:] @ start_sensitivity + log_q10_jacobian[7:]
    elasticity_matrix = total / diagnostics[:, None]
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(map_state))))
    return cycle_start, diagnostics, elasticity_matrix, spectral_radius

import numpy as np
 
 
def filippov_q10_sensitivity_norm(
    initial_cycle_start: "np.ndarray",
    reference_q10: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
    closure_tolerance: float,
    max_cycles: int,
) -> float:
    elasticity_matrix = filippov_periodic_q10_elasticities(
        initial_cycle_start,
        reference_q10,
        parameters,
        forcing_parameters,
        renewal_state,
        retention_fraction,
        time_step,
        closure_tolerance,
        max_cycles,
    )[2]
    return float(np.linalg.norm(elasticity_matrix))
SCICODE_GOLD_EOF
