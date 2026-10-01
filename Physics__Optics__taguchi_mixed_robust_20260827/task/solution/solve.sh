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

import numpy as np

def taguchi_initial_levels(v_min, v_max, s=3):
    lower, upper = float(v_min), float(v_max)
    count = int(s)
    if count != s or count < 1 or not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Require finite v_min<v_max and positive integer s")
    wide = np.longdouble
    spacing = (wide(upper) - wide(lower)) / (count + 1)
    levels = wide(lower) + np.arange(1, count + 1, dtype=wide) * spacing
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(np.append(levels, spacing), dtype=float)
    if not np.isfinite(output).all() or output[-1] <= 0:
        raise ValueError("Levels or positive spacing are not representable")
    return output

import numpy as np

def _is_prime_integer(n: int) -> bool:
    if n < 2:
        return False
    return all(n % d for d in range(2, int(np.sqrt(n)) + 1))

def prime_strength2_oa(q: int, k: int) -> np.ndarray:
    q = int(q)
    k = int(k)
    if not _is_prime_integer(q) or not 2 <= k <= q + 1:
        raise ValueError("Require prime q and 2 <= k <= q+1")
    rows = []
    for a in range(q):
        for b in range(q):
            row = [a, b]
            row.extend((a + m * b) % q for m in range(1, k - 1))
            rows.append(row)
    return np.asarray(rows, dtype=int)

import numpy as np

def sech_soliton_peak_power_mw(beta2_fs2_per_m, gamma_w_inv_km, fwhm_ps):
    dispersion, gamma, width = map(float, (beta2_fs2_per_m, gamma_w_inv_km, fwhm_ps))
    if not np.isfinite([dispersion, gamma, width]).all() or gamma <= 0 or width <= 0:
        raise ValueError("Require finite dispersion and positive gamma and FWHM")
    wide = np.longdouble
    factor = 2 * np.arccosh(np.sqrt(wide(2)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        power = float(abs(wide(dispersion)) * factor**2 / wide(gamma) / wide(width)**2)
    if not np.isfinite(power) or (dispersion != 0 and power <= 0):
        raise ValueError("Positive peak power is not representable as float64")
    return power

import numpy as np

def guiding_center_theory_candidates(alpha_db_per_km, span_km, p0_mw):
    loss, span, power = map(float, (alpha_db_per_km, span_km, p0_mw))
    if not np.isfinite([loss, span, power]).all() or loss < 0 or span < 0 or power <= 0:
        raise ValueError("Require finite nonnegative loss/span and positive power")
    wide = np.longdouble
    exponent = wide(loss) * wide(span) * np.log(wide(10)) / 10
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gain = np.exp(exponent)
        if exponent == 0:
            multipliers = np.ones(4, dtype=wide)
        else:
            decrement = -np.expm1(-exponent)
            increment = np.expm1(exponent)
            multipliers = np.array([decrement / exponent, gain * increment / exponent,
                                    exponent / decrement, exponent / increment], dtype=wide)
        output = np.asarray(np.column_stack((np.full(4, exponent, dtype=wide),
                            np.full(4, gain, dtype=wide), wide(power) * multipliers,
                            np.sqrt(multipliers))), dtype=float)
    if not np.isfinite(output).all() or np.any(output[:, 1:] <= 0):
        raise ValueError("Positive launch states are not representable as float64")
    return output

import numpy as np

def guiding_center_response_candidates(order_residuals, power_error, width_error):
    residuals = np.asarray(order_residuals, dtype=float)
    power, width = float(power_error), float(width_error)
    if residuals.ndim != 1 or residuals.size == 0 or not np.isfinite(residuals).all() or not np.isfinite([power, width]).all():
        raise ValueError("Require finite nonempty residuals and finite terminal errors")
    scale = float(np.max(np.abs(residuals)))
    order_rms = 0.0 if scale == 0 else scale * np.sqrt(np.mean((residuals / scale)**2))
    with np.errstate(over="ignore", invalid="ignore"):
        terminal = np.hypot(power, width)
        output = np.array([terminal, max(scale, abs(power), abs(width)),
                           order_rms, np.hypot(order_rms, terminal)])
    if not np.isfinite(output).all():
        raise ValueError("Response is not representable as float64")
    return output

import numpy as np

def dispersion_profile_candidates(log_a, c0, z_km):
    logs = np.asarray(log_a, dtype=float)
    distances = np.asarray(z_km, dtype=float)
    coefficient = float(c0)
    if logs.shape != (3,) or distances.ndim != 1 or distances.size == 0:
        raise ValueError("Require three logarithms and nonempty one-dimensional distances")
    if not np.isfinite(logs).all() or not np.isfinite(distances).all() or not np.isfinite(coefficient) or np.any(distances < 0):
        raise ValueError("Require finite inputs and nonnegative distances")
    output = np.ones((4, distances.size))
    active = distances > 0
    if coefficient == 0 or not np.any(active):
        return output
    wide = np.longdouble
    distance = distances[active].astype(wide)
    factor = wide(coefficient)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scaled = np.exp(logs.astype(wide)[:, None] * np.log(wide(10)) + np.log(distance)[None, :])
        first, second, third = scaled
        common_linear = factor * first
        common_quadratic = factor * second**2 / 2
        composite_quadratic = (factor * second)**2 / 2
        common_cubic = factor * third**3 / 6
        composite_cubic = (factor * third)**3 / 6
        output[:, active] = np.asarray(np.vstack((
            1 + common_linear + composite_quadratic + common_cubic,
            1 + common_linear + common_quadratic + common_cubic,
            1 + common_linear + composite_quadratic + composite_cubic,
            1 + common_linear + factor * second * distance / 2 + factor * third * distance**2 / 6,
        )), dtype=float)
    if not np.isfinite(output).all():
        raise ValueError("Profiles are not representable as finite float64 values")
    return output

import numpy as np

def taguchi_protocol_candidates(levels, oa, responses, level_differences, reduction_rate):
    values = np.asarray(levels, dtype=float)
    design = np.asarray(oa)
    response = np.asarray(responses, dtype=float)
    spacing = np.asarray(level_differences, dtype=float)
    rate = float(reduction_rate)
    if values.ndim != 2 or design.ndim != 2:
        raise ValueError("Levels and OA must be two-dimensional")
    factors, count = values.shape
    if factors < 1 or count < 2 or design.shape[0] < 1 or design.shape[1] != factors:
        raise ValueError("Invalid level table or OA shape")
    if response.shape != (design.shape[0],) or spacing.shape != (factors,):
        raise ValueError("Incompatible response or spacing shape")
    if not np.issubdtype(design.dtype, np.integer) or np.any(design < 0) or np.any(design >= count):
        raise ValueError("OA entries must be valid zero-based integer levels")
    if not all(np.isfinite(array).all() for array in (values, response, spacing)):
        raise ValueError("Inputs must be finite")
    if np.any(values[:, 1:] <= values[:, :-1]) or np.any(response < 0) or np.any(spacing <= 0) or not 0 < rate < 1:
        raise ValueError("Require ascending levels, nonnegative responses, positive spacing and 0<rate<1")
    snr = np.full(response.shape, np.inf)
    positive = response > 0
    snr[positive] = -20 * np.log10(response[positive])
    mean_snr = np.empty((factors, count))
    mean_response = np.empty((factors, count))
    median_snr = np.empty((factors, count))
    for factor in range(factors):
        for level in range(count):
            mask = design[:, factor] == level
            if not np.any(mask):
                raise ValueError("Every factor level must occur")
            selected = response[mask]
            scale = float(np.max(selected))
            mean_response[factor, level] = 0 if scale == 0 else scale * np.mean(selected / scale)
            mean_snr[factor, level] = np.mean(snr[mask])
            median_snr[factor, level] = np.median(snr[mask])
    indices = np.vstack((design[np.argmin(response)], np.argmax(mean_snr, axis=1),
                         np.argmin(mean_response, axis=1), np.argmax(median_snr, axis=1)))
    output = np.empty((4, factors, count + 3))
    contracted = spacing * rate
    offsets = np.arange(count) - (count - 1) / 2
    with np.errstate(over="ignore", invalid="ignore"):
        for mode in range(4):
            centers = values[np.arange(factors), indices[mode]]
            next_levels = centers[:, None] + contracted[:, None] * offsets
            output[mode] = np.column_stack((indices[mode], centers, contracted, next_levels))
    if not np.isfinite(output).all() or np.any(contracted <= 0):
        raise ValueError("Update or positive contracted spacing is not representable")
    return output

import numpy as np

def ddf_truncation_selector(log_a, c0, validation_z_km, references, dense_z_km, relative_tolerance=0.03):
    logs = np.asarray(log_a, dtype=float)
    validation_z = np.asarray(validation_z_km, dtype=float)
    refs = np.asarray(references, dtype=float)
    dense_z = np.asarray(dense_z_km, dtype=float)
    coefficient, tolerance = float(c0), float(relative_tolerance)
    if logs.shape != (3,) or validation_z.ndim != 2 or min(validation_z.shape) < 1: raise ValueError("Require three logarithms and nonempty (F,Z) validation grid")
    if refs.ndim != 3 or refs.shape[0] != validation_z.shape[0] or refs.shape[2] != validation_z.shape[1] or refs.shape[1] < 1: raise ValueError("References must have matching shape (F,R,Z)")
    if dense_z.ndim != 1 or dense_z.size == 0: raise ValueError("Require a nonempty one-dimensional dense grid")
    if not all(np.isfinite(values).all() for values in (logs, validation_z, refs, dense_z, [coefficient, tolerance])): raise ValueError("All inputs must be finite")
    if coefficient == 0 or tolerance < 0 or np.any(refs <= 0) or np.any(validation_z < 0) or np.any(dense_z < 0): raise ValueError("Invalid coefficient, tolerance, references or distances")
    wide = np.longdouble
    def profiles(points):
        result = np.ones((3, points.size), dtype=wide)
        active = points.ravel() > 0
        if np.any(active):
            distances = points.ravel()[active].astype(wide)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                exponent = logs.astype(wide)[:, None] * np.log(wide(10))
                exponent = exponent + np.log(abs(wide(coefficient))) + np.log(distances)
                factors = np.sign(coefficient) * np.exp(exponent)
                linear = factors[0]
                quadratic = factors[1] ** 2 / 2
                cubic = factors[2] ** 3 / 6
                result[:, active] = np.stack((1 + linear, 1 + linear + quadratic, 1 + linear + quadratic + cubic))
        if not np.isfinite(result).all(): raise ValueError("Nested profile evaluation is not representable")
        return result.reshape((3,) + points.shape)
    validation = profiles(validation_z)
    minima = np.min(profiles(dense_z), axis=1)
    scores = np.full(3, wide(1e6), dtype=wide)
    for order, profile in enumerate(validation):
        if np.all(profile > 0):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                errors = abs(1 - np.sqrt(refs.astype(wide)) / np.sqrt(profile[:, None, :]))
                scale = np.max(errors, axis=2, keepdims=True)
                scaled = np.divide(errors, scale, out=np.zeros_like(errors), where=scale != 0)
                scores[order] = np.max(scale[..., 0] * np.sqrt(np.mean(scaled ** 2, axis=2)))
    if not np.isfinite(scores).all(): raise ValueError("Nested scores must be finite")
    feasible = minima > 0
    if not np.any(feasible): raise ValueError("No positive nested profile on the dense grid")
    best = np.min(scores[feasible])
    threshold = best * (1 + wide(tolerance)) + wide(1e-15)
    selected = int(np.flatnonzero(feasible & (scores <= threshold))[0])
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(np.concatenate((scores, minima, [selected + 1, scores[selected]])), dtype=float)
    if not np.isfinite(result).all(): raise ValueError("Returned diagnostics must be representable as float64")
    return result

import numpy as np

def cross_example_closure_score(ddf_error, unit_order_response,
                                      theory_order_response, alpha_ddf,
                                      alpha_guiding, p0_relative_errors, tolerances):
    responses = np.asarray([ddf_error, unit_order_response, theory_order_response], dtype=float)
    slopes = np.asarray([alpha_ddf, alpha_guiding], dtype=float)
    errors = np.asarray(p0_relative_errors, dtype=float)
    scales = np.asarray(tolerances, dtype=float)
    if errors.ndim != 1 or errors.size == 0 or scales.shape != (5,):
        raise ValueError("Require nonempty error vector and five tolerances")
    if not all(np.isfinite(array).all() for array in (responses, slopes, errors, scales)):
        raise ValueError("Inputs must be finite")
    if np.any(responses < 0) or np.any(scales <= 0):
        raise ValueError("Responses must be nonnegative and tolerances positive")
    peak_error = np.max(np.abs(errors))
    wide = np.longdouble
    mismatch = abs(wide(slopes[0]) - wide(slopes[1]))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        channels = np.asarray(np.array([*responses, mismatch, peak_error], dtype=wide)
                              / scales.astype(wide), dtype=float)
    output = np.concatenate((responses, slopes, [peak_error], channels, [np.max(channels)]))
    if not np.isfinite(output).all():
        raise ValueError("Closure diagnostics are not representable as float64")
    return output

import numpy as np

def _default_surrogate_tensor() -> np.ndarray:
    b0 = np.array([
        [0.026,-0.021,0.017,-0.014,0.010,-0.008],
        [0.470,0.390,0.540,0.440,0.260,-0.200],
        [0.310,0.370,0.270,0.340,0.320,0.250],
        [0.130,-0.070,0.090,-0.050,-0.100,0.080],
        [0.070,0.060,0.080,0.050,0.040,0.010],
        [-0.030,-0.050,-0.020,-0.040,0.010,0.030],
    ], dtype=float)
    b1 = np.array([
        [0.034,-0.028,0.023,-0.017,0.014,-0.011],
        [0.560,0.460,0.630,0.500,0.310,-0.240],
        [0.360,0.440,0.320,0.400,0.380,0.300],
        [0.100,-0.090,0.120,-0.070,-0.120,0.100],
        [0.090,0.040,0.060,0.070,0.060,-0.010],
        [-0.050,-0.070,-0.040,-0.060,-0.010,0.050],
    ], dtype=float)
    return np.stack((b0, b1))

def cross_example_taguchi_gap(reduction_candidates: np.ndarray = None,
                                      surrogate_tensor: np.ndarray = None,
                                      closure_tolerances: np.ndarray = None,
                                      ddf_iterations: int = 4,
                                      guiding_iterations: int = 5) -> float:
    if reduction_candidates is None:
        reduction_candidates = np.array([0.50, 0.65, 0.75], dtype=float)
    if surrogate_tensor is None:
        surrogate_tensor = _default_surrogate_tensor()
    if closure_tolerances is None:
        closure_tolerances = np.array([0.05, 0.24, 0.06, 0.002, 0.10], dtype=float)
    rates = np.asarray(reduction_candidates, dtype=float)
    tensor = np.asarray(surrogate_tensor, dtype=float)
    tau = np.asarray(closure_tolerances, dtype=float)
    if rates.shape != (3,) or tensor.shape != (2,6,6) or tau.shape != (5,):
        raise ValueError("Require three rates, a 2x6x6 archive, and five tolerances")
    if not np.isfinite(rates).all() or not np.isfinite(tensor).all() or not np.isfinite(tau).all():
        raise ValueError("All task inputs must be finite")
    if np.any(rates <= 0.0) or np.any(rates >= 1.0) or np.unique(rates).size != 3 or np.any(tau <= 0.0):
        raise ValueError("Rates must be distinct in (0,1) and tolerances positive")
    ni = int(ddf_iterations)
    ng = int(guiding_iterations)
    if ni != ddf_iterations or ng != guiding_iterations or ni < 1 or ng < 1:
        raise ValueError("Iteration counts must be positive integers")

    # Candidate rows are deliberately numerical: source selection fixes these indices.
    profile_mode = 2
    response_mode = 3
    protocol_mode = 1
    theory_mode = 2

    oa4 = prime_strength2_oa(3, 4)
    ai0 = taguchi_initial_levels(-1.9, -0.8, 3)
    c00 = taguchi_initial_levels(-1.25, -0.75, 3)
    initial_ddf = np.vstack((ai0[:3], ai0[:3], ai0[:3], c00[:3]))
    initial_ddf_spacing = np.array([ai0[3], ai0[3], ai0[3], c00[3]], dtype=float)
    train_z = np.array([0.75,2.25,3.75,5.25,6.75,8.25,9.75], dtype=float)
    folds = np.array([[1.5,4.5,7.5],[3.0,6.0,9.0]], dtype=float)
    dense_z = np.linspace(0.0, 10.0, 201)
    alpha_refs = np.array([0.244,0.281], dtype=float) * np.log(10.0) / 10.0
    references = np.exp(-alpha_refs[None,:,None] * folds[:,None,:])

    p0_mw = sech_soliton_peak_power_mw(-7650.0, 1.3, 50.0)
    spans = np.array([0.17,0.24], dtype=float) * 105.5
    theory_states = np.stack([
        guiding_center_theory_candidates(0.2611, float(span), p0_mw)[theory_mode]
        for span in spans
    ])
    oa2 = prime_strength2_oa(3, 2)
    start = taguchi_initial_levels(1.0, 10.0, 3)

    records = []
    for rate in rates:
        current = initial_ddf.copy()
        spacings = initial_ddf_spacing.copy()
        ddf_center = np.zeros(4, dtype=float)
        for _ in range(ni):
            responses = np.empty(oa4.shape[0], dtype=float)
            for row_index, row in enumerate(oa4):
                settings = current[np.arange(4), row]
                q = dispersion_profile_candidates(settings[:3], settings[3], train_z)[profile_mode]
                if np.any(q <= 0.0):
                    responses[row_index] = 1.0e6
                else:
                    response_by_loss = [
                        np.sqrt(np.mean((1.0 - np.sqrt(np.exp(-alpha*train_z)/q))**2))
                        for alpha in alpha_refs
                    ]
                    responses[row_index] = float(max(response_by_loss))
            update = taguchi_protocol_candidates(current, oa4, responses, spacings, float(rate))[protocol_mode]
            ddf_center = update[:,1].copy()
            spacings = update[:,2].copy()
            current = update[:,3:].copy()

        order_diag = ddf_truncation_selector(
            ddf_center[:3], ddf_center[3], folds, references, dense_z, 0.03
        )
        selected_order = int(order_diag[6])
        terms = ddf_center[3] * np.power(10.0, ddf_center[:3])
        validation_profiles = []
        for fold in folds:
            first = terms[0] * fold
            second = (terms[1] * fold)**2 / 2.0
            third = (terms[2] * fold)**3 / 6.0
            pieces = (first, second, third)
            validation_profiles.append(1.0 + sum(pieces[:selected_order]))
        q_holdout = np.stack(validation_profiles)
        if np.any(q_holdout <= 0.0):
            raise ValueError("Selected DDF profile is nonpositive on a validation fold")
        flat_z = folds.ravel()
        alpha_ddf = float(-np.dot(flat_z, np.log(q_holdout.ravel())) / np.dot(flat_z, flat_z))

        branch_centers = np.empty((2, 2, 2), dtype=float)
        branch_responses = np.empty((2, 2), dtype=float)
        for branch in range(2):
            for span_index in range(2):
                current_gc = np.vstack((start[:3], start[:3]))
                gc_spacings = np.array([start[3], start[3]], dtype=float)
                center = np.zeros(2, dtype=float)
                state = theory_states[span_index]
                for _ in range(ng):
                    responses = np.empty(oa2.shape[0], dtype=float)
                    for row_index, row in enumerate(oa2):
                        gain, p_fac = current_gc[np.arange(2), row]
                        x = gain / state[1] - 1.0
                        y = p_fac / (state[2] / p0_mw) - 1.0
                        basis = np.array([1.0,x,y,x*y,x*x,y*y], dtype=float)
                        residual = basis @ tensor[span_index]
                        order_residual = residual[:4].copy()
                        if branch == 0:
                            order_residual += state[3] - 1.0
                        responses[row_index] = guiding_center_response_candidates(
                            order_residual, residual[4], residual[5]
                        )[response_mode]
                    update = taguchi_protocol_candidates(
                        current_gc, oa2, responses, gc_spacings, float(rate)
                    )[protocol_mode]
                    center = update[:,1].copy()
                    gc_spacings = update[:,2].copy()
                    current_gc = update[:,3:].copy()
                branch_centers[branch, span_index] = center
                x = center[0] / state[1] - 1.0
                y = center[1] / (state[2] / p0_mw) - 1.0
                basis = np.array([1.0,x,y,x*y,x*x,y*y], dtype=float)
                residual = basis @ tensor[span_index]
                order_residual = residual[:4].copy()
                if branch == 0:
                    order_residual += state[3] - 1.0
                branch_responses[branch, span_index] = guiding_center_response_candidates(
                    order_residual, residual[4], residual[5]
                )[response_mode]

        theory_centers = branch_centers[1]
        if not np.isfinite(theory_centers).all() or np.any(theory_centers <= 0):
            raise ValueError("Final theoretical gain and launch factor must be finite and positive")
        alpha_guiding = float(np.dot(spans, np.log(theory_centers[:,0])) / np.dot(spans, spans))
        p0_relative_errors = np.empty(2, dtype=float)
        for span_index, (gain, p_fac) in enumerate(theory_centers):
            x = float(np.log(gain))
            inverse_multiplier = 1.0 if x == 0.0 else float(np.expm1(x) / (np.exp(x) * x))
            p0_relative_errors[span_index] = p_fac * inverse_multiplier - 1.0
        closure = cross_example_closure_score(
            float(order_diag[7]), float(np.max(branch_responses[0])),
            float(np.max(branch_responses[1])), alpha_ddf, alpha_guiding,
            p0_relative_errors, tau
        )
        records.append({
            "rate": float(rate),
            "order": selected_order,
            "ddf_response": float(order_diag[7]),
            "alpha_ddf": alpha_ddf,
            "alpha_guiding": alpha_guiding,
            "unit_response": float(np.max(branch_responses[0])),
            "theory_response": float(np.max(branch_responses[1])),
            "p0_error": float(np.max(np.abs(p0_relative_errors))),
            "loss_mismatch": float(abs(alpha_ddf-alpha_guiding)),
            "active_channel": int(np.argmax(closure[6:11])),
            "joint_score": float(closure[11]),
        })

    ranked = sorted(records, key=lambda item: (item["joint_score"], item["rate"]))
    value = float(ranked[1]["joint_score"] - ranked[0]["joint_score"])
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Joint-score gap must be finite and nonnegative")
    return value

def _archive_tensor():
    return _default_surrogate_tensor()
SCICODE_GOLD_EOF
