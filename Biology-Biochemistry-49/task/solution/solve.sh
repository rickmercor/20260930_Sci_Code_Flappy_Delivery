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

def pcv_progress_geometry(
    points: "np.ndarray",
    strings: "np.ndarray",
    alpha: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    points = np.asarray(points, dtype=np.float64)
    strings = np.asarray(strings, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("points must have shape (n,d) with n,d > 0")
    if strings.ndim != 3 or strings.shape[0] == 0 or strings.shape[1] < 2:
        raise ValueError("strings must have shape (w,m,d) with w > 0 and m >= 2")
    if strings.shape[2] != points.shape[1]:
        raise ValueError("points and strings must have the same coordinate dimension")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(strings)):
        raise ValueError("points and strings must be finite")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    difference = strings[:, None, :, :] - points[None, :, None, :]
    squared = np.sum(difference * difference, axis=-1)
    log_kernel = -float(alpha) * squared
    shift = np.max(log_kernel, axis=2, keepdims=True)
    kernel = np.exp(log_kernel - shift)
    rho = kernel / np.sum(kernel, axis=2, keepdims=True)
    fractions = np.linspace(0.0, 1.0, strings.shape[1], dtype=np.float64)
    progress = np.sum(rho * fractions[None, None, :], axis=2)
    jacobian = (
        -2.0 * float(alpha) * rho[:, :, :, None]
        * (fractions[None, None, :, None] - progress[:, :, None, None])
        * difference
    )
    if not np.all(np.isfinite(progress)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("progress outputs must be finite")
    return progress, jacobian

import numpy as np

def pcv_orthogonal_geometry(
    points: "np.ndarray",
    strings: "np.ndarray",
    alpha: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    points = np.asarray(points, dtype=np.float64)
    strings = np.asarray(strings, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("points must have shape (n,d) with n,d > 0")
    if strings.ndim != 3 or strings.shape[0] == 0 or strings.shape[1] < 2:
        raise ValueError("strings must have shape (w,m,d) with w > 0 and m >= 2")
    if strings.shape[2] != points.shape[1]:
        raise ValueError("points and strings must have the same coordinate dimension")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(strings)):
        raise ValueError("points and strings must be finite")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    difference = strings[:, None, :, :] - points[None, :, None, :]
    squared = np.sum(difference * difference, axis=-1)
    log_kernel = -float(alpha) * squared
    shift = np.max(log_kernel, axis=2, keepdims=True)
    exp_shifted = np.exp(log_kernel - shift)
    sum_exp = np.sum(exp_shifted, axis=2, keepdims=True)
    rho = exp_shifted / sum_exp
    orthogonal = -(shift[:, :, 0] + np.log(sum_exp[:, :, 0])) / float(alpha)
    jacobian = 2.0 * rho[:, :, :, None] * difference
    if not np.all(np.isfinite(orthogonal)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("orthogonal outputs must be finite")
    return orthogonal, jacobian

import numpy as np

def pathway_bias_geometry(
    progress_values: "np.ndarray",
    progress_jacobian: "np.ndarray",
    orthogonal_values: "np.ndarray",
    orthogonal_jacobian: "np.ndarray",
    linear_coefficients: "np.ndarray",
    curvature_coefficients: "np.ndarray",
    orthogonal_scales: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    progress_values = np.asarray(progress_values, dtype=np.float64)
    progress_jacobian = np.asarray(progress_jacobian, dtype=np.float64)
    orthogonal_values = np.asarray(orthogonal_values, dtype=np.float64)
    orthogonal_jacobian = np.asarray(orthogonal_jacobian, dtype=np.float64)
    linear_coefficients = np.asarray(linear_coefficients, dtype=np.float64)
    curvature_coefficients = np.asarray(curvature_coefficients, dtype=np.float64)
    orthogonal_scales = np.asarray(orthogonal_scales, dtype=np.float64)
    if progress_values.ndim != 2 or progress_values.shape[0] == 0 or progress_values.shape[1] == 0:
        raise ValueError("progress_values must have shape (w,n) with w,n > 0")
    w, n = progress_values.shape
    if orthogonal_values.shape != (w, n):
        raise ValueError("orthogonal_values must have shape (w,n)")
    if progress_jacobian.ndim != 4 or progress_jacobian.shape[:2] != (w, n):
        raise ValueError("progress_jacobian must have shape (w,n,m,d)")
    if progress_jacobian.shape[2] < 2 or progress_jacobian.shape[3] < 1:
        raise ValueError("the Jacobian requires at least two images and one dimension")
    if orthogonal_jacobian.shape != progress_jacobian.shape:
        raise ValueError("coordinate Jacobians must have matching shapes")
    for values in (linear_coefficients, curvature_coefficients, orthogonal_scales):
        if values.shape != (w,):
            raise ValueError("coefficient vectors must have shape (w,)")
    arrays = (
        progress_values, progress_jacobian, orthogonal_values,
        orthogonal_jacobian, linear_coefficients, curvature_coefficients,
        orthogonal_scales,
    )
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("all bias inputs must be finite")
    if np.any(progress_values < 0.0) or np.any(progress_values > 1.0):
        raise ValueError("progress values must lie in [0,1]")
    if np.any(orthogonal_scales < 0.0):
        raise ValueError("orthogonal scales must be nonnegative")
    profile = (
        linear_coefficients[:, None] * progress_values
        + curvature_coefficients[:, None] * progress_values * (1.0 - progress_values)
    )
    bias_values = -profile + orthogonal_scales[:, None] * orthogonal_values * orthogonal_values
    profile_derivative = (
        linear_coefficients[:, None]
        + curvature_coefficients[:, None] * (1.0 - 2.0 * progress_values)
    )
    bias_jacobian = (
        -profile_derivative[:, :, None, None] * progress_jacobian
        + 2.0 * orthogonal_scales[:, None, None, None]
        * orthogonal_values[:, :, None, None] * orthogonal_jacobian
    )
    if not np.all(np.isfinite(bias_values)) or not np.all(np.isfinite(bias_jacobian)):
        raise ValueError("bias outputs must be finite")
    return bias_values, bias_jacobian

import numpy as np

def binless_reweighting_factors(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
) -> "np.ndarray":
    bias_values = np.asarray(bias_values, dtype=np.float64)
    sample_counts = np.asarray(sample_counts, dtype=np.float64)
    free_energy_offsets = np.asarray(free_energy_offsets, dtype=np.float64)
    if bias_values.ndim != 2 or bias_values.shape[0] == 0 or bias_values.shape[1] == 0:
        raise ValueError("bias_values must have shape (w, n) with w,n > 0")
    expected = (bias_values.shape[0],)
    if sample_counts.shape != expected or free_energy_offsets.shape != expected:
        raise ValueError("counts and offsets must have shape (w,)")
    if not np.all(np.isfinite(bias_values)) or not np.all(np.isfinite(sample_counts)):
        raise ValueError("bias values and counts must be finite")
    if not np.all(np.isfinite(free_energy_offsets)):
        raise ValueError("offsets must be finite")
    if np.any(sample_counts <= 0.0):
        raise ValueError("sample counts must be positive")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    log_terms = (
        np.log(sample_counts)[:, None]
        - beta * (bias_values - free_energy_offsets[:, None])
    )
    maximum = np.max(log_terms, axis=0)
    log_denominator = maximum + np.log(
        np.sum(np.exp(log_terms - maximum[None, :]), axis=0)
    )
    result = np.exp(-log_denominator)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("reweighting factors are not finite and positive")
    return result

import numpy as np

def wham_offset_update(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
    reference_index: int,
) -> "np.ndarray":
    bias_values = np.asarray(bias_values, dtype=np.float64)
    if bias_values.ndim != 2 or bias_values.shape[0] == 0 or bias_values.shape[1] == 0:
        raise ValueError("bias_values must have shape (w, n) with w,n > 0")
    if isinstance(reference_index, (bool, np.bool_)) or not isinstance(
        reference_index, (int, np.integer)
    ):
        raise ValueError("reference_index must be an integer")
    reference_index = int(reference_index)
    if reference_index < 0 or reference_index >= bias_values.shape[0]:
        raise ValueError("reference_index is out of range")
    factors = binless_reweighting_factors(
        bias_values, sample_counts, beta, free_energy_offsets
    )
    log_terms = -float(beta) * bias_values + np.log(factors)[None, :]
    maximum = np.max(log_terms, axis=1)
    log_sums = maximum + np.log(
        np.sum(np.exp(log_terms - maximum[:, None]), axis=1)
    )
    updated = -(log_sums - log_sums[reference_index]) / float(beta)
    updated[reference_index] = 0.0
    return updated

import numpy as np

def solve_wham_offsets(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    tolerance: float,
    max_iterations: int,
    reference_index: int,
) -> tuple["np.ndarray", int]:
    bias_values = np.asarray(bias_values, dtype=np.float64)
    if bias_values.ndim != 2 or bias_values.shape[0] == 0 or bias_values.shape[1] == 0:
        raise ValueError("bias_values must have shape (w, n) with w,n > 0")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ) or int(max_iterations) <= 0:
        raise ValueError("max_iterations must be a positive integer")
    current = np.zeros(bias_values.shape[0], dtype=np.float64)
    for iteration in range(1, int(max_iterations) + 1):
        updated = wham_offset_update(
            bias_values,
            sample_counts,
            beta,
            current,
            reference_index,
        )
        if np.max(np.abs(updated - current)) <= tolerance:
            return updated, iteration
        current = updated
    raise RuntimeError("WHAM offsets did not converge")

import numpy as np

def wham_geometry_gradient(
    bias_values: "np.ndarray",
    bias_jacobian: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
    committor_traces: "np.ndarray",
    lag_steps: "np.ndarray",
    reference_index: int,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    bias_values = np.asarray(bias_values, dtype=np.float64)
    bias_jacobian = np.asarray(bias_jacobian, dtype=np.float64)
    sample_counts = np.asarray(sample_counts, dtype=np.float64)
    free_energy_offsets = np.asarray(free_energy_offsets, dtype=np.float64)
    committor_traces = np.asarray(committor_traces, dtype=np.float64)
    lag_steps = np.asarray(lag_steps)
    if bias_values.ndim != 2 or bias_values.shape[0] == 0 or bias_values.shape[1] == 0:
        raise ValueError("bias_values must have shape (w,n) with w,n > 0")
    w, n = bias_values.shape
    if bias_jacobian.ndim != 4 or bias_jacobian.shape[:2] != (w, n):
        raise ValueError("bias_jacobian must have shape (w,n,m,d)")
    m, d = bias_jacobian.shape[2:]
    if m < 2 or d < 1:
        raise ValueError("bias_jacobian requires m >= 2 and d >= 1")
    if sample_counts.shape != (w,) or free_energy_offsets.shape != (w,):
        raise ValueError("counts and offsets must have shape (w,)")
    arrays = (bias_values, bias_jacobian, sample_counts, free_energy_offsets)
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("WHAM inputs must be finite")
    if np.any(sample_counts <= 0.0) or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("counts and beta must be finite and positive")
    if isinstance(reference_index, (bool, np.bool_)) or not isinstance(reference_index, (int, np.integer)):
        raise ValueError("reference_index must be an integer")
    reference_index = int(reference_index)
    if reference_index < 0 or reference_index >= w:
        raise ValueError("reference_index is out of range")
    if not np.isclose(free_energy_offsets[reference_index], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("the reference offset must be zero")
    if committor_traces.ndim != 2 or committor_traces.shape[0] != n or committor_traces.shape[1] < 2:
        raise ValueError("committor_traces must have shape (n,t) with t >= 2")
    if not np.all(np.isfinite(committor_traces)) or np.any(committor_traces < 0.0) or np.any(committor_traces > 1.0):
        raise ValueError("committor traces must be finite and lie in [0,1]")
    if lag_steps.ndim != 1 or lag_steps.size == 0 or not np.issubdtype(lag_steps.dtype, np.integer):
        raise ValueError("lag_steps must be a nonempty one-dimensional integer-dtype array")
    lag_steps = lag_steps.astype(int, copy=False)
    if np.any(lag_steps <= 0) or np.any(lag_steps >= committor_traces.shape[1]):
        raise ValueError("each lag must satisfy 1 <= lag < trace length")

    parameter_count = w * m * d
    bias_gradient = np.zeros((w, n, parameter_count), dtype=np.float64)
    for pathway in range(w):
        start = pathway * m * d
        stop = start + m * d
        bias_gradient[pathway, :, start:stop] = bias_jacobian[pathway].reshape(n, m * d)

    factors = binless_reweighting_factors(
        bias_values, sample_counts, float(beta), free_energy_offsets
    )
    log_mixture = np.log(sample_counts)[:, None] - float(beta) * (
        bias_values - free_energy_offsets[:, None]
    )
    mixture_shift = np.max(log_mixture, axis=0, keepdims=True)
    mixture_probabilities = np.exp(log_mixture - mixture_shift)
    mixture_probabilities /= np.sum(mixture_probabilities, axis=0, keepdims=True)

    log_path = np.log(factors)[None, :] - float(beta) * bias_values
    path_shift = np.max(log_path, axis=1, keepdims=True)
    path_probabilities = np.exp(log_path - path_shift)
    path_probabilities /= np.sum(path_probabilities, axis=1, keepdims=True)

    active = np.array([i for i in range(w) if i != reference_index], dtype=int)
    offset_gradient = np.zeros((w, parameter_count), dtype=np.float64)
    if active.size:
        jacobian = np.empty((active.size, active.size), dtype=np.float64)
        for row, i in enumerate(active):
            for column, k in enumerate(active):
                jacobian[row, column] = (
                    (1.0 if i == k else 0.0)
                    - np.sum(
                        (path_probabilities[i] - path_probabilities[reference_index])
                        * mixture_probabilities[k]
                    )
                )
        mean_bias_gradient = np.einsum(
            "jn,jnp->np", mixture_probabilities, bias_gradient
        )
        direct_response = (
            np.einsum("in,np->ip", path_probabilities, mean_bias_gradient)
            - np.einsum("in,inp->ip", path_probabilities, bias_gradient)
        )
        right_hand_side = -(
            direct_response[active] - direct_response[reference_index]
        )
        try:
            offset_gradient[active] = np.linalg.solve(jacobian, right_hand_side)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the gauge-fixed WHAM Jacobian is singular") from exc

    log_factor_gradient = float(beta) * np.einsum(
        "in,inp->np",
        mixture_probabilities,
        bias_gradient - offset_gradient[:, None, :],
    )
    conditional = np.empty((n, lag_steps.size), dtype=np.float64)
    for column, lag in enumerate(lag_steps):
        differences = committor_traces[:, lag:] - committor_traces[:, :-lag]
        conditional[:, column] = np.mean(differences * differences, axis=1)
    normalized_weights = factors / np.sum(factors)
    correlations = np.einsum("n,nl->l", normalized_weights, conditional)
    correlation_gradient = np.einsum(
        "n,nl,np->lp",
        normalized_weights,
        conditional - correlations[None, :],
        log_factor_gradient,
    )
    offset_gradient = offset_gradient.reshape(w, w, m, d)
    correlation_gradient = correlation_gradient.reshape(lag_steps.size, w, m, d)
    if not np.all(np.isfinite(offset_gradient)) or not np.all(np.isfinite(correlations)) or not np.all(np.isfinite(correlation_gradient)):
        raise ValueError("geometry-gradient outputs must be finite")
    return offset_gradient, correlations, correlation_gradient

import numpy as np

def binless_wham_geometry_sensitivity(
    points: "np.ndarray",
    strings: "np.ndarray",
    committor_traces: "np.ndarray",
    alpha: float,
    linear_coefficients: "np.ndarray",
    curvature_coefficients: "np.ndarray",
    orthogonal_scales: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    lag_steps: "np.ndarray",
    tolerance: float,
    max_iterations: int,
    target_lag_step: int,
) -> float:
    strings_array = np.asarray(strings)
    if strings_array.ndim != 3 or strings_array.shape[1] < 3:
        raise ValueError("strings must contain at least three images")
    if isinstance(target_lag_step, (bool, np.bool_)) or not isinstance(target_lag_step, (int, np.integer)):
        raise ValueError("target_lag_step must be an integer")
    progress, progress_jacobian = pcv_progress_geometry(
        points, strings, alpha
    )
    orthogonal, orthogonal_jacobian = pcv_orthogonal_geometry(
        points, strings, alpha
    )
    bias_values, bias_jacobian = pathway_bias_geometry(
        progress,
        progress_jacobian,
        orthogonal,
        orthogonal_jacobian,
        linear_coefficients,
        curvature_coefficients,
        orthogonal_scales,
    )
    offsets, _ = solve_wham_offsets(
        bias_values,
        sample_counts,
        beta,
        tolerance,
        max_iterations,
        0,
    )
    _, _, correlation_gradient = wham_geometry_gradient(
        bias_values,
        bias_jacobian,
        sample_counts,
        beta,
        offsets,
        committor_traces,
        lag_steps,
        0,
    )
    lag_array = np.asarray(lag_steps)
    matches = np.flatnonzero(lag_array == int(target_lag_step))
    if matches.size != 1:
        raise ValueError("target_lag_step must occur exactly once")
    interior_gradient = correlation_gradient[int(matches[0]), :, 1:-1, :]
    result = float(np.linalg.norm(interior_gradient))
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("geometry sensitivity must be finite and nonnegative")
    return result
SCICODE_GOLD_EOF
