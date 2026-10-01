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


def _fixed_factor_array(value, shape, name):
    """Return a finite nonnegative float array of the given shape."""
    import numpy as np

    try:
        array = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not (np.all(np.isfinite(array)) and np.all(array >= 0.0)):
        raise ValueError(f"{name} must hold finite nonnegative values")
    return array


def _fixed_pair_channel(five, three):
    """Return 1 for neutral, 2 for Watson–Crick and 3 for G–U."""
    if (five, three) in ((0, 3), (3, 0), (2, 1), (1, 2)):
        return 2
    if (five, three) in ((2, 3), (3, 2)):
        return 3
    return 1


def compute_stacked_inside_weights(
    sequence: "np.ndarray",
    rule_weights: "np.ndarray",
    left_emission: "np.ndarray",
    right_emission: "np.ndarray",
    pair_emission: "np.ndarray",
    stacking_factors: "np.ndarray",
    min_loop: int,
) -> "np.ndarray":
    """Reference class-resolved inside recursion in increasing span length."""
    import numpy as np

    raw = np.asarray(sequence)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype == bool:
        raise ValueError("sequence must be a nonempty one-dimensional code array")
    try:
        codes = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("sequence must hold integer codes") from None
    if not np.all(np.isfinite(codes)) or np.any(codes != np.round(codes)):
        raise ValueError("sequence must hold integer codes")
    if np.any(codes < 0) or np.any(codes > 3):
        raise ValueError("sequence codes must lie in 0..3")
    x = codes.astype(int)
    t_p, t_l, t_r, t_b, t_e = _fixed_factor_array(rule_weights, (5,), "rule_weights")
    e_l = _fixed_factor_array(left_emission, (4,), "left_emission")
    e_r = _fixed_factor_array(right_emission, (4,), "right_emission")
    e_p = _fixed_factor_array(pair_emission, (4, 4), "pair_emission")
    stack = _fixed_factor_array(stacking_factors, (4,), "stacking_factors")
    if isinstance(min_loop, bool) or not isinstance(min_loop, (int, np.integer)) or min_loop < 0:
        raise ValueError("min_loop must be a nonnegative integer")

    n = x.size
    weights = np.zeros((4, n + 2, n + 1), dtype=float)
    for i in range(1, n + 2):
        weights[0, i, i - 1] = t_e

    for span in range(1, n + 1):
        for i in range(1, n - span + 2):
            j = i + span - 1
            total = t_l * e_l[x[i - 1]] * weights[0, i + 1, j]
            total += t_r * e_r[x[j - 1]] * weights[0, i, j - 1]
            if span > 1:
                total += t_b * float(np.dot(weights[0, i, i:j], weights[0, i + 1:j + 1, j]))

            if j - i - 1 >= min_loop:
                outer = _fixed_pair_channel(int(x[i - 1]), int(x[j - 1]))
                inner_neutral = weights[1, i + 1, j - 1]
                inner_wc = weights[2, i + 1, j - 1]
                inner_gu = weights[3, i + 1, j - 1]
                nonpair_inner = weights[0, i + 1, j - 1] - inner_neutral - inner_wc - inner_gu
                if outer == 1:
                    paired_inner = inner_neutral + inner_wc + inner_gu
                elif outer == 2:
                    paired_inner = inner_neutral + stack[0] * inner_wc + stack[1] * inner_gu
                else:
                    paired_inner = inner_neutral + stack[2] * inner_wc + stack[3] * inner_gu
                weights[outer, i, j] = (
                    t_p * e_p[x[i - 1], x[j - 1]] * (nonpair_inner + paired_inner)
                )
            weights[0, i, j] = total + float(np.sum(weights[1:, i, j]))
    return weights

import numpy as np


def _grouped_validated_codes(sequence):
    """Return validated integer nucleotide codes."""
    import numpy as np

    raw = np.asarray(sequence)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype == bool:
        raise ValueError("sequence must be a nonempty one-dimensional code array")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("sequence must hold integer codes") from None
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.round(numeric)):
        raise ValueError("sequence must hold integer codes")
    if np.any(numeric < 0) or np.any(numeric > 3):
        raise ValueError("sequence codes must lie in 0..3")
    return numeric.astype(int)


def _grouped_validated_kernel(mutation_kernel):
    """Return a valid conditional replacement kernel."""
    import numpy as np

    try:
        kernel = np.array(mutation_kernel, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("mutation_kernel must be numeric") from None
    if kernel.shape != (4, 4):
        raise ValueError("mutation_kernel must have shape (4, 4)")
    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
        raise ValueError("mutation_kernel must be finite and nonnegative")
    if not np.allclose(np.diag(kernel), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("mutation_kernel must have a zero diagonal")
    if not np.allclose(np.sum(kernel, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("mutation_kernel rows must sum to one")
    return kernel


def _grouped_validated_labels(position_groups, length):
    """Return contiguous integer group labels for all positions."""
    import numpy as np

    raw = np.asarray(position_groups)
    if raw.ndim != 1 or raw.shape != (length,) or raw.dtype == bool:
        raise ValueError("position_groups must be a one-dimensional length-L integer array")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("position_groups must hold integer labels") from None
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.round(numeric)):
        raise ValueError("position_groups must hold integer labels")
    groups = numeric.astype(int)
    if np.any(groups < 0):
        raise ValueError("position_groups must be nonnegative")
    unique = np.unique(groups)
    if unique.size == 0 or unique.size > 4 or not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("group labels must be contiguous 0..G-1 with 1 <= G <= 4")
    return groups


def build_mutation_profile_polynomials(
    sequence: "np.ndarray",
    mutation_kernel: "np.ndarray",
    position_groups: "np.ndarray",
) -> "np.ndarray":
    """Reference construction of grouped affine site probabilities."""
    import numpy as np

    codes = _grouped_validated_codes(sequence)
    kernel = _grouped_validated_kernel(mutation_kernel)
    groups = _grouped_validated_labels(position_groups, codes.size)
    group_count = int(np.max(groups)) + 1
    profile = np.zeros((codes.size, 4) + (2,) * group_count, dtype=float)
    zero = (0,) * group_count
    for i, code in enumerate(codes):
        linear = [0] * group_count
        linear[int(groups[i])] = 1
        profile[(i, int(code)) + zero] = 1.0
        profile[(i, slice(None)) + tuple(linear)] = kernel[int(code)]
        profile[(i, int(code)) + tuple(linear)] -= 1.0
    return profile

import numpy as np
from scipy.signal import convolve as _scipy_nd_convolve


def _multivar_factor_array(value, shape, name):
    """Return a finite nonnegative float array of the requested shape."""
    import numpy as np

    try:
        array = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must hold finite nonnegative values")
    return array


def _multivar_stack_table(stacking_factors):
    """Expand four ordered WC/GU factors to endpoint identities."""
    import numpy as np

    kind = np.full((4, 4), -1, dtype=int)
    for a, b in ((0, 3), (3, 0), (2, 1), (1, 2)):
        kind[a, b] = 0
    for a, b in ((2, 3), (3, 2)):
        kind[a, b] = 1
    table = np.ones((4, 4, 4, 4), dtype=float)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    if kind[a, b] >= 0 and kind[c, d] >= 0:
                        table[a, b, c, d] = stacking_factors[2 * kind[a, b] + kind[c, d]]
    return table


def _multivar_profile_and_groups(profile_polynomials, position_groups):
    """Validate a grouped site-probability tensor and return it with labels."""
    import numpy as np

    try:
        profile = np.array(profile_polynomials, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("profile_polynomials must be numeric") from None
    if profile.ndim < 3 or profile.ndim > 6 or profile.shape[0] == 0 or profile.shape[1] != 4:
        raise ValueError("profile_polynomials must have shape (L, 4) + (2,) * G")
    group_count = profile.ndim - 2
    if group_count < 1 or group_count > 4 or profile.shape[2:] != (2,) * group_count:
        raise ValueError("profile coefficient axes must all have length two with 1 <= G <= 4")
    if not np.all(np.isfinite(profile)):
        raise ValueError("profile_polynomials must be finite")

    raw = np.asarray(position_groups)
    if raw.ndim != 1 or raw.shape != (profile.shape[0],) or raw.dtype == bool:
        raise ValueError("position_groups must be a one-dimensional length-L integer array")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("position_groups must hold integer labels") from None
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.round(numeric)):
        raise ValueError("position_groups must hold integer labels")
    groups = numeric.astype(int)
    if np.any(groups < 0) or not np.array_equal(np.unique(groups), np.arange(group_count)):
        raise ValueError("position_groups must use every contiguous label 0..G-1")

    zero = (0,) * group_count
    for i, group in enumerate(groups):
        linear = [0] * group_count
        linear[int(group)] = 1
        allowed = np.zeros((2,) * group_count, dtype=bool)
        allowed[zero] = True
        allowed[tuple(linear)] = True
        if np.any(np.abs(profile[i][:, ~allowed]) > 1e-12):
            raise ValueError("each site may depend only on its assigned group variable")
        constant = profile[(i, slice(None)) + zero]
        slope = profile[(i, slice(None)) + tuple(linear)]
        if np.any(constant < -1e-12) or np.any(constant + slope < -1e-12):
            raise ValueError("site probabilities must be nonnegative at both active endpoints")
        if not np.isclose(np.sum(constant), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("site probabilities at the zero endpoint must sum to one")
        if not np.isclose(np.sum(slope), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("site probability polynomials must sum identically to one")
    return profile, groups


def _trim_multivar(poly, counts):
    """Return the coefficient box supported by a position-count vector."""
    return poly[tuple(slice(0, int(count) + 1) for count in counts)]


def _accumulate_multivar_product(destination, factors, count_vectors, scale=1.0):
    """Accumulate a bounded direct multidimensional convolution."""
    import numpy as np

    dimension = destination.ndim
    product = np.ones((1,) * dimension, dtype=float)
    total_counts = np.zeros(dimension, dtype=int)
    for factor, counts in zip(factors, count_vectors):
        counts = np.asarray(counts, dtype=int)
        product = _scipy_nd_convolve(
            product, _trim_multivar(factor, counts), mode="full", method="direct"
        )
        total_counts += counts
    destination[tuple(slice(0, int(count) + 1) for count in total_counts)] += float(scale) * product


def compute_profile_partition_polynomial(
    profile_polynomials: "np.ndarray",
    position_groups: "np.ndarray",
    rule_weights: "np.ndarray",
    left_emission: "np.ndarray",
    right_emission: "np.ndarray",
    pair_emission: "np.ndarray",
    stacking_factors: "np.ndarray",
    min_loop: int,
) -> "np.ndarray":
    """Reference endpoint-conditioned multivariate inside recursion."""
    import numpy as np

    profile, groups = _multivar_profile_and_groups(profile_polynomials, position_groups)
    n = profile.shape[0]
    group_count = profile.ndim - 2
    group_sizes = np.bincount(groups, minlength=group_count).astype(int)
    coefficient_shape = tuple(int(size) + 1 for size in group_sizes)
    t_p, t_l, t_r, t_b, t_e = _multivar_factor_array(rule_weights, (5,), "rule_weights")
    e_l = _multivar_factor_array(left_emission, (4,), "left_emission")
    e_r = _multivar_factor_array(right_emission, (4,), "right_emission")
    e_p = _multivar_factor_array(pair_emission, (4, 4), "pair_emission")
    stack = _multivar_stack_table(
        _multivar_factor_array(stacking_factors, (4,), "stacking_factors")
    )
    if isinstance(min_loop, bool) or not isinstance(min_loop, (int, np.integer)) or min_loop < 0:
        raise ValueError("min_loop must be a nonnegative integer")

    prefix = np.zeros((n + 1, group_count), dtype=int)
    site_counts = np.zeros((n, group_count), dtype=int)
    for position, group in enumerate(groups):
        site_counts[position, int(group)] = 1
        prefix[position + 1] = prefix[position] + site_counts[position]

    total = np.zeros((n + 2, n + 1) + coefficient_shape, dtype=float)
    pair_mean = np.zeros_like(total)
    pair_context = np.zeros((n + 2, n + 1, 4, 4) + coefficient_shape, dtype=float)
    zero = (0,) * group_count
    for i in range(1, n + 2):
        total[(i, i - 1) + zero] = t_e

    active_pairs = [(a, b) for a in range(4) for b in range(4) if e_p[a, b] != 0.0]
    for span in range(1, n + 1):
        for i in range(1, n - span + 2):
            j = i + span - 1
            current_counts = prefix[j] - prefix[i - 1]
            left_inner_counts = prefix[j] - prefix[i]
            right_inner_counts = prefix[j - 1] - prefix[i - 1]

            if j - i - 1 >= min_loop:
                inner_counts = prefix[j - 1] - prefix[i]
                nonpair_inner = total[i + 1, j - 1] - pair_mean[i + 1, j - 1]
                for a, b in active_pairs:
                    conditioned = nonpair_inner.copy()
                    if span >= 4:
                        deep_counts = inner_counts - site_counts[i] - site_counts[j - 2]
                        for c, d in active_pairs:
                            _accumulate_multivar_product(
                                conditioned,
                                (
                                    profile[i, c],
                                    profile[j - 2, d],
                                    pair_context[i + 1, j - 1, c, d],
                                ),
                                (site_counts[i], site_counts[j - 2], deep_counts),
                                stack[a, b, c, d],
                            )
                    pair_context[i, j, a, b] = t_p * e_p[a, b] * conditioned
                    _accumulate_multivar_product(
                        pair_mean[i, j],
                        (profile[i - 1, a], profile[j - 1, b], pair_context[i, j, a, b]),
                        (site_counts[i - 1], site_counts[j - 1], inner_counts),
                    )

            value = pair_mean[i, j].copy()
            for nucleotide in range(4):
                _accumulate_multivar_product(
                    value,
                    (profile[i - 1, nucleotide], total[i + 1, j]),
                    (site_counts[i - 1], left_inner_counts),
                    t_l * e_l[nucleotide],
                )
                _accumulate_multivar_product(
                    value,
                    (profile[j - 1, nucleotide], total[i, j - 1]),
                    (site_counts[j - 1], right_inner_counts),
                    t_r * e_r[nucleotide],
                )
            if span > 1:
                for split in range(i, j):
                    left_counts = prefix[split] - prefix[i - 1]
                    right_counts = prefix[j] - prefix[split]
                    _accumulate_multivar_product(
                        value,
                        (total[i, split], total[split + 1, j]),
                        (left_counts, right_counts),
                        t_b,
                    )
            total[i, j] = value
    return total[1, n].copy()

import numpy as np


def normalize_order_coefficients(
    fixed_inside_weights: "np.ndarray",
    library_coefficients: "np.ndarray",
    relative_tolerance: float = 1e-10,
) -> "np.ndarray":
    """Reference diagonal contraction, validation and normalization."""
    import numpy as np

    try:
        inside = np.array(fixed_inside_weights, dtype=float)
        coefficients = np.array(library_coefficients, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("weights and coefficients must be numeric") from None
    if inside.ndim != 3 or inside.shape[0] != 4 or inside.shape[2] < 2:
        raise ValueError("fixed_inside_weights has an invalid shape")
    length = inside.shape[2] - 1
    if inside.shape != (4, length + 2, length + 1):
        raise ValueError("fixed_inside_weights has an invalid shape")
    if coefficients.ndim < 1 or coefficients.ndim > 4 or any(size < 1 for size in coefficients.shape):
        raise ValueError("library_coefficients must have one through four nonempty axes")
    if sum(size - 1 for size in coefficients.shape) != length:
        raise ValueError("library coefficient degree capacities must sum to L")
    if not np.all(np.isfinite(inside)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("weights and coefficients must be finite")
    if isinstance(relative_tolerance, bool):
        raise ValueError("relative_tolerance must be a finite nonnegative scalar")
    try:
        tolerance = float(relative_tolerance)
    except (TypeError, ValueError):
        raise ValueError("relative_tolerance must be a finite nonnegative scalar") from None
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("relative_tolerance must be a finite nonnegative scalar")

    reference = float(inside[0, 1, length])
    constant = float(coefficients[(0,) * coefficients.ndim])
    if reference <= 0.0:
        raise ValueError("the reference partition function must be positive")
    scale = max(abs(reference), abs(constant))
    if abs(constant - reference) > tolerance * scale:
        raise ValueError("the polynomial constant does not match the reference partition function")

    diagonal = np.zeros(length + 1, dtype=float)
    for multi_index in np.ndindex(coefficients.shape):
        diagonal[sum(multi_index)] += coefficients[multi_index]
    return diagonal / reference

import numpy as np


def compute_log_series_coefficients(
    normalized_coefficients: "np.ndarray",
    max_order: int,
) -> "np.ndarray":
    """Reference formal-series logarithm from A' = (log A)' A."""
    import numpy as np

    try:
        coefficients = np.array(normalized_coefficients, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("normalized_coefficients must be numeric") from None
    if coefficients.ndim != 1 or coefficients.size < 2:
        raise ValueError("normalized_coefficients must be one-dimensional with at least two entries")
    if not np.all(np.isfinite(coefficients)) or coefficients[0] <= 0.0:
        raise ValueError("normalized_coefficients must be finite with a positive constant")
    if isinstance(max_order, bool) or not isinstance(max_order, (int, np.integer)):
        raise ValueError("max_order must be an integer")
    if max_order < 1 or max_order >= coefficients.size:
        raise ValueError("max_order is outside the supplied polynomial degree")

    unit = coefficients / coefficients[0]
    log_coefficients = np.zeros(max_order + 1, dtype=float)
    log_coefficients[0] = np.log(coefficients[0])
    for order in range(1, max_order + 1):
        correction = 0.0
        for lower in range(1, order):
            correction += lower * log_coefficients[lower] * unit[order - lower]
        log_coefficients[order] = unit[order] - correction / order
    return log_coefficients

import numpy as np


def _remainder_inputs(normalized_coefficients, log_coefficients, mutation_rate):
    """Validate inputs shared by evaluation and root location."""
    import numpy as np

    try:
        normalized = np.array(normalized_coefficients, dtype=float)
        logarithm = np.array(log_coefficients, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("coefficient arrays must be numeric") from None
    if normalized.ndim != 1 or normalized.size < 2 or not np.all(np.isfinite(normalized)):
        raise ValueError("normalized_coefficients must be a finite one-dimensional vector")
    if logarithm.ndim != 1 or logarithm.size < 2 or logarithm.size > normalized.size:
        raise ValueError("log_coefficients has an invalid shape")
    if not np.all(np.isfinite(logarithm)):
        raise ValueError("log_coefficients must be finite")
    if not np.isclose(normalized[0], 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("normalized_coefficients must have constant one")
    if not np.isclose(logarithm[0], 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("log_coefficients must have constant zero")
    if isinstance(mutation_rate, bool):
        raise ValueError("mutation_rate must be a finite scalar in [0, 1]")
    try:
        rate = float(mutation_rate)
    except (TypeError, ValueError):
        raise ValueError("mutation_rate must be a finite scalar in [0, 1]") from None
    if not np.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ValueError("mutation_rate must be a finite scalar in [0, 1]")
    return normalized, logarithm, rate


def evaluate_log_remainder_share(
    normalized_coefficients: "np.ndarray",
    log_coefficients: "np.ndarray",
    mutation_rate: float,
) -> float:
    """Reference polynomial evaluation and relative log remainder."""
    import numpy as np

    normalized, logarithm, rate = _remainder_inputs(
        normalized_coefficients, log_coefficients, mutation_rate
    )
    exact_polynomial = float(np.polynomial.polynomial.polyval(rate, normalized))
    if exact_polynomial <= 0.0 or not np.isfinite(exact_polynomial):
        raise ValueError("the normalized partition polynomial must be positive at mutation_rate")
    exact_log = float(np.log(exact_polynomial))
    if exact_log == 0.0:
        raise ValueError("the exact log change is zero at mutation_rate")
    truncated_log = float(np.polynomial.polynomial.polyval(rate, logarithm))
    return float(abs(exact_log - truncated_log) / abs(exact_log))

import numpy as np


def _root_scalar(value, name, *, positive=False, nonnegative=False):
    """Validate and return a finite non-boolean scalar."""
    import numpy as np

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite scalar") from None
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def locate_log_remainder_threshold(
    normalized_coefficients: "np.ndarray",
    log_coefficients: "np.ndarray",
    target_share: float,
    lower_rate: float,
    upper_rate: float,
    rate_tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Reference bisection of the relative log-remainder crossing."""
    target = _root_scalar(target_share, "target_share", nonnegative=True)
    lower = _root_scalar(lower_rate, "lower_rate")
    upper = _root_scalar(upper_rate, "upper_rate")
    tolerance = _root_scalar(rate_tolerance, "rate_tolerance", positive=True)
    if not (0.0 < lower < upper <= 1.0):
        raise ValueError("rates must satisfy 0 < lower_rate < upper_rate <= 1")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be a positive integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    f_lower = evaluate_log_remainder_share(
        normalized_coefficients, log_coefficients, lower
    ) - target
    f_upper = evaluate_log_remainder_share(
        normalized_coefficients, log_coefficients, upper
    ) - target
    if f_lower == 0.0:
        return float(lower)
    if f_upper == 0.0:
        return float(upper)
    if f_lower * f_upper > 0.0:
        raise ValueError("the target is not bracketed")

    for _ in range(max_iterations):
        if upper - lower <= tolerance:
            return float(0.5 * (lower + upper))
        middle = 0.5 * (lower + upper)
        f_middle = evaluate_log_remainder_share(
            normalized_coefficients, log_coefficients, middle
        ) - target
        if f_middle == 0.0:
            return float(middle)
        if f_lower * f_middle < 0.0:
            upper = middle
            f_upper = f_middle
        else:
            lower = middle
            f_lower = f_middle
    raise ValueError("max_iterations reached before rate_tolerance")

import numpy as np


def run_log_nonadditivity_threshold(
    sequence: "np.ndarray | None" = None,
    mutation_kernel: "np.ndarray | None" = None,
    rule_weights: "np.ndarray | None" = None,
    left_emission: "np.ndarray | None" = None,
    right_emission: "np.ndarray | None" = None,
    pair_emission: "np.ndarray | None" = None,
    stacking_factors: "np.ndarray | None" = None,
    min_loop: "int | None" = None,
    log_order: "int | None" = None,
    target_share: "float | None" = None,
    lower_rate: "float | None" = None,
    upper_rate: "float | None" = None,
    rate_tolerance: float = 1e-12,
) -> float:
    """Reference orchestrator using every preceding oracle stage."""
    import numpy as np

    if sequence is None:
        sequence = np.array(["ACGU".index(ch) for ch in "GGACAUCGAGUCCUAG"])
    if mutation_kernel is None:
        mutation_kernel = np.ones((4, 4), dtype=float) / 3.0
        np.fill_diagonal(mutation_kernel, 0.0)
    if rule_weights is None:
        rule_weights = np.array([0.32, 0.24, 0.18, 0.14, 0.70])
    if left_emission is None:
        left_emission = np.array([0.33, 0.21, 0.26, 0.20])
    if right_emission is None:
        right_emission = np.array([0.19, 0.30, 0.22, 0.29])
    if pair_emission is None:
        pair_emission = np.zeros((4, 4), dtype=float)
        for a, b, value in (
            (0, 3, 1.15), (3, 0, 0.85), (2, 1, 1.75),
            (1, 2, 1.40), (2, 3, 0.50), (3, 2, 0.65),
        ):
            pair_emission[a, b] = value
    if stacking_factors is None:
        stacking_factors = np.array([2.2, 1.5, 1.3, 0.8])
    if min_loop is None:
        min_loop = 3
    if log_order is None:
        log_order = 4
    if target_share is None:
        target_share = 0.30
    if lower_rate is None:
        lower_rate = 0.045
    if upper_rate is None:
        upper_rate = 0.070

    sequence = _grouped_validated_codes(sequence)
    _, position_groups = np.unique(sequence, return_inverse=True)
    position_groups = position_groups.astype(int)
    profile = build_mutation_profile_polynomials(
        sequence, mutation_kernel, position_groups,
    )
    fixed = compute_stacked_inside_weights(
        sequence, rule_weights, left_emission, right_emission,
        pair_emission, stacking_factors, min_loop,
    )
    polynomial = compute_profile_partition_polynomial(
        profile, position_groups, rule_weights, left_emission, right_emission,
        pair_emission, stacking_factors, min_loop,
    )
    normalized = normalize_order_coefficients(fixed, polynomial)
    logarithm = compute_log_series_coefficients(normalized, log_order)
    return locate_log_remainder_threshold(
        normalized, logarithm, target_share, lower_rate, upper_rate,
        rate_tolerance, 200,
    )
SCICODE_GOLD_EOF
