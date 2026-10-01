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
def map_two_threshold_variable(
    s: complex,
    s_plus: float,
    s_inelastic: float,
    sheet: str,
) -> complex:
    """Reference implementation (principal roots, upper-half-plane boundary values)."""
    import numpy as np

    def _is_real(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_number(value):
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(np.isfinite(value))
        if isinstance(value, (complex, np.complexfloating)):
            return bool(np.isfinite(value.real) and np.isfinite(value.imag))
        return False

    if not _is_number(s):
        raise ValueError("s must be a finite real or complex number")
    if not (_is_real(s_plus) and s_plus > 0.0):
        raise ValueError("s_plus must be a finite positive number")
    if not (_is_real(s_inelastic) and s_inelastic > float(s_plus)):
        raise ValueError("s_inelastic must be a finite number above s_plus")
    if sheet not in ("11", "21", "22", "12"):
        raise ValueError("sheet must be one of '11', '21', '22', '12'")

    point = complex(s)
    low, high = float(s_plus), float(s_inelastic)

    def _root(value):
        # Principal square root, except that a negative real argument is the
        # limit reached from s + i0, which sits just below the branch cut.
        value = complex(value)
        if value.imag == 0.0 and value.real < 0.0:
            return complex(0.0, -np.sqrt(-value.real))
        return np.sqrt(value)

    # phi_(11) = sqrt(s_plus - s) / (sqrt(s_in - s) + sqrt(s_in - s_plus)); this
    # rationalised form stays accurate at the elastic threshold, where the
    # difference form collapses to 0/0.
    denominator = _root(high - point) + np.sqrt(high - low)
    if denominator == 0.0:
        raise ValueError("the uniformising variable is not finite at this s")
    physical = _root(low - point) / denominator

    if sheet == "11":
        result = physical
    elif sheet == "21":
        result = -physical
    else:
        if physical == 0.0:
            raise ValueError("the requested branch is not finite at this s")
        result = 1.0 / physical if sheet == "22" else -1.0 / physical

    if not (np.isfinite(result.real) and np.isfinite(result.imag)):
        raise ValueError("the requested branch is not finite at this s")
    return complex(result)

import numpy as np
def map_left_hand_cut(
    point: complex,
    cut_branch_point: float,
    origin_point: float,
) -> complex:
    """Reference implementation of the slit-disc map."""
    import numpy as np

    def _is_real(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_number(value):
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(np.isfinite(value))
        if isinstance(value, (complex, np.complexfloating)):
            return bool(np.isfinite(value.real) and np.isfinite(value.imag))
        return False

    if not _is_number(point):
        raise ValueError("point must be a finite real or complex number")
    if not (_is_real(cut_branch_point) and _is_real(origin_point)):
        raise ValueError("cut_branch_point and origin_point must be finite real numbers")
    if not (-1.0 < float(cut_branch_point) < float(origin_point) < 1.0):
        raise ValueError("require -1 < cut_branch_point < origin_point < 1")
    value = complex(point)
    if abs(value) > 1.0:
        raise ValueError("point must satisfy abs(point) <= 1")

    edge = float(cut_branch_point)
    centre = float(origin_point)

    # The slit disc is carried to an auxiliary plane in which the slit is a
    # subthreshold cut and then back with a second square-root map; composing
    # the two leaves a ratio of square roots whose arguments are
    #   A ~ (s_a - s)  and  B ~ (s_a - s_0)
    # up to one common factor, with s the image of `point`.
    first = (value * (edge - 1.0) ** 2 - edge * (value - 1.0) ** 2) * (centre - 1.0) ** 2
    second = (centre * (edge - 1.0) ** 2 - edge * (centre - 1.0) ** 2) * (value - 1.0) ** 2
    root_first = np.sqrt(complex(first))
    root_second = np.sqrt(complex(second))
    total = root_first + root_second
    if total == 0.0:
        raise ValueError("the slit-disc map is not finite at this point")
    result = (root_first - root_second) / total
    if not (np.isfinite(result.real) and np.isfinite(result.imag)):
        raise ValueError("the slit-disc map is not finite at this point")
    return complex(result)

import numpy as np
def map_four_sheet_variable(
    s: complex,
    s_plus: float,
    s_inelastic: float,
    s_lhc: float,
    s_origin: float,
    sheet: str,
    threshold_fn: "Callable[..., complex]",
    lhc_fn: "Callable[..., complex]",
) -> complex:
    """Reference implementation (composition with reflection outside the disc)."""
    import numpy as np

    def _finite(value):
        value = complex(value)
        return bool(np.isfinite(value.real) and np.isfinite(value.imag))

    def _is_function(value):
        return callable(value)

    if not _is_function(threshold_fn):
        raise ValueError("threshold_fn must be callable")
    if not _is_function(lhc_fn):
        raise ValueError("lhc_fn must be callable")

    edge = complex(threshold_fn(s_lhc, s_plus, s_inelastic, "21"))
    centre = complex(threshold_fn(s_origin, s_plus, s_inelastic, "11"))
    if not (_finite(edge) and _finite(centre)):
        raise ValueError("the reference points must be finite")
    if abs(edge.imag) > 1.0e-12 or abs(centre.imag) > 1.0e-12:
        raise ValueError("the reference points must be real")

    variable = complex(threshold_fn(s, s_plus, s_inelastic, sheet))
    if not _finite(variable):
        raise ValueError("the two-threshold variable must be finite")

    if abs(variable) <= 1.0:
        result = complex(lhc_fn(variable, edge.real, centre.real))
    else:
        # Reflection in the unit circle is the unique continuation that keeps
        # the boundary fixed, so the exterior value is the reciprocal of the
        # image of the reciprocal point.
        inner = complex(lhc_fn(1.0 / variable, edge.real, centre.real))
        if inner == 0.0:
            raise ValueError("the continued map is not finite at this s")
        result = 1.0 / inner
    if not _finite(result):
        raise ValueError("the composed map must return a finite value")
    return complex(result)

import numpy as np
def solve_asymptotic_behaviour(
    variable_fn: "Callable[[float], complex]",
    probe: float = -1.0e10,
    ratio: float = 1.0e4,
) -> "np.ndarray":
    """Reference implementation (two-point log-log slope)."""
    import numpy as np

    def _is_real(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not _is_function(variable_fn):
        raise ValueError("variable_fn must be callable")
    if not (_is_real(probe) and probe < 0.0):
        raise ValueError("probe must be a finite negative number")
    if not (_is_real(ratio) and ratio > 1.0):
        raise ValueError("ratio must be a finite number greater than one")

    near = float(probe)
    far = float(probe) * float(ratio)

    def _offset(point):
        value = complex(variable_fn(point))
        if not (np.isfinite(value.real) and np.isfinite(value.imag)):
            raise ValueError("the variable must be finite at the probe points")
        if abs(value.imag) > 1.0e-12:
            raise ValueError("the variable must be real at a spacelike probe")
        gap = value.real - 1.0
        if not (gap < 0.0):
            raise ValueError("the variable must stay below one at the probes")
        return -gap

    near_gap = _offset(near)
    far_gap = _offset(far)
    slope = np.log(near_gap / far_gap)
    if not (np.isfinite(slope) and slope > 0.0):
        raise ValueError("the probes do not resolve a decaying approach to unity")
    exponent = float(np.log(abs(far / near)) / slope)
    if not (np.isfinite(exponent) and exponent > 0.0):
        raise ValueError("the estimated exponent must be finite and positive")

    # (1 - psi)**p -> c / abs(s), read off at the farther probe.
    coefficient = float(abs(far) * far_gap ** exponent)
    if not np.isfinite(coefficient):
        raise ValueError("the estimated coefficient must be finite")
    return np.array([exponent, coefficient], dtype=float)

import numpy as np
def evaluate_pole_factor(
    points: "np.ndarray",
    pole_points: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the conjugate-pair pole product."""
    import numpy as np

    def _as_line(value, label):
        try:
            arr = np.asarray(value, dtype=complex)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be an array of numbers") from None
        arr = np.atleast_1d(arr)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(f"{label} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(arr.real) & np.isfinite(arr.imag)):
            raise ValueError(f"{label} must contain only finite numbers")
        return arr

    grid = _as_line(points, "points")
    poles = _as_line(pole_points, "pole_points")
    if np.any(np.abs(poles.imag) <= 0.0):
        raise ValueError("pole images must lie off the real axis")

    total = np.ones(grid.size, dtype=complex)
    for pole in poles:
        gap = grid - pole
        mirror = grid - np.conj(pole)
        if np.any(gap == 0.0) or np.any(mirror == 0.0):
            raise ValueError("a requested point coincides with a pole")
        total = total / (gap * mirror)
    if not np.all(np.isfinite(total.real) & np.isfinite(total.imag)):
        raise ValueError("the pole product overflowed at a requested point")
    return total

import numpy as np
def solve_series_coefficients(
    real_points: "np.ndarray",
    real_values: "np.ndarray",
    complex_points: "np.ndarray",
    complex_values: "np.ndarray",
    pole_points: "np.ndarray",
    vanishing_order: int,
    factor_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Reference implementation (square real linear system)."""
    import numpy as np

    def _as_line(value, label):
        try:
            arr = np.atleast_1d(np.asarray(value, dtype=complex))
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be an array of numbers") from None
        if arr.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        if arr.size and not np.all(np.isfinite(arr.real) & np.isfinite(arr.imag)):
            raise ValueError(f"{label} must contain only finite numbers")
        return arr

    def _is_function(value):
        return callable(value)

    def _is_integer(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, np.integer)))

    if not _is_function(factor_fn):
        raise ValueError("factor_fn must be callable")
    if not _is_integer(vanishing_order):
        raise ValueError("vanishing_order must be an integer")
    order_count = int(vanishing_order)
    if order_count < 1:
        raise ValueError("vanishing_order must be at least one")

    flat_points = _as_line(real_points, "real_points")
    flat_values = _as_line(real_values, "real_values")
    wide_points = _as_line(complex_points, "complex_points")
    wide_values = _as_line(complex_values, "complex_values")
    if flat_points.size != flat_values.size:
        raise ValueError("real_points and real_values must have equal length")
    if wide_points.size != wide_values.size:
        raise ValueError("complex_points and complex_values must have equal length")
    if np.any(flat_points.imag != 0.0) or np.any(flat_values.imag != 0.0):
        raise ValueError("real_points and real_values must be real")
    if np.any(wide_points.imag == 0.0):
        raise ValueError("complex_points must lie off the real axis")

    width = flat_points.size + 2 * wide_points.size + order_count
    if flat_points.size + wide_points.size == 0:
        raise ValueError("at least one measurement is required")

    nodes = np.concatenate([flat_points, wide_points])
    factors = np.atleast_1d(np.asarray(factor_fn(nodes, pole_points), dtype=complex))
    if factors.shape != nodes.shape:
        raise ValueError("factor_fn must return one value per point")
    if np.any(factors == 0.0):
        raise ValueError("the pole product vanishes at a measurement point")

    powers = np.arange(width, dtype=float)
    rows, right = [], []
    for index in range(flat_points.size):
        basis = flat_points[index] ** powers
        rows.append(basis.real)
        right.append(float((flat_values[index] / factors[index]).real))
    for index in range(wide_points.size):
        basis = wide_points[index] ** powers
        target = wide_values[index] / factors[flat_points.size + index]
        rows.append(basis.real); right.append(float(target.real))
        rows.append(basis.imag); right.append(float(target.imag))
    for step in range(order_count):
        falling = np.ones(width, dtype=float)
        for shift in range(step):
            falling = falling * (powers - float(shift))
        rows.append(falling)
        right.append(0.0)

    matrix = np.array(rows, dtype=float)
    vector = np.array(right, dtype=float)
    try:
        coefficients = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        raise ValueError("the conditions do not determine the coefficients") from None
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("the conditions do not determine the coefficients")
    return coefficients.astype(float)

import numpy as np
def evaluate_series_form_factor(
    points: "np.ndarray",
    coefficients: "np.ndarray",
    pole_points: "np.ndarray",
    factor_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Reference implementation (Horner evaluation times the pole product)."""
    import numpy as np

    def _is_function(value):
        return callable(value)

    if not _is_function(factor_fn):
        raise ValueError("factor_fn must be callable")
    try:
        grid = np.atleast_1d(np.asarray(points, dtype=complex))
        weights = np.atleast_1d(np.asarray(coefficients, dtype=complex))
    except (TypeError, ValueError):
        raise ValueError("points and coefficients must be arrays of numbers") from None
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("points must be a non-empty one-dimensional array")
    if weights.ndim != 1 or weights.size < 1:
        raise ValueError("coefficients must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(grid.real) & np.isfinite(grid.imag)):
        raise ValueError("points must contain only finite numbers")
    if not np.all(np.isfinite(weights.real) & np.isfinite(weights.imag)):
        raise ValueError("coefficients must contain only finite numbers")
    if np.any(weights.imag != 0.0):
        raise ValueError("coefficients must be real")

    series = np.zeros(grid.size, dtype=complex)
    for weight in weights[::-1]:
        series = series * grid + weight

    factors = np.atleast_1d(np.asarray(factor_fn(grid, pole_points), dtype=complex))
    if factors.shape != grid.shape:
        raise ValueError("factor_fn must return one value per point")
    result = factors * series
    if not np.all(np.isfinite(result.real) & np.isfinite(result.imag)):
        raise ValueError("the parametrisation is not finite at a requested point")
    return result

import numpy as np
def extract_pole_residue(
    pole_position: complex,
    function_fn: "Callable[[np.ndarray], np.ndarray]",
    radius: float,
    nodes: int,
) -> complex:
    """Reference implementation (trapezoidal Cauchy integral on a circle)."""
    import numpy as np

    def _is_real(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_number(value):
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(np.isfinite(value))
        if isinstance(value, (complex, np.complexfloating)):
            return bool(np.isfinite(value.real) and np.isfinite(value.imag))
        return False

    def _is_function(value):
        return callable(value)

    def _is_integer(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, np.integer)))

    if not _is_number(pole_position):
        raise ValueError("pole_position must be a finite number")
    if not _is_function(function_fn):
        raise ValueError("function_fn must be callable")
    if not (_is_real(radius) and radius > 0.0):
        raise ValueError("radius must be a finite positive number")
    if not _is_integer(nodes):
        raise ValueError("nodes must be an integer")
    count = int(nodes)
    if count < 2:
        raise ValueError("nodes must be at least two")

    centre = complex(pole_position)
    angles = 2.0 * np.pi * np.arange(count, dtype=float) / float(count)
    offsets = float(radius) * np.exp(1j * angles)
    samples = np.atleast_1d(np.asarray(function_fn(centre + offsets), dtype=complex))
    if samples.shape != offsets.shape:
        raise ValueError("function_fn must return one value per sample point")
    if not np.all(np.isfinite(samples.real) & np.isfinite(samples.imag)):
        raise ValueError("the sampled values must be finite")

    # (1 / 2 pi i) * contour integral of f, with ds = i * offset * dtheta.
    result = complex(np.sum(samples * offsets) / float(count))
    if not (np.isfinite(result.real) and np.isfinite(result.imag)):
        raise ValueError("the residue estimate is not finite")
    return result

import numpy as np
def estimate_pole_residue(
    s_plus: float = 4.0 * 0.13957 ** 2,
    s_inelastic: float = 4.0 * 0.493677 ** 2,
    s_lhc: float = 0.0,
    s_origin: float = -0.30,
    pole_roots: tuple = (0.462 - 0.271j, 0.9935 - 0.0285j, 0.9705 - 0.0655j, 0.700 - 0.350j),
    pole_sheets: tuple = ("21", "21", "22", "22"),
    real_nodes: tuple = (-2.00, -0.40),
    real_values: tuple = (0.45, 0.80),
    timelike_nodes: tuple = (0.20, 0.50, 0.80),
    timelike_values: tuple = (1.70 + 0.55j, 2.30 + 1.30j, 2.90 + 3.60j),
    target: int = 0,
    radius: float = 0.01,
    nodes: int = 128,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _line(value, label):
        try:
            arr = np.atleast_1d(np.asarray(tuple(value), dtype=complex))
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a sequence of numbers") from None
        if arr.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        return arr

    roots = _line(pole_roots, "pole_roots")
    if roots.size < 1:
        raise ValueError("pole_roots must not be empty")
    sheets = tuple(pole_sheets)
    if len(sheets) != roots.size:
        raise ValueError("pole_sheets must match pole_roots in length")
    flat_nodes = _line(real_nodes, "real_nodes")
    flat_values = _line(real_values, "real_values")
    wide_nodes = _line(timelike_nodes, "timelike_nodes")
    wide_values = _line(timelike_values, "timelike_values")
    def _is_integer(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, np.integer)))

    if not _is_integer(target):
        raise ValueError("target must be an integer")
    if not 0 <= int(target) < roots.size:
        raise ValueError("target must index pole_roots")

    def _threshold(point, low, high, sheet):
        return map_two_threshold_variable(point, low, high, sheet)

    def _slit(point, edge, centre):
        return map_left_hand_cut(point, edge, centre)

    def _variable(point, sheet):
        return map_four_sheet_variable(
            point, s_plus, s_inelastic, s_lhc, s_origin, sheet, _threshold, _slit
        )

    def _factor(points, poles):
        return evaluate_pole_factor(points, poles)

    behaviour = solve_asymptotic_behaviour(lambda point: _variable(point, "11"))
    exponent = float(np.asarray(behaviour, dtype=float).ravel()[0])
    vanishing = int(np.rint(exponent))
    if vanishing < 1:
        raise ValueError("the fall-off condition must impose at least one constraint")

    pole_points = np.array(
        [_variable(complex(root) ** 2, sheet) for root, sheet in zip(roots, sheets)],
        dtype=complex,
    )
    flat_points = np.array(
        [complex(_variable(complex(point).real, "11")).real for point in flat_nodes],
        dtype=float,
    )
    wide_points = np.array(
        [_variable(complex(point).real, "11") for point in wide_nodes], dtype=complex
    )
    coefficients = solve_series_coefficients(
        flat_points,
        np.asarray(flat_values, dtype=complex).real.astype(float),
        wide_points,
        wide_values,
        pole_points,
        vanishing,
        _factor,
    )

    target_sheet = sheets[int(target)]

    def _amplitude(points):
        images = np.array(
            [_variable(complex(point), target_sheet) for point in np.atleast_1d(points)],
            dtype=complex,
        )
        return evaluate_series_form_factor(images, coefficients, pole_points, _factor)

    residue = extract_pole_residue(
        complex(roots[int(target)]) ** 2, _amplitude, radius, nodes
    )
    result = float(abs(residue))
    if not np.isfinite(result):
        raise ValueError("the residue modulus must be finite")
    return result
SCICODE_GOLD_EOF
