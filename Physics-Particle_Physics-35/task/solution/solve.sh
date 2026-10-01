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


def channel_geometry(light_mass, heavy_mass, cut_opening, normalization_point):
    def _real(value, name):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite real scalar")
        if np.iscomplexobj(value) and np.imag(value) != 0.0:
            raise ValueError(name + " must be a finite real scalar")
        try:
            numeric = float(np.real(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite real scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite real scalar")
        return numeric

    light = _real(light_mass, "light_mass")
    heavy = _real(heavy_mass, "heavy_mass")
    if light <= 0.0:
        raise ValueError("light_mass must be a finite positive scalar")
    if heavy <= 0.0:
        raise ValueError("heavy_mass must be a finite positive scalar")

    opening = _real(cut_opening, "cut_opening")
    normalization = _real(normalization_point, "normalization_point")

    lower = 4.0 * light * light
    upper = 4.0 * heavy * heavy
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("the squared thresholds must be finite")
    if not lower < upper:
        raise ValueError("light_mass must place the lower threshold strictly below the upper one")

    return (float(lower), float(upper), float(opening), float(normalization))

import numpy as np


def threshold_branch_map(s, lower_threshold, upper_threshold, sheet):
    def _complex_scalar(value, name):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite real or complex scalar")
        try:
            numeric = np.asarray(value, dtype=np.complex128).item()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite real or complex scalar") from exc
        if not np.isfinite(numeric.real) or not np.isfinite(numeric.imag):
            raise ValueError(name + " must be a finite real or complex scalar")
        return numeric

    def _real_scalar(value, name):
        numeric = _complex_scalar(value, name)
        if numeric.imag != 0.0:
            raise ValueError(name + " must be a finite real scalar")
        return float(numeric.real)

    energy = _complex_scalar(s, "s")
    if energy.imag == 0.0:
        energy = complex(energy.real, 0.0)
    lower = _real_scalar(lower_threshold, "lower_threshold")
    upper = _real_scalar(upper_threshold, "upper_threshold")
    if not lower < upper:
        raise ValueError("lower_threshold must be strictly below upper_threshold")
    if not isinstance(sheet, str) or sheet not in {"11", "21", "22", "12"}:
        raise ValueError("sheet must be one of '11', '21', '22' or '12'")

    if energy == complex(lower) and sheet in {"22", "12"}:
        raise ValueError("the outer branch diverges at the lower threshold")

    def _root_difference(threshold, point):
        # Halving before an overflowing subtraction preserves a representable root.
        real_part = threshold - point.real
        if np.isfinite(real_part):
            return complex(np.sqrt(complex(real_part, -point.imag)))
        half_difference = complex(threshold / 2.0 - point.real / 2.0, -point.imag / 2.0)
        return complex(np.sqrt(half_difference)) * np.sqrt(2.0)

    c = _root_difference(upper, complex(lower))
    root_upper = _root_difference(upper, energy)
    root_lower = _root_difference(lower, energy)

    if sheet in {"11", "21"}:
        if energy == complex(lower):
            value = complex(0.0)
        else:
            value = root_lower / (root_upper + c)
        if sheet == "21":
            value = -value
    else:
        value = (root_upper + c) / root_lower
        if sheet == "12":
            value = -value

    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError("the branch result is not representable as a finite complex scalar")
    if value == 0.0 and energy != complex(lower):
        raise ValueError("the nonzero branch result underflows the supported numeric range")
    return complex(value)

import numpy as np


def leftcut_map(x, cut_reference, normalization_reference):
    def _complex_scalar(value, name):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite real or complex scalar")
        try:
            numeric = np.asarray(value, dtype=np.complex128).item()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite real or complex scalar") from exc
        if not np.isfinite(numeric.real) or not np.isfinite(numeric.imag):
            raise ValueError(name + " must be a finite real or complex scalar")
        return numeric

    value = _complex_scalar(x, "x")
    cut = _complex_scalar(cut_reference, "cut_reference")
    normalization = _complex_scalar(normalization_reference, "normalization_reference")
    if cut == normalization:
        raise ValueError("cut_reference and normalization_reference must differ")

    with np.errstate(all="ignore"):
        try:
            factors_a = (value - cut, 1.0 - cut * value, normalization - 1.0, normalization - 1.0)
            factors_b = (normalization - cut, 1.0 - cut * normalization, value - 1.0, value - 1.0)
            a = factors_a[0] * factors_a[1] * factors_a[2] * factors_a[3]
            b = factors_b[0] * factors_b[1] * factors_b[2] * factors_b[3]
        except OverflowError as exc:
            raise ValueError("the radicands overflow for these arguments") from exc
    for radicand in (a, b):
        if not np.isfinite(radicand.real) or not np.isfinite(radicand.imag):
            raise ValueError("the radicands overflow for these arguments")
    if (a == 0.0 and all(f != 0.0 for f in factors_a)) or (b == 0.0 and all(f != 0.0 for f in factors_b)):
        raise ValueError("a nonzero radicand underflows the supported numeric range")
    # A signed zero is not a choice of lip for a complete negative real radicand.
    root_a = np.sqrt(complex(a.real, 0.0 if a.imag == 0.0 else a.imag))
    root_b = np.sqrt(complex(b.real, 0.0 if b.imag == 0.0 else b.imag))
    denominator = root_a + root_b
    if denominator == 0.0:
        raise ValueError("the sum of the principal square roots must be nonzero")
    result = complex((root_a - root_b) / denominator)
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError("the map is not finite for these arguments")

    return result

import numpy as np


def bounded_variable(s, lower_threshold, upper_threshold, cut_opening, normalization_point, sheet):
    for value in (cut_opening, normalization_point):
        if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
            raise ValueError("reference points must be finite real scalars")
        numeric = complex(value)
        if numeric.imag != 0.0 or not np.isfinite(numeric.real):
            raise ValueError("reference points must be finite real scalars")
    x_l = threshold_branch_map(cut_opening, lower_threshold, upper_threshold, "21")
    x_0 = threshold_branch_map(normalization_point, lower_threshold, upper_threshold, "11")
    if not float(np.real(normalization_point)) < float(np.real(lower_threshold)):
        raise ValueError("normalization_point must be below the lower threshold")
    if x_0 in (1.0, -1.0):
        raise ValueError("the normalization image is numerically unresolved from a unit endpoint")
    if x_l == x_0:
        raise ValueError("mapped cut and normalization references must differ")
    intermediate = threshold_branch_map(s, lower_threshold, upper_threshold, sheet)
    if abs(intermediate) <= 1.0:
        return complex(leftcut_map(intermediate, x_l, x_0))
    denominator = leftcut_map(1.0 / intermediate, x_l, x_0)
    if denominator == 0.0:
        raise ValueError("the outer conformal denominator must be nonzero")
    result = complex(1.0 / denominator)
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError("the composed map must be finite")
    return result

from collections.abc import Sequence
import numpy as np


def pole_factor(variable_value, mapped_poles):
    from collections.abc import Sequence

    def _finite_complex_scalar(value, name):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = complex(float(np.real(value)), float(np.imag(value)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric.real) or not np.isfinite(numeric.imag):
            raise ValueError(name + " must be a finite scalar")
        return numeric

    variable = _finite_complex_scalar(variable_value, "variable_value")
    if isinstance(mapped_poles, (str, bytes)) or not isinstance(mapped_poles, Sequence) or len(mapped_poles) == 0:
        raise ValueError("mapped_poles must be a non-empty sequence")

    poles = []
    for index, pole in enumerate(mapped_poles):
        numeric = _finite_complex_scalar(pole, "mapped_poles[" + str(index) + "]")
        if variable == numeric or variable == numeric.conjugate():
            raise ValueError("variable_value must not coincide with a pole or its conjugate")
        poles.append(numeric)

    import math

    # Accumulate a binary-scaled denominator, not intermediate reciprocals.
    mantissa = 1.0 + 0.0j
    exponent = 0
    for pole in poles:
        for partner in (pole, pole.conjugate()):
            difference = variable - partner
            extra = 0
            if not np.isfinite(difference.real) or not np.isfinite(difference.imag):
                difference = variable * 0.5 - partner * 0.5
                extra = 1
            scale = max(abs(difference.real), abs(difference.imag))
            if scale == 0.0:
                raise ValueError("the pole difference is numerically singular")
            _, power = math.frexp(scale)
            normalized = complex(math.ldexp(difference.real, -power), math.ldexp(difference.imag, -power))
            mantissa *= normalized
            exponent += power + extra
            _, power = math.frexp(max(abs(mantissa.real), abs(mantissa.imag)))
            mantissa = complex(math.ldexp(mantissa.real, -power), math.ldexp(mantissa.imag, -power))
            exponent += power
    inverse = 1.0 / mantissa
    try:
        result = complex(math.ldexp(inverse.real, -exponent), math.ldexp(inverse.imag, -exponent))
    except OverflowError as exc:
        raise ValueError("the reciprocal product overflows the supported numeric range") from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag) or result == 0.0:
        raise ValueError("the nonzero reciprocal product is outside the supported numeric range")
    return result

from collections.abc import Sequence
import numpy as np


def form_factor(s, sheet, lower_threshold, upper_threshold, cut_opening, normalization_point, poles, coefficients):
    from collections.abc import Sequence

    if isinstance(poles, (str, bytes)) or not isinstance(poles, Sequence) or len(poles) == 0:
        raise ValueError("poles must be a non-empty sequence of energy and sheet pairs")
    mapped_poles = []
    for index, entry in enumerate(poles):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise ValueError("poles[" + str(index) + "] must be an energy and sheet pair")
        pole_energy, pole_sheet = entry
        if isinstance(pole_energy, (bool, np.bool_)) or not np.isscalar(pole_energy) or isinstance(pole_energy, (str, bytes)):
            raise ValueError("poles[" + str(index) + "][0] must be a finite scalar")
        try:
            squared = complex(pole_energy) ** 2
        except OverflowError as exc:
            raise ValueError("the squared pole energy must be finite") from exc
        if not np.isfinite(squared.real) or not np.isfinite(squared.imag):
            raise ValueError("the squared pole energy must be finite")
        mapped_poles.append(
            bounded_variable(squared, lower_threshold, upper_threshold, cut_opening, normalization_point, pole_sheet)
        )

    if isinstance(coefficients, (str, bytes)) or not isinstance(coefficients, Sequence) or len(coefficients) == 0:
        raise ValueError("coefficients must be a non-empty sequence of finite reals")
    validated_coefficients = []
    for index, value in enumerate(coefficients):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError("coefficients[" + str(index) + "] must be a finite real scalar")
        numeric = complex(float(np.real(value)), float(np.imag(value)))
        if numeric.imag != 0.0 or not np.isfinite(numeric.real):
            raise ValueError("coefficients[" + str(index) + "] must be a finite real scalar")
        validated_coefficients.append(float(numeric.real))

    variable = bounded_variable(s, lower_threshold, upper_threshold, cut_opening, normalization_point, sheet)
    factor = pole_factor(variable, mapped_poles)
    try:
        polynomial = sum(
            coefficient * variable**index
            for index, coefficient in enumerate(validated_coefficients)
        )
        result = complex(factor * polynomial)
    except OverflowError as exc:
        raise ValueError("the form factor must be finite") from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError("the form factor must be finite")
    return result

from collections.abc import Sequence
import numpy as np


def run_pipeline(light_mass, heavy_mass, cut_opening, normalization_point, poles, coefficients, evaluations):
    from collections.abc import Sequence

    lower, upper, opening, normalization = channel_geometry(
        light_mass, heavy_mass, cut_opening, normalization_point
    )

    if isinstance(poles, (str, bytes)) or not isinstance(poles, Sequence) or len(poles) == 0:
        raise ValueError("poles must be a non-empty sequence of energy and sheet pairs")
    for index, entry in enumerate(poles):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise ValueError("poles[" + str(index) + "] must be an energy and sheet pair")

    if isinstance(evaluations, (str, bytes)) or not isinstance(evaluations, Sequence) or len(evaluations) == 0:
        raise ValueError("evaluations must be a non-empty sequence of squared energy and sheet pairs")
    for index, entry in enumerate(evaluations):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise ValueError("evaluations[" + str(index) + "] must be a squared energy and sheet pair")

    product = 1.0
    for squared_energy, evaluation_sheet in evaluations:
        value = form_factor(
            squared_energy,
            evaluation_sheet,
            lower,
            upper,
            opening,
            normalization,
            poles,
            coefficients,
        )
        try:
            product *= abs(value) ** 2
        except OverflowError as exc:
            raise ValueError("the squared-modulus product must be finite") from exc
        if not np.isfinite(product):
            raise ValueError("the squared-modulus product must be finite")
    return float(product)
SCICODE_GOLD_EOF
