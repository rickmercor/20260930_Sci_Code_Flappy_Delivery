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

def _validated_problem_data(alpha, gamma, row_mixing, u, v, determinant_degree):
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    row_mixing = np.asarray(row_mixing, dtype=float)
    if alpha.shape != (5, 3) or gamma.shape != (4, 2) or row_mixing.shape != (7, 7):
        raise ValueError("alpha, gamma, and row_mixing have invalid shapes")
    if (
        not np.all(np.isfinite(alpha))
        or not np.all(np.isfinite(gamma))
        or not np.all(np.isfinite(row_mixing))
    ):
        raise ValueError("all coefficient data must be finite")
    if (
        not np.isfinite(u)
        or not np.isfinite(v)
        or np.isclose(u, v, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("u and v must be finite and distinct")
    if not isinstance(determinant_degree, (int, np.integer)) or determinant_degree < 1:
        raise ValueError("determinant_degree must be a positive integer")
    if abs(abs(np.linalg.det(row_mixing)) - 1.0) > 1e-9:
        raise ValueError("row_mixing must have unit determinant magnitude")
    return alpha, gamma, row_mixing, float(u), float(v), int(determinant_degree)


def construct_resultant_tensor(
    alpha: np.ndarray,
    gamma: np.ndarray,
    u: float,
    v: float,
    row_mixing: np.ndarray,
    determinant_degree: int = 12,
) -> np.ndarray:
    """Reference construction of the padded dense elimination tensor."""
    alpha, gamma, row_mixing, u, v, determinant_degree = _validated_problem_data(
        alpha, gamma, row_mixing, u, v, determinant_degree
    )
    base = np.zeros((determinant_degree + 1, 7, 7), dtype=float)
    q = np.array([u * v, -(u + v), 1.0])
    for column, y_degree in enumerate((4, 3, 2, 1, 0)):
        scaled = np.polynomial.polynomial.polymul(q, alpha[y_degree])
        if scaled.size > determinant_degree + 1:
            raise ValueError("determinant_degree is too small for the data")
        base[: scaled.size, 0, column] = scaled
        base[: alpha[y_degree].size, 1, column + 1] = alpha[y_degree]
        base[: alpha[y_degree].size, 2, column + 2] = alpha[y_degree]
    for shift in range(4):
        for column, y_degree in enumerate((3, 2, 1, 0)):
            base[: gamma[y_degree].size, 3 + shift, shift + column] = gamma[y_degree]
    return np.einsum("ij,djk->dik", row_mixing, base)

import numpy as np

def generate_unit_circle_samples(determinant_degree: int) -> np.ndarray:
    """Reference roots-of-unity construction."""
    if not isinstance(determinant_degree, (int, np.integer)) or determinant_degree < 1:
        raise ValueError("determinant_degree must be a positive integer")
    count = int(determinant_degree) + 1
    return np.exp(-2j * np.pi * np.arange(count) / count)

import numpy as np

def evaluate_resultant_fft(
    coefficient_tensor: np.ndarray, sample_points: np.ndarray,
    derivative_order: int = 0,
) -> np.ndarray:
    tensor = np.asarray(coefficient_tensor)
    points = np.asarray(sample_points)
    if (tensor.ndim != 3 or tensor.shape[0] < 1 or tensor.shape[1] != tensor.shape[2]
            or points.ndim != 1 or points.size < 2):
        raise ValueError("invalid tensor or sample shape")
    if not isinstance(derivative_order, (int, np.integer)) or not 0 <= derivative_order <= 3:
        raise ValueError("derivative_order must be an integer in 0..3")
    if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(points)):
        raise ValueError("inputs must be finite")
    count = points.size
    expected = np.exp(-2j*np.pi*np.arange(count)/count)
    if not np.allclose(points, expected, rtol=0, atol=1e-12):
        raise ValueError("incorrect sampling grid")
    output = []
    for order in range(derivative_order+1):
        folded = np.zeros((count,)+tensor.shape[1:], dtype=complex)
        for degree in range(order, tensor.shape[0]):
            factor = 1
            for j in range(order):
                factor *= degree-j
            folded[(degree-order) % count] += factor*tensor[degree]
        output.append(np.fft.fft(folded, axis=0))
    return output[0] if derivative_order == 0 else np.asarray(output)

import numpy as np

def _pivoted_determinant(matrix):
    work = np.array(matrix, dtype=complex, copy=True)
    determinant = 1.0 + 0.0j
    for column in range(work.shape[0]):
        pivot = column + int(np.argmax(np.abs(work[column:, column])))
        if abs(work[pivot, column]) <= np.finfo(float).tiny:
            return 0.0 + 0.0j
        if pivot != column:
            work[[column, pivot]] = work[[pivot, column]]
            determinant = -determinant
        value = work[column, column]
        determinant *= value
        if column+1 < work.shape[0]:
            factors = work[column+1:, column]/value
            work[column+1:, column+1:] -= np.outer(factors, work[column, column+1:])
    return determinant


def compute_determinant_samples(matrix_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(matrix_samples)
    if samples.ndim not in (3,4):
        raise ValueError("input rank must be three or four")
    scalar = samples.ndim == 3
    jets = samples[None] if scalar else samples
    if not 1 <= jets.shape[0] <= 4 or jets.shape[1] < 1 or jets.shape[2] != jets.shape[3]:
        raise ValueError("invalid jet or matrix shape")
    if not np.all(np.isfinite(jets)):
        raise ValueError("matrix data must be finite")
    if scalar:
        return np.array([_pivoted_determinant(m) for m in samples])
    channels, count, size, _ = jets.shape
    factorials = np.array([1,1,2,6][:channels], dtype=float)
    coefficients = jets/factorials[:,None,None,None]
    out = np.zeros((channels,count), dtype=complex)
    # Subset expansion over columns; products are truncated Taylor series.
    # The sign counts inversions contributed by the newly chosen column.
    for sample in range(count):
        dp = np.zeros((1 << size,channels), dtype=complex)
        dp[0,0] = 1
        for mask in range((1 << size)-1):
            row = mask.bit_count()
            for column in range(size):
                if mask & (1 << column):
                    continue
                sign = -1 if (mask >> (column+1)).bit_count() % 2 else 1
                product = np.convolve(dp[mask], coefficients[:,sample,row,column])[:channels]
                dp[mask | (1 << column)] += sign*product
        out[:,sample] = dp[-1]*factorials
    return out

import numpy as np

def recover_determinant_coefficients(
    determinant_samples: np.ndarray, coefficient_tensor: np.ndarray,
    validation_point: float = 0.37, validation_tolerance: float = 1e-9,
) -> np.ndarray:
    raw = np.asarray(determinant_samples)
    tensor = np.asarray(coefficient_tensor)
    if raw.ndim not in (1,2) or tensor.ndim != 3 or tensor.shape[0] < 1 or tensor.shape[1] != tensor.shape[2]:
        raise ValueError("invalid sample or tensor rank")
    data = raw[None] if raw.ndim == 1 else raw
    channels,count = data.shape
    length = tensor.shape[0]
    if not 1 <= channels <= 4 or count < 2 or channels*count < length:
        raise ValueError("insufficient or invalid confluent data")
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(tensor)):
        raise ValueError("data must be finite")
    if not np.isfinite(validation_point) or not np.isfinite(validation_tolerance) or validation_tolerance <= 0:
        raise ValueError("invalid validation settings")
    points = np.exp(-2j*np.pi*np.arange(count)/count)
    if np.min(np.abs(points-validation_point)) <= 1e-12:
        raise ValueError("validation point must be off grid")
    # Multiplication by x_j**r aligns each derivative with degree modulo S.
    moments = np.fft.ifft(data*points[None,:]**np.arange(channels)[:,None], axis=1)
    coefficients = np.zeros(length,dtype=complex)
    for residue in range(count):
        degrees = np.arange(residue,length,count)
        if not degrees.size:
            continue
        vandermonde = np.ones((channels,degrees.size),dtype=float)
        for order in range(1,channels):
            vandermonde[order] = vandermonde[order-1]*(degrees-order+1)
        # Scale equations together with their right-hand sides for stability.
        row_scale = np.maximum(1.0,np.max(np.abs(vandermonde),axis=1))
        coefficients[degrees] = np.linalg.lstsq(vandermonde/row_scale[:,None],moments[:,residue]/row_scale,rcond=None)[0]
    scale = max(1.0,float(np.max(np.abs(coefficients))))
    if np.max(np.abs(coefficients.imag)) > validation_tolerance*scale:
        raise ValueError("coefficients are not numerically real")
    coefficients = coefficients.real
    recovered = np.zeros_like(data,dtype=complex)
    for order in range(channels):
        derivative = np.polynomial.polynomial.polyder(coefficients,order)
        recovered[order] = np.polynomial.polynomial.polyval(points,derivative)
    if np.max(np.abs(recovered-data)) > validation_tolerance*max(1.0,float(np.max(np.abs(data)))):
        raise ValueError("inconsistent derivative samples")
    matrix = np.polynomial.polynomial.polyval(validation_point,tensor)
    direct = compute_determinant_samples(matrix[None])[0]
    value = np.polynomial.polynomial.polyval(validation_point,coefficients)
    if abs(direct-value) > validation_tolerance*(1+abs(direct)):
        raise ValueError("off-grid determinant validation failed")
    return coefficients

import numpy as np

def extract_hidden_candidates(
    coefficients: np.ndarray, imaginary_tolerance: float = 1e-8,
    interval: tuple | None = None
) -> np.ndarray:
    """Reference companion-root extraction with ascending-order handling."""
    coefficients = np.asarray(coefficients)
    if (
        coefficients.ndim != 1
        or coefficients.size < 2
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("coefficients must be a finite one-dimensional polynomial")
    if not np.isfinite(imaginary_tolerance) or imaginary_tolerance <= 0:
        raise ValueError("imaginary_tolerance must be positive and finite")
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    if abs(coefficients[-1]) <= 100 * np.finfo(float).eps * scale:
        raise ValueError("the leading coefficient must be nonzero")
    if interval is not None:
        bounds = np.asarray(interval)
        if bounds.shape != (2,) or not np.isrealobj(bounds) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
            raise ValueError("invalid root interval")
    roots = np.roots(coefficients[::-1])
    candidates = roots[np.abs(roots.imag) <= imaginary_tolerance].real
    if interval is not None:
        candidates = candidates[(candidates >= bounds[0]) & (candidates <= bounds[1])]
    if candidates.size == 0:
        raise ValueError("no root passes the imaginary-part gate")
    return np.sort(candidates)

import numpy as np

def _evaluate_polynomial_matrix(
    coefficient_tensor: np.ndarray, x_value: float
) -> np.ndarray:
    powers = float(x_value) ** np.arange(coefficient_tensor.shape[0])
    return np.tensordot(powers, coefficient_tensor, axes=(0, 0))


def _cofactor_null_vector(matrix: np.ndarray, deletion_row: int) -> np.ndarray:
    cofactors = []
    for column in range(matrix.shape[1]):
        minor = np.delete(np.delete(matrix, deletion_row, axis=0), column, axis=1)
        determinant = compute_determinant_samples(minor[None, :, :])[0]  # noqa: F821
        cofactors.append((-1) ** (deletion_row + column) * determinant)
    return np.asarray(cofactors)


def recover_and_filter_solutions(
    coefficient_tensor: np.ndarray,
    candidates: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    deletion_row: int = 0,
    anchor_column: int = 6,
    residual_threshold: float = 1e-3,
) -> np.ndarray:
    """Reference cofactor recovery and original-system residual filter."""
    tensor = np.asarray(coefficient_tensor)
    candidates = np.asarray(candidates, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if tensor.ndim != 3 or tensor.shape[1:] != (7, 7) or candidates.ndim != 1:
        raise ValueError("tensor or candidate shape is invalid")
    if alpha.shape != (5, 3) or gamma.shape != (4, 2):
        raise ValueError("equation coefficient shapes are invalid")
    if deletion_row not in range(7) or anchor_column not in range(1, 7):
        raise ValueError("deletion_row or anchor_column is invalid")
    if not np.isfinite(residual_threshold) or residual_threshold <= 0:
        raise ValueError("residual_threshold must be positive and finite")
    if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(candidates)):
        raise ValueError("inputs must be finite")

    polyval = np.polynomial.polynomial.polyval
    records = []
    for x_value in candidates:
        matrix = _evaluate_polynomial_matrix(tensor, x_value)
        null_vector = _cofactor_null_vector(matrix, deletion_row)
        scale = max(1.0, float(np.max(np.abs(null_vector))))
        if abs(null_vector[anchor_column]) <= 100 * np.finfo(float).eps * scale:
            continue
        y_value = null_vector[anchor_column - 1] / null_vector[anchor_column]
        if abs(y_value.imag) > 1e-8:
            continue
        y_value = float(y_value.real)
        f_value = sum(polyval(x_value, alpha[m]) * y_value**m for m in range(5))
        g_value = sum(polyval(x_value, gamma[m]) * y_value**m for m in range(4))
        denominator = max(1.0, float(np.hypot(x_value, y_value)))
        residual = float(max(abs(f_value), abs(g_value)) / denominator)
        if residual < residual_threshold:
            records.append((float(x_value), y_value, residual))
    if not records:
        raise ValueError("no candidate satisfies the residual gate")
    records.sort(key=lambda row: (row[0], row[1], row[2]))
    return np.asarray(records)

import numpy as np

def compute_hidden_root_condition(
    alpha: np.ndarray,
    gamma: np.ndarray,
    u: float,
    v: float,
    row_mixing: np.ndarray,
    determinant_degree: int = 12,
    imaginary_tolerance: float = 1e-8,
    residual_threshold: float = 1e-3,
) -> float:
    """Reference end-to-end solver composed from the seven earlier oracles."""
    if not np.isfinite(imaginary_tolerance) or imaginary_tolerance <= 0:
        raise ValueError("imaginary_tolerance must be positive and finite")
    if not np.isfinite(residual_threshold) or residual_threshold <= 0:
        raise ValueError("residual_threshold must be positive and finite")
    tensor = construct_resultant_tensor(  # noqa: F821
        alpha, gamma, u, v, row_mixing, determinant_degree
    )
    sample_count = max(2, (int(determinant_degree) + 4) // 4)
    points = generate_unit_circle_samples(sample_count - 1)  # noqa: F821
    matrices = evaluate_resultant_fft(tensor, points, derivative_order=3)  # noqa: F821
    samples = compute_determinant_samples(matrices)  # noqa: F821
    coefficients = recover_determinant_coefficients(samples, tensor)  # noqa: F821
    candidates = extract_hidden_candidates(coefficients, imaginary_tolerance, interval=(0.0,3.0))  # noqa: F821
    solutions = recover_and_filter_solutions(  # noqa: F821
        tensor,
        candidates,
        alpha,
        gamma,
        residual_threshold=residual_threshold,
    )
    derivative_coefficients = np.arange(1, coefficients.size) * coefficients[1:]
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    conditions = []
    for x_value in solutions[:, 0]:
        derivative = np.polynomial.polynomial.polyval(x_value, derivative_coefficients)
        if abs(derivative) <= 100 * np.finfo(float).eps * scale:
            raise ValueError("a retained root is numerically multiple")
        powers = x_value ** np.arange(coefficients.size)
        conditions.append(float(np.linalg.norm(powers) / abs(derivative)))
    result = max(conditions)
    if not np.isfinite(result):
        raise ValueError("the condition number is not finite")
    return result

import numpy as np

def _response_product(a, b, length):
    return np.convolve(a,b)[:length]

def _response_evaluate(coefficients, root):
    length = len(root)
    value = np.zeros(length)
    for degree in range(coefficients.shape[1]-1,-1,-1):
        value = _response_product(value,root,length) + coefficients[:,degree]
    return value

def _response_log_positive(series):
    length = len(series)
    if series[0] <= 0:
        raise ValueError("positive logarithm base required")
    inverse = np.zeros(length)
    inverse[0] = 1.0/series[0]
    for order in range(1,length):
        inverse[order] = -np.dot(series[1:order+1],inverse[order-1::-1])/series[0]
    result = np.zeros(length)
    result[0] = np.log(series[0])
    if length > 1:
        derivative = np.arange(1,length)*series[1:]
        result[1:] = np.convolve(derivative,inverse)[:length-1]/np.arange(1,length)
    return result

def continue_condition_jets(
    coefficient_jets: np.ndarray, anchor_root: float,
) -> np.ndarray:
    original = np.asarray(coefficient_jets)
    if (original.ndim != 2 or not 1 <= original.shape[0] <= 7
            or original.shape[1] < 2 or not np.isrealobj(original)
            or not np.all(np.isfinite(original))):
        raise ValueError("invalid real coefficient jets")
    if not np.isscalar(anchor_root) or not np.isrealobj(anchor_root) or not np.isfinite(anchor_root):
        raise ValueError("invalid anchor root")
    scale = float(np.max(np.abs(original[0])))
    if scale == 0:
        raise ValueError("zero base polynomial")
    coefficients = np.asarray(original,dtype=float)/scale
    length, width = coefficients.shape
    root = np.zeros(length)
    root[0] = float(anchor_root)
    base = coefficients[0]
    derivative = coefficients[:,1:]*np.arange(1,width)
    bound = float(np.dot(np.abs(base),max(1.0,abs(root[0]))**np.arange(width)))
    if abs(np.polynomial.polynomial.polyval(root[0],base)) > 1e-8*bound:
        raise ValueError("anchor is not a base root")
    for _ in range(8):
        slope = float(np.polynomial.polynomial.polyval(root[0],derivative[0]))
        slope_bound = float(np.dot(np.abs(derivative[0]),max(1.0,abs(root[0]))**np.arange(width-1)))
        if abs(slope) <= 100*np.finfo(float).eps*slope_bound:
            raise ValueError("multiple base root")
        correction = np.polynomial.polynomial.polyval(root[0],base)/slope
        root[0] -= correction
        if abs(correction) <= 2*np.finfo(float).eps*max(1.0,abs(root[0])):
            break
    slope = float(np.polynomial.polynomial.polyval(root[0],derivative[0]))
    for order in range(1,length):
        root[order] = -_response_evaluate(coefficients,root)[order]/slope
    moving_slope = _response_evaluate(derivative,root)
    numerator_squared = np.zeros(length)
    power = np.zeros(length)
    power[0] = 1.0
    squared_root = _response_product(root,root,length)
    for _ in range(width):
        numerator_squared += power
        power = _response_product(power,squared_root,length)
    log_condition = 0.5*_response_log_positive(numerator_squared)
    log_condition -= _response_log_positive(moving_slope*np.sign(moving_slope[0]))
    log_condition[0] -= np.log(scale)
    result = np.column_stack([root,log_condition])
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite continuation")
    return result

import numpy as np

def compute_condition_response(
    alpha: np.ndarray, gamma: np.ndarray, u: float, v: float,
    row_mixing: np.ndarray, alpha_direction: np.ndarray,
    gamma_direction: np.ndarray, response_order: int = 6,
    determinant_degree: int = 12, imaginary_tolerance: float = 1e-8,
    residual_threshold: float = 1e-3,
) -> float:
    da, dg = np.asarray(alpha_direction), np.asarray(gamma_direction)
    if (da.shape != (5,3) or dg.shape != (4,2) or not np.isrealobj(da)
            or not np.isrealobj(dg) or not np.all(np.isfinite(da))
            or not np.all(np.isfinite(dg))):
        raise ValueError("invalid coefficient directions")
    if not isinstance(response_order,(int,np.integer)) or not 0 <= response_order <= 6:
        raise ValueError("invalid response order")
    base_maximum = compute_hidden_root_condition(
        alpha,gamma,u,v,row_mixing,determinant_degree,
        imaginary_tolerance,residual_threshold)
    tensor = construct_resultant_tensor(alpha,gamma,u,v,row_mixing,determinant_degree)
    direction = construct_resultant_tensor(da,dg,u,v,row_mixing,determinant_degree)
    width = tensor.shape[0]
    count = max(2,(width+3)//4)
    points = generate_unit_circle_samples(count-1)
    parameters = generate_unit_circle_samples(7)
    sampled = np.asarray([
        compute_determinant_samples(
            evaluate_resultant_fft(tensor+t*direction,points,3))
        for t in parameters])
    moments = np.fft.ifft(sampled*points[None,None,:]**np.arange(4)[None,:,None],axis=2)
    at_parameters = np.zeros((8,width),dtype=complex)
    for residue in range(count):
        degrees = np.arange(residue,width,count)
        if not degrees.size:
            continue
        system = np.ones((4,len(degrees)))
        for order in range(1,4):
            system[order] = system[order-1]*(degrees-order+1)
        scale = np.maximum(1,np.max(np.abs(system),axis=1))
        at_parameters[:,degrees] = np.linalg.lstsq(
            system/scale[:,None],moments[:,:,residue].T/scale[:,None],rcond=None)[0].T
    jets = np.fft.ifft(at_parameters,axis=0)
    if np.max(np.abs(jets.imag)) > 1e-7*max(1,float(np.max(np.abs(jets)))):
        raise ValueError("coefficient response is not real")
    jets = jets.real
    recovered = np.zeros_like(sampled)
    for h,t in enumerate(parameters):
        coefficients = np.polynomial.polynomial.polyval(t,jets)
        for order in range(4):
            recovered[h,order] = np.polynomial.polynomial.polyval(
                points,np.polynomial.polynomial.polyder(coefficients,order))
    if np.max(np.abs(recovered-sampled)) > 1e-7*max(1,float(np.max(np.abs(sampled)))):
        raise ValueError("inconsistent confluent response")
    direct = compute_determinant_samples(np.asarray([
        np.polynomial.polynomial.polyval(0.37,tensor+t*direction) for t in parameters]))
    predicted = np.asarray([
        np.polynomial.polynomial.polyval(0.37,np.polynomial.polynomial.polyval(t,jets))
        for t in parameters])
    if np.max(np.abs(predicted-direct)) > 1e-7*(1+float(np.max(np.abs(direct)))):
        raise ValueError("off-grid response validation failed")
    roots = extract_hidden_candidates(jets[0],imaginary_tolerance,interval=(0,3))
    solutions = recover_and_filter_solutions(
        tensor,roots,alpha,gamma,residual_threshold=residual_threshold)
    responses = [continue_condition_jets(jets[:response_order+1],x) for x in solutions[:,0]]
    log_values = np.array([value[0,1] for value in responses])
    selected = int(np.argmin(np.abs(log_values-np.log(base_maximum))))
    if len(log_values)>1:
        ordered = np.sort(np.exp(log_values))
        if ordered[-1]-ordered[-2] <= 1e-8*max(1,base_maximum):
            raise ValueError("nonunique maximizing branch")
    factorial = 1
    for n in range(2,response_order+1):
        factorial *= n
    result = float(factorial*responses[selected][response_order,1])
    if not np.isfinite(result):
        raise ValueError("nonfinite condition response")
    return result
SCICODE_GOLD_EOF
