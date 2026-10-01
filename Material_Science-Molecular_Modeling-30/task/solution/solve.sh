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


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numerical") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _symmetric(value, name, positive=False):
    result = _real(value, name)
    if (
        result.ndim != 2
        or result.shape[0] != result.shape[1]
        or not 1 <= len(result) <= 15
    ):
        raise ValueError(f"{name} must be a nonempty square matrix of order at most 15")
    if np.linalg.norm(result - result.T) > 1e-12 * max(1.0, np.linalg.norm(result)):
        raise ValueError(f"{name} must be symmetric")
    result = (result + result.T) * 0.5
    if positive and np.linalg.eigvalsh(result)[0] <= 0:
        raise ValueError(f"{name} must be positive definite")
    return result


def _vector(value, size, name):
    result = _real(value, name)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return result


def _temperature(value):
    result = _real(value, "temperature")
    if result.ndim != 0 or result < 0:
        raise ValueError("temperature must be a nonnegative scalar")
    return float(result)


def _basis(dimension):
    result = []
    for row in range(dimension):
        for col in range(row, dimension):
            entry = np.zeros((dimension, dimension))
            entry[row, col] = entry[col, row] = (
                1.0 if row == col else 1.0 / np.sqrt(2.0)
            )
            result.append(entry)
    return np.asarray(result)


def _pack(matrix, basis):
    return np.einsum("aij,ij->a", basis, matrix)


def _unpack(vector, basis):
    return np.einsum("a,aij->ij", vector, basis)


def quantum_matrix_response(auxiliary: np.ndarray, temperature: float) -> tuple:
    auxiliary = _symmetric(auxiliary, "auxiliary", positive=True)
    dimension = len(auxiliary)
    if dimension > 5:
        raise ValueError("the optical dimension must not exceed five")
    temperature = _temperature(temperature)
    values, modes = np.linalg.eigh(auxiliary)
    frequencies = np.sqrt(values)
    if temperature == 0.0:
        occupations = np.zeros(dimension)
    else:
        decay = np.exp(-frequencies / temperature)
        occupations = decay / (-np.expm1(-frequencies / temperature))
    variances = (occupations + 0.5) / frequencies
    covariance = (modes * variances) @ modes.T
    spectral = np.empty((dimension, dimension))
    for row, left in enumerate(frequencies):
        for col, right in enumerate(frequencies):
            low, high = min(left, right), max(left, right)
            gap = high - low
            quotient = 0.0
            if temperature > 0.0:
                if gap == 0.0:
                    quotient = (
                        -occupations[row] * (1.0 + occupations[row]) / temperature
                    )
                else:
                    quotient = (
                        np.exp(-low / temperature)
                        * np.expm1(-gap / temperature)
                        / (
                            gap
                            * (-np.expm1(-low / temperature))
                            * (-np.expm1(-high / temperature))
                        )
                    )
            spectral[row, col] = -(
                (1.0 + occupations[row] + occupations[col]) / (left + right) - quotient
            ) / (4.0 * left * right)
    basis = _basis(dimension)
    rotated = np.einsum("ia,kij,jb->kab", modes, basis, modes)
    pair_response = np.einsum("aij,ij,bij->ab", rotated, spectral, rotated)
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(pair_response)):
        raise ValueError("quantum response exceeds the numerical range")
    return covariance, (pair_response + pair_response.T) * 0.5

import numpy as np


def gaussian_vertex_averages(
    means: np.ndarray,
    variances: np.ndarray,
    reference_variances: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
) -> np.ndarray:
    means = _real(means, "means")
    if means.ndim != 1 or not 1 <= means.size <= 32:
        raise ValueError("means must contain one to 32 directions")
    size = means.size
    variances = _vector(variances, size, "variances")
    reference_variances = _vector(reference_variances, size, "reference_variances")
    cubic = _vector(cubic, size, "cubic")
    quartic = _vector(quartic, size, "quartic")
    sextic = _vector(sextic, size, "sextic")
    if (
        np.any(variances < 0)
        or np.any(reference_variances < 0)
        or np.any(quartic < 0)
        or np.any(sextic < 0)
    ):
        raise ValueError(
            "variances, quartic and sextic coefficients must be nonnegative"
        )
    shift = variances - reference_variances
    first = (
        cubic * (means**2 + shift) / 2
        + quartic * (means**3 + 3 * means * shift) / 6
        + sextic * (means**5 + 10 * means**3 * shift + 15 * means * shift**2) / 120
    )
    second = (
        cubic * means
        + quartic * (means**2 + shift) / 2
        + sextic * (means**4 + 6 * means**2 * shift + 3 * shift**2) / 24
    )
    third = cubic + quartic * means + sextic * (means**3 + 3 * means * shift) / 6
    fourth = quartic + sextic * (means**2 + shift) / 2
    result = np.column_stack((first, second, third, fourth))
    if not np.all(np.isfinite(result)):
        raise ValueError("vertex averages exceed the numerical range")
    return result

import numpy as np


def _model_data(
    bare, centroid, directions, cubic, quartic, sextic, reference_variances
):
    bare = _symmetric(bare, "bare", positive=True)
    dimension = len(bare)
    if dimension > 5:
        raise ValueError("the optical dimension must not exceed five")
    centroid = _vector(centroid, dimension, "centroid")
    directions = _real(directions, "directions")
    if (
        directions.ndim != 2
        or directions.shape[1] != dimension
        or not 1 <= len(directions) <= 32
    ):
        raise ValueError("directions must have shape (R,d), with one to 32 rows")
    size = len(directions)
    cubic = _vector(cubic, size, "cubic")
    quartic = _vector(quartic, size, "quartic")
    sextic = _vector(sextic, size, "sextic")
    reference_variances = _vector(reference_variances, size, "reference_variances")
    if np.any(quartic < 0) or np.any(sextic < 0) or np.any(reference_variances < 0):
        raise ValueError(
            "even coefficients and reference variances must be nonnegative"
        )
    return bare, centroid, directions, cubic, quartic, sextic, reference_variances


def _stationarity_data(
    auxiliary,
    temperature,
    bare,
    centroid,
    directions,
    cubic,
    quartic,
    sextic,
    reference_variances,
):
    covariance, pair_response = quantum_matrix_response(auxiliary, temperature)
    variances = np.einsum("ri,ij,rj->r", directions, covariance, directions)
    vertices = gaussian_vertex_averages(
        directions @ centroid, variances, reference_variances, cubic, quartic, sextic
    )
    target = bare + np.einsum("r,ri,rj->ij", vertices[:, 1], directions, directions)
    return target, covariance, pair_response, vertices


def relax_auxiliary_matrix(
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_variances: np.ndarray,
    temperature: float,
    tolerance: float = 1e-12,
    max_steps: int = 80,
) -> np.ndarray:
    data = _model_data(
        bare, centroid, directions, cubic, quartic, sextic, reference_variances
    )
    bare, centroid, directions, cubic, quartic, sextic, reference_variances = data
    temperature = _temperature(temperature)
    if (
        not np.isscalar(tolerance)
        or not np.isfinite(tolerance)
        or not 1e-14 <= tolerance <= 1e-6
    ):
        raise ValueError("tolerance must lie between 1e-14 and 1e-6")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 1
    ):
        raise ValueError("max_steps must be a positive integer")
    basis = _basis(len(bare))
    directional_pairs = np.einsum("aij,ri,rj->ra", basis, directions, directions)
    auxiliary = bare.copy()
    for _ in range(max_steps):
        target, covariance, pair_response, vertices = _stationarity_data(
            auxiliary, temperature, *data
        )
        residual = _pack(auxiliary - target, basis)
        norm = np.linalg.norm(residual)
        if norm <= tolerance * max(1.0, np.linalg.norm(auxiliary)):
            return auxiliary
        quartic_matrix = np.einsum(
            "r,ra,rb->ab", vertices[:, 3], directional_pairs, directional_pairs
        )
        jacobian = np.eye(len(basis)) - quartic_matrix @ pair_response
        try:
            direction = _unpack(np.linalg.solve(jacobian, -residual), basis)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("stationarity Jacobian is singular") from error
        fraction = 1.0
        for _ in range(45):
            trial = auxiliary + fraction * direction
            if np.linalg.eigvalsh(trial)[0] > 0:
                trial_target = _stationarity_data(trial, temperature, *data)[0]
                if np.linalg.norm(trial - trial_target) <= (1 - 1e-4 * fraction) * norm:
                    auxiliary = trial
                    break
            fraction *= 0.5
        else:
            raise RuntimeError("positive-definite stationarity iteration failed")
    raise RuntimeError("stationarity tolerance not reached")

import numpy as np


def build_correlated_ensemble(
    auxiliary: np.ndarray,
    temperature: float,
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_variances: np.ndarray,
) -> tuple:
    data = _model_data(
        bare, centroid, directions, cubic, quartic, sextic, reference_variances
    )
    bare, centroid, directions, cubic, quartic, sextic, reference_variances = data
    auxiliary = _symmetric(auxiliary, "auxiliary", positive=True)
    if auxiliary.shape != bare.shape:
        raise ValueError("auxiliary and bare dimensions must agree")
    covariance, _ = quantum_matrix_response(auxiliary, temperature)
    dimension = len(bare)
    root10 = np.sqrt(10.0)
    nodes = np.array(
        [
            -np.sqrt(5 + root10),
            -np.sqrt(5 - root10),
            0.0,
            np.sqrt(5 - root10),
            np.sqrt(5 + root10),
        ]
    )
    weights1d = np.array(
        [
            (7 - 2 * root10) / 60,
            (7 + 2 * root10) / 60,
            8 / 15,
            (7 + 2 * root10) / 60,
            (7 - 2 * root10) / 60,
        ]
    )
    indices = np.indices((5,) * dimension).reshape(dimension, -1).T
    displacements = nodes[indices] @ np.linalg.cholesky(covariance).T
    weights = np.prod(weights1d[indices], axis=1)
    positions = displacements + centroid
    projections = positions @ directions.T
    fixed = reference_variances
    radial = (
        cubic / 2 * (projections**2 - fixed)
        + quartic / 6 * (projections**3 - 3 * fixed * projections)
        + sextic
        / 120
        * (projections**5 - 10 * fixed * projections**3 + 15 * fixed**2 * projections)
    )
    residual_forces = (
        -positions @ bare.T - radial @ directions + displacements @ auxiliary.T
    )
    precision = np.linalg.solve(covariance, np.eye(dimension))
    precision_displacements = displacements @ precision.T
    if not np.all(np.isfinite(residual_forces)):
        raise ValueError("force ensemble exceeds the numerical range")
    return precision, precision_displacements, residual_forces, weights

import numpy as np


def _force_data(precision, precision_displacements, residual_forces, weights):
    precision = _symmetric(precision, "precision", positive=True)
    dimension = len(precision)
    if dimension > 5:
        raise ValueError("the optical dimension must not exceed five")
    vectors = _real(precision_displacements, "precision_displacements")
    forces = _real(residual_forces, "residual_forces")
    if (
        vectors.ndim != 2
        or vectors.shape[1] != dimension
        or len(vectors) < 1
        or forces.shape != vectors.shape
    ):
        raise ValueError("force and displacement arrays must have equal shape (K,d)")
    weights = _vector(weights, len(vectors), "weights")
    if np.any(weights < 0) or abs(float(weights.sum()) - 1) > 1e-12:
        raise ValueError("weights must be nonnegative and normalized")
    return precision, vectors, forces, weights


def apply_anharmonic_response(
    precision: np.ndarray,
    precision_displacements: np.ndarray,
    residual_forces: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    precision, vectors, forces, weights = _force_data(
        precision, precision_displacements, residual_forces, weights
    )
    dimension = len(precision)
    basis = _basis(dimension)
    state = _vector(state, dimension + len(basis), "state")
    centroid = state[:dimension]
    pair = _unpack(state[dimension:], basis)
    scalar2 = np.einsum("ki,ij,kj->k", vectors, pair, vectors) - np.trace(
        precision @ pair
    )
    mixed = np.einsum("ki,ij,kj->k", vectors, pair, forces)
    psforce = forces @ pair @ precision
    top = (
        -np.einsum(
            "k,ki->i",
            weights,
            forces * scalar2[:, None] + 2 * vectors * mixed[:, None] - 2 * psforce,
        )
        / 3
    )
    score2 = np.einsum("ki,kj->kij", vectors, vectors) - precision
    rf = forces @ centroid
    rv = vectors @ centroid
    pr = precision @ centroid
    cubic = (
        rf[:, None, None] * score2
        + rv[:, None, None]
        * (
            np.einsum("ki,kj->kij", forces, vectors)
            + np.einsum("ki,kj->kij", vectors, forces)
        )
        - np.einsum("i,kj->kij", pr, forces)
        - np.einsum("ki,j->kij", forces, pr)
    )
    score3 = vectors * scalar2[:, None] - 2 * vectors @ pair @ precision
    quartic = (
        np.einsum("ki,kj->kij", forces, score3)
        + np.einsum("ki,kj->kij", score3, forces)
        + 2
        * (
            mixed[:, None, None] * score2
            - np.einsum("ki,kj->kij", vectors, psforce)
            - np.einsum("ki,kj->kij", psforce, vectors)
        )
    )
    bottom = -np.einsum("k,kij->ij", weights, cubic / 3 + quartic / 4)
    result = np.concatenate((top, _pack(bottom, basis)))
    if not np.all(np.isfinite(result)):
        raise ValueError("anharmonic response exceeds the numerical range")
    return result

import numpy as np


def _response_data(auxiliary, pair_response, precision, vectors, forces, weights):
    auxiliary = _symmetric(auxiliary, "auxiliary", positive=True)
    precision, vectors, forces, weights = _force_data(
        precision, vectors, forces, weights
    )
    if auxiliary.shape != precision.shape:
        raise ValueError("auxiliary and precision dimensions must agree")
    size = len(auxiliary) * (len(auxiliary) + 1) // 2
    pair_response = _symmetric(pair_response, "pair_response")
    if (
        pair_response.shape != (size, size)
        or np.linalg.eigvalsh(pair_response)[-1] >= 0
    ):
        raise ValueError("pair_response must have shape (p,p) and be negative definite")
    return auxiliary, pair_response, precision, vectors, forces, weights


def apply_static_response(
    auxiliary: np.ndarray,
    pair_response: np.ndarray,
    precision: np.ndarray,
    precision_displacements: np.ndarray,
    residual_forces: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    auxiliary, pair_response, precision, vectors, forces, weights = _response_data(
        auxiliary,
        pair_response,
        precision,
        precision_displacements,
        residual_forces,
        weights,
    )
    dimension = len(auxiliary)
    state = _vector(state, dimension + len(pair_response), "state")
    harmonic = np.concatenate(
        (
            auxiliary @ state[:dimension],
            -np.linalg.solve(pair_response, state[dimension:]),
        )
    )
    return harmonic + apply_anharmonic_response(
        precision, vectors, forces, weights, state
    )

import numpy as np


def relaxed_positional_hessian(
    auxiliary: np.ndarray,
    pair_response: np.ndarray,
    precision: np.ndarray,
    precision_displacements: np.ndarray,
    residual_forces: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    data = _response_data(
        auxiliary,
        pair_response,
        precision,
        precision_displacements,
        residual_forces,
        weights,
    )
    auxiliary, pair_response, precision, vectors, forces, weights = data
    dimension = len(auxiliary)
    total = dimension + len(pair_response)
    columns = [apply_static_response(*data, column) for column in np.eye(total)]
    operator = np.column_stack(columns)
    asymmetry = np.linalg.norm(operator - operator.T) / max(
        1.0, np.linalg.norm(operator)
    )
    if asymmetry > 1e-10:
        raise RuntimeError("static response violates reciprocity")
    operator = (operator + operator.T) * 0.5
    coupling = operator[:dimension, dimension:]
    pair_block = operator[dimension:, dimension:]
    if np.linalg.eigvalsh(pair_block)[0] <= 0:
        raise RuntimeError("the covariance sector is not locally stable")
    try:
        response = np.linalg.solve(pair_block, coupling.T)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("pair response is singular") from error
    residual = np.linalg.norm(pair_block @ response - coupling.T) / max(
        1.0, np.linalg.norm(coupling)
    )
    if not np.isfinite(residual) or residual > 1e-10:
        raise RuntimeError("pair response tolerance not reached")
    hessian = auxiliary - coupling @ response
    if not np.all(np.isfinite(hessian)):
        raise RuntimeError("positional Hessian is not finite")
    return (hessian + hessian.T) * 0.5

import numpy as np
from decimal import Context, Decimal, localcontext


def _scalar_direction(value, name):
    value = _real(value, name)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(value)


def quantum_response_tangent(
    auxiliary: np.ndarray,
    temperature: float,
    auxiliary_direction: np.ndarray,
    temperature_direction: float,
) -> tuple:
    auxiliary = _symmetric(auxiliary, "auxiliary", positive=True)
    direction = _symmetric(auxiliary_direction, "auxiliary_direction")
    if len(auxiliary) > 5 or direction.shape != auxiliary.shape:
        raise ValueError(
            "auxiliary dimension must be one to five and directions must match"
        )
    temperature = _temperature(temperature)
    rate = _scalar_direction(temperature_direction, "temperature_direction")
    if temperature == 0 and rate < 0:
        raise ValueError(
            "a zero-temperature directional path cannot enter negative temperature"
        )
    values, vectors = np.linalg.eigh(auxiliary)
    basis = _basis(len(auxiliary))
    rotated_basis = np.einsum("ji,ajk,kl->ail", vectors, basis, vectors)
    rotated_direction = vectors.T @ direction @ vectors
    with localcontext(Context(prec=80)):
        thermal = Decimal.from_float(temperature)
        nodes = [Decimal.from_float(float(value)) for value in values]

        def _scalar_values(x):
            omega = x.sqrt()
            if temperature == 0:
                occupation = Decimal(0)
            else:
                ratio = omega / thermal
                decay = (-ratio).exp() if ratio < 100000 else Decimal(0)
                occupation = decay / (1 - decay)
            factor = 1 + 2 * occupation
            product = occupation * (1 + occupation)
            value = factor / (2 * omega)
            first_value = -factor / (4 * omega**3)
            second_value = 3 * factor / (8 * omega**5)
            thermal_value = Decimal(0)
            thermal_first_value = Decimal(0)
            if temperature > 0:
                first_value -= product / (2 * thermal * omega**2)
                second_value += 3 * product / (4 * thermal * omega**4)
                second_value += product * factor / (4 * thermal**2 * omega**3)
                thermal_value = product / thermal**2
                thermal_first_value = -product * factor / (2 * thermal**3 * omega)
            return value, first_value, second_value, thermal_value, thermal_first_value

        scalar_values = {x: _scalar_values(x) for x in nodes}

        def _first(x, y, thermal_only=False):
            if x == y:
                return scalar_values[x][4 if thermal_only else 1]
            index = 3 if thermal_only else 0
            return (scalar_values[y][index] - scalar_values[x][index]) / (y - x)

        def _second(x, y, z):
            x, y, z = sorted((x, y, z))
            if x == z:
                return scalar_values[x][2] / 2
            return (_first(y, z) - _first(x, y)) / (z - x)

        dimension = len(values)
        first = np.empty((dimension, dimension))
        thermal_first = np.empty_like(first)
        second = np.empty((dimension, dimension, dimension))
        thermal_values = np.asarray([float(scalar_values[x][3]) for x in nodes])
        for i in range(dimension):
            for j in range(dimension):
                first[i, j] = float(_first(nodes[i], nodes[j]))
                thermal_first[i, j] = float(_first(nodes[i], nodes[j], True))
                for k in range(dimension):
                    second[i, k, j] = float(_second(nodes[i], nodes[k], nodes[j]))
    covariance_direction = (
        vectors
        @ (first * rotated_direction + rate * np.diag(thermal_values))
        @ vectors.T
    )
    columns = []
    for entry in rotated_basis:
        action = np.einsum("ikj,ik,kj->ij", second, rotated_direction, entry)
        action += np.einsum("ikj,ik,kj->ij", second, entry, rotated_direction)
        action += rate * thermal_first * entry
        columns.append(0.5 * np.einsum("aij,ij->a", rotated_basis, action))
    pair_direction = np.column_stack(columns)
    if not np.all(np.isfinite(covariance_direction)) or not np.all(
        np.isfinite(pair_direction)
    ):
        raise RuntimeError("quantum response tangent is not finite")
    return (covariance_direction + covariance_direction.T) * 0.5, (
        pair_direction + pair_direction.T
    ) * 0.5

import numpy as np


def stationary_hessian_tangent(
    auxiliary: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_variances: np.ndarray,
    temperature: float,
    cubic_direction: np.ndarray,
    temperature_direction: float,
) -> np.ndarray:
    data = _model_data(
        auxiliary, centroid, directions, cubic, quartic, sextic, reference_variances
    )
    auxiliary, centroid, directions, cubic, quartic, sextic, reference_variances = data
    cubic_direction = _vector(cubic_direction, len(cubic), "cubic_direction")
    temperature = _temperature(temperature)
    rate = _scalar_direction(temperature_direction, "temperature_direction")
    if temperature == 0 and rate < 0:
        raise ValueError(
            "a zero-temperature directional path cannot enter negative temperature"
        )
    covariance, pair_response = quantum_matrix_response(auxiliary, temperature)
    basis = _basis(len(auxiliary))
    means = directions @ centroid
    variances = np.einsum("ri,ij,rj->r", directions, covariance, directions)
    vertices = gaussian_vertex_averages(
        means, variances, reference_variances, cubic, quartic, sextic
    )
    pairs = np.einsum("aij,ri,rj->ra", basis, directions, directions)
    coupling = np.einsum("r,ri,ra->ia", vertices[:, 2], directions, pairs)
    quartic_block = np.einsum("r,ra,rb->ab", vertices[:, 3], pairs, pairs)
    inverse_pair = np.linalg.solve(pair_response, np.eye(len(pair_response)))
    pair_block = quartic_block - inverse_pair
    if np.linalg.eigvalsh(pair_block)[0] <= 0:
        raise RuntimeError("the covariance sector is not locally stable")
    thermal_covariance, _ = quantum_response_tangent(
        auxiliary, temperature, np.zeros_like(auxiliary), rate
    )
    direct = np.einsum("r,r,ra->a", cubic_direction, means, pairs)
    direct += 0.5 * quartic_block @ _pack(thermal_covariance, basis)
    jacobian = np.eye(len(pair_response)) - quartic_block @ pair_response
    try:
        packed_direction = np.linalg.solve(jacobian, direct)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("implicit stationarity response is singular") from error
    if np.linalg.norm(jacobian @ packed_direction - direct) > 1e-10 * max(
        1.0, np.linalg.norm(direct)
    ):
        raise RuntimeError("implicit stationarity response did not converge")
    auxiliary_direction = _unpack(packed_direction, basis)
    covariance_direction, pair_direction = quantum_response_tangent(
        auxiliary, temperature, auxiliary_direction, rate
    )
    variance_direction = np.einsum(
        "ri,ij,rj->r", directions, covariance_direction, directions
    )
    cubic_vertex_direction = cubic_direction + 0.5 * sextic * means * variance_direction
    quartic_vertex_direction = 0.5 * sextic * variance_direction
    coupling_direction = np.einsum(
        "r,ri,ra->ia", cubic_vertex_direction, directions, pairs
    )
    quartic_direction = np.einsum("r,ra,rb->ab", quartic_vertex_direction, pairs, pairs)
    block_direction = quartic_direction + inverse_pair @ pair_direction @ inverse_pair
    response = np.linalg.solve(pair_block, coupling.T)
    hessian_direction = (
        auxiliary_direction
        - coupling_direction @ response
        - response.T @ coupling_direction.T
        + response.T @ block_direction @ response
    )
    result = np.stack(
        [
            auxiliary_direction,
            covariance_direction,
            (hessian_direction + hessian_direction.T) * 0.5,
        ]
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("stationary Hessian tangent is not finite")
    return result

import numpy as np
from scipy.optimize import brentq


def _critical_state(
    coupling,
    bare,
    centroid,
    directions,
    cubic,
    quartic,
    sextic,
    reference_variances,
    temperature,
):
    scaled = coupling * cubic
    auxiliary = relax_auxiliary_matrix(
        bare,
        centroid,
        directions,
        scaled,
        quartic,
        sextic,
        reference_variances,
        temperature,
        tolerance=2e-13,
    )
    covariance, pair_response = quantum_matrix_response(auxiliary, temperature)
    ensemble = build_correlated_ensemble(
        auxiliary,
        temperature,
        bare,
        centroid,
        directions,
        scaled,
        quartic,
        sextic,
        reference_variances,
    )
    hessian = relaxed_positional_hessian(auxiliary, pair_response, *ensemble)
    return auxiliary, covariance, hessian


def _locate_instability(
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_temperature: float,
    temperature: float,
    bracket: np.ndarray,
) -> float:
    bare = _symmetric(bare, "bare", positive=True)
    reference_covariance, _ = quantum_matrix_response(
        bare, reference_temperature
    )
    directions = _real(directions, "directions")
    if directions.ndim != 2 or directions.shape[1] != len(bare):
        raise ValueError("directions must have shape (R,d)")
    fixed = np.einsum("ri,ij,rj->r", directions, reference_covariance, directions)
    data = _model_data(bare, centroid, directions, cubic, quartic, sextic, fixed)
    temperature = _temperature(temperature)
    bracket = _vector(bracket, 2, "bracket")
    if bracket[0] < 0 or bracket[1] <= bracket[0]:
        raise ValueError("bracket must be increasing and nonnegative")

    def _curvature(coupling):
        hessian = _critical_state(coupling, *data, temperature)[2]
        return float(np.linalg.eigvalsh(hessian)[0])

    endpoint_hessians = [
        _critical_state(value, *data, temperature)[2] for value in bracket
    ]
    left, right = [float(np.linalg.eigvalsh(matrix)[0]) for matrix in endpoint_hessians]
    endpoint_tolerances = [
        64 * np.finfo(float).eps * max(1.0, np.linalg.norm(matrix, 2))
        for matrix in endpoint_hessians
    ]
    if left < -endpoint_tolerances[0] or right > endpoint_tolerances[1]:
        raise ValueError("bracket must enclose a stable-to-unstable crossing")
    if abs(left) <= endpoint_tolerances[0]:
        return float(bracket[0])
    if abs(right) <= endpoint_tolerances[1]:
        return float(bracket[1])
    try:
        result = brentq(
            _curvature, bracket[0], bracket[1], xtol=5e-12, rtol=2e-14, maxiter=100
        )
    except RuntimeError as error:
        raise RuntimeError("critical coupling did not converge") from error
    return float(result)


def critical_temperature_slope(
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_temperature: float,
    temperature: float,
    bracket: np.ndarray,
) -> float:
    critical = _locate_instability(
        bare,
        centroid,
        directions,
        cubic,
        quartic,
        sextic,
        reference_temperature,
        temperature,
        bracket,
    )
    reference_covariance, _ = quantum_matrix_response(
        bare, reference_temperature
    )
    fixed = np.einsum("ri,ij,rj->r", directions, reference_covariance, directions)
    data = _model_data(bare, centroid, directions, cubic, quartic, sextic, fixed)
    auxiliary, _, hessian = _critical_state(critical, *data, temperature)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    if len(eigenvalues) > 1 and eigenvalues[1] - eigenvalues[0] <= 1e-10 * max(
        1.0, np.linalg.norm(hessian, 2)
    ):
        raise RuntimeError("critical positional eigenvalue is not numerically simple")
    args = (
        auxiliary,
        data[1],
        data[2],
        critical * data[3],
        data[4],
        data[5],
        fixed,
        temperature,
    )
    coupling_response = stationary_hessian_tangent(*args, data[3], 0.0)[2]
    thermal_response = stationary_hessian_tangent(
        *args, np.zeros_like(data[3]), 1.0
    )[2]
    mode = eigenvectors[:, 0]
    coupling_partial = float(mode @ coupling_response @ mode)
    thermal_partial = float(mode @ thermal_response @ mode)
    if abs(coupling_partial) <= 1e-12 * max(1.0, np.linalg.norm(coupling_response, 2)):
        raise RuntimeError("critical coupling derivative cannot be resolved")
    result = -thermal_partial / coupling_partial
    if not np.isfinite(result):
        raise RuntimeError("critical thermal slope is not finite")
    return float(result)
SCICODE_GOLD_EOF
