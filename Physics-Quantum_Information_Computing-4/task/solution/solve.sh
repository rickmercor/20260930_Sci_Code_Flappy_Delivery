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


def _validate_probability_table(table: "np.ndarray", d_local: int, name: str) -> "np.ndarray":
    """Return the table as a float array after checking shape, finiteness and normalisation."""
    array = np.asarray(table)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = array.astype(float)
    if array.shape != (d_local, d_local):
        raise ValueError(f"{name} must have shape ({d_local}, {d_local})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite entries")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must have nonnegative entries")
    if abs(float(array.sum()) - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to one within 1e-9")
    return array


def _hermitian_square_root(matrix: "np.ndarray") -> "np.ndarray":
    """Return the positive square root of a Hermitian positive semidefinite operator."""
    spectrum, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    return (vectors * np.sqrt(np.clip(spectrum, 0.0, None))) @ vectors.conj().T


def build_protocol_data(d_local: int, p_z: "np.ndarray", p_x: "np.ndarray", epsilon: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    if isinstance(d_local, bool) or not isinstance(d_local, (int, np.integer)):
        raise ValueError("d_local must be an integer")
    d_local = int(d_local)
    if d_local < 2:
        raise ValueError("d_local must be at least 2")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, np.integer, np.floating)):
        raise ValueError("epsilon must be a real scalar")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must satisfy 0 < epsilon <= 1")

    table_z = _validate_probability_table(p_z, d_local, "p_z")
    table_x = _validate_probability_table(p_x, d_local, "p_x")

    omega = np.exp(2j * np.pi / d_local)
    powers = np.outer(np.arange(d_local), np.arange(d_local))
    fourier = omega ** powers / np.sqrt(d_local)      # column j is |x_j>, row index is l
    computational = np.eye(d_local, dtype=complex)

    settings = ((computational, computational, table_z), (fourier, fourier.conj(), table_x))
    operators = []
    moments = []
    for basis_a, basis_b, table in settings:
        for a in range(d_local):
            for b in range(d_local):
                if a == d_local - 1 and b == d_local - 1:
                    continue
                vec_a = basis_a[:, a].reshape(-1, 1)
                vec_b = basis_b[:, b].reshape(-1, 1)
                operators.append(np.kron(vec_a @ vec_a.conj().T, vec_b @ vec_b.conj().T))
                moments.append(table[a, b])

    identity_bob = np.eye(d_local, dtype=complex)
    detector = []
    for a in range(d_local):
        element = (epsilon / d_local) * np.eye(d_local, dtype=complex)
        element[a, a] += 1.0 - epsilon
        detector.append(np.kron(_hermitian_square_root(element), identity_bob))

    return (np.array(operators, dtype=complex), np.array(moments, dtype=float),
            np.array(detector, dtype=complex))

import numpy as np


def _validate_detector_ops(operator: "np.ndarray", detector_ops: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Return the operator and detector stack as validated complex arrays."""
    matrix = np.asarray(operator, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("operator must be a square 2D array with at least one row")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("operator must have finite entries")
    stack = np.asarray(detector_ops, dtype=complex)
    dim = matrix.shape[0]
    if stack.ndim != 3 or stack.shape[0] < 1 or stack.shape[1:] != (dim, dim):
        raise ValueError("detector_ops must have shape (m, d, d) with m >= 1 matching operator")
    if not np.all(np.isfinite(stack)):
        raise ValueError("detector_ops must have finite entries")
    return matrix, stack


def key_map_blocks(operator: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    matrix, stack = _validate_detector_ops(operator, detector_ops)
    return np.array([element @ matrix @ element.conj().T for element in stack], dtype=complex)

import numpy as np


def _validate_density_operator(rho: "np.ndarray") -> "np.ndarray":
    """Return rho as a complex array after checking it is a valid density operator."""
    matrix = np.asarray(rho, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("rho must be a square 2D array with at least one row")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rho must have finite entries")
    if not np.allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-9):
        raise ValueError("rho must be Hermitian within 1e-9")
    matrix = 0.5 * (matrix + matrix.conj().T)
    if float(np.linalg.eigvalsh(matrix)[0]) < -1e-9:
        raise ValueError("rho must be positive semidefinite within 1e-9")
    if abs(float(np.trace(matrix).real) - 1.0) > 1e-9:
        raise ValueError("rho must have unit trace within 1e-9")
    return matrix


def _von_neumann_entropy(matrix: "np.ndarray") -> float:
    """Return the von Neumann entropy in nats, discarding non-positive eigenvalues."""
    spectrum = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
    spectrum = spectrum[spectrum > 0.0]
    return float(-np.sum(spectrum * np.log(spectrum)))


def entropy_production(rho: "np.ndarray", detector_ops: "np.ndarray") -> float:
    matrix = _validate_density_operator(rho)
    blocks = key_map_blocks(matrix, detector_ops)
    recorded = sum(_von_neumann_entropy(block) for block in blocks)
    return float(recorded - _von_neumann_entropy(matrix))

import numpy as np


def _positive_definite_logarithm(matrix: "np.ndarray") -> "np.ndarray":
    """Return the matrix logarithm of a Hermitian operator with eigenvalues above 1e-12."""
    spectrum, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    if float(spectrum[0]) <= 1e-12:
        raise ValueError("operator must have smallest eigenvalue above 1e-12")
    return (vectors * np.log(spectrum)) @ vectors.conj().T


def moving_reference_logarithm(rho: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    matrix = _validate_density_operator(rho)
    blocks = key_map_blocks(matrix, detector_ops)
    stack = np.asarray(detector_ops, dtype=complex)
    reference = np.zeros_like(matrix)
    for element, block in zip(stack, blocks):
        reference = reference + element.conj().T @ _positive_definite_logarithm(block) @ element
    return 0.5 * (reference + reference.conj().T)

import numpy as np


def objective_gradient(rho: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    matrix = _validate_density_operator(rho)
    if float(np.linalg.eigvalsh(matrix)[0]) <= 1e-12:
        raise ValueError("rho must have smallest eigenvalue above 1e-12")
    reference = moving_reference_logarithm(matrix, detector_ops)
    gradient = _positive_definite_logarithm(matrix) - reference
    return 0.5 * (gradient + gradient.conj().T)

import numpy as np


def _validate_exponential_family(log_reference: "np.ndarray", constraint_ops: "np.ndarray",
                                 weights: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return validated copies of the reference logarithm, observable stack and weights."""
    reference = np.asarray(log_reference, dtype=complex)
    if reference.ndim != 2 or reference.shape[0] != reference.shape[1] or reference.shape[0] < 1:
        raise ValueError("log_reference must be a square 2D array with at least one row")
    if not np.all(np.isfinite(reference)):
        raise ValueError("log_reference must have finite entries")
    if not np.allclose(reference, reference.conj().T, rtol=0.0, atol=1e-9):
        raise ValueError("log_reference must be Hermitian within 1e-9")
    dim = reference.shape[0]

    operators = np.asarray(constraint_ops, dtype=complex)
    if operators.ndim != 3 or operators.shape[1:] != (dim, dim):
        raise ValueError("constraint_ops must have shape (n, d, d) matching log_reference")
    if operators.size and not np.all(np.isfinite(operators)):
        raise ValueError("constraint_ops must have finite entries")
    for index in range(operators.shape[0]):
        block = operators[index]
        if not np.allclose(block, block.conj().T, rtol=0.0, atol=1e-9):
            raise ValueError("every constraint observable must be Hermitian within 1e-9")

    multipliers = np.asarray(weights)
    if np.iscomplexobj(multipliers):
        raise ValueError("weights must be real")
    multipliers = multipliers.astype(float)
    if multipliers.shape != (operators.shape[0], ):
        raise ValueError("weights must have shape (n, ) matching constraint_ops")
    if multipliers.size and not np.all(np.isfinite(multipliers)):
        raise ValueError("weights must have finite entries")

    return 0.5 * (reference + reference.conj().T), operators, multipliers


def _effective_hamiltonian(log_reference: "np.ndarray", constraint_ops: "np.ndarray",
                           weights: "np.ndarray") -> "np.ndarray":
    """Return the Hermitian generator of the exponential family."""
    shift = np.tensordot(weights, constraint_ops, axes=(0, 0)) if weights.size else 0.0
    generator = log_reference - shift
    return 0.5 * (generator + generator.conj().T)


def gibbs_state(log_reference: "np.ndarray", constraint_ops: "np.ndarray", weights: "np.ndarray") -> "tuple[np.ndarray, float]":
    reference, operators, multipliers = _validate_exponential_family(
        log_reference, constraint_ops, weights)
    generator = _effective_hamiltonian(reference, operators, multipliers)
    spectrum, vectors = np.linalg.eigh(generator)
    peak = float(spectrum[-1])
    shifted = np.exp(spectrum - peak)
    total = float(shifted.sum())
    populations = shifted / total
    state = (vectors * populations) @ vectors.conj().T
    state = 0.5 * (state + state.conj().T)
    return state, peak + float(np.log(total))

import numpy as np


def _validate_moment_vector(moments: "np.ndarray", count: int) -> "np.ndarray":
    """Return the observed-moment vector as a validated float array."""
    values = np.asarray(moments)
    if np.iscomplexobj(values):
        raise ValueError("moments must be real")
    values = values.astype(float)
    if values.shape != (count, ):
        raise ValueError("moments must have shape (n, ) matching constraint_ops")
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("moments must have finite entries")
    return values


def multiplier_objective(log_reference: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", weights: "np.ndarray") -> "tuple[float, np.ndarray]":
    reference, operators, multipliers = _validate_exponential_family(
        log_reference, constraint_ops, weights)
    observed = _validate_moment_vector(moments, operators.shape[0])

    state, log_partition = gibbs_state(reference, operators, multipliers)
    if operators.shape[0]:
        expectations = np.einsum('iab,ba->i', operators, state).real
    else:
        expectations = np.zeros(0, dtype=float)

    value = float(log_partition + float(multipliers @ observed))
    return value, observed - expectations

import numpy as np


def _logarithmic_mean_kernel(exponents: "np.ndarray", populations: "np.ndarray") -> "np.ndarray":
    """Return the matrix of logarithmic means of the populations, keyed by their exponents."""
    gaps = exponents[:, None] - exponents[None, :]
    safe = np.where(gaps == 0.0, 1.0, gaps)
    near = populations[None, :] * np.expm1(np.clip(gaps, -50.0, 50.0)) / safe
    near = np.where(gaps == 0.0, np.broadcast_to(populations[None, :], gaps.shape), near)
    far = (populations[:, None] - populations[None, :]) / safe
    return np.where(np.abs(gaps) <= 1.0, near, far)


def bkm_susceptibility(log_reference: "np.ndarray", constraint_ops: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    reference, operators, multipliers = _validate_exponential_family(
        log_reference, constraint_ops, weights)
    count = operators.shape[0]
    if count == 0:
        return np.zeros((0, 0), dtype=float)

    generator = _effective_hamiltonian(reference, operators, multipliers)
    exponents, vectors = np.linalg.eigh(generator)
    populations = np.exp(exponents - float(exponents[-1]))
    populations = populations / populations.sum()

    rotated = np.einsum('pa,ipq,qb->iab', vectors.conj(), operators, vectors)
    expectations = np.einsum('a,iaa->i', populations, rotated).real
    identity = np.eye(reference.shape[0], dtype=complex)
    centered = rotated - expectations[:, None, None] * identity[None, :, :]

    kernel = _logarithmic_mean_kernel(exponents, populations)
    curvature = np.einsum('ab,iab,jba->ij', kernel, centered, centered).real
    return 0.5 * (curvature + curvature.T)

import numpy as np


def solve_multipliers(log_reference: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", initial_weights: "np.ndarray", tol: float, max_iter: int) -> "tuple[np.ndarray, float]":
    reference, operators, weights = _validate_exponential_family(
        log_reference, constraint_ops, initial_weights)
    observed = _validate_moment_vector(moments, operators.shape[0])
    if isinstance(tol, bool) or not isinstance(tol, (int, float, np.integer, np.floating)):
        raise ValueError("tol must be a real scalar")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be positive and finite")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError("max_iter must be an integer")
    if int(max_iter) < 0:
        raise ValueError("max_iter must be nonnegative")

    threshold = float(tol)
    budget = int(max_iter)
    value, gradient = multiplier_objective(reference, operators, observed, weights)

    for _ in range(budget):
        residual = float(np.abs(gradient).max()) if gradient.size else 0.0
        if residual <= threshold:
            break
        curvature = bkm_susceptibility(reference, operators, weights)
        try:
            displacement = -np.linalg.solve(curvature, gradient)
        except np.linalg.LinAlgError:
            displacement = -np.linalg.lstsq(curvature, gradient, rcond=None)[0]
        predicted = float(gradient @ displacement)
        scale = 1.0
        trial_value, trial_gradient = multiplier_objective(
            reference, operators, observed, weights + scale * displacement)
        for _ in range(60):
            if trial_value <= value + 1e-4 * scale * predicted:
                break
            scale *= 0.5
            trial_value, trial_gradient = multiplier_objective(
                reference, operators, observed, weights + scale * displacement)
        weights = weights + scale * displacement
        value, gradient = trial_value, trial_gradient

    residual = float(np.abs(gradient).max()) if gradient.size else 0.0
    return weights, residual

import numpy as np


def outer_iteration(rho: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", detector_ops: "np.ndarray", tol: float, max_iter: int) -> "tuple[np.ndarray, np.ndarray]":
    state = _validate_density_operator(rho)
    if float(np.linalg.eigvalsh(state)[0]) <= 1e-12:
        raise ValueError("rho must have smallest eigenvalue above 1e-12")

    reference = moving_reference_logarithm(state, detector_ops)
    count = np.shape(constraint_ops)[0] if np.ndim(constraint_ops) == 3 else 0
    _, operators, _ = _validate_exponential_family(reference, constraint_ops, np.zeros(count))
    start = np.zeros(operators.shape[0], dtype=float)

    weights, _ = solve_multipliers(reference, operators, moments, start, tol, max_iter)
    next_state, _ = gibbs_state(reference, operators, weights)
    return next_state, weights

import numpy as np


def entropy_certificate(rho: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", detector_ops: "np.ndarray", weights: "np.ndarray") -> float:
    gradient = objective_gradient(rho, detector_ops)
    _, operators, multipliers = _validate_exponential_family(gradient, constraint_ops, weights)
    observed = _validate_moment_vector(moments, operators.shape[0])

    shift = np.tensordot(multipliers, operators, axes=(0, 0)) if multipliers.size else 0.0
    pencil = gradient + shift
    pencil = 0.5 * (pencil + pencil.conj().T)
    return float(np.linalg.eigvalsh(pencil)[0] - float(multipliers @ observed))

import numpy as np


def certified_entropy_bracket(d_local: int, p_z: "np.ndarray", p_x: "np.ndarray", epsilon: float, n_outer: int, tol: float, max_iter: int) -> "tuple[float, float]":
    operators, moments, detector = build_protocol_data(d_local, p_z, p_x, epsilon)
    if isinstance(n_outer, bool) or not isinstance(n_outer, (int, np.integer)):
        raise ValueError("n_outer must be an integer")
    if int(n_outer) < 1:
        raise ValueError("n_outer must be at least 1")

    dimension = int(d_local) ** 2
    state = np.eye(dimension, dtype=complex) / dimension
    bound = 0.0
    for _ in range(int(n_outer)):
        linearisation = state
        state, weights = outer_iteration(
            linearisation, operators, moments, detector, tol, max_iter)
        bound = entropy_certificate(
            linearisation, operators, moments, detector, weights)

    candidate = entropy_production(state, detector)
    return candidate, bound
SCICODE_GOLD_EOF
