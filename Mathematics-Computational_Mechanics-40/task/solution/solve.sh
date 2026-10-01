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


def select_pod_basis(snapshots: np.ndarray, energy_tolerance: float) -> np.ndarray:
    """Reference implementation."""
    snapshots = np.asarray(snapshots, dtype=float)
    if snapshots.ndim != 2 or 0 in snapshots.shape:
        raise ValueError("snapshots must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(snapshots)):
        raise ValueError("snapshots must contain finite values")
    if not np.isscalar(energy_tolerance):
        raise ValueError("energy_tolerance must be a scalar")
    energy_tolerance = float(energy_tolerance)
    if not 0.0 < energy_tolerance < 1.0:
        raise ValueError("energy_tolerance must lie in (0, 1)")

    left_vectors, singular_values, _ = np.linalg.svd(snapshots, full_matrices=False)
    total_energy = float(singular_values @ singular_values)
    if total_energy == 0.0:
        raise ValueError("snapshots must have positive energy")
    discarded = 1.0 - np.cumsum(singular_values**2) / total_energy
    retained = int(np.flatnonzero(discarded < energy_tolerance)[0] + 1)
    basis = left_vectors[:, :retained].copy()
    for column in range(retained):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis

import numpy as np


def assemble_quadratic_operator(n_state: int, terms: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    if not isinstance(n_state, (int, np.integer)) or int(n_state) < 1:
        raise ValueError("n_state must be a positive integer")
    n_state = int(n_state)
    terms = np.asarray(terms, dtype=float)
    if terms.ndim != 2 or terms.shape[1] != 4:
        raise ValueError("terms must have shape (n_terms, 4)")
    if not np.all(np.isfinite(terms)):
        raise ValueError("terms must contain finite values")
    indices = terms[:, :3]
    if not np.all(indices == np.floor(indices)):
        raise ValueError("term indices must be integers")
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= n_state):
        raise ValueError("term indices are outside the state range")

    quadratic = np.zeros((n_state, n_state * n_state), dtype=float)
    for (output, first, second), coefficient in zip(indices, terms[:, 3]):
        quadratic[output, first * n_state + second] += coefficient
    return quadratic

import numpy as np


def assemble_residual_polynomial(
    basis: np.ndarray,
    constant: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    input_vector: np.ndarray,
    bilinear: np.ndarray,
    previous_state: np.ndarray,
    current_input: float,
    previous_input: float,
    time_step: float,
) -> dict:
    """Reference implementation."""
    basis = np.asarray(basis, dtype=float)
    constant = np.asarray(constant, dtype=float)
    linear = np.asarray(linear, dtype=float)
    quadratic = np.asarray(quadratic, dtype=float)
    input_vector = np.asarray(input_vector, dtype=float)
    bilinear = np.asarray(bilinear, dtype=float)
    previous_state = np.asarray(previous_state, dtype=float)
    if basis.ndim != 2 or 0 in basis.shape:
        raise ValueError("basis must be a nonempty matrix")
    n_full, n_reduced = basis.shape
    if constant.shape != (n_full,) or input_vector.shape != (n_full,):
        raise ValueError("vector operators have incompatible shapes")
    if linear.shape != (n_full, n_full) or bilinear.shape != (n_full, n_full):
        raise ValueError("matrix operators have incompatible shapes")
    if quadratic.shape != (n_full, n_full * n_full):
        raise ValueError("quadratic operator has an incompatible shape")
    if previous_state.shape != (n_reduced,):
        raise ValueError("previous_state has an incompatible shape")
    arrays = (
        basis,
        constant,
        linear,
        quadratic,
        input_vector,
        bilinear,
        previous_state,
    )
    scalars = np.asarray([current_input, previous_input, time_step], dtype=float)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all arrays must contain finite values")
    if not np.all(np.isfinite(scalars)) or float(time_step) <= 0.0:
        raise ValueError("inputs must be finite and time_step must be positive")

    current_input = float(current_input)
    previous_input = float(previous_input)
    time_step = float(time_step)
    previous_full = basis @ previous_state
    previous_dynamics = (
        constant
        + linear @ previous_full
        + quadratic @ np.kron(previous_full, previous_full)
        + input_vector * previous_input
        + bilinear @ (previous_input * previous_full)
    )
    residual_constant = -previous_full - 0.5 * time_step * (
        constant + input_vector * current_input + previous_dynamics
    )
    residual_linear = basis - 0.5 * time_step * (
        linear @ basis + current_input * (bilinear @ basis)
    )
    residual_quadratic = (
        -0.5 * time_step * (quadratic @ np.kron(basis, basis))
    ).reshape(n_full, n_reduced, n_reduced)
    return {
        "constant": residual_constant,
        "linear": residual_linear,
        "quadratic": residual_quadratic,
    }

import numpy as np



def precompute_hrf_tensors(
    basis: np.ndarray, residual_polynomial: dict
) -> dict:
    """Reference implementation."""
    _TENSOR_KEYS = {
        "galerkin_constant",
        "galerkin_linear",
        "galerkin_quadratic",
        "lspg_gradient_constant",
        "lspg_gradient_linear",
        "lspg_gradient_quadratic",
        "lspg_gradient_cubic",
        "lspg_normal_constant",
        "lspg_normal_linear",
        "lspg_normal_quadratic",
    }

    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2 or 0 in basis.shape:
        raise ValueError("basis must be a nonempty matrix")
    if not isinstance(residual_polynomial, dict):
        raise ValueError("residual_polynomial must be a dictionary")
    if set(residual_polynomial) != {"constant", "linear", "quadratic"}:
        raise ValueError("residual_polynomial has invalid keys")
    n_full, n_reduced = basis.shape
    constant = np.asarray(residual_polynomial["constant"], dtype=float)
    linear = np.asarray(residual_polynomial["linear"], dtype=float)
    quadratic = np.asarray(residual_polynomial["quadratic"], dtype=float)
    if constant.shape != (n_full,):
        raise ValueError("constant coefficient has an incompatible shape")
    if linear.shape != (n_full, n_reduced):
        raise ValueError("linear coefficient has an incompatible shape")
    if quadratic.shape != (n_full, n_reduced, n_reduced):
        raise ValueError("quadratic coefficient has an incompatible shape")
    if not all(
        np.all(np.isfinite(array))
        for array in (basis, constant, linear, quadratic)
    ):
        raise ValueError("all coefficients must contain finite values")

    derivative_coefficients = np.empty(
        (n_reduced, n_full, n_reduced), dtype=float
    )
    for coefficient_index in range(n_reduced):
        for column_index in range(n_reduced):
            derivative_coefficients[coefficient_index, :, column_index] = (
                quadratic[:, coefficient_index, column_index]
                + quadratic[:, column_index, coefficient_index]
            )

    galerkin_constant = basis.T @ constant
    galerkin_linear = basis.T @ linear
    galerkin_quadratic = np.einsum(
        "Ni,Njk->ijk", basis, quadratic, optimize=True
    )

    gradient_constant = linear.T @ constant
    gradient_linear = np.empty((n_reduced, n_reduced), dtype=float)
    gradient_quadratic = np.empty(
        (n_reduced, n_reduced, n_reduced), dtype=float
    )
    gradient_cubic = np.empty(
        (n_reduced, n_reduced, n_reduced, n_reduced), dtype=float
    )
    normal_constant = linear.T @ linear
    normal_linear = np.empty(
        (n_reduced, n_reduced, n_reduced), dtype=float
    )
    normal_quadratic = np.empty(
        (n_reduced, n_reduced, n_reduced, n_reduced), dtype=float
    )
    for first_index in range(n_reduced):
        derivative = derivative_coefficients[first_index]
        gradient_linear[:, first_index] = (
            linear.T @ linear[:, first_index] + derivative.T @ constant
        )
        normal_linear[first_index] = (
            linear.T @ derivative + derivative.T @ linear
        )
        for second_index in range(n_reduced):
            gradient_quadratic[:, first_index, second_index] = (
                linear.T @ quadratic[:, first_index, second_index]
                + derivative.T @ linear[:, second_index]
            )
            normal_quadratic[first_index, second_index] = (
                derivative.T @ derivative_coefficients[second_index]
            )
            for third_index in range(n_reduced):
                gradient_cubic[
                    :, first_index, second_index, third_index
                ] = derivative.T @ quadratic[:, second_index, third_index]

    tensors = {
        "galerkin_constant": galerkin_constant,
        "galerkin_linear": galerkin_linear,
        "galerkin_quadratic": galerkin_quadratic,
        "lspg_gradient_constant": gradient_constant,
        "lspg_gradient_linear": gradient_linear,
        "lspg_gradient_quadratic": gradient_quadratic,
        "lspg_gradient_cubic": gradient_cubic,
        "lspg_normal_constant": normal_constant,
        "lspg_normal_linear": normal_linear,
        "lspg_normal_quadratic": normal_quadratic,
    }
    if set(tensors) != _TENSOR_KEYS:
        raise RuntimeError("internal tensor key mismatch")
    return tensors

import numpy as np



def _validated_tensors(tensors):
    """Convert and validate the complete contracted tensor dictionary."""
    _TENSOR_KEYS = {
        "galerkin_constant",
        "galerkin_linear",
        "galerkin_quadratic",
        "lspg_gradient_constant",
        "lspg_gradient_linear",
        "lspg_gradient_quadratic",
        "lspg_gradient_cubic",
        "lspg_normal_constant",
        "lspg_normal_linear",
        "lspg_normal_quadratic",
    }

    if not isinstance(tensors, dict) or set(tensors) != _TENSOR_KEYS:
        raise ValueError("tensors has invalid keys")
    converted = {key: np.asarray(value, dtype=float) for key, value in tensors.items()}
    constant = converted["galerkin_constant"]
    if constant.ndim != 1 or constant.size == 0:
        raise ValueError("galerkin_constant must be a nonempty vector")
    n = constant.size
    shapes = {
        "galerkin_constant": (n,),
        "galerkin_linear": (n, n),
        "galerkin_quadratic": (n, n, n),
        "lspg_gradient_constant": (n,),
        "lspg_gradient_linear": (n, n),
        "lspg_gradient_quadratic": (n, n, n),
        "lspg_gradient_cubic": (n, n, n, n),
        "lspg_normal_constant": (n, n),
        "lspg_normal_linear": (n, n, n),
        "lspg_normal_quadratic": (n, n, n, n),
    }
    if any(converted[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("contracted tensor shapes are incompatible")
    if not all(np.all(np.isfinite(value)) for value in converted.values()):
        raise ValueError("contracted tensors must contain finite values")
    return converted, n


def evaluate_hrf_galerkin(tensors: dict, state: np.ndarray) -> dict:
    """Reference implementation."""
    tensors, n_reduced = _validated_tensors(tensors)
    state = np.asarray(state, dtype=float)
    if state.shape != (n_reduced,) or not np.all(np.isfinite(state)):
        raise ValueError("state has an incompatible shape or nonfinite value")
    constant = tensors["galerkin_constant"]
    linear = tensors["galerkin_linear"]
    quadratic = tensors["galerkin_quadratic"]
    residual = (
        constant
        + linear @ state
        + np.einsum("ijk,j,k->i", quadratic, state, state, optimize=True)
    )
    derivative_coefficients = quadratic + quadratic.swapaxes(1, 2)
    jacobian = linear + np.einsum(
        "ijk,k->ij", derivative_coefficients, state, optimize=True
    )
    return {"residual": residual, "jacobian": jacobian}

import numpy as np



def _validated_tensors(tensors):
    """Convert and validate the complete contracted tensor dictionary."""
    _TENSOR_KEYS = {
        "galerkin_constant",
        "galerkin_linear",
        "galerkin_quadratic",
        "lspg_gradient_constant",
        "lspg_gradient_linear",
        "lspg_gradient_quadratic",
        "lspg_gradient_cubic",
        "lspg_normal_constant",
        "lspg_normal_linear",
        "lspg_normal_quadratic",
    }

    if not isinstance(tensors, dict) or set(tensors) != _TENSOR_KEYS:
        raise ValueError("tensors has invalid keys")
    converted = {key: np.asarray(value, dtype=float) for key, value in tensors.items()}
    constant = converted["galerkin_constant"]
    if constant.ndim != 1 or constant.size == 0:
        raise ValueError("galerkin_constant must be a nonempty vector")
    n = constant.size
    shapes = {
        "galerkin_constant": (n,),
        "galerkin_linear": (n, n),
        "galerkin_quadratic": (n, n, n),
        "lspg_gradient_constant": (n,),
        "lspg_gradient_linear": (n, n),
        "lspg_gradient_quadratic": (n, n, n),
        "lspg_gradient_cubic": (n, n, n, n),
        "lspg_normal_constant": (n, n),
        "lspg_normal_linear": (n, n, n),
        "lspg_normal_quadratic": (n, n, n, n),
    }
    if any(converted[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("contracted tensor shapes are incompatible")
    if not all(np.all(np.isfinite(value)) for value in converted.values()):
        raise ValueError("contracted tensors must contain finite values")
    return converted, n


def evaluate_hrf_lspg(tensors: dict, state: np.ndarray) -> dict:
    """Reference implementation."""
    tensors, n_reduced = _validated_tensors(tensors)
    state = np.asarray(state, dtype=float)
    if state.shape != (n_reduced,) or not np.all(np.isfinite(state)):
        raise ValueError("state has an incompatible shape or nonfinite value")
    residual = (
        tensors["lspg_gradient_constant"]
        + tensors["lspg_gradient_linear"] @ state
        + np.einsum(
            "ijk,j,k->i",
            tensors["lspg_gradient_quadratic"],
            state,
            state,
            optimize=True,
        )
        + np.einsum(
            "ijkl,j,k,l->i",
            tensors["lspg_gradient_cubic"],
            state,
            state,
            state,
            optimize=True,
        )
    )
    jacobian = (
        tensors["lspg_normal_constant"]
        + np.einsum(
            "kij,k->ij",
            tensors["lspg_normal_linear"],
            state,
            optimize=True,
        )
        + np.einsum(
            "klij,k,l->ij",
            tensors["lspg_normal_quadratic"],
            state,
            state,
            optimize=True,
        )
    )
    return {"residual": residual, "jacobian": jacobian}

import numpy as np



def _validated_certificate_inputs(basis, residual_polynomial, tensors, states):
    """Validate and convert all certificate inputs."""
    _TENSOR_KEYS = {
        "galerkin_constant",
        "galerkin_linear",
        "galerkin_quadratic",
        "lspg_gradient_constant",
        "lspg_gradient_linear",
        "lspg_gradient_quadratic",
        "lspg_gradient_cubic",
        "lspg_normal_constant",
        "lspg_normal_linear",
        "lspg_normal_quadratic",
    }

    basis = np.asarray(basis, dtype=float)
    states = np.asarray(states, dtype=float)
    if basis.ndim != 2 or 0 in basis.shape:
        raise ValueError("basis must be a nonempty matrix")
    n_full, n_reduced = basis.shape
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] != n_reduced:
        raise ValueError("states must have shape (m, n) with m positive")
    if not isinstance(residual_polynomial, dict):
        raise ValueError("residual_polynomial must be a dictionary")
    if set(residual_polynomial) != {"constant", "linear", "quadratic"}:
        raise ValueError("residual_polynomial has invalid keys")
    polynomial = {
        key: np.asarray(value, dtype=float)
        for key, value in residual_polynomial.items()
    }
    expected_polynomial_shapes = {
        "constant": (n_full,),
        "linear": (n_full, n_reduced),
        "quadratic": (n_full, n_reduced, n_reduced),
    }
    if any(
        polynomial[key].shape != shape
        for key, shape in expected_polynomial_shapes.items()
    ):
        raise ValueError("residual polynomial shapes are incompatible")
    if not isinstance(tensors, dict) or set(tensors) != _TENSOR_KEYS:
        raise ValueError("tensors has invalid keys")
    tensors = {key: np.asarray(value, dtype=float) for key, value in tensors.items()}
    n = n_reduced
    expected_tensor_shapes = {
        "galerkin_constant": (n,),
        "galerkin_linear": (n, n),
        "galerkin_quadratic": (n, n, n),
        "lspg_gradient_constant": (n,),
        "lspg_gradient_linear": (n, n),
        "lspg_gradient_quadratic": (n, n, n),
        "lspg_gradient_cubic": (n, n, n, n),
        "lspg_normal_constant": (n, n),
        "lspg_normal_linear": (n, n, n),
        "lspg_normal_quadratic": (n, n, n, n),
    }
    if any(
        tensors[key].shape != shape
        for key, shape in expected_tensor_shapes.items()
    ):
        raise ValueError("contracted tensor shapes are incompatible")
    arrays = [basis, states, *polynomial.values(), *tensors.values()]
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all numerical inputs must be finite")
    return basis, polynomial, tensors, states


def certify_hrf_tensors(
    basis: np.ndarray,
    residual_polynomial: dict,
    tensors: dict,
    states: np.ndarray,
) -> dict:
    """Reference implementation."""
    basis, polynomial, tensors, states = _validated_certificate_inputs(
        basis, residual_polynomial, tensors, states
    )
    constant = polynomial["constant"]
    linear = polynomial["linear"]
    quadratic = polynomial["quadratic"]
    derivative_coefficients = quadratic + quadratic.swapaxes(1, 2)
    errors = np.empty((states.shape[0], 4), dtype=float)
    for row_index, state in enumerate(states):
        direct_residual = (
            constant
            + linear @ state
            + np.einsum(
                "Nij,i,j->N", quadratic, state, state, optimize=True
            )
        )
        direct_jacobian = linear + np.einsum(
            "Nij,j->Ni", derivative_coefficients, state, optimize=True
        )
        compact_galerkin_residual = (
            tensors["galerkin_constant"]
            + tensors["galerkin_linear"] @ state
            + np.einsum(
                "ijk,j,k->i",
                tensors["galerkin_quadratic"],
                state,
                state,
                optimize=True,
            )
        )
        compact_galerkin_jacobian = tensors["galerkin_linear"] + np.einsum(
            "ijk,k->ij",
            tensors["galerkin_quadratic"]
            + tensors["galerkin_quadratic"].swapaxes(1, 2),
            state,
            optimize=True,
        )
        compact_lspg_gradient = (
            tensors["lspg_gradient_constant"]
            + tensors["lspg_gradient_linear"] @ state
            + np.einsum(
                "ijk,j,k->i",
                tensors["lspg_gradient_quadratic"],
                state,
                state,
                optimize=True,
            )
            + np.einsum(
                "ijkl,j,k,l->i",
                tensors["lspg_gradient_cubic"],
                state,
                state,
                state,
                optimize=True,
            )
        )
        compact_lspg_normal = (
            tensors["lspg_normal_constant"]
            + np.einsum(
                "kij,k->ij",
                tensors["lspg_normal_linear"],
                state,
                optimize=True,
            )
            + np.einsum(
                "klij,k,l->ij",
                tensors["lspg_normal_quadratic"],
                state,
                state,
                optimize=True,
            )
        )
        errors[row_index] = (
            np.max(
                np.abs(compact_galerkin_residual - basis.T @ direct_residual)
            ),
            np.max(
                np.abs(compact_galerkin_jacobian - basis.T @ direct_jacobian)
            ),
            np.max(
                np.abs(
                    compact_lspg_gradient
                    - direct_jacobian.T @ direct_residual
                )
            ),
            np.max(
                np.abs(
                    compact_lspg_normal
                    - direct_jacobian.T @ direct_jacobian
                )
            ),
        )
    return {"errors": errors}

import numpy as np



def _validated_solver_tensors(tensors):
    """Validate and convert the contracted tensor dictionary."""
    _TENSOR_KEYS = {
        "galerkin_constant",
        "galerkin_linear",
        "galerkin_quadratic",
        "lspg_gradient_constant",
        "lspg_gradient_linear",
        "lspg_gradient_quadratic",
        "lspg_gradient_cubic",
        "lspg_normal_constant",
        "lspg_normal_linear",
        "lspg_normal_quadratic",
    }

    if not isinstance(tensors, dict) or set(tensors) != _TENSOR_KEYS:
        raise ValueError("tensors has invalid keys")
    tensors = {key: np.asarray(value, dtype=float) for key, value in tensors.items()}
    constant = tensors["galerkin_constant"]
    if constant.ndim != 1 or constant.size == 0:
        raise ValueError("galerkin_constant must be a nonempty vector")
    n = constant.size
    shapes = {
        "galerkin_constant": (n,),
        "galerkin_linear": (n, n),
        "galerkin_quadratic": (n, n, n),
        "lspg_gradient_constant": (n,),
        "lspg_gradient_linear": (n, n),
        "lspg_gradient_quadratic": (n, n, n),
        "lspg_gradient_cubic": (n, n, n, n),
        "lspg_normal_constant": (n, n),
        "lspg_normal_linear": (n, n, n),
        "lspg_normal_quadratic": (n, n, n, n),
    }
    if any(tensors[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("contracted tensor shapes are incompatible")
    if not all(np.all(np.isfinite(value)) for value in tensors.values()):
        raise ValueError("contracted tensors must contain finite values")
    return tensors, n


def _tensor_system(tensors, state, scheme_index):
    """Evaluate one tensor-expanded Newton system."""
    if scheme_index == 0:
        residual = (
            tensors["galerkin_constant"]
            + tensors["galerkin_linear"] @ state
            + np.einsum(
                "ijk,j,k->i",
                tensors["galerkin_quadratic"],
                state,
                state,
                optimize=True,
            )
        )
        jacobian = tensors["galerkin_linear"] + np.einsum(
            "ijk,k->ij",
            tensors["galerkin_quadratic"]
            + tensors["galerkin_quadratic"].swapaxes(1, 2),
            state,
            optimize=True,
        )
    else:
        residual = (
            tensors["lspg_gradient_constant"]
            + tensors["lspg_gradient_linear"] @ state
            + np.einsum(
                "ijk,j,k->i",
                tensors["lspg_gradient_quadratic"],
                state,
                state,
                optimize=True,
            )
            + np.einsum(
                "ijkl,j,k,l->i",
                tensors["lspg_gradient_cubic"],
                state,
                state,
                state,
                optimize=True,
            )
        )
        jacobian = (
            tensors["lspg_normal_constant"]
            + np.einsum(
                "kij,k->ij",
                tensors["lspg_normal_linear"],
                state,
                optimize=True,
            )
            + np.einsum(
                "klij,k,l->ij",
                tensors["lspg_normal_quadratic"],
                state,
                state,
                optimize=True,
            )
        )
    return residual, jacobian


def solve_hrf_pair(
    tensors: dict,
    previous_state: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation."""
    tensors, n_reduced = _validated_solver_tensors(tensors)
    previous_state = np.asarray(previous_state, dtype=float)
    if previous_state.shape != (n_reduced,) or not np.all(
        np.isfinite(previous_state)
    ):
        raise ValueError("previous_state has an incompatible shape")
    if not np.isscalar(tolerance) or not np.isfinite(float(tolerance)):
        raise ValueError("tolerance must be a finite scalar")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive")
    if not isinstance(max_iterations, (int, np.integer)) or int(max_iterations) < 1:
        raise ValueError("max_iterations must be a positive integer")
    tolerance = float(tolerance)
    max_iterations = int(max_iterations)

    states = []
    iterations = []
    convergence_norms = []
    for scheme_index in range(2):
        state = previous_state.copy()
        update_count = 0
        while True:
            residual, jacobian = _tensor_system(tensors, state, scheme_index)
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm < tolerance:
                break
            if update_count >= max_iterations:
                raise ValueError("reduced Newton iteration did not converge")
            state = state + np.linalg.solve(jacobian, -residual)
            update_count += 1
        states.append(state)
        iterations.append(update_count)
        convergence_norms.append(residual_norm)
    return {
        "states": np.vstack(states),
        "iterations": np.asarray(iterations, dtype=int),
        "convergence_norms": np.asarray(convergence_norms, dtype=float),
    }

import numpy as np



def _validate_gap_inputs(
    basis, residual_polynomial, tensors, previous_state, solution, component_index
):
    """Validate and convert projection-gap inputs."""
    _TENSOR_KEYS = {
        "galerkin_constant",
        "galerkin_linear",
        "galerkin_quadratic",
        "lspg_gradient_constant",
        "lspg_gradient_linear",
        "lspg_gradient_quadratic",
        "lspg_gradient_cubic",
        "lspg_normal_constant",
        "lspg_normal_linear",
        "lspg_normal_quadratic",
    }

    basis = np.asarray(basis, dtype=float)
    previous_state = np.asarray(previous_state, dtype=float)
    if basis.ndim != 2 or 0 in basis.shape:
        raise ValueError("basis must be a nonempty matrix")
    n_full, n_reduced = basis.shape
    if previous_state.shape != (n_reduced,):
        raise ValueError("previous_state has an incompatible shape")
    if not isinstance(residual_polynomial, dict) or set(
        residual_polynomial
    ) != {"constant", "linear", "quadratic"}:
        raise ValueError("residual_polynomial has invalid keys")
    polynomial = {
        key: np.asarray(value, dtype=float)
        for key, value in residual_polynomial.items()
    }
    polynomial_shapes = {
        "constant": (n_full,),
        "linear": (n_full, n_reduced),
        "quadratic": (n_full, n_reduced, n_reduced),
    }
    if any(
        polynomial[key].shape != shape
        for key, shape in polynomial_shapes.items()
    ):
        raise ValueError("residual polynomial shapes are incompatible")
    if not isinstance(tensors, dict) or set(tensors) != _TENSOR_KEYS:
        raise ValueError("tensors has invalid keys")
    tensors = {key: np.asarray(value, dtype=float) for key, value in tensors.items()}
    n = n_reduced
    tensor_shapes = {
        "galerkin_constant": (n,),
        "galerkin_linear": (n, n),
        "galerkin_quadratic": (n, n, n),
        "lspg_gradient_constant": (n,),
        "lspg_gradient_linear": (n, n),
        "lspg_gradient_quadratic": (n, n, n),
        "lspg_gradient_cubic": (n, n, n, n),
        "lspg_normal_constant": (n, n),
        "lspg_normal_linear": (n, n, n),
        "lspg_normal_quadratic": (n, n, n, n),
    }
    if any(tensors[key].shape != shape for key, shape in tensor_shapes.items()):
        raise ValueError("contracted tensor shapes are incompatible")
    if not isinstance(solution, dict) or "states" not in solution:
        raise ValueError("solution must contain states")
    states = np.asarray(solution["states"], dtype=float)
    if states.shape != (2, n_reduced):
        raise ValueError("solution states have an incompatible shape")
    if not isinstance(component_index, (int, np.integer)):
        raise ValueError("component_index must be an integer")
    component_index = int(component_index)
    if not 0 <= component_index < n_full:
        raise ValueError("component_index is outside the full-order state")
    arrays = [basis, previous_state, states, *polynomial.values(), *tensors.values()]
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all numerical inputs must be finite")
    return basis, polynomial, tensors, previous_state, states, component_index


def _direct_and_compact_systems(basis, polynomial, tensors, state):
    """Evaluate direct and tensor-expanded systems at one state."""
    constant = polynomial["constant"]
    linear = polynomial["linear"]
    quadratic = polynomial["quadratic"]
    direct_residual = (
        constant
        + linear @ state
        + np.einsum("Nij,i,j->N", quadratic, state, state, optimize=True)
    )
    direct_jacobian = linear + np.einsum(
        "Nij,j->Ni",
        quadratic + quadratic.swapaxes(1, 2),
        state,
        optimize=True,
    )
    compact_galerkin_residual = (
        tensors["galerkin_constant"]
        + tensors["galerkin_linear"] @ state
        + np.einsum(
            "ijk,j,k->i",
            tensors["galerkin_quadratic"],
            state,
            state,
            optimize=True,
        )
    )
    compact_galerkin_jacobian = tensors["galerkin_linear"] + np.einsum(
        "ijk,k->ij",
        tensors["galerkin_quadratic"]
        + tensors["galerkin_quadratic"].swapaxes(1, 2),
        state,
        optimize=True,
    )
    compact_lspg_gradient = (
        tensors["lspg_gradient_constant"]
        + tensors["lspg_gradient_linear"] @ state
        + np.einsum(
            "ijk,j,k->i",
            tensors["lspg_gradient_quadratic"],
            state,
            state,
            optimize=True,
        )
        + np.einsum(
            "ijkl,j,k,l->i",
            tensors["lspg_gradient_cubic"],
            state,
            state,
            state,
            optimize=True,
        )
    )
    compact_lspg_normal = (
        tensors["lspg_normal_constant"]
        + np.einsum(
            "kij,k->ij", tensors["lspg_normal_linear"], state, optimize=True
        )
        + np.einsum(
            "klij,k,l->ij",
            tensors["lspg_normal_quadratic"],
            state,
            state,
            optimize=True,
        )
    )
    direct_systems = (
        basis.T @ direct_residual,
        basis.T @ direct_jacobian,
        direct_jacobian.T @ direct_residual,
        direct_jacobian.T @ direct_jacobian,
    )
    compact_systems = (
        compact_galerkin_residual,
        compact_galerkin_jacobian,
        compact_lspg_gradient,
        compact_lspg_normal,
    )
    errors = np.asarray(
        [
            np.max(np.abs(compact - direct))
            for compact, direct in zip(compact_systems, direct_systems)
        ],
        dtype=float,
    )
    return direct_residual, errors


def compute_projection_gap(
    basis: np.ndarray,
    residual_polynomial: dict,
    tensors: dict,
    previous_state: np.ndarray,
    solution: dict,
    component_index: int = 0,
) -> dict:
    """Reference implementation."""
    basis, polynomial, tensors, previous_state, states, component_index = (
        _validate_gap_inputs(
            basis,
            residual_polynomial,
            tensors,
            previous_state,
            solution,
            component_index,
        )
    )
    checked_states = np.vstack((previous_state, states))
    compact_errors = np.empty((3, 4), dtype=float)
    residual_norms = np.empty(2, dtype=float)
    for row_index, state in enumerate(checked_states):
        direct_residual, compact_errors[row_index] = _direct_and_compact_systems(
            basis, polynomial, tensors, state
        )
        if row_index > 0:
            residual_norms[row_index - 1] = np.linalg.norm(direct_residual)
    reconstructed_components = (states @ basis.T)[:, component_index]
    return {
        "gap": float(reconstructed_components[1] - reconstructed_components[0]),
        "reconstructed_components": reconstructed_components,
        "direct_residual_norms": residual_norms,
        "compact_errors": compact_errors,
    }

import numpy as np



def run_full_pipeline(energy_tolerance: float = 0.035) -> float:
    """Reference implementation chaining every earlier step."""
    _SNAPSHOTS = np.array(
        [
            [1.20, 1.00, 0.75, -0.35, -0.65, -0.90],
            [0.35, 0.60, 0.90, 1.10, 1.30, 1.45],
            [-0.55, -0.25, 0.05, 0.45, 0.85, 1.10],
            [0.85, 0.55, 0.20, -0.10, -0.45, -0.70],
            [-0.25, 0.05, 0.35, 0.70, 1.00, 1.25],
        ],
        dtype=float,
    )
    _CONSTANT = np.array([0.08, -0.04, 0.03, 0.05, -0.02], dtype=float)
    _LINEAR = np.array(
        [
            [-0.70, 0.20, 0.00, -0.10, 0.05],
            [0.15, -0.50, 0.25, 0.00, -0.08],
            [-0.10, 0.12, -0.60, 0.18, 0.10],
            [0.05, -0.14, 0.22, -0.45, 0.16],
            [0.10, 0.02, -0.12, 0.20, -0.55],
        ],
        dtype=float,
    )
    _QUADRATIC_TERMS = np.array(
        [
            [0, 0, 1, 0.50],
            [0, 2, 2, -0.35],
            [0, 3, 4, 0.12],
            [1, 0, 0, -0.42],
            [1, 1, 3, 0.30],
            [1, 2, 4, -0.18],
            [2, 0, 2, 0.28],
            [2, 1, 1, 0.40],
            [2, 3, 4, -0.25],
            [3, 1, 2, -0.33],
            [3, 0, 4, 0.22],
            [3, 3, 3, 0.15],
            [4, 2, 3, 0.31],
            [4, 1, 4, -0.27],
            [4, 0, 0, 0.20],
        ],
        dtype=float,
    )
    _INPUT_VECTOR = np.array([0.15, -0.08, 0.04, 0.10, -0.12], dtype=float)
    _BILINEAR = np.array(
        [
            [0.10, -0.05, 0.00, 0.03, 0.00],
            [0.02, 0.08, -0.04, 0.00, 0.01],
            [-0.03, 0.02, 0.06, -0.02, 0.00],
            [0.00, 0.04, 0.01, 0.07, -0.03],
            [0.05, 0.00, -0.02, 0.03, 0.09],
        ],
        dtype=float,
    )
    _PREVIOUS_FULL_STATE = np.array(
        [1.75, -1.92, -0.96, -0.42, -2.00], dtype=float
    )
    _CURRENT_INPUT = -0.85
    _PREVIOUS_INPUT = 0.41
    _TIME_STEP = 1.6
    _TOLERANCE = 1e-10
    _MAX_ITERATIONS = 25

    if not np.isscalar(energy_tolerance):
        raise ValueError("energy_tolerance must be a scalar")
    energy_tolerance = float(energy_tolerance)
    if not 0.0 < energy_tolerance < 1.0:
        raise ValueError("energy_tolerance must lie in (0, 1)")

    basis = select_pod_basis(_SNAPSHOTS, energy_tolerance)
    quadratic = assemble_quadratic_operator(
        _SNAPSHOTS.shape[0], _QUADRATIC_TERMS
    )
    previous_state = basis.T @ _PREVIOUS_FULL_STATE
    residual_polynomial = assemble_residual_polynomial(
        basis,
        _CONSTANT,
        _LINEAR,
        quadratic,
        _INPUT_VECTOR,
        _BILINEAR,
        previous_state,
        _CURRENT_INPUT,
        _PREVIOUS_INPUT,
        _TIME_STEP,
    )
    tensors = precompute_hrf_tensors(basis, residual_polynomial)
    initial_galerkin = evaluate_hrf_galerkin(tensors, previous_state)
    initial_lspg = evaluate_hrf_lspg(tensors, previous_state)
    if not all(
        np.all(np.isfinite(value))
        for value in (
            initial_galerkin["residual"],
            initial_galerkin["jacobian"],
            initial_lspg["residual"],
            initial_lspg["jacobian"],
        )
    ):
        raise ValueError("initial projected system is not finite")
    initial_certificate = certify_hrf_tensors(
        basis,
        residual_polynomial,
        tensors,
        previous_state[None, :],
    )
    if np.max(initial_certificate["errors"]) > 1e-12:
        raise ValueError("initial tensor/direct certificate failed")
    solution = solve_hrf_pair(
        tensors,
        previous_state,
        _TOLERANCE,
        _MAX_ITERATIONS,
    )
    certificate = compute_projection_gap(
        basis,
        residual_polynomial,
        tensors,
        previous_state,
        solution,
        0,
    )
    if np.max(certificate["compact_errors"]) > 1e-12:
        raise ValueError("terminal tensor/direct certificate failed")
    return certificate["gap"]
SCICODE_GOLD_EOF
