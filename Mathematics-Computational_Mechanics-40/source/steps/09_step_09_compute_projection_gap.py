"""
The projection gap is certified from the polynomial residual rather than from the online solver alone. Re-evaluating both projected systems at the shared initial state and at the two terminal states detects internally consistent solves built from incorrectly contracted coefficient tensors.

Returns
-------
dict with gap float, reconstructed_components (2,), direct_residual_norms (2,), and compact_errors (3, 4)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_projection_gap(
    basis: np.ndarray,
    residual_polynomial: dict,
    tensors: dict,
    previous_state: np.ndarray,
    solution: dict,
    component_index: int = 0,
) -> dict:
    """Certify two reduced solutions and compute their signed component gap.

    Parameters
    ----------
    basis : np.ndarray
        Finite trial basis Phi of shape (N, n).
    residual_polynomial : dict
        Exactly constant (N,), linear (N, n), and quadratic (N, n, n).
    tensors : dict
        Ten finite arrays with the key and axis conventions documented for
        precompute_hrf_tensors. They need not be consistent with the supplied
        polynomial; the returned errors diagnose consistency.
    previous_state : np.ndarray
        Finite reduced state at the previous time level, shape (n,).
    solution : dict
        Dictionary containing finite states of shape (2, n), with Galerkin in
        row 0 and LSPG in row 1.
    component_index : int
        Full-order component index in the half-open range [0, N).

    Returns
    -------
    certificate : dict
        Exactly gap, reconstructed_components (2,), direct_residual_norms
        (2,), and compact_errors (3, 4). Error rows correspond to the shared
        initial, Galerkin terminal, and LSPG terminal states; columns follow
        certify_hrf_tensors.

    Raises
    ------
    ValueError
        If a key, shape, finiteness, or component-index requirement fails.
    """
    return certificate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_compute_projection_gap(
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def _make_gap_tensors(basis, residual_polynomial):
    """Construct a consistent tensor fixture inside the test field."""
    a = residual_polynomial["constant"]
    linear = residual_polynomial["linear"]
    quadratic = residual_polynomial["quadratic"]
    n = basis.shape[1]
    derivative = np.empty((n, basis.shape[0], n))
    for j in range(n):
        for k in range(n):
            derivative[j, :, k] = quadratic[:, j, k] + quadratic[:, k, j]
    tensors = {
        "galerkin_constant": basis.T @ a,
        "galerkin_linear": basis.T @ linear,
        "galerkin_quadratic": np.einsum("Ni,Njk->ijk", basis, quadratic),
        "lspg_gradient_constant": linear.T @ a,
        "lspg_gradient_linear": np.empty((n, n)),
        "lspg_gradient_quadratic": np.empty((n, n, n)),
        "lspg_gradient_cubic": np.empty((n, n, n, n)),
        "lspg_normal_constant": linear.T @ linear,
        "lspg_normal_linear": np.empty((n, n, n)),
        "lspg_normal_quadratic": np.empty((n, n, n, n)),
    }
    for j in range(n):
        tensors["lspg_gradient_linear"][:, j] = (
            linear.T @ linear[:, j] + derivative[j].T @ a
        )
        tensors["lspg_normal_linear"][j] = (
            linear.T @ derivative[j] + derivative[j].T @ linear
        )
        for k in range(n):
            tensors["lspg_gradient_quadratic"][:, j, k] = (
                linear.T @ quadratic[:, j, k]
                + derivative[j].T @ linear[:, k]
            )
            tensors["lspg_normal_quadratic"][j, k] = (
                derivative[j].T @ derivative[k]
            )
            for l in range(n):
                tensors["lspg_gradient_cubic"][:, j, k, l] = (
                    derivative[j].T @ quadratic[:, k, l]
                )
    return tensors


def _pack(value):
    """Flatten a returned value into native numbers for comparison.

    Container sizes and array shapes are packed alongside the numbers, so a
    result carrying the right values in the wrong structure cannot compare equal.
    """
    if isinstance(value, dict):
        packed = [len(value)]
        for key in sorted(value):
            packed.extend(_pack(value[key]))
        return packed
    if isinstance(value, np.ndarray):
        packed = list(value.shape)
        packed.extend(round(float(entry), 12) for entry in value.ravel().tolist())
        return packed
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [round(float(value), 12)]


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
basis = np.array([[0.8, -0.2], [0.3, 0.9], [-0.5, 0.4]])
residual_polynomial = {
    "constant": np.array([0.4, -0.7, 0.2]),
    "linear": np.array([[1.1, -0.3], [0.2, 0.8], [-0.6, 0.5]]),
    "quadratic": np.array([
        [[0.3, -0.4], [0.1, 0.2]],
        [[-0.2, 0.5], [0.7, -0.1]],
        [[0.6, 0.2], [-0.3, 0.4]],
    ]),
}
tensors = _make_gap_tensors(basis, residual_polynomial)
previous_state = np.array([-0.7, 0.45])
solution = {"states": np.array([[0.2, -0.35], [0.6, 0.1]])}
component_index = 1
""",
            "call": "_pack(compute_projection_gap(basis, residual_polynomial, tensors, previous_state, solution, component_index))",
            "gold_call": "_pack(_oracle_compute_projection_gap(basis, residual_polynomial, tensors, previous_state, solution, component_index))",
        },
        {
            "setup": """import numpy as np
basis = np.array([[1.0]])
residual_polynomial = {
    "constant": np.array([0.2]),
    "linear": np.array([[1.3]]),
    "quadratic": np.array([[[-0.4]]]),
}
tensors = _make_gap_tensors(basis, residual_polynomial)
tensors["lspg_gradient_cubic"][0, 0, 0, 0] += 0.25
previous_state = np.array([0.0])
solution = {"states": np.array([[0.3], [-0.2]])}
""",
            "call": "_pack(compute_projection_gap(basis, residual_polynomial, tensors, previous_state, solution))",
            "gold_call": "_pack(_oracle_compute_projection_gap(basis, residual_polynomial, tensors, previous_state, solution))",
        },
        {
            "setup": """import numpy as np
basis = np.eye(2)
residual_polynomial = {
    "constant": np.zeros(2),
    "linear": np.eye(2),
    "quadratic": np.zeros((2, 2, 2)),
}
tensors = _make_gap_tensors(basis, residual_polynomial)
previous_state = np.zeros(2)
solution = {"states": np.zeros((2, 2))}
def run_model():
    try:
        compute_projection_gap(basis, residual_polynomial, tensors, previous_state, solution, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_projection_gap(basis, residual_polynomial, tensors, previous_state, solution, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
