"""
The paired reduced solve performs all nonlinear work through precontracted coefficient tensors. Galerkin and LSPG share an initial reduced state but evaluate polynomials of different degrees and stop against different stationarity residuals. No full-order state, residual, Jacobian, or residual-library Gram matrix is available inside the iteration.

Returns
-------
dict with states (2, n), iterations (2,), and convergence_norms (2,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_hrf_pair(
    tensors: dict,
    previous_state: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Solve the tensor-expanded Galerkin and LSPG reduced systems.

    Parameters
    ----------
    tensors : dict
        The exact 10-array output of precompute_hrf_tensors.
    previous_state : np.ndarray
        Finite, nonempty shared initial reduced state, shape (n,).
    tolerance : float
        Finite, positive convergence tolerance for each scheme's own
        stationarity residual norm.
    max_iterations : int
        Positive maximum number of unit Newton updates per scheme.

    Returns
    -------
    solution : dict
        Exactly states (2, n), iterations (2,), and convergence_norms (2,),
        with Galerkin in row 0 and LSPG in row 1.

    Raises
    ------
    ValueError
        If an input contract fails or either iteration does not converge.
    """
    return solution

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_solve_hrf_pair(
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def _make_solver_tensors(basis, residual_polynomial):
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
        tensors["lspg_gradient_linear"][:, j] = linear.T @ linear[:, j] + derivative[j].T @ a
        tensors["lspg_normal_linear"][j] = linear.T @ derivative[j] + derivative[j].T @ linear
        for k in range(n):
            tensors["lspg_gradient_quadratic"][:, j, k] = linear.T @ quadratic[:, j, k] + derivative[j].T @ linear[:, k]
            tensors["lspg_normal_quadratic"][j, k] = derivative[j].T @ derivative[k]
            for l in range(n):
                tensors["lspg_gradient_cubic"][:, j, k, l] = derivative[j].T @ quadratic[:, k, l]
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
basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.35, -0.2]])
residual_polynomial = {
    "constant": np.array([-0.2, 0.15, 0.05]),
    "linear": np.array([[1.0, 0.1], [-0.1, 1.0], [0.2, -0.2]]),
    "quadratic": np.array([
        [[0.08, -0.03], [0.02, 0.04]],
        [[-0.05, 0.06], [0.01, -0.07]],
        [[0.03, -0.02], [0.05, 0.02]],
    ]),
}
tensors = _make_solver_tensors(basis, residual_polynomial)
previous_state = np.array([0.4, -0.3])
""",
            "call": "_pack(solve_hrf_pair(tensors, previous_state, 1e-10, 25))",
            "gold_call": "_pack(_oracle_solve_hrf_pair(tensors, previous_state, 1e-10, 25))",
        },
        {
            "setup": """import numpy as np
basis = np.array([[1.0]])
residual_polynomial = {
    "constant": np.array([-0.2]),
    "linear": np.array([[1.2]]),
    "quadratic": np.array([[[0.15]]]),
}
tensors = _make_solver_tensors(basis, residual_polynomial)
previous_state = np.array([0.5])
""",
            "call": "_pack(solve_hrf_pair(tensors, previous_state, 1e-10, 15))",
            "gold_call": "_pack(_oracle_solve_hrf_pair(tensors, previous_state, 1e-10, 15))",
        },
        {
            "setup": """import numpy as np
basis = np.array([[1.0]])
residual_polynomial = {
    "constant": np.array([0.0]),
    "linear": np.array([[1.0]]),
    "quadratic": np.array([[[0.0]]]),
}
tensors = _make_solver_tensors(basis, residual_polynomial)
previous_state = np.array([0.0])
def run_model():
    try:
        solve_hrf_pair(tensors, previous_state, 0.0, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_solve_hrf_pair(tensors, previous_state, 0.0, 5)
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
