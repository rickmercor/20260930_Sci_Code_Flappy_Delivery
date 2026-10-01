"""
Exactness is a defining property of the hyper-reduction-free construction. At arbitrary reduced states, contracted Galerkin and LSPG systems can be compared with independently evaluated full residuals and residual Jacobians. This detects tensor-axis mistakes, missing quadratic derivative placements, and omitted cubic interactions without accepting a sampled approximation.

Returns
-------
dict with errors (m, 4) in the documented projection-system order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def certify_hrf_tensors(
    basis: np.ndarray,
    residual_polynomial: dict,
    tensors: dict,
    states: np.ndarray,
) -> dict:
    """Measure compact-versus-direct errors at one or more reduced states.

    Parameters
    ----------
    basis : np.ndarray
        Finite trial basis Phi of shape (N, n).
    residual_polynomial : dict
        Exactly constant (N,), linear (N, n), and quadratic (N, n, n).
    tensors : dict
        Ten finite arrays with the key and axis conventions documented for
        precompute_hrf_tensors. They need not already be consistent with the
        supplied polynomial; the returned errors diagnose consistency.
    states : np.ndarray
        Finite array of shape (m, n), with m positive.

    Returns
    -------
    certificate : dict
        Exactly errors of shape (m, 4). Columns are maximum absolute errors
        for Galerkin residual, Galerkin Jacobian, LSPG gradient, and LSPG
        normal matrix, in that order.

    Raises
    ------
    ValueError
        If dictionary keys, shapes, or finiteness requirements fail.
    """
    return certificate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_certify_hrf_tensors(
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def _make_consistent_tensors(basis, residual_polynomial):
    """Construct a test fixture independently inside the test field."""
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
tensors = _make_consistent_tensors(basis, residual_polynomial)
states = np.array([[-0.7, 0.45], [0.2, -0.35], [0.0, 0.0]])
""",
            "call": "_pack(certify_hrf_tensors(basis, residual_polynomial, tensors, states))",
            "gold_call": "_pack(_oracle_certify_hrf_tensors(basis, residual_polynomial, tensors, states))",
        },
        {
            "setup": """import numpy as np
basis = np.array([[1.0]])
residual_polynomial = {
    "constant": np.array([0.2]),
    "linear": np.array([[1.3]]),
    "quadratic": np.array([[[-0.4]]]),
}
tensors = _make_consistent_tensors(basis, residual_polynomial)
tensors["galerkin_quadratic"][0, 0, 0] += 0.1
tensors["lspg_gradient_cubic"][0, 0, 0, 0] -= 0.2
states = np.array([[0.0], [0.6]])
""",
            "call": "_pack(certify_hrf_tensors(basis, residual_polynomial, tensors, states))",
            "gold_call": "_pack(_oracle_certify_hrf_tensors(basis, residual_polynomial, tensors, states))",
        },
        {
            "setup": """import numpy as np
basis = np.eye(2)
residual_polynomial = {
    "constant": np.zeros(2),
    "linear": np.eye(2),
    "quadratic": np.zeros((2, 2, 2)),
}
tensors = _make_consistent_tensors(basis, residual_polynomial)
states = np.zeros((3, 3))
def run_model():
    try:
        certify_hrf_tensors(basis, residual_polynomial, tensors, states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_certify_hrf_tensors(basis, residual_polynomial, tensors, states)
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
