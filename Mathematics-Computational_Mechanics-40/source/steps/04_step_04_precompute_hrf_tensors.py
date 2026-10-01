"""
The hyper-reduction-free formulation expands the projected Newton operators as low-degree polynomials of the reduced state. For a quadratic residual, the Galerkin residual remains quadratic, the residual Jacobian is affine, the least-squares normal matrix is quadratic, and the least-squares gradient is cubic. Their coefficient tensors can be contracted before the online nonlinear iteration.

Returns
-------
dict with 10 float64 coefficient arrays following the documented degree and axis conventions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def precompute_hrf_tensors(basis: np.ndarray, residual_polynomial: dict) -> dict:
    """Precompute coefficient tensors for exact Galerkin and LSPG evaluation.

    Parameters
    ----------
    basis : np.ndarray
        Finite trial basis Phi of shape (N, n).
    residual_polynomial : dict
        Exactly constant (N,), linear (N, n), and quadratic (N, n, n).
        Quadratic axes are ordered as output, first state index, second state
        index and are not assumed symmetric.

    Returns
    -------
    tensors : dict
        Exactly galerkin_constant (n,), galerkin_linear (n, n),
        galerkin_quadratic (n, n, n), lspg_gradient_constant (n,),
        lspg_gradient_linear (n, n), lspg_gradient_quadratic (n, n, n),
        lspg_gradient_cubic (n, n, n, n), lspg_normal_constant (n, n),
        lspg_normal_linear (n, n, n), and lspg_normal_quadratic
        (n, n, n, n). State-polynomial coefficient indices precede normal-
        matrix row and column indices.

    Raises
    ------
    ValueError
        If the dictionary keys, shapes, or finiteness requirements fail.
    """
    return tensors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _oracle_precompute_hrf_tensors(
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


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
""",
            "call": "_pack(precompute_hrf_tensors(basis, residual_polynomial))",
            "gold_call": "_pack(_oracle_precompute_hrf_tensors(basis, residual_polynomial))",
        },
        {
            "setup": """import numpy as np
basis = np.array([[1.0]])
residual_polynomial = {
    "constant": np.array([0.2]),
    "linear": np.array([[1.3]]),
    "quadratic": np.array([[[-0.4]]]),
}
""",
            "call": "_pack(precompute_hrf_tensors(basis, residual_polynomial))",
            "gold_call": "_pack(_oracle_precompute_hrf_tensors(basis, residual_polynomial))",
        },
        {
            "setup": """import numpy as np
basis = np.eye(2)
residual_polynomial = {
    "constant": np.zeros(2),
    "linear": np.eye(2),
}
def run_model():
    try:
        precompute_hrf_tensors(basis, residual_polynomial)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_precompute_hrf_tensors(basis, residual_polynomial)
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
