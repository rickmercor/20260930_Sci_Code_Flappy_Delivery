"""
The hyper-reduction-free LSPG online kernel evaluates a cubic coefficient expansion for the normal residual and a quadratic expansion for the Gauss-Newton matrix. These degrees arise from multiplying an affine residual Jacobian by a quadratic residual, and they retain the nested reduced-state interactions that a generic one-line Gram shortcut conceals.

Returns
-------
dict with residual (n,) and jacobian (n, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_hrf_lspg(tensors: dict, state: np.ndarray) -> dict:
    """Evaluate the LSPG gradient and normal matrix from contracted tensors.

    Parameters
    ----------
    tensors : dict
        The exact 10-array output of precompute_hrf_tensors.
    state : np.ndarray
        Finite, nonempty reduced state z of shape (n,).

    Returns
    -------
    system : dict
        Exactly residual (n,) for the LSPG gradient and jacobian (n, n) for
        the Gauss-Newton normal matrix.

    Raises
    ------
    ValueError
        If tensor keys, tensor shapes, state shape, or finiteness is invalid.
    """
    return system

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_evaluate_hrf_lspg(tensors: dict, state: np.ndarray) -> dict:
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
n = 2
tensors = {
    "galerkin_constant": np.array([0.4, -0.2]),
    "galerkin_linear": np.array([[1.1, -0.3], [0.2, 0.8]]),
    "galerkin_quadratic": np.arange(8, dtype=float).reshape(n, n, n) / 20.0,
    "lspg_gradient_constant": np.array([0.1, -0.2]),
    "lspg_gradient_linear": np.array([[1.2, -0.4], [0.3, 0.9]]),
    "lspg_gradient_quadratic": np.arange(8, dtype=float).reshape(n, n, n) / 13.0,
    "lspg_gradient_cubic": (np.arange(16, dtype=float).reshape(n, n, n, n) - 5.0) / 17.0,
    "lspg_normal_constant": np.array([[2.0, -0.3], [-0.3, 1.4]]),
    "lspg_normal_linear": np.arange(8, dtype=float).reshape(n, n, n) / 19.0,
    "lspg_normal_quadratic": (np.arange(16, dtype=float).reshape(n, n, n, n) - 7.0) / 23.0,
}
state = np.array([-0.7, 0.45])
""",
            "call": "_pack(evaluate_hrf_lspg(tensors, state))",
            "gold_call": "_pack(_oracle_evaluate_hrf_lspg(tensors, state))",
        },
        {
            "setup": """import numpy as np
tensors = {
    "galerkin_constant": np.array([0.2]),
    "galerkin_linear": np.array([[1.3]]),
    "galerkin_quadratic": np.array([[[-0.4]]]),
    "lspg_gradient_constant": np.array([0.1]),
    "lspg_gradient_linear": np.array([[0.2]]),
    "lspg_gradient_quadratic": np.array([[[0.3]]]),
    "lspg_gradient_cubic": np.array([[[[0.4]]]]),
    "lspg_normal_constant": np.array([[1.0]]),
    "lspg_normal_linear": np.array([[[0.5]]]),
    "lspg_normal_quadratic": np.array([[[[0.25]]]]),
}
state = np.array([0.0])
""",
            "call": "_pack(evaluate_hrf_lspg(tensors, state))",
            "gold_call": "_pack(_oracle_evaluate_hrf_lspg(tensors, state))",
        },
        {
            "setup": """import numpy as np
tensors = {"lspg_gradient_constant": np.zeros(2)}
state = np.zeros(2)
def run_model():
    try:
        evaluate_hrf_lspg(tensors, state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_evaluate_hrf_lspg(tensors, state)
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
