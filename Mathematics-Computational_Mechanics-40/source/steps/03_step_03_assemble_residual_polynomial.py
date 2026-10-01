"""
For a fixed implicit time step, a quadratic full-order dynamical system restricted to a trial basis has a full residual that is a quadratic polynomial in the unknown reduced state. Separating its constant, linear, and ordered quadratic coefficients exposes the structure needed for hyper-reduction-free tensor contraction while preserving direct full-order verification.

Returns
-------
dict with constant (N,), linear (N, n), and quadratic (N, n, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Assemble the Crank-Nicolson residual polynomial in reduced coordinates.

    Parameters
    ----------
    basis : np.ndarray
        Finite trial basis Phi of shape (N, n).
    constant : np.ndarray
        Finite full-order constant operator C of shape (N,).
    linear : np.ndarray
        Finite full-order linear operator A of shape (N, N).
    quadratic : np.ndarray
        Finite quadratic operator F of shape (N, N * N), using row-major
        Kronecker ordering.
    input_vector : np.ndarray
        Finite scalar-input operator B of shape (N,).
    bilinear : np.ndarray
        Finite scalar-input/state operator N of shape (N, N).
    previous_state : np.ndarray
        Finite reduced state at the previous time level, shape (n,).
    current_input, previous_input : float
        Finite scalar inputs at the current and previous time levels.
    time_step : float
        Finite, positive Crank-Nicolson step size.

    Returns
    -------
    polynomial : dict
        Exactly constant (N,), linear (N, n), and quadratic (N, n, n), such
        that the ordered last two axes multiply z_j z_k.

    Raises
    ------
    ValueError
        If a documented shape, finiteness, or positivity requirement fails.
    """
    return polynomial

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_residual_polynomial(
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
constant = np.array([0.1, -0.2, 0.05])
linear = np.array([[-0.4, 0.2, 0.1], [0.0, -0.3, 0.25], [-0.1, 0.05, -0.2]])
quadratic = np.zeros((3, 9))
quadratic[0, 1] = 0.7
quadratic[1, 3] = -0.25
quadratic[2, 8] = 0.4
input_vector = np.array([0.2, -0.1, 0.3])
bilinear = np.array([[0.1, 0.0, -0.05], [0.02, -0.08, 0.04], [0.0, 0.03, 0.06]])
previous_state = np.array([0.2, -0.35])
current_input = -0.6
previous_input = 0.25
time_step = 0.9
""",
            "call": "_pack(assemble_residual_polynomial(basis, constant, linear, quadratic, input_vector, bilinear, previous_state, current_input, previous_input, time_step))",
            "gold_call": "_pack(_oracle_assemble_residual_polynomial(basis, constant, linear, quadratic, input_vector, bilinear, previous_state, current_input, previous_input, time_step))",
        },
        {
            "setup": """import numpy as np
basis = np.array([[1.0]])
constant = np.array([0.2])
linear = np.array([[-0.5]])
quadratic = np.array([[0.3]])
input_vector = np.array([-0.1])
bilinear = np.array([[0.4]])
previous_state = np.array([0.0])
current_input = 0.0
previous_input = 0.0
time_step = 0.1
""",
            "call": "_pack(assemble_residual_polynomial(basis, constant, linear, quadratic, input_vector, bilinear, previous_state, current_input, previous_input, time_step))",
            "gold_call": "_pack(_oracle_assemble_residual_polynomial(basis, constant, linear, quadratic, input_vector, bilinear, previous_state, current_input, previous_input, time_step))",
        },
        {
            "setup": """import numpy as np
basis = np.eye(2)
constant = np.zeros(2)
linear = np.zeros((2, 2))
quadratic = np.zeros((2, 5))
input_vector = np.zeros(2)
bilinear = np.zeros((2, 2))
previous_state = np.zeros(2)
def run_model():
    try:
        assemble_residual_polynomial(basis, constant, linear, quadratic, input_vector, bilinear, previous_state, 0.0, 0.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_assemble_residual_polynomial(basis, constant, linear, quadratic, input_vector, bilinear, previous_state, 0.0, 0.0, 0.1)
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
