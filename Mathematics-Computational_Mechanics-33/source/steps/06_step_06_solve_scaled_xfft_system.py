"""
Matrix-free preconditioned linear conjugate gradients for the scaled X-FFT system.




Element stiffness actions are gathered and scattered without assembling a global matrix. The standard residual block is transformed by a three-dimensional FFT, multiplied by the cached Green operator, and inverse transformed. The internally scaled enriched block is preconditioned by the identity. The paper's energy residual is the square root of the preconditioned residual product, and convergence compares it with the norm of the current volume-averaged Mandel stress.




Inputs

------

elements : dict

    Required fields are float64 arrays element_matrices (e, 24, 24), element_stress (e, 6, 24), rhs (d,), and stress_macro (6,); integer array element_dofs (e, 24); and native int n_standard_dofs, where e and d count elements and total degrees of freedom.

green : np.ndarray of shape (n, n, n, 3, 3)

tol : float

max_iter : int




Returns

-------

solution : dict

    Keys displacement, residual_trace, iterations, converged, and effective_stress, with shapes and dtypes specified below.

Returns
-------
dict with float64 arrays displacement (d,), residual_trace (iterations + 1,), and effective_stress (6,); native int iterations; and native bool converged.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_scaled_xfft_system(
    elements: dict,
    green: np.ndarray,
    tol: float,
    max_iter: int,
) -> dict:
    """Solve the scaled X-FFT equilibrium system by preconditioned linear CG.

    Parameters
    ----------
    elements : dict
        Required fields are float64 arrays element_matrices (e, 24, 24), element_stress (e, 6, 24), rhs (d,), and stress_macro (6,); integer array element_dofs (e, 24); and native int n_standard_dofs, where e and d count elements and total degrees of freedom.
    green : np.ndarray
        Fourier inverse symbols for the standard displacement block.
    tol : float
        Positive relative residual tolerance.
    max_iter : int
        Positive maximum iteration count.

    Returns
    -------
    solution : dict
        Keys displacement, residual_trace, iterations, converged, and effective_stress, with shapes and dtypes specified below.
    Raises
    ------
    ValueError
        If elements is not a dict carrying element_matrices, element_dofs,
        element_stress, rhs, stress_macro and n_standard_dofs, if green is not
        five-dimensional with trailing shape (3, 3), if tol is not a finite
        positive number, if max_iter is not a positive integer, if
        n_standard_dofs differs from 3 * n_voxels**3 implied by green, or if
        the conjugate-gradient iteration meets non-positive curvature.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _apply_operator(elements, vector):
    result = np.zeros_like(vector)
    for matrix, dofs in zip(elements["element_matrices"], elements["element_dofs"]):
        active = dofs >= 0
        local = np.zeros(matrix.shape[0], dtype=float)
        local[active] = vector[dofs[active]]
        product = matrix @ local
        np.add.at(result, dofs[active], product[active])
    return result


def _effective_stress(elements, vector):
    result = np.asarray(elements["stress_macro"], dtype=float).copy()
    for matrix, dofs in zip(elements["element_stress"], elements["element_dofs"]):
        active = dofs >= 0
        local = np.zeros(matrix.shape[1], dtype=float)
        local[active] = vector[dofs[active]]
        result += matrix @ local
    return result


def _apply_preconditioner(green, residual, n_standard_dofs):
    n_voxels = green.shape[0]
    expected = 3 * n_voxels**3
    if n_standard_dofs != expected:
        raise ValueError("standard block size is inconsistent with green")
    standard = residual[:n_standard_dofs].reshape(n_voxels, n_voxels, n_voxels, 3)
    transformed = np.fft.fftn(standard, axes=(0, 1, 2))
    solution_hat = np.einsum("...ij,...j->...i", green, transformed)
    standard_solution = np.fft.ifftn(solution_hat, axes=(0, 1, 2)).real.reshape(-1)
    return np.concatenate([standard_solution, residual[n_standard_dofs:]])


def _oracle_solve_scaled_xfft_system(
    elements: dict,
    green: np.ndarray,
    tol: float,
    max_iter: int,
) -> dict:
    """Reference implementation."""
    if not isinstance(elements, dict):
        raise ValueError("elements must be a dictionary")  # noqa: TRY004
    required = {
        "element_matrices",
        "element_dofs",
        "element_stress",
        "rhs",
        "stress_macro",
        "n_standard_dofs",
    }
    if not required.issubset(elements):
        raise ValueError("elements is missing required fields")
    green = np.asarray(green, dtype=complex)
    if green.ndim != 5 or green.shape[-2:] != (3, 3):
        raise ValueError("green must have shape (n, n, n, 3, 3)")
    if not np.isfinite(tol) or float(tol) <= 0.0:
        raise ValueError("tol must be positive")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")

    right_hand_side = np.asarray(elements["rhs"], dtype=float)
    displacement = np.zeros_like(right_hand_side)
    residual = right_hand_side - _apply_operator(elements, displacement)
    preconditioned = _apply_preconditioner(
        green, residual, int(elements["n_standard_dofs"])
    )
    direction = preconditioned.copy()
    residual_product = float(residual @ preconditioned)
    residual_trace = [np.sqrt(max(residual_product, 0.0))]
    converged = residual_trace[-1] <= float(tol) * np.linalg.norm(
        _effective_stress(elements, displacement)
    )
    iteration = 0
    while not converged and iteration < int(max_iter):
        action = _apply_operator(elements, direction)
        curvature = float(direction @ action)
        if curvature <= 0.0:
            raise ValueError("CG encountered non-positive curvature")
        step = residual_product / curvature
        displacement += step * direction
        residual -= step * action
        preconditioned = _apply_preconditioner(
            green, residual, int(elements["n_standard_dofs"])
        )
        new_product = float(residual @ preconditioned)
        residual_trace.append(np.sqrt(max(new_product, 0.0)))
        iteration += 1
        converged = residual_trace[-1] <= float(tol) * np.linalg.norm(
            _effective_stress(elements, displacement)
        )
        if converged:
            residual_product = new_product
            break
        direction = preconditioned + (new_product / residual_product) * direction
        residual_product = new_product
    return {
        "displacement": displacement,
        "residual_trace": np.asarray(residual_trace, dtype=float),
        "iterations": iteration,
        "converged": bool(converged),
        "effective_stress": _effective_stress(elements, displacement),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
n = 2
green = np.zeros((n, n, n, 3, 3), dtype=complex)
for frequency in np.ndindex(n, n, n):
    if frequency != (0, 0, 0):
        green[frequency] = np.eye(3)
rhs_grid = np.arange(24, dtype=float).reshape(2, 2, 2, 3)
rhs_grid -= rhs_grid.mean(axis=(0, 1, 2), keepdims=True)
rhs = rhs_grid.reshape(-1)
elements = {
    "element_matrices": np.eye(24)[None, :, :],
    "element_dofs": np.arange(24, dtype=int)[None, :],
    "element_stress": np.zeros((1, 6, 24)),
    "rhs": rhs,
    "stress_macro": np.array([10., 4., 4., 0., 0., 0.]),
    "n_standard_dofs": 24,
}
def summarize(result):
    return (
        np.round(result["displacement"], 12).tolist(),
        np.round(result["residual_trace"], 12).tolist(),
        result["iterations"],
        result["converged"],
    )
""",
            "call": "summarize(solve_scaled_xfft_system(elements, green, 1e-12, 20))",
            "gold_call": "summarize(_oracle_solve_scaled_xfft_system(elements, green, 1e-12, 20))",
        },
        {
            "setup": """import numpy as np
n = 2
green = np.zeros((n, n, n, 3, 3), dtype=complex)
elements = {
    "element_matrices": np.eye(24)[None, :, :],
    "element_dofs": np.arange(24, dtype=int)[None, :],
    "element_stress": np.zeros((1, 6, 24)),
    "rhs": np.zeros(24),
    "stress_macro": np.ones(6),
    "n_standard_dofs": 24,
}
def summarize(result):
    return (
        result["iterations"],
        result["converged"],
        result["residual_trace"].tolist(),
    )
""",
            "call": "summarize(solve_scaled_xfft_system(elements, green, 1e-12, 20))",
            "gold_call": "summarize(_oracle_solve_scaled_xfft_system(elements, green, 1e-12, 20))",
        },
        {
            "setup": """import numpy as np
elements = {}
green = np.zeros((2, 2, 2, 3, 3), dtype=complex)
def run_model():
    try:
        solve_scaled_xfft_system(elements, green, 0.0, 20)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_solve_scaled_xfft_system(elements, green, 0.0, 20)
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
