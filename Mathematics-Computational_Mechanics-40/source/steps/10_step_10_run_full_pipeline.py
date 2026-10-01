"""
The end-to-end calculation must preserve one basis orientation and one tensor-axis convention through residual assembly, offline contraction, two different reduced Newton systems, and independent compact/direct certification. The final scalar is formed only after both terminal states satisfy their own stationarity conditions.

Returns
-------
float, the signed component-0 LSPG-minus-Galerkin projection gap
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(energy_tolerance: float = 0.035) -> float:
    """Run the deterministic HRF projection-gap calculation.

    Parameters
    ----------
    energy_tolerance : float
        POD discarded-energy threshold in the open interval (0, 1).

    Returns
    -------
    projection_gap : float
        Component-0 reconstructed LSPG state minus reconstructed Galerkin state.

    Raises
    ------
    ValueError
        If energy_tolerance is not a scalar strictly between 0 and 1.
    """
    return projection_gap

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _oracle_run_full_pipeline(energy_tolerance: float = 0.035) -> float:
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

    basis = _oracle_select_pod_basis(_SNAPSHOTS, energy_tolerance)
    quadratic = _oracle_assemble_quadratic_operator(
        _SNAPSHOTS.shape[0], _QUADRATIC_TERMS
    )
    previous_state = basis.T @ _PREVIOUS_FULL_STATE
    residual_polynomial = _oracle_assemble_residual_polynomial(
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
    tensors = _oracle_precompute_hrf_tensors(basis, residual_polynomial)
    initial_galerkin = _oracle_evaluate_hrf_galerkin(tensors, previous_state)
    initial_lspg = _oracle_evaluate_hrf_lspg(tensors, previous_state)
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
    initial_certificate = _oracle_certify_hrf_tensors(
        basis,
        residual_polynomial,
        tensors,
        previous_state[None, :],
    )
    if np.max(initial_certificate["errors"]) > 1e-12:
        raise ValueError("initial tensor/direct certificate failed")
    solution = _oracle_solve_hrf_pair(
        tensors,
        previous_state,
        _TOLERANCE,
        _MAX_ITERATIONS,
    )
    certificate = _oracle_compute_projection_gap(
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
            "setup": "energy_tolerance = 0.035\n",
            "call": "_pack(run_full_pipeline(energy_tolerance))",
            "gold_call": "_pack(_oracle_run_full_pipeline(energy_tolerance))",
        },
        {
            "setup": "energy_tolerance = 0.0003\n",
            "call": "_pack(run_full_pipeline(energy_tolerance))",
            "gold_call": "_pack(_oracle_run_full_pipeline(energy_tolerance))",
        },
        {
            "setup": """def run_model():
    try:
        run_full_pipeline(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(0.0)
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
