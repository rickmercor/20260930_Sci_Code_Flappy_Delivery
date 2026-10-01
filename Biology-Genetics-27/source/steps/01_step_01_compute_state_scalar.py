"""
Compute the target-indexed scalar associated with the supplied matrix state.

target_index identifies one row and column of rg_matrix. Remove that row and column to form A, and form b from the corresponding correlations with the indexed row in the remaining row order. Solve A x = b and return b.T @ x.

Returns
-------
scalar : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_state_scalar(
    rg_matrix: np.ndarray,
    target_index: int,
) -> float:
    """Return the scalar for the supplied matrix state.

    Parameters
    ----------
    rg_matrix : np.ndarray
        Square symmetric correlation matrix with unit diagonal.

    target_index : int
        Zero-based index identifying the target trait.

    Returns
    -------
    float
        Computed scalar for the supplied state.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_state_scalar(
    rg_matrix: np.ndarray,
    target_index: int,
) -> float:
    rg = np.asarray(rg_matrix, dtype=float)

    if (
        rg.ndim != 2
        or rg.shape[0] != rg.shape[1]
        or rg.shape[0] < 2
    ):
        raise ValueError(
            "rg_matrix must be square with at least two traits"
        )

    if not np.all(np.isfinite(rg)):
        raise ValueError(
            "rg_matrix must contain only finite values"
        )

    if not isinstance(target_index, (int, np.integer)):
        raise ValueError(
            "target_index must be an integer"
        )

    target = int(target_index)
    n_traits = rg.shape[0]

    if target < 0 or target >= n_traits:
        raise ValueError(
            "target_index is out of range"
        )

    if not np.allclose(
        rg,
        rg.T,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must be symmetric"
        )

    if not np.allclose(
        np.diag(rg),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "rg_matrix must have a unit diagonal"
        )

    if np.any(np.abs(rg) > 1.0 + 1e-12):
        raise ValueError(
            "correlations must lie in [-1, 1]"
        )

    remaining = np.array(
        [
            idx
            for idx in range(n_traits)
            if idx != target
        ],
        dtype=int,
    )

    b = rg[remaining, target]
    A = rg[np.ix_(remaining, remaining)]

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "remaining-trait matrix must be nonsingular"
        ) from exc

    scalar = float(b @ x)

    if not np.isfinite(scalar):
        raise ValueError(
            "state scalar must be finite"
        )

    return scalar

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for the state scalar."""

    return [
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00,  0.25, -0.15, 0.10],
    [0.25,  1.00,  0.20, 0.05],
    [-0.15, 0.20,  1.00, 0.30],
    [0.10,  0.05,  0.30, 1.00],
], dtype=float)

target_index = 0
""",
            "call": (
                "compute_state_scalar("
                "rg_matrix, target_index)"
            ),
            "gold_call": (
                "_oracle_compute_state_scalar("
                "rg_matrix, target_index)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.40],
    [0.40, 1.00],
], dtype=float)

target_index = 0
""",
            "call": (
                "compute_state_scalar("
                "rg_matrix, target_index)"
            ),
            "gold_call": (
                "_oracle_compute_state_scalar("
                "rg_matrix, target_index)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.05,  0.25,  0.20],
    [0.05, 1.00,  0.10,  0.30],
    [0.25, 0.10,  1.00, -0.15],
    [0.20, 0.30, -0.15,  1.00],
], dtype=float)

target_index = 2
""",
            "call": (
                "compute_state_scalar("
                "rg_matrix, target_index)"
            ),
            "gold_call": (
                "_oracle_compute_state_scalar("
                "rg_matrix, target_index)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.0, 0.2, 0.2],
    [0.2, 1.0, 1.0],
    [0.2, 1.0, 1.0],
], dtype=float)

target_index = 0

def run_model():
    try:
        compute_state_scalar(
            rg_matrix,
            target_index,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_state_scalar(
            rg_matrix,
            target_index,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
