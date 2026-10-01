"""
Construct the per-state numerical components required by the subsequent scalar calculation.

Each row of perturbed_states contains one packed symmetric zero-diagonal perturbation followed by its proposal count. Reconstruct the perturbation in the packed coordinate order associated with rg_matrix; the final proposal-count column is not part of the perturbation.



After excluding target_index, let r be the point-correlation vector between the target and the remaining traits. For retained state i, let A_i be the remaining-trait block of the complete perturbed correlation matrix and let u_i be the target-to-remaining-trait portion of the perturbation.



Return one row [a_i, b_i, c_i] for each retained state such that



v_i(s) = a_i * s**2 + b_i * s + c_i



equals



(s * r + u_i).T @ solve(A_i, s * r + u_i)



for any scalar s.

Returns
-------
components : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def construct_state_components(
    rg_matrix: np.ndarray,
    perturbed_states: np.ndarray,
    target_index: int,
) -> np.ndarray:
    """Return per-state numerical components for a scalar calculation.

    Parameters
    ----------
    rg_matrix : np.ndarray
        Square point correlation matrix for the current trait state.
    perturbed_states : np.ndarray
        Retained packed perturbations with a final proposal-count
        column.
    target_index : int
        Zero-based index identifying the target trait.

    Returns
    -------
    np.ndarray
        Matrix with one row per retained state and columns containing
        the quadratic, linear, and constant components.
    """
    return np.empty((0, 3), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_construct_state_components(
    rg_matrix: np.ndarray,
    perturbed_states: np.ndarray,
    target_index: int,
) -> np.ndarray:
    rg = np.asarray(rg_matrix, dtype=float)
    states = np.asarray(
        perturbed_states,
        dtype=float,
    )

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

    if (
        not isinstance(
            target_index,
            (int, np.integer),
        )
        or isinstance(
            target_index,
            (bool, np.bool_),
        )
    ):
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

    try:
        np.linalg.cholesky(rg)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "rg_matrix must be positive definite"
        ) from exc

    n_coordinates = (
        n_traits * (n_traits - 1) // 2
    )

    if (
        states.ndim != 2
        or states.shape[0] < 1
        or states.shape[1]
        != n_coordinates + 1
    ):
        raise ValueError(
            "perturbed_states has an incompatible shape"
        )

    if not np.all(np.isfinite(states)):
        raise ValueError(
            "perturbed_states must contain only finite values"
        )

    proposal_counts = states[:, -1]

    if (
        np.any(proposal_counts < 1.0)
        or not np.allclose(
            proposal_counts,
            np.round(proposal_counts),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "proposal counts must be positive integers"
        )

    packed_pairs = [
        (row, column)
        for column in range(n_traits - 1)
        for row in range(column + 1, n_traits)
    ]

    remaining = np.array(
        [
            index
            for index in range(n_traits)
            if index != target
        ],
        dtype=int,
    )

    point_vector = rg[
        remaining,
        target,
    ]

    components = np.empty(
        (states.shape[0], 3),
        dtype=float,
    )

    for state_index, state_row in enumerate(
        states
    ):
        error_matrix = np.zeros(
            (n_traits, n_traits),
            dtype=float,
        )

        for value, (row, column) in zip(
            state_row[:-1],
            packed_pairs,
        ):
            error_matrix[row, column] = value
            error_matrix[column, row] = value

        perturbed_matrix = rg + error_matrix

        try:
            np.linalg.cholesky(
                perturbed_matrix
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "each perturbed state must be "
                "positive definite"
            ) from exc

        auxiliary_matrix = perturbed_matrix[
            np.ix_(remaining, remaining)
        ]

        error_vector = error_matrix[
            remaining,
            target,
        ]

        try:
            solved_point = np.linalg.solve(
                auxiliary_matrix,
                point_vector,
            )
            solved_error = np.linalg.solve(
                auxiliary_matrix,
                error_vector,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "each auxiliary state must be nonsingular"
            ) from exc

        components[
            state_index,
            0,
        ] = float(
            point_vector @ solved_point
        )

        components[
            state_index,
            1,
        ] = float(
            point_vector @ solved_error
            + error_vector @ solved_point
        )

        components[
            state_index,
            2,
        ] = float(
            error_vector @ solved_error
        )

    if not np.all(np.isfinite(components)):
        raise ValueError(
            "state components must be finite"
        )

    return components

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for construct_state_components."""
    return [
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.30, 0.20],
    [0.30, 1.00, 0.10],
    [0.20, 0.10, 1.00],
], dtype=float)

perturbed_states = np.array([
    [ 0.010, -0.020,  0.005, 1.0],
    [-0.015,  0.010, -0.010, 2.0],
    [ 0.000,  0.000,  0.000, 1.0],
], dtype=float)

target_index = 0
""",
            "call": (
                "construct_state_components("
                "rg_matrix, perturbed_states, target_index)"
            ),
            "gold_call": (
                "_oracle_construct_state_components("
                "rg_matrix, perturbed_states, target_index)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00,  0.10,  0.25, 0.05],
    [0.10,  1.00, -0.10, 0.20],
    [0.25, -0.10,  1.00, 0.15],
    [0.05,  0.20,  0.15, 1.00],
], dtype=float)

perturbed_states = np.array([
    [ 0.005, -0.010,  0.008,  0.004, -0.003,  0.006, 1.0],
    [-0.004,  0.012, -0.006, -0.005,  0.007, -0.004, 1.0],
    [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000, 1.0],
], dtype=float)

target_index = 2
""",
            "call": (
                "construct_state_components("
                "rg_matrix, perturbed_states, target_index)"
            ),
            "gold_call": (
                "_oracle_construct_state_components("
                "rg_matrix, perturbed_states, target_index)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.40],
    [0.40, 1.00],
], dtype=float)

perturbed_states = np.array([
    [ 0.020, 1.0],
    [-0.010, 2.0],
    [ 0.000, 1.0],
], dtype=float)

target_index = 0
""",
            "call": (
                "construct_state_components("
                "rg_matrix, perturbed_states, target_index)"
            ),
            "gold_call": (
                "_oracle_construct_state_components("
                "rg_matrix, perturbed_states, target_index)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.30, 0.20],
    [0.30, 1.00, 0.10],
    [0.20, 0.10, 1.00],
], dtype=float)

perturbed_states = np.array([
    [0.01, -0.02, 1.0],
], dtype=float)

target_index = 0

def run_model():
    try:
        construct_state_components(
            rg_matrix,
            perturbed_states,
            target_index,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_construct_state_components(
            rg_matrix,
            perturbed_states,
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
