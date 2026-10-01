"""
Select the numerical trait-retention mask for one supplied matrix threshold.

target_scores follows the current non-target trait order after target_index is excluded.

For each pair of non-target traits whose squared correlation is strictly greater than threshold, retain the member with the larger target score and mark the other member for removal. Apply the comparison to every qualifying pair in the current matrix. Never remove the target trait.

If a qualifying pair has target scores equal within 1e-12, raise ValueError. Return a length-n numerical mask containing 1 for retained traits and 0 for removed traits.

Returns
-------
keep_mask : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_state_mask(
    rg_matrix: np.ndarray,
    target_index: int,
    target_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return the trait-retention mask for one matrix threshold.

    Parameters
    ----------
    rg_matrix : np.ndarray
        Square correlation matrix for the current trait state.
    target_index : int
        Zero-based index identifying the target trait.
    target_scores : np.ndarray
        Target-associated scores in current non-target trait order.
    threshold : float
        Squared-correlation threshold for the current comparison.

    Returns
    -------
    np.ndarray
        Length-n numerical mask containing 1 for retained traits
        and 0 for removed traits.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_state_mask(
    rg_matrix: np.ndarray,
    target_index: int,
    target_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    rg = np.asarray(rg_matrix, dtype=float)
    scores = np.asarray(
        target_scores,
        dtype=float,
    )

    if (
        rg.ndim != 2
        or rg.shape[0] != rg.shape[1]
        or rg.shape[0] < 3
    ):
        raise ValueError(
            "rg_matrix must be square with at least three traits"
        )

    if not np.all(np.isfinite(rg)):
        raise ValueError(
            "rg_matrix must contain only finite values"
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

    auxiliary = np.array(
        [
            index
            for index in range(n_traits)
            if index != target
        ],
        dtype=int,
    )

    if (
        scores.shape != (auxiliary.size,)
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError(
            "target_scores must align with non-target traits"
        )

    if (
        not isinstance(
            threshold,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            threshold,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "threshold must be a real number"
        )

    threshold_value = float(threshold)

    if (
        not np.isfinite(threshold_value)
        or threshold_value < 0.0
        or threshold_value >= 1.0
    ):
        raise ValueError(
            "threshold must lie in [0, 1)"
        )

    keep_mask = np.ones(
        n_traits,
        dtype=float,
    )

    for first_position in range(
        auxiliary.size - 1
    ):
        for second_position in range(
            first_position + 1,
            auxiliary.size,
        ):
            first_trait = int(
                auxiliary[
                    first_position
                ]
            )

            second_trait = int(
                auxiliary[
                    second_position
                ]
            )

            squared_correlation = float(
                rg[
                    first_trait,
                    second_trait,
                ]
                ** 2
            )

            if (
                squared_correlation
                > threshold_value
            ):
                first_score = float(
                    scores[
                        first_position
                    ]
                )

                second_score = float(
                    scores[
                        second_position
                    ]
                )

                if (
                    abs(
                        first_score
                        - second_score
                    )
                    <= 1e-12
                ):
                    raise ValueError(
                        "a qualifying pair has tied target scores"
                    )

                if first_score > second_score:
                    keep_mask[
                        second_trait
                    ] = 0.0
                else:
                    keep_mask[
                        first_trait
                    ] = 0.0

    keep_mask[target] = 1.0

    return keep_mask

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for select_state_mask."""
    return [
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.17, 0.60, 0.22, 0.16],
    [0.17, 1.00, 0.72, 0.18, 0.08],
    [0.60, 0.72, 1.00, 0.15, 0.10],
    [0.22, 0.18, 0.15, 1.00, 0.25],
    [0.16, 0.08, 0.10, 0.25, 1.00],
], dtype=float)

target_index = 0

target_scores = np.array([
    6.7011850127,
    3.9444714937,
    1.0898703479,
    6.3013302554,
], dtype=float)

threshold = 0.5
""",
            "call": (
                "select_state_mask("
                "rg_matrix, target_index, "
                "target_scores, threshold)"
            ),
            "gold_call": (
                "_oracle_select_state_mask("
                "rg_matrix, target_index, "
                "target_scores, threshold)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.10, 0.12, 0.08],
    [0.10, 1.00, 0.75, 0.30],
    [0.12, 0.75, 1.00, 0.72],
    [0.08, 0.30, 0.72, 1.00],
], dtype=float)

target_index = 0

target_scores = np.array([
    5.0,
    4.0,
    3.0,
], dtype=float)

threshold = 0.5
""",
            "call": (
                "select_state_mask("
                "rg_matrix, target_index, "
                "target_scores, threshold)"
            ),
            "gold_call": (
                "_oracle_select_state_mask("
                "rg_matrix, target_index, "
                "target_scores, threshold)"
            ),
        },
        {
            "setup": """import numpy as np

rg_matrix = np.array([
    [1.00, 0.75, 0.15, 0.10],
    [0.75, 1.00, 0.12, 0.20],
    [0.15, 0.12, 1.00, 0.18],
    [0.10, 0.20, 0.18, 1.00],
], dtype=float)

target_index = 2

target_scores = np.array([
    2.0,
    5.0,
    1.0,
], dtype=float)

threshold = 0.5
""",
            "call": (
                "select_state_mask("
                "rg_matrix, target_index, "
                "target_scores, threshold)"
            ),
            "gold_call": (
                "_oracle_select_state_mask("
                "rg_matrix, target_index, "
                "target_scores, threshold)"
            ),
        },
        {
            "setup": """import numpy as np

correlation = 0.7

rg_matrix = np.array([
    [1.00, 0.20, 0.10],
    [0.20, 1.00, correlation],
    [0.10, correlation, 1.00],
], dtype=float)

target_index = 0

target_scores = np.array([
    4.0,
    3.0,
], dtype=float)

threshold = float(correlation ** 2)
""",
            "call": (
                "select_state_mask("
                "rg_matrix, target_index, "
                "target_scores, threshold)"
            ),
            "gold_call": (
                "_oracle_select_state_mask("
                "rg_matrix, target_index, "
                "target_scores, threshold)"
            ),
        },
    ]
