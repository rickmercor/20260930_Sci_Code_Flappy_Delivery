"""
Compute phenotype prediction intervals for multiple test individuals using common cross-validated training data and each test individual's fold-specific predictions.

A single cross-validated training cohort provides the residual information required to construct prediction intervals for multiple new individuals. Each test individual has one prediction from every fold-specific model. The interval construction is performed independently for each test individual while preserving the correspondence between training residuals and the appropriate fold-specific test prediction. The resulting output contains one lower and one upper prediction bound for every test individual.

Returns
-------
A NumPy float array of shape (n_test, 2). Each row corresponds to one test individual, with column 0 containing the unrounded lower prediction bound and column 1 containing the unrounded upper prediction bound.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_prediction_intervals_for_tests(
    observed: np.ndarray,
    heldout_predictions: np.ndarray,
    fold_ids: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    confidence_level: float
) -> np.ndarray:
    """
    Compute prediction intervals for multiple test individuals.

    Parameters
    ----------
    observed : np.ndarray
        Observed training phenotypes.
    heldout_predictions : np.ndarray
        Cross-validated held-out training predictions.
    fold_ids : np.ndarray
        One-based fold identifier for each training individual.
    test_fold_predictions_matrix : np.ndarray
        Two-dimensional array with shape (n_test, n_folds),
        containing one fold-specific prediction per test
        individual and fold.
    confidence_level : float
        Requested prediction-interval confidence level.

    Returns
    -------
    intervals : np.ndarray
        Float array of shape (n_test, 2), with columns
        [lower_bound, upper_bound].
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_prediction_intervals_for_tests(
    observed: np.ndarray,
    heldout_predictions: np.ndarray,
    fold_ids: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    confidence_level: float
) -> np.ndarray:
    import numpy as np

    test_matrix = np.asarray(
        test_fold_predictions_matrix,
        dtype=float
    )

    if test_matrix.ndim != 2:
        raise ValueError(
            "test_fold_predictions_matrix must be two-dimensional."
        )

    if (
        test_matrix.shape[0] == 0
        or test_matrix.shape[1] == 0
    ):
        raise ValueError(
            "test_fold_predictions_matrix must be non-empty."
        )

    if not np.all(
        np.isfinite(test_matrix)
    ):
        raise ValueError(
            "Test fold predictions must be finite."
        )

    intervals = []

    for row in test_matrix:
        interval = (
            _oracle_compute_prediction_interval(
                observed,
                heldout_predictions,
                fold_ids,
                row,
                confidence_level
            )
        )

        interval = np.asarray(
            interval,
            dtype=float
        )

        if interval.shape != (2,):
            raise ValueError(
                "Each prediction interval must contain two bounds."
            )

        intervals.append(
            interval
        )

    return np.vstack(
        intervals
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

observed = np.array([
    0.0,
    1.0,
    2.0,
    3.0
])

heldout_predictions = np.zeros(4)

fold_ids = np.array([
    1,
    1,
    2,
    2
])

test_fold_predictions_matrix = np.array([
    [1.0, 3.0],
    [2.0, 4.0]
])

confidence_level = 0.50
""",
            "call": """
compute_prediction_intervals_for_tests(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    confidence_level
)
""",
            "gold_call": """
_oracle_compute_prediction_intervals_for_tests(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    confidence_level
)
""",
        },

        {
            "setup": """
import numpy as np

observed = np.array([
    -1.0,
     0.0,
     1.0,
     2.0,
     3.0
])

heldout_predictions = np.zeros(5)

fold_ids = np.array([
    1,
    2,
    1,
    2,
    1
])

test_fold_predictions_matrix = np.array([
    [ 0.0,  1.0],
    [-1.0,  2.0],
    [ 3.0, -2.0]
])

confidence_level = 0.60
""",
            "call": """
compute_prediction_intervals_for_tests(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    confidence_level
)
""",
            "gold_call": """
_oracle_compute_prediction_intervals_for_tests(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    confidence_level
)
""",
        },

        {
            "setup": """
import numpy as np

observed = np.array([
    1.0,
    2.0,
    3.0,
    4.0,
    5.0
])

heldout_predictions = np.zeros(5)

fold_ids = np.ones(
    5,
    dtype=int
)

test_fold_predictions_matrix = np.array([
    [0.0],
    [2.0]
])

confidence_level = 0.50
""",
            "call": """
compute_prediction_intervals_for_tests(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    confidence_level
)
""",
            "gold_call": """
_oracle_compute_prediction_intervals_for_tests(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    confidence_level
)
""",
        },
    ]
