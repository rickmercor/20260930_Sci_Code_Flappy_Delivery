"""
Compute the final requested number of test individuals classified as high risk by constructing their prediction intervals, deriving the source-study quantitative-trait high-risk threshold from training phenotypes, and applying interval-based screening.

The final benchmark target depends on the full scientific chain. Cross-validated training information is first used to construct prediction intervals for every test individual while preserving fold-specific model predictions. The quantitative-trait high-risk threshold is then derived from the training phenotype distribution according to the source-study definition. The requested final scalar is the total number of test individuals whose prediction intervals extend into that derived high-risk phenotype range.

Returns
-------
A single integer equal to the total number of test individuals classified as high risk by the interval-based criterion after deriving the source-study quantitative-trait risk threshold from the training phenotypes. For the benchmark data, the expected result is 3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_target_interval_high_risk_count(
    observed: np.ndarray,
    heldout_predictions: np.ndarray,
    fold_ids: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    observed_test: np.ndarray,
    confidence_level: float
) -> int:
    """
    Compute the interval-based high-risk count.

    Parameters
    ----------
    observed : np.ndarray
        Observed training phenotypes.
    heldout_predictions : np.ndarray
        Cross-validated held-out training predictions.
    fold_ids : np.ndarray
        One-based training-fold identifiers.
    test_fold_predictions_matrix : np.ndarray
        Fold-specific predictions for all test individuals,
        with shape (n_test, n_folds).
    observed_test : np.ndarray
        Held-out observed phenotypes for test individuals.
    confidence_level : float
        Requested prediction-interval confidence level.

    Returns
    -------
    interval_high_risk_count : int
        Number of test individuals whose prediction interval
        extends above the source-derived quantitative-trait
        high-risk threshold.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_target_interval_high_risk_count(
    observed: np.ndarray,
    heldout_predictions: np.ndarray,
    fold_ids: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    observed_test: np.ndarray,
    confidence_level: float
) -> int:
    intervals = (
        _oracle_compute_prediction_intervals_for_tests(
            observed,
            heldout_predictions,
            fold_ids,
            test_fold_predictions_matrix,
            confidence_level
        )
    )

    metrics = (
        _oracle_compute_screening_metrics(
            intervals,
            test_fold_predictions_matrix,
            observed,
            observed_test
        )
    )

    return int(
        metrics[1]
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
    [ 1.0, 3.0],
    [ 0.0, 0.0],
    [-1.0, 2.0]
])

observed_test = np.array([
    6.0,
    0.0,
    4.0
])

confidence_level = 0.50
""",
            "call": """
compute_target_interval_high_risk_count(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    observed_test,
    confidence_level
)
""",
            "gold_call": """
_oracle_compute_target_interval_high_risk_count(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    observed_test,
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

observed_test = np.array([
    2.0,
    0.0,
    4.0
])

confidence_level = 0.60
""",
            "call": """
compute_target_interval_high_risk_count(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    observed_test,
    confidence_level
)
""",
            "gold_call": """
_oracle_compute_target_interval_high_risk_count(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    observed_test,
    confidence_level
)
""",
        },

        {
            "setup": """
import numpy as np

observed = np.array([
    0.0,
    1.0,
    2.0,
    3.0,
    4.0
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
    [-1.0,  0.0],
    [-1.0,  2.0],
    [ 3.0, -2.0]
])

observed_test = np.array([
    2.0,
    0.0,
    5.0
])

confidence_level = 0.60
""",
            "call": """
compute_target_interval_high_risk_count(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    observed_test,
    confidence_level
)
""",
            "gold_call": """
_oracle_compute_target_interval_high_risk_count(
    observed,
    heldout_predictions,
    fold_ids,
    test_fold_predictions_matrix,
    observed_test,
    confidence_level
)
""",
        },
    ]
