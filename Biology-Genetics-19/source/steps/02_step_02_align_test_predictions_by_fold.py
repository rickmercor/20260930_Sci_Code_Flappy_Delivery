"""
Assign to each training individual the test-individual prediction produced by the model that held out the same cross-validation fold.

CV+ preserves the cross-validation structure when transferring training prediction errors to a new individual. A training residual from fold k must be combined with the test prediction from the model that also held out fold k. Averaging the fold-specific test predictions before this step would discard information required by the prediction-interval construction.

Returns
-------
A one-dimensional float array of length N containing the test prediction associated with each training individual's fold.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def align_test_predictions_by_fold(
    fold_ids: np.ndarray,
    test_fold_predictions: np.ndarray
) -> np.ndarray:
    """
    Match each training individual to the test prediction from its fold model.

    Parameters
    ----------
    fold_ids : np.ndarray
        One-dimensional array of one-based fold identifiers for the training
        individuals.
    test_fold_predictions : np.ndarray
        One-dimensional array containing one test-individual prediction for
        each fold, ordered by one-based fold number.

    Returns
    -------
    matched_test_predictions : np.ndarray
        One-dimensional float array containing one fold-matched test
        prediction per training individual.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_align_test_predictions_by_fold(
    fold_ids: np.ndarray,
    test_fold_predictions: np.ndarray
) -> np.ndarray:
    import numpy as np

    fold_ids = np.asarray(fold_ids)
    test_fold_predictions = np.asarray(
        test_fold_predictions,
        dtype=float
    )

    if fold_ids.ndim != 1 or test_fold_predictions.ndim != 1:
        raise ValueError(
            "Inputs must be one-dimensional."
        )

    if fold_ids.size == 0 or test_fold_predictions.size == 0:
        raise ValueError(
            "Inputs must be non-empty."
        )

    if not np.all(np.isfinite(fold_ids)):
        raise ValueError(
            "Fold identifiers must be finite."
        )

    if not np.all(np.isfinite(test_fold_predictions)):
        raise ValueError(
            "Test predictions must be finite."
        )

    if not np.all(fold_ids == np.floor(fold_ids)):
        raise ValueError(
            "Fold identifiers must be integers."
        )

    fold_index = fold_ids.astype(int) - 1

    if np.any(fold_index < 0):
        raise ValueError(
            "Fold identifiers must be one-based positive integers."
        )

    if np.any(fold_index >= test_fold_predictions.size):
        raise ValueError(
            "Fold identifier is outside the available fold range."
        )

    return test_fold_predictions[fold_index]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

fold_ids = np.array([1, 1, 2, 3, 3])
test_predictions = np.array([0.1, 0.2, 0.3])
""",
            "call": "align_test_predictions_by_fold(fold_ids, test_predictions)",
            "gold_call": "_oracle_align_test_predictions_by_fold(fold_ids, test_predictions)",
        },
        {
            "setup": """
import numpy as np

fold_ids = np.array([5, 4, 3, 2, 1])
test_predictions = np.array([-0.2, 0.4, 1.1, -1.3, 2.0])
""",
            "call": "align_test_predictions_by_fold(fold_ids, test_predictions)",
            "gold_call": "_oracle_align_test_predictions_by_fold(fold_ids, test_predictions)",
        },
        {
            "setup": """
import numpy as np

fold_ids = np.array([2, 2, 2, 1])
test_predictions = np.array([10.0, 20.0])
""",
            "call": "align_test_predictions_by_fold(fold_ids, test_predictions)",
            "gold_call": "_oracle_align_test_predictions_by_fold(fold_ids, test_predictions)",
        },
    ]
