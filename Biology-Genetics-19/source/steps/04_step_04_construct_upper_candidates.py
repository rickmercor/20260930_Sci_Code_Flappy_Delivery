"""
Construct one candidate upper-bound value for each training individual by adding that individual's absolute cross-validated residual to the corresponding fold-matched test prediction.

The upper candidate set is the counterpart of the lower candidate set. Absolute cross-validated prediction errors are added to the fold-matched test predictions, producing a finite empirical set from which the upper prediction bound can be selected.

Returns
-------
A one-dimensional float array of length N with one upper candidate per training individual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_upper_candidates(
    matched_test_predictions: np.ndarray,
    residuals: np.ndarray
) -> np.ndarray:
    """
    Construct candidate values for the upper prediction-interval bound.

    Parameters
    ----------
    matched_test_predictions : np.ndarray
        One-dimensional array of fold-matched test predictions.
    residuals : np.ndarray
        One-dimensional array of non-negative absolute cross-validated
        residuals with the same length.

    Returns
    -------
    upper_candidates : np.ndarray
        One-dimensional float array formed by adding each residual
        to its matched test prediction.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_construct_upper_candidates(
    matched_test_predictions: np.ndarray,
    residuals: np.ndarray
) -> np.ndarray:
    import numpy as np
    matched_test_predictions = np.asarray(matched_test_predictions, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    if matched_test_predictions.ndim != 1 or residuals.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if matched_test_predictions.size != residuals.size:
        raise ValueError("Inputs must have the same length.")
    if matched_test_predictions.size == 0:
        raise ValueError("Inputs must be non-empty.")
    if np.any(residuals < 0):
        raise ValueError("Residual magnitudes must be non-negative.")

    return matched_test_predictions + residuals

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

matched = np.array([1.0, 2.0, 3.0])
residuals = np.array([0.2, 0.5, 1.0])
""",
            "call": "construct_upper_candidates(matched, residuals)",
            "gold_call": "_oracle_construct_upper_candidates(matched, residuals)",
        },
        {
            "setup": """
import numpy as np

matched = np.array([-1.0, -1.0, 0.0])
residuals = np.array([0.0, 2.0, 0.5])
""",
            "call": "construct_upper_candidates(matched, residuals)",
            "gold_call": "_oracle_construct_upper_candidates(matched, residuals)",
        },
        {
            "setup": """
import numpy as np

matched = np.array([0.1, 0.1, 0.1, 0.1])
residuals = np.array([3.0, 1.0, 2.0, 4.0])
""",
            "call": "construct_upper_candidates(matched, residuals)",
            "gold_call": "_oracle_construct_upper_candidates(matched, residuals)",
        },
    ]
