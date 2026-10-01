"""
Derive the source-study quantitative-trait high-risk threshold from the training phenotypes and compare interval-based screening, point-prediction screening, and held-out observed test phenotypes at that threshold.

The source study defines high risk for quantitative traits using an upper-tail phenotype criterion derived from the training phenotype distribution. Once that source-specific threshold has been derived, interval-based screening identifies individuals whose prediction intervals extend into the high-risk phenotype range, whereas point-based screening compares only the arithmetic-mean prediction with the same threshold. Held-out observed test phenotypes provide retrospective true-risk labels and allow identification success to be measured among truly high-risk individuals.

Returns
-------
A NumPy float array of shape (6,) containing, in order: the source-derived quantitative-trait high-risk phenotype threshold, the number classified high risk by the interval-based criterion, the number classified high risk by the arithmetic-mean point-prediction criterion, the number truly high risk according to held-out observed phenotypes, the number of truly high-risk individuals identified by the interval-based criterion, and the corresponding identification success rate. For the benchmark data, the expected values are approximately [1.6117416873409185, 3, 0, 3, 2, 0.6666666666666666].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_screening_metrics(
    intervals: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    observed_train: np.ndarray,
    observed_test: np.ndarray
) -> np.ndarray:
    """
    Compute source-derived high-risk screening metrics.

    Parameters
    ----------
    intervals : np.ndarray
        Prediction intervals with shape (n_test, 2).
    test_fold_predictions_matrix : np.ndarray
        Fold-specific predictions with shape
        (n_test, n_folds).
    observed_train : np.ndarray
        Observed training phenotypes used to derive the
        quantitative-trait high-risk threshold.
    observed_test : np.ndarray
        Held-out observed phenotypes for the test individuals.

    Returns
    -------
    metrics : np.ndarray
        Float array containing:
        [high_risk_threshold,
         interval_high_risk_count,
         point_high_risk_count,
         true_high_risk_count,
         true_high_risk_identified,
         identification_success_rate].
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_screening_metrics(
    intervals: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    observed_train: np.ndarray,
    observed_test: np.ndarray
) -> np.ndarray:
    import numpy as np

    intervals = np.asarray(
        intervals,
        dtype=float
    )

    test_matrix = np.asarray(
        test_fold_predictions_matrix,
        dtype=float
    )

    observed_train = np.asarray(
        observed_train,
        dtype=float
    )

    observed_test = np.asarray(
        observed_test,
        dtype=float
    )

    if (
        intervals.ndim != 2
        or intervals.shape[1] != 2
    ):
        raise ValueError(
            "intervals must have shape (n_test, 2)."
        )

    if test_matrix.ndim != 2:
        raise ValueError(
            "test_fold_predictions_matrix must be two-dimensional."
        )

    if observed_train.ndim != 1:
        raise ValueError(
            "observed_train must be one-dimensional."
        )

    if observed_test.ndim != 1:
        raise ValueError(
            "observed_test must be one-dimensional."
        )

    if observed_train.size == 0:
        raise ValueError(
            "At least one training phenotype is required."
        )

    n_test = intervals.shape[0]

    if (
        test_matrix.shape[0] != n_test
        or observed_test.size != n_test
    ):
        raise ValueError(
            "All test inputs must describe the same individuals."
        )

    if n_test == 0:
        raise ValueError(
            "At least one test individual is required."
        )

    if not (
        np.all(np.isfinite(intervals))
        and np.all(np.isfinite(test_matrix))
        and np.all(np.isfinite(observed_train))
        and np.all(np.isfinite(observed_test))
    ):
        raise ValueError(
            "All screening inputs must be finite."
        )

    sorted_train = np.sort(
        observed_train
    )

    position = (
        (sorted_train.size - 1)
        * 0.95
    )

    lower_index = int(
        np.floor(position)
    )

    upper_index = int(
        np.ceil(position)
    )

    fraction = (
        position
        - lower_index
    )

    high_risk_threshold = (
        sorted_train[lower_index]
        + fraction
        * (
            sorted_train[upper_index]
            - sorted_train[lower_index]
        )
    )

    point_predictions = np.mean(
        test_matrix,
        axis=1
    )

    interval_high_risk = (
        intervals[:, 1]
        > high_risk_threshold
    )

    point_high_risk = (
        point_predictions
        > high_risk_threshold
    )

    true_high_risk = (
        observed_test
        > high_risk_threshold
    )

    true_count = int(
        np.sum(
            true_high_risk
        )
    )

    if true_count == 0:
        raise ValueError(
            "Identification success is undefined when no "
            "test individual is truly high risk."
        )

    true_identified = int(
        np.sum(
            interval_high_risk
            & true_high_risk
        )
    )

    success_rate = (
        true_identified
        / true_count
    )

    return np.array(
        [
            float(high_risk_threshold),
            float(np.sum(interval_high_risk)),
            float(np.sum(point_high_risk)),
            float(true_count),
            float(true_identified),
            float(success_rate)
        ],
        dtype=float
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

intervals = np.array([
    [-1.0, 3.0],
    [-2.0, 2.0],
    [ 0.0, 4.0]
])

test_fold_predictions_matrix = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0]
])

observed_train = np.array([
    0.0,
    1.0,
    2.0,
    3.0
])

observed_test = np.array([
    4.0,
    1.0,
    5.0
])
""",
            "call": """
compute_screening_metrics(
    intervals,
    test_fold_predictions_matrix,
    observed_train,
    observed_test
)
""",
            "gold_call": """
_oracle_compute_screening_metrics(
    intervals,
    test_fold_predictions_matrix,
    observed_train,
    observed_test
)
""",
        },

        {
            "setup": """
import numpy as np

intervals = np.array([
    [-2.0, 1.0],
    [ 0.0, 4.0],
    [-1.0, 2.5],
    [ 1.0, 5.0]
])

test_fold_predictions_matrix = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [4.0, 4.0],
    [2.0, 2.0]
])

observed_train = np.array([
    -1.0,
     0.0,
     1.0,
     2.0,
     3.0
])

observed_test = np.array([
    0.0,
    4.5,
    1.0,
    4.0
])
""",
            "call": """
compute_screening_metrics(
    intervals,
    test_fold_predictions_matrix,
    observed_train,
    observed_test
)
""",
            "gold_call": """
_oracle_compute_screening_metrics(
    intervals,
    test_fold_predictions_matrix,
    observed_train,
    observed_test
)
""",
        },

        {
            "setup": """
import numpy as np

intervals = np.array([
    [-1.0, 5.0],
    [-2.0, 0.0],
    [ 0.0, 6.0]
])

test_fold_predictions_matrix = np.array([
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    [5.0, 5.0, 5.0]
])

observed_train = np.array([
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0
])

observed_test = np.array([
    5.0,
    1.0,
    6.0
])
""",
            "call": """
compute_screening_metrics(
    intervals,
    test_fold_predictions_matrix,
    observed_train,
    observed_test
)
""",
            "gold_call": """
_oracle_compute_screening_metrics(
    intervals,
    test_fold_predictions_matrix,
    observed_train,
    observed_test
)
""",
        },
    ]
