#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_cv_residuals(
    observed: np.ndarray,
    heldout_predictions: np.ndarray
) -> np.ndarray:
    import numpy as np

    observed = np.asarray(observed, dtype=float)
    heldout_predictions = np.asarray(heldout_predictions, dtype=float)

    if observed.ndim != 1 or heldout_predictions.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if observed.size != heldout_predictions.size:
        raise ValueError("Inputs must have the same length.")
    if observed.size == 0:
        raise ValueError("Inputs must be non-empty.")

    return np.abs(observed - heldout_predictions)

def align_test_predictions_by_fold(
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

def construct_lower_candidates(
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

    return matched_test_predictions - residuals

def construct_upper_candidates(
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

def compute_interval_ranks(
    n_training: int,
    confidence_level: float
) -> np.ndarray:
    import numpy as np
    from decimal import Decimal, ROUND_CEILING

    if isinstance(n_training, bool):
        raise ValueError(
            "n_training must be a positive integer."
        )

    try:
        n_value = float(n_training)
    except (TypeError, ValueError):
        raise ValueError(
            "n_training must be a positive integer."
        )

    if not np.isfinite(n_value):
        raise ValueError(
            "n_training must be finite."
        )

    if not n_value.is_integer() or n_value <= 0:
        raise ValueError(
            "n_training must be a positive integer."
        )

    try:
        confidence_value = float(confidence_level)
    except (TypeError, ValueError):
        raise ValueError(
            "confidence_level must be numeric."
        )

    if not np.isfinite(confidence_value):
        raise ValueError(
            "confidence_level must be finite."
        )

    if not (0.0 < confidence_value < 1.0):
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
        )

    n_training = int(n_value)

    confidence_decimal = Decimal(
        str(confidence_value)
    )

    alpha = (
        Decimal("1")
        - confidence_decimal
    )

    n_plus_one = Decimal(
        n_training + 1
    )

    lower_rank = int(
        (
            alpha
            * n_plus_one
        ).to_integral_value(
            rounding=ROUND_CEILING
        )
    )

    upper_rank = int(
        (
            confidence_decimal
            * n_plus_one
        ).to_integral_value(
            rounding=ROUND_CEILING
        )
    )

    if (
        lower_rank < 1
        or lower_rank > n_training
    ):
        raise ValueError(
            "Lower rank falls outside the candidate set."
        )

    if (
        upper_rank < 1
        or upper_rank > n_training
    ):
        raise ValueError(
            "Upper rank falls outside the candidate set."
        )

    if lower_rank > upper_rank:
        raise ValueError(
            "Lower rank cannot exceed upper rank."
        )

    return np.array(
        [
            lower_rank,
            upper_rank
        ],
        dtype=int
    )

def select_order_statistic(
    values: np.ndarray,
    rank: int
) -> float:
    import numpy as np

    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "values must be a one-dimensional array."
        )

    if values.size == 0:
        raise ValueError(
            "values must be non-empty."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "values must contain only finite numbers."
        )

    if isinstance(rank, bool):
        raise ValueError(
            "rank must be an integer."
        )

    try:
        rank_value = float(rank)
    except (TypeError, ValueError):
        raise ValueError(
            "rank must be an integer."
        )

    if not np.isfinite(rank_value):
        raise ValueError(
            "rank must be finite."
        )

    if not rank_value.is_integer():
        raise ValueError(
            "rank must be an integer."
        )

    rank = int(rank_value)

    if rank < 1 or rank > values.size:
        raise ValueError(
            "rank is outside the candidate set."
        )

    ordered_values = np.sort(values)

    return float(
        ordered_values[rank - 1]
    )

def compute_prediction_interval(
    observed: np.ndarray,
    heldout_predictions: np.ndarray,
    fold_ids: np.ndarray,
    test_fold_predictions: np.ndarray,
    confidence_level: float
) -> np.ndarray:
    import numpy as np

    observed_array = np.asarray(
        observed,
        dtype=float
    )

    if observed_array.ndim != 1:
        raise ValueError(
            "observed must be one-dimensional."
        )

    if observed_array.size == 0:
        raise ValueError(
            "At least one training individual is required."
        )

    residuals = compute_cv_residuals(
        observed,
        heldout_predictions
    )

    matched_test_predictions = (
        align_test_predictions_by_fold(
            fold_ids,
            test_fold_predictions
        )
    )

    lower_candidates = (
        construct_lower_candidates(
            matched_test_predictions,
            residuals
        )
    )

    upper_candidates = (
        construct_upper_candidates(
            matched_test_predictions,
            residuals
        )
    )

    ranks = compute_interval_ranks(
        observed_array.size,
        confidence_level
    )

    lower_bound = select_order_statistic(
        lower_candidates,
        int(ranks[0])
    )

    upper_bound = select_order_statistic(
        upper_candidates,
        int(ranks[1])
    )

    if not (
        np.isfinite(lower_bound)
        and np.isfinite(upper_bound)
    ):
        raise ValueError(
            "Prediction interval bounds must be finite."
        )

    if lower_bound > upper_bound:
        raise ValueError(
            "Lower prediction bound cannot exceed upper bound."
        )

    return np.array(
        [
            float(lower_bound),
            float(upper_bound)
        ],
        dtype=float
    )

def compute_prediction_intervals_for_tests(
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
            compute_prediction_interval(
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

def compute_screening_metrics(
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

def compute_target_interval_high_risk_count(
    observed: np.ndarray,
    heldout_predictions: np.ndarray,
    fold_ids: np.ndarray,
    test_fold_predictions_matrix: np.ndarray,
    observed_test: np.ndarray,
    confidence_level: float
) -> int:
    intervals = (
        compute_prediction_intervals_for_tests(
            observed,
            heldout_predictions,
            fold_ids,
            test_fold_predictions_matrix,
            confidence_level
        )
    )

    metrics = (
        compute_screening_metrics(
            intervals,
            test_fold_predictions_matrix,
            observed,
            observed_test
        )
    )

    return int(
        metrics[1]
    )
SCICODE_GOLD_EOF
