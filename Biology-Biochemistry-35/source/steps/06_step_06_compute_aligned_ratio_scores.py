"""
Compute elementwise ratios for two aligned positive value arrays.

numerator_values and denominator_values are finite, positive, one-dimensional arrays with the same shape.



For every aligned position, calculate:



numerator_values / denominator_values



Preserve the input order.

Returns
-------
ratio_scores : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_aligned_ratio_scores(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
) -> np.ndarray:
    """Compute elementwise ratios for aligned positive values.

    Parameters
    ----------
    numerator_values
        Finite positive one-dimensional array.
    denominator_values
        Finite positive one-dimensional array with the same shape as
        numerator_values.

    Returns
    -------
    np.ndarray
        Finite positive one-dimensional array preserving input order.
    """
    ratio_scores = np.empty_like(
        numerator_values,
        dtype=float,
    )
    return ratio_scores

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_aligned_ratio_scores(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
) -> np.ndarray:
    """Reference implementation for compute_aligned_ratio_scores."""
    import numpy as np

    numerators = np.asarray(
        numerator_values,
        dtype=float,
    )

    denominators = np.asarray(
        denominator_values,
        dtype=float,
    )

    if (
        numerators.ndim != 1
        or numerators.size < 1
    ):
        raise ValueError(
            "numerator_values must be a non-empty "
            "one-dimensional array"
        )

    if denominators.shape != numerators.shape:
        raise ValueError(
            "denominator_values must have the same shape "
            "as numerator_values"
        )

    if not np.all(
        np.isfinite(numerators)
    ):
        raise ValueError(
            "numerator_values must contain only finite values"
        )

    if not np.all(
        np.isfinite(denominators)
    ):
        raise ValueError(
            "denominator_values must contain only finite values"
        )

    if np.any(
        numerators <= 0.0
    ):
        raise ValueError(
            "numerator_values must be strictly positive"
        )

    if np.any(
        denominators <= 0.0
    ):
        raise ValueError(
            "denominator_values must be strictly positive"
        )

    log_scores = (
        np.log(numerators)
        - np.log(denominators)
    )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        ratio_scores = np.exp(
            log_scores
        )

    if (
        not np.all(
            np.isfinite(ratio_scores)
        )
        or np.any(
            ratio_scores <= 0.0
        )
    ):
        raise ValueError(
            "ratio scores must be finite and strictly positive"
        )

    return ratio_scores.astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_aligned_ratio_scores."""
    return [
        {
            "setup": """import numpy as np
numerator_values = np.array(
    [2.0, 9.0, 5.0],
    dtype=float,
)
denominator_values = np.array(
    [1.0, 3.0, 10.0],
    dtype=float,
)
""",
            "call": "compute_aligned_ratio_scores(numerator_values, denominator_values)",
            "gold_call": "_oracle_compute_aligned_ratio_scores(numerator_values, denominator_values)",
        },
        {
            "setup": """import numpy as np
numerator_values = np.array(
    [1.0e-15, 4.0e-9, 7.5e4, 3.2e11],
    dtype=float,
)
denominator_values = np.array(
    [2.0e-16, 8.0e-8, 2.5e3, 1.6e10],
    dtype=float,
)
""",
            "call": "compute_aligned_ratio_scores(numerator_values, denominator_values)",
            "gold_call": "_oracle_compute_aligned_ratio_scores(numerator_values, denominator_values)",
        },
        {
            "setup": """import numpy as np
numerator_values = np.array(
    [0.73, 1.19, 2.41, 8.06, 0.11],
    dtype=float,
)
denominator_values = np.array(
    [1.37, 0.52, 3.28, 2.15, 0.44],
    dtype=float,
)
""",
            "call": "compute_aligned_ratio_scores(numerator_values, denominator_values)",
            "gold_call": "_oracle_compute_aligned_ratio_scores(numerator_values, denominator_values)",
        },
        {
            "setup": """import numpy as np
numerator_values = np.array(
    [1.0, 2.0, 3.0],
    dtype=float,
)
denominator_values = np.array(
    [1.0, 0.0, 3.0],
    dtype=float,
)

def run_model():
    try:
        compute_aligned_ratio_scores(
            numerator_values,
            denominator_values,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_aligned_ratio_scores(
            numerator_values,
            denominator_values,
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
