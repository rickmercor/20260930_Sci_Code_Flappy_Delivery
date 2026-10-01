"""
Compute the one-based finite-sample order-statistic ranks used for the lower and upper prediction-interval bounds from the training-sample size and requested confidence level.

The prediction-interval construction uses finite-sample order statistics rather than interpolated empirical percentiles. For N training individuals and confidence level 1 - alpha, the lower and upper candidate ranks follow the ceiling-based finite-sample rule used in the source implementation. The returned ranks are one-based and are subsequently applied to the sorted lower and upper candidate arrays. For the benchmark training cohort of 39 individuals at 95% confidence, the required ranks are 2 and 38.

Returns
-------
A NumPy integer array of shape (2,) containing [lower_rank, upper_rank], where both ranks use one-based indexing. For N = 39 and confidence_level = 0.95, the expected result is [2, 38].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_interval_ranks(
    n_training: int,
    confidence_level: float
) -> np.ndarray:
    """
    Compute one-based lower and upper finite-sample order-statistic ranks.

    Parameters
    ----------
    n_training : int
        Positive number of training individuals.
    confidence_level : float
        Requested prediction-interval confidence level strictly between
        0 and 1.

    Returns
    -------
    ranks : np.ndarray
        Integer array of shape (2,) containing
        [lower_rank, upper_rank].
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_interval_ranks(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

n_training = 39
confidence_level = 0.95
""",
            "call": """
compute_interval_ranks(
    n_training,
    confidence_level
)
""",
            "gold_call": """
_oracle_compute_interval_ranks(
    n_training,
    confidence_level
)
""",
        },

        {
            "setup": """
import numpy as np

n_training = 19
confidence_level = 0.90
""",
            "call": """
compute_interval_ranks(
    n_training,
    confidence_level
)
""",
            "gold_call": """
_oracle_compute_interval_ranks(
    n_training,
    confidence_level
)
""",
        },

        {
            "setup": """
import numpy as np

n_training = 9
confidence_level = 0.80
""",
            "call": """
compute_interval_ranks(
    n_training,
    confidence_level
)
""",
            "gold_call": """
_oracle_compute_interval_ranks(
    n_training,
    confidence_level
)
""",
        },
    ]
