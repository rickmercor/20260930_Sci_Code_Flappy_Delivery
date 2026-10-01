"""
Sort a numerical candidate set in ascending order and return the value at a specified one-based rank without percentile interpolation.

Once the candidate phenotype values have been formed, the prediction bounds are selected as finite order statistics. This differs from applying a continuous or interpolated quantile function because a specific observed candidate value must be selected from the finite set.

Returns
-------
A single float equal to the requested one-based order statistic.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_order_statistic(
    values: np.ndarray,
    rank: int
) -> float:
    """
    Select a one-based ascending order statistic without interpolation.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional finite numeric candidate array.
    rank : int
        One-based integer rank between 1 and len(values).

    Returns
    -------
    selected_value : float
        Candidate value at the requested ascending rank.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_order_statistic(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

values = np.array([4.0, 1.0, 3.0, 2.0])
rank = 2
""",
            "call": "select_order_statistic(values, rank)",
            "gold_call": "_oracle_select_order_statistic(values, rank)",
        },
        {
            "setup": """
import numpy as np

values = np.array([-1.0, -5.0, 2.0, 0.0, 7.0])
rank = 4
""",
            "call": "select_order_statistic(values, rank)",
            "gold_call": "_oracle_select_order_statistic(values, rank)",
        },
        {
            "setup": """
import numpy as np

values = np.array([3.5, 3.5, 1.2, 9.1])
rank = 1
""",
            "call": "select_order_statistic(values, rank)",
            "gold_call": "_oracle_select_order_statistic(values, rank)",
        },
    ]
