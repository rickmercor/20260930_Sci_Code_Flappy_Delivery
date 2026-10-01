"""
Select one candidate using inclusive bounds on one aligned value and a maximum on another.

candidate_ids is a one-dimensional integer array containing one unique identifier per candidate.



eligibility_values and priority_values are finite positive one-dimensional arrays aligned with candidate_ids.



A candidate is eligible when:



lower_bound <= eligibility_value <= upper_bound



Among eligible candidates, select the candidate with the largest priority value.



If multiple eligible candidates have exactly the same largest priority value, select the lowest candidate ID.



If no candidate satisfies the inclusive bounds, raise ValueError.



Return:



[selected candidate ID, selected priority value]

Returns
-------
selection : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_bounded_maximum(
    candidate_ids: np.ndarray,
    eligibility_values: np.ndarray,
    priority_values: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Select the eligible candidate with the largest priority value.

    Parameters
    ----------
    candidate_ids
        Non-empty one-dimensional integer array of unique candidate
        identifiers.
    eligibility_values
        Finite positive one-dimensional array aligned with candidate_ids.
    priority_values
        Finite positive one-dimensional array aligned with candidate_ids.
    lower_bound
        Finite positive inclusive lower eligibility bound.
    upper_bound
        Finite positive inclusive upper eligibility bound.

    Returns
    -------
    np.ndarray
        Shape-(2,) floating-point array containing
        [selected candidate ID, selected priority value].
    """
    selection = np.empty(
        2,
        dtype=float,
    )
    return selection

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_bounded_maximum(
    candidate_ids: np.ndarray,
    eligibility_values: np.ndarray,
    priority_values: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Reference implementation for select_bounded_maximum."""
    import numpy as np

    ids_raw = np.asarray(
        candidate_ids,
    )

    eligibility = np.asarray(
        eligibility_values,
        dtype=float,
    )

    priority = np.asarray(
        priority_values,
        dtype=float,
    )

    if (
        ids_raw.ndim != 1
        or ids_raw.size < 1
        or not np.issubdtype(
            ids_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "candidate_ids must be a non-empty "
            "one-dimensional integer array"
        )

    ids = ids_raw.astype(
        int,
        copy=False,
    )

    if np.unique(
        ids
    ).size != ids.size:
        raise ValueError(
            "candidate_ids must be unique"
        )

    if eligibility.shape != ids.shape:
        raise ValueError(
            "eligibility_values must align with candidate_ids"
        )

    if priority.shape != ids.shape:
        raise ValueError(
            "priority_values must align with candidate_ids"
        )

    if not np.all(
        np.isfinite(eligibility)
    ):
        raise ValueError(
            "eligibility_values must contain only finite values"
        )

    if not np.all(
        np.isfinite(priority)
    ):
        raise ValueError(
            "priority_values must contain only finite values"
        )

    if np.any(
        eligibility <= 0.0
    ):
        raise ValueError(
            "eligibility_values must be strictly positive"
        )

    if np.any(
        priority <= 0.0
    ):
        raise ValueError(
            "priority_values must be strictly positive"
        )

    if isinstance(
        lower_bound,
        (bool, np.bool_),
    ) or isinstance(
        upper_bound,
        (bool, np.bool_),
    ):
        raise ValueError(
            "bounds must be real numerical scalars"
        )

    try:
        lower = float(
            lower_bound
        )

        upper = float(
            upper_bound
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "bounds must be real numerical scalars"
        ) from exc

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or upper <= 0.0
        or lower > upper
    ):
        raise ValueError(
            "bounds must be finite, positive, and ordered"
        )

    eligible = (
        (eligibility >= lower)
        & (eligibility <= upper)
    )

    if not np.any(
        eligible
    ):
        raise ValueError(
            "no candidate satisfies the supplied bounds"
        )

    largest_priority = np.max(
        priority[eligible]
    )

    tied = (
        eligible
        & (priority == largest_priority)
    )

    selected_id = int(
        np.min(
            ids[tied]
        )
    )

    selected_priority = float(
        priority[
            np.flatnonzero(
                ids == selected_id
            )[0]
        ]
    )

    selection = np.array(
        [
            float(selected_id),
            selected_priority,
        ],
        dtype=float,
    )

    return selection

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for select_bounded_maximum."""
    return [
        {
            "setup": """import numpy as np
candidate_ids = np.array(
    [11, 12, 13],
    dtype=int,
)
eligibility_values = np.array(
    [0.8, 1.6, 1.2],
    dtype=float,
)
priority_values = np.array(
    [2.0, 9.0, 3.0],
    dtype=float,
)
lower_bound = 0.6
upper_bound = 1.4
""",
            "call": "select_bounded_maximum(candidate_ids, eligibility_values, priority_values, lower_bound, upper_bound)",
            "gold_call": "_oracle_select_bounded_maximum(candidate_ids, eligibility_values, priority_values, lower_bound, upper_bound)",
        },
        {
            "setup": """import numpy as np
candidate_ids = np.array(
    [5, 3, 1],
    dtype=int,
)
eligibility_values = np.array(
    [0.4, 1.8, 1.8001],
    dtype=float,
)
priority_values = np.array(
    [2.0, 4.0, 9.0],
    dtype=float,
)
lower_bound = 0.4
upper_bound = 1.8
""",
            "call": "select_bounded_maximum(candidate_ids, eligibility_values, priority_values, lower_bound, upper_bound)",
            "gold_call": "_oracle_select_bounded_maximum(candidate_ids, eligibility_values, priority_values, lower_bound, upper_bound)",
        },
        {
            "setup": """import numpy as np
candidate_ids = np.array(
    [8, 2, 1],
    dtype=int,
)
eligibility_values = np.array(
    [0.9, 1.1, 1.3],
    dtype=float,
)
priority_values = np.array(
    [5.0, 5.0, 4.0],
    dtype=float,
)
lower_bound = 0.7
upper_bound = 1.4
""",
            "call": "select_bounded_maximum(candidate_ids, eligibility_values, priority_values, lower_bound, upper_bound)",
            "gold_call": "_oracle_select_bounded_maximum(candidate_ids, eligibility_values, priority_values, lower_bound, upper_bound)",
        },
        {
            "setup": """import numpy as np
candidate_ids = np.array(
    [1, 2, 3],
    dtype=int,
)
eligibility_values = np.array(
    [0.2, 1.8, 2.1],
    dtype=float,
)
priority_values = np.array(
    [3.0, 4.0, 5.0],
    dtype=float,
)
lower_bound = 0.9
upper_bound = 1.6

def run_model():
    try:
        select_bounded_maximum(
            candidate_ids,
            eligibility_values,
            priority_values,
            lower_bound,
            upper_bound,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_select_bounded_maximum(
            candidate_ids,
            eligibility_values,
            priority_values,
            lower_bound,
            upper_bound,
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
