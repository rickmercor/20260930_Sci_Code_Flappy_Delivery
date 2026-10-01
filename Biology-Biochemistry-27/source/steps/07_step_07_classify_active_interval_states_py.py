"""
Classify aligned states using active-only inclusive interval requirements.

validation_values contains one row per state and one column per validation coordinate.

active_mask has the same shape and identifies which validation coordinates are active in each state.

lower_bounds and upper_bounds define one inclusive interval per validation coordinate.

For state i and validation coordinate j:

- if active_mask[i,j] is True, the coordinate satisfies its requirement exactly when

  lower_bounds[j] <= validation_values[i,j] <= upper_bounds[j];

- if active_mask[i,j] is False, that coordinate does not affect classification for that state, regardless of its finite validation value.

A state is classified True only when every active validation coordinate satisfies its inclusive interval.

A state with no active validation coordinates is classified True.

Return one Boolean value per state.

validation_values must be a finite non-empty two-dimensional numerical array.

active_mask must be a Boolean array with exactly the same shape.

lower_bounds and upper_bounds must be finite one-dimensional numerical arrays aligned with the validation columns, and lower_bounds must not exceed upper_bounds elementwise.

Raise ValueError if these requirements are not satisfied.

Returns
-------
1D NumPy Boolean array of length n_states indicating which states satisfy every active inclusive interval requirement.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def classify_active_interval_states(
    validation_values: np.ndarray,
    active_mask: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """
    Classify states by active-only inclusive interval requirements.

    Returns
    -------
    np.ndarray
        Boolean array of shape (n_states,).
    """
    return np.empty(0, dtype=bool)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_classify_active_interval_states(
    validation_values: np.ndarray,
    active_mask: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    import numpy as np

    values = np.asarray(
        validation_values,
        dtype=float,
    )

    mask = np.asarray(
        active_mask
    )

    lower = np.asarray(
        lower_bounds,
        dtype=float,
    )

    upper = np.asarray(
        upper_bounds,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
    ):
        raise ValueError(
            "validation_values must be a non-empty two-dimensional array"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "validation_values must contain only finite values"
        )

    if mask.dtype != np.bool_:
        raise ValueError(
            "active_mask must contain Boolean values"
        )

    if mask.shape != values.shape:
        raise ValueError(
            "active_mask must have the same shape as validation_values"
        )

    n_validation = values.shape[1]

    if (
        lower.ndim != 1
        or lower.size != n_validation
    ):
        raise ValueError(
            "lower_bounds must align with validation columns"
        )

    if (
        upper.ndim != 1
        or upper.size != n_validation
    ):
        raise ValueError(
            "upper_bounds must align with validation columns"
        )

    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
    ):
        raise ValueError(
            "bounds must contain only finite values"
        )

    if np.any(
        lower > upper
    ):
        raise ValueError(
            "lower_bounds must not exceed upper_bounds"
        )

    within = (
        (values >= lower.reshape(1, -1))
        & (values <= upper.reshape(1, -1))
    )

    result = np.all(
        (~mask) | within,
        axis=1,
    )

    return result.astype(
        bool,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for classify_active_interval_states."""
    return [
        {
            "setup": """import numpy as np

validation_values = np.array(
    [
        [-3.0, -2.0, -6.0],
        [-5.0, -2.0, -6.0],
        [-3.0, -6.0, -6.0],
        [-3.0, -2.0, -9.0],
    ],
    dtype=float,
)

active_mask = np.ones(
    validation_values.shape,
    dtype=bool,
)

lower_bounds = np.array(
    [-4.5, -5.0, -8.5],
    dtype=float,
)

upper_bounds = np.array(
    [-2.0, -1.5, -4.3],
    dtype=float,
)
""",
            "call": "classify_active_interval_states(validation_values, active_mask, lower_bounds, upper_bounds)",
            "gold_call": "_oracle_classify_active_interval_states(validation_values, active_mask, lower_bounds, upper_bounds)",
        },
        {
            "setup": """import numpy as np

validation_values = np.array(
    [
        [-3.0,  500.0, -6.0],
        [99.0,  -2.0, -100.0],
        [50.0, -80.0,  20.0],
        [-5.0,   9.0,  40.0],
    ],
    dtype=float,
)

active_mask = np.array(
    [
        [ True, False,  True],
        [False,  True, False],
        [False, False, False],
        [ True, False, False],
    ],
    dtype=bool,
)

lower_bounds = np.array(
    [-4.5, -5.0, -8.5],
    dtype=float,
)

upper_bounds = np.array(
    [-2.0, -1.5, -4.3],
    dtype=float,
)
""",
            "call": "classify_active_interval_states(validation_values, active_mask, lower_bounds, upper_bounds)",
            "gold_call": "_oracle_classify_active_interval_states(validation_values, active_mask, lower_bounds, upper_bounds)",
        },
        {
            "setup": """import numpy as np

validation_values = np.array(
    [
        [-4.5, -1.5, -8.5],
        [-2.0, -5.0, -4.3],
        [-4.5000001, -3.0, -6.0],
        [-3.0, -1.4999999, -6.0],
    ],
    dtype=float,
)

active_mask = np.ones(
    validation_values.shape,
    dtype=bool,
)

lower_bounds = np.array(
    [-4.5, -5.0, -8.5],
    dtype=float,
)

upper_bounds = np.array(
    [-2.0, -1.5, -4.3],
    dtype=float,
)
""",
            "call": "classify_active_interval_states(validation_values, active_mask, lower_bounds, upper_bounds)",
            "gold_call": "_oracle_classify_active_interval_states(validation_values, active_mask, lower_bounds, upper_bounds)",
        },
        {
            "setup": """import numpy as np

validation_values = np.array(
    [
        [-3.0, -2.5, -6.0,  20.0],
        [-4.0, -4.5, -5.0, -10.0],
        [-2.5, -3.0, -7.0,   0.5],
    ],
    dtype=float,
)

active_mask = np.array(
    [
        [ True,  True,  True, False],
        [ True, False,  True,  True],
        [ True,  True, False,  True],
    ],
    dtype=bool,
)

lower_bounds = np.array(
    [-4.5, -5.0, -8.0, -12.0],
    dtype=float,
)

upper_bounds = np.array(
    [-2.0, -1.5, -4.0,   1.0],
    dtype=float,
)

state_order = np.array(
    [2, 0, 1],
    dtype=int,
)

column_order = np.array(
    [3, 1, 0, 2],
    dtype=int,
)

validation_values = validation_values[
    state_order
][:, column_order]

active_mask = active_mask[
    state_order
][:, column_order]

lower_bounds = lower_bounds[
    column_order
]

upper_bounds = upper_bounds[
    column_order
]
""",
            "call": "classify_active_interval_states(validation_values, active_mask, lower_bounds, upper_bounds)",
            "gold_call": "_oracle_classify_active_interval_states(validation_values, active_mask, lower_bounds, upper_bounds)",
        },
    ]
