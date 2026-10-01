"""
Classify aligned signed values as active or inactive relative to a supplied magnitude threshold.

For every aligned signed value x, define its activity state using the supplied non-negative threshold τ.

The value is active exactly when |x| > τ.

Values with |x| = τ are inactive.

signed_values must be a finite, non-empty two-dimensional numerical array. active_threshold must be finite and non-negative.

Return a Boolean array with the same shape as signed_values. Raise ValueError if these requirements are not satisfied.

Returns
-------
2D NumPy Boolean array with the same shape as signed_values, where True denotes absolute magnitude strictly greater than active_threshold.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_activity_mask(
    signed_values: np.ndarray,
    active_threshold: float,
) -> np.ndarray:
    """
    Classify aligned signed values by absolute-magnitude activity.

    Parameters
    ----------
    signed_values : np.ndarray
        Finite array of shape (n_states, n_variables).
    active_threshold : float
        Finite non-negative activity threshold.

    Returns
    -------
    np.ndarray
        Boolean array with the same shape as signed_values.
    """
    return np.empty((0, 0), dtype=bool)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_activity_mask(
    signed_values: np.ndarray,
    active_threshold: float,
) -> np.ndarray:
    import numpy as np

    values = np.asarray(
        signed_values,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
    ):
        raise ValueError(
            "signed_values must be a non-empty two-dimensional array"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "signed_values must contain only finite values"
        )

    threshold = float(
        active_threshold
    )

    if (
        not np.isfinite(threshold)
        or threshold < 0.0
    ):
        raise ValueError(
            "active_threshold must be finite and non-negative"
        )

    return (
        np.abs(values)
        > threshold
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_activity_mask."""
    return [
        {
            "setup": """import numpy as np
signed_values = np.array(
    [
        [0.0, 0.2, -0.2, 1.5],
        [2.0, -3.0, 0.05, -0.08],
    ],
    dtype=float,
)
active_threshold = 0.1
""",
            "call": "compute_activity_mask(signed_values, active_threshold)",
            "gold_call": "_oracle_compute_activity_mask(signed_values, active_threshold)",
        },
        {
            "setup": """import numpy as np
signed_values = np.array(
    [
        [
            0.0,
            1.0e-6,
            -1.0e-6,
            1.0000001e-6,
            -1.0000001e-6,
        ],
        [
            2.0e-6,
            -2.0e-6,
            5.0e-7,
            -5.0e-7,
            0.0,
        ],
    ],
    dtype=float,
)
active_threshold = 1.0e-6
""",
            "call": "compute_activity_mask(signed_values, active_threshold)",
            "gold_call": "_oracle_compute_activity_mask(signed_values, active_threshold)",
        },
        {
            "setup": """import numpy as np
signed_values = np.array(
    [
        [2.5, -0.3, 0.0, 7.2],
        [-8.0, 0.4, 0.25, -1.1],
        [0.249999, -0.250001, 3.0, -4.0],
    ],
    dtype=float,
)

row_order = np.array(
    [2, 0, 1],
    dtype=int,
)

column_order = np.array(
    [3, 1, 0, 2],
    dtype=int,
)

signed_values = signed_values[
    row_order
][:, column_order]

active_threshold = 0.25
""",
            "call": "compute_activity_mask(signed_values, active_threshold)",
            "gold_call": "_oracle_compute_activity_mask(signed_values, active_threshold)",
        },
        {
            "setup": """import numpy as np
signed_values = np.array(
    [
        [0.0, 1.0, -1.0],
        [1.0e-12, -1.0e-12, 0.0],
    ],
    dtype=float,
)
active_threshold = 0.0
""",
            "call": "compute_activity_mask(signed_values, active_threshold)",
            "gold_call": "_oracle_compute_activity_mask(signed_values, active_threshold)",
        },
    ]
