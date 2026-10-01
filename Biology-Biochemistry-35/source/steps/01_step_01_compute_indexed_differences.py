"""
Compute directed column differences for supplied index pairs across aligned numerical profiles.

profile_values is a finite two-dimensional array containing one aligned numerical profile per row.



difference_pairs is an integer array of shape (m, 2). Each row [a, b] requests the directed difference:



profile_values[:, a] - profile_values[:, b]



Preserve the input profile-row order and the supplied difference-pair order.

Returns
-------
differences : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_indexed_differences(
    profile_values: np.ndarray,
    difference_pairs: np.ndarray,
) -> np.ndarray:
    """Compute directed column differences across aligned profiles.

    Parameters
    ----------
    profile_values
        Finite two-dimensional array of shape (n_profiles, n_states).
    difference_pairs
        Integer array of shape (m, 2). Each row [a, b] requests
        profile_values[:, a] - profile_values[:, b].

    Returns
    -------
    np.ndarray
        Floating-point array of shape (n_profiles, m), preserving the
        supplied profile-row and difference-pair order.
    """
    differences = np.empty(
        (
            profile_values.shape[0],
            difference_pairs.shape[0],
        ),
        dtype=float,
    )
    return differences

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_indexed_differences(
    profile_values,
    difference_pairs,
):
    """Reference implementation for compute_indexed_differences."""
    import numpy as np

    values = np.asarray(
        profile_values,
        dtype=float,
    )

    pairs = np.asarray(
        difference_pairs,
    )

    if values.ndim != 2:
        raise ValueError(
            "profile_values must be a two-dimensional array"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "profile_values must contain only finite values"
        )

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            "difference_pairs must have shape (m, 2)"
        )

    if not np.issubdtype(
        pairs.dtype,
        np.integer,
    ):
        raise ValueError(
            "difference_pairs must contain integer indices"
        )

    if (
        np.any(pairs < 0)
        or np.any(pairs >= values.shape[1])
    ):
        raise ValueError(
            "difference-pair indices are out of bounds"
        )

    return (
        values[:, pairs[:, 0]]
        - values[:, pairs[:, 1]]
    ).astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_indexed_differences."""
    return [
        {
            "setup": """import numpy as np

profile_values = np.array(
    [
        [2.0, -1.0, 4.5, 0.5, 7.0],
        [-3.0, 2.0, 1.5, 6.0, -2.0],
    ],
    dtype=float,
)

difference_pairs = np.array(
    [
        [2, 0],
        [4, 1],
        [3, 2],
    ],
    dtype=int,
)
""",
            "call": "compute_indexed_differences(profile_values, difference_pairs)",
            "gold_call": "_oracle_compute_indexed_differences(profile_values, difference_pairs)",
        },
        {
            "setup": """import numpy as np

profile_values = np.array(
    [
        [1.2, 3.4, -2.0, 5.1],
        [-4.0, 0.5, 7.2, 2.1],
        [6.3, -1.7, 0.0, 4.8],
    ],
    dtype=float,
)

difference_pairs = np.array(
    [
        [0, 3],
        [1, 2],
        [3, 1],
    ],
    dtype=int,
)
""",
            "call": "compute_indexed_differences(profile_values, difference_pairs)",
            "gold_call": "_oracle_compute_indexed_differences(profile_values, difference_pairs)",
        },
        {
            "setup": """import numpy as np

profile_values = np.array(
    [
        [0.25, -0.75, 2.5, 8.0, -3.5],
        [4.0, 1.0, -2.0, 0.5, 6.5],
    ],
    dtype=float,
)

difference_pairs = np.array(
    [
        [2, 2],
        [0, 4],
        [4, 0],
        [1, 3],
    ],
    dtype=int,
)
""",
            "call": "compute_indexed_differences(profile_values, difference_pairs)",
            "gold_call": "_oracle_compute_indexed_differences(profile_values, difference_pairs)",
        },
        {
            "setup": """import numpy as np

profile_values = np.array(
    [
        [3.1, -2.4, 7.7, 1.2, -5.6, 4.3],
    ],
    dtype=float,
)

difference_pairs = np.array(
    [
        [5, 1],
        [0, 3],
        [4, 2],
        [1, 5],
        [3, 0],
    ],
    dtype=int,
)
""",
            "call": "compute_indexed_differences(profile_values, difference_pairs)",
            "gold_call": "_oracle_compute_indexed_differences(profile_values, difference_pairs)",
        },
    ]
