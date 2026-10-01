"""
Evaluate aligned violations of linear balance, fixed-value, and one-way signed constraints.

signed_values contains aligned signed states, with one state per row and one variable per column.

constraint_matrix acts on each signed state through

signed_values @ constraint_matrix.T.

For every state, return three non-negative diagnostics in this exact column order:

1. the maximum absolute linear-constraint residual;

2. the maximum absolute deviation from the supplied fixed coordinate values;

3. the maximum violation of the supplied one-way requirements, where a one-way coordinate is violated only when its signed value is negative.

If fixed_indices is empty, its diagnostic is 0. If one_way_indices is empty, its diagnostic is 0.

signed_values and constraint_matrix must be finite non-empty two-dimensional numerical arrays with matching variable dimensions.

fixed_indices and one_way_indices must be one-dimensional integer arrays containing valid, non-duplicated variable indices. fixed_values must be a finite one-dimensional array aligned with fixed_indices.

Raise ValueError if these requirements are not satisfied.

Returns
-------
2D NumPy float array of shape (n_states, 3), with columns [maximum linear residual, maximum fixed-value deviation, maximum one-way violation].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_signed_constraint_violations(
    signed_values: np.ndarray,
    constraint_matrix: np.ndarray,
    fixed_indices: np.ndarray,
    fixed_values: np.ndarray,
    one_way_indices: np.ndarray,
) -> np.ndarray:
    """
    Evaluate three classes of aligned signed-state constraint violation.

    Returns
    -------
    np.ndarray
        Float array of shape (n_states, 3).
    """
    return np.empty((0, 3), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_signed_constraint_violations(
    signed_values: np.ndarray,
    constraint_matrix: np.ndarray,
    fixed_indices: np.ndarray,
    fixed_values: np.ndarray,
    one_way_indices: np.ndarray,
) -> np.ndarray:
    import numpy as np

    values = np.asarray(
        signed_values,
        dtype=float,
    )

    matrix = np.asarray(
        constraint_matrix,
        dtype=float,
    )

    raw_fixed = np.asarray(
        fixed_indices
    )

    fixed_values = np.asarray(
        fixed_values,
        dtype=float,
    )

    raw_one_way = np.asarray(
        one_way_indices
    )

    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
    ):
        raise ValueError(
            "signed_values must be a non-empty two-dimensional array"
        )

    if (
        matrix.ndim != 2
        or matrix.shape[0] < 1
        or matrix.shape[1] != values.shape[1]
    ):
        raise ValueError(
            "constraint_matrix must be non-empty and align with signed_values columns"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "signed_values must contain only finite values"
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "constraint_matrix must contain only finite values"
        )

    n_variables = values.shape[1]

    def _parse_indices(
        raw: np.ndarray,
        name: str,
    ) -> np.ndarray:
        if raw.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional"
            )

        if not np.issubdtype(
            raw.dtype,
            np.integer,
        ):
            raise ValueError(
                f"{name} must contain integers"
            )

        indices = raw.astype(
            int,
            copy=False,
        )

        if (
            np.any(indices < 0)
            or np.any(indices >= n_variables)
        ):
            raise ValueError(
                f"{name} contains an out-of-range index"
            )

        if (
            np.unique(indices).size
            != indices.size
        ):
            raise ValueError(
                f"{name} must not contain duplicates"
            )

        return indices

    fixed = _parse_indices(
        raw_fixed,
        "fixed_indices",
    )

    one_way = _parse_indices(
        raw_one_way,
        "one_way_indices",
    )

    if (
        fixed_values.ndim != 1
        or fixed_values.size != fixed.size
    ):
        raise ValueError(
            "fixed_values must align with fixed_indices"
        )

    if not np.all(
        np.isfinite(fixed_values)
    ):
        raise ValueError(
            "fixed_values must contain only finite values"
        )

    linear_residual = (
        values @ matrix.T
    )

    maximum_linear = np.max(
        np.abs(linear_residual),
        axis=1,
    )

    maximum_fixed = np.zeros(
        values.shape[0],
        dtype=float,
    )

    if fixed.size:
        maximum_fixed = np.max(
            np.abs(
                values[:, fixed]
                - fixed_values.reshape(
                    1,
                    -1,
                )
            ),
            axis=1,
        )

    maximum_one_way = np.zeros(
        values.shape[0],
        dtype=float,
    )

    if one_way.size:
        maximum_one_way = np.max(
            np.maximum(
                -values[:, one_way],
                0.0,
            ),
            axis=1,
        )

    result = np.column_stack(
        (
            maximum_linear,
            maximum_fixed,
            maximum_one_way,
        )
    )

    return result.astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for evaluate_signed_constraint_violations."""
    return [
        {
            "setup": """import numpy as np

signed_values = np.array(
    [
        [5.0, 3.0, 2.0],
        [5.0, 4.0, 1.0],
        [5.0, -1.0, 6.0],
        [4.5, 3.0, 2.0],
    ],
    dtype=float,
)

constraint_matrix = np.array(
    [
        [1.0, -1.0, -1.0],
    ],
    dtype=float,
)

fixed_indices = np.array(
    [0],
    dtype=int,
)

fixed_values = np.array(
    [5.0],
    dtype=float,
)

one_way_indices = np.array(
    [1, 2],
    dtype=int,
)
""",
            "call": "evaluate_signed_constraint_violations(signed_values, constraint_matrix, fixed_indices, fixed_values, one_way_indices)",
            "gold_call": "_oracle_evaluate_signed_constraint_violations(signed_values, constraint_matrix, fixed_indices, fixed_values, one_way_indices)",
        },
        {
            "setup": """import numpy as np

signed_values = np.array(
    [
        [2.0, -1.0, 3.0, 0.0],
        [1.0,  2.0, 3.0, 4.0],
    ],
    dtype=float,
)

constraint_matrix = np.array(
    [
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
    ],
    dtype=float,
)

fixed_indices = np.array(
    [],
    dtype=int,
)

fixed_values = np.array(
    [],
    dtype=float,
)

one_way_indices = np.array(
    [],
    dtype=int,
)
""",
            "call": "evaluate_signed_constraint_violations(signed_values, constraint_matrix, fixed_indices, fixed_values, one_way_indices)",
            "gold_call": "_oracle_evaluate_signed_constraint_violations(signed_values, constraint_matrix, fixed_indices, fixed_values, one_way_indices)",
        },
        {
            "setup": """import numpy as np

signed_values = np.array(
    [
        [5.0, -4.0, 0.0, 1.0, 4.0],
        [5.0, -6.0, 1.0, 0.0, 4.0],
    ],
    dtype=float,
)

constraint_matrix = np.array(
    [
        [ 1.0, 1.0, 1.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0, -1.0, 1.0],
    ],
    dtype=float,
)

fixed_indices = np.array(
    [0],
    dtype=int,
)

fixed_values = np.array(
    [5.0],
    dtype=float,
)

one_way_indices = np.array(
    [2, 3],
    dtype=int,
)

permutation = np.array(
    [3, 0, 4, 2, 1],
    dtype=int,
)

new_index = np.empty(
    permutation.size,
    dtype=int,
)

new_index[permutation] = np.arange(
    permutation.size,
    dtype=int,
)

signed_values = signed_values[
    :,
    permutation,
]

constraint_matrix = constraint_matrix[
    :,
    permutation,
]

fixed_indices = new_index[
    fixed_indices
]

one_way_indices = new_index[
    one_way_indices
]
""",
            "call": "evaluate_signed_constraint_violations(signed_values, constraint_matrix, fixed_indices, fixed_values, one_way_indices)",
            "gold_call": "_oracle_evaluate_signed_constraint_violations(signed_values, constraint_matrix, fixed_indices, fixed_values, one_way_indices)",
        },
        {
            "setup": """import numpy as np

signed_values = np.array(
    [
        [5.0, 0.0, 5.0],
        [5.0, -1.0e-10, 5.0000000001],
        [5.0, 1.0e-10, 4.9999999999],
    ],
    dtype=float,
)

constraint_matrix = np.array(
    [
        [1.0, 1.0, -1.0],
    ],
    dtype=float,
)

fixed_indices = np.array(
    [0],
    dtype=int,
)

fixed_values = np.array(
    [5.0],
    dtype=float,
)

one_way_indices = np.array(
    [1],
    dtype=int,
)
""",
            "call": "evaluate_signed_constraint_violations(signed_values, constraint_matrix, fixed_indices, fixed_values, one_way_indices)",
            "gold_call": "_oracle_evaluate_signed_constraint_violations(signed_values, constraint_matrix, fixed_indices, fixed_values, one_way_indices)",
        },
    ]
