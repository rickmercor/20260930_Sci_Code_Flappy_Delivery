"""
Compute signed differences between aligned pairs of numerical components.

component_pairs is a finite three-dimensional numerical array with shape (n_states, n_variables, 2). For every state i and variable j, define the signed value
d[i,j] = component_pairs[i,j,0] - component_pairs[i,j,1].

Return the complete aligned signed-value array.

component_pairs must contain at least one state and one variable, and its final dimension must have length 2. Raise ValueError if the array does not satisfy these requirements or contains a non-finite value.

Returns
-------
2D NumPy float array of shape (n_states, n_variables) containing the aligned first-component-minus-second-component differences.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_paired_component_differences(
    component_pairs: np.ndarray,
) -> np.ndarray:
    """
    Compute signed differences between aligned component pairs.

    Parameters
    ----------
    component_pairs : np.ndarray
        Finite array of shape (n_states, n_variables, 2).

    Returns
    -------
    np.ndarray
        Float array of shape (n_states, n_variables).
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_paired_component_differences(
    component_pairs: np.ndarray,
) -> np.ndarray:
    import numpy as np

    components = np.asarray(
        component_pairs,
        dtype=float,
    )

    if (
        components.ndim != 3
        or components.shape[0] < 1
        or components.shape[1] < 1
        or components.shape[2] != 2
    ):
        raise ValueError(
            "component_pairs must be a non-empty three-dimensional array with final dimension 2"
        )

    if not np.all(
        np.isfinite(components)
    ):
        raise ValueError(
            "component_pairs must contain only finite values"
        )

    result = (
        components[..., 0]
        - components[..., 1]
    )

    if not np.all(
        np.isfinite(result)
    ):
        raise ValueError(
            "computed signed differences must be finite"
        )

    return result.astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_paired_component_differences."""
    return [
        {
            "setup": """import numpy as np
component_pairs = np.array(
    [
        [
            [4.0, 1.0],
            [1.2, 3.2],
            [0.5, 0.5],
        ],
        [
            [2.0, 5.0],
            [7.5, 0.25],
            [0.0001, 0.0004],
        ],
    ],
    dtype=float,
)
""",
            "call": "compute_paired_component_differences(component_pairs)",
            "gold_call": "_oracle_compute_paired_component_differences(component_pairs)",
        },
        {
            "setup": """import numpy as np
component_pairs = np.array(
    [
        [
            [0.0, 0.0],
            [-2.0, -5.0],
            [-7.0, 1.0],
            [3.5, -4.5],
        ],
    ],
    dtype=float,
)
""",
            "call": "compute_paired_component_differences(component_pairs)",
            "gold_call": "_oracle_compute_paired_component_differences(component_pairs)",
        },
        {
            "setup": """import numpy as np
component_pairs = np.array(
    [
        [
            [1.5, 4.0],
            [8.0, 2.0],
            [0.75, 0.25],
            [6.0, 9.0],
        ],
        [
            [2.2, 0.1],
            [5.5, 5.5],
            [3.0, 7.0],
            [1.0, 0.2],
        ],
        [
            [0.04, 0.8],
            [4.2, 1.1],
            [9.0, 3.5],
            [2.5, 6.5],
        ],
    ],
    dtype=float,
)

permutation = np.array(
    [2, 0, 3, 1],
    dtype=int,
)

component_pairs = component_pairs[
    :,
    permutation,
    :,
]
""",
            "call": "compute_paired_component_differences(component_pairs)",
            "gold_call": "_oracle_compute_paired_component_differences(component_pairs)",
        },
        {
            "setup": """import numpy as np
component_pairs = np.array(
    [
        [
            [1.000000001, 1.0],
        ],
        [
            [2.0, 2.000000003],
        ],
        [
            [-1.5, -1.5],
        ],
    ],
    dtype=float,
)
""",
            "call": "compute_paired_component_differences(component_pairs)",
            "gold_call": "_oracle_compute_paired_component_differences(component_pairs)",
        },
    ]
