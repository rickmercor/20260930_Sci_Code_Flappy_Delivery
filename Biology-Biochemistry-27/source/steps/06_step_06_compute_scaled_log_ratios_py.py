"""
Compute aligned scaled logarithmic ratios from strictly positive paired components.

component_pairs is a finite three-dimensional numerical array with shape (n_states, n_variables, 2).

For each state i and variable j, define

L[i,j] = scale × ln(component_pairs[i,j,1] / component_pairs[i,j,0]).

Return the complete aligned array L with shape (n_states, n_variables).

Every component must be strictly positive.

scale must be finite and strictly positive.

Raise ValueError if component_pairs is not a non-empty three-dimensional numerical array with final dimension 2, if any component is non-finite or non-positive, or if scale is non-finite or non-positive.

Returns
-------
2D NumPy float array of shape (n_states, n_variables) containing the aligned scaled logarithmic ratios.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_scaled_log_ratios(
    component_pairs: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Compute aligned scaled logarithmic component ratios.

    Parameters
    ----------
    component_pairs : np.ndarray
        Finite strictly positive array of shape
        (n_states, n_variables, 2).
    scale : float
        Finite strictly positive scalar multiplier.

    Returns
    -------
    np.ndarray
        Float array of shape (n_states, n_variables).
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_scaled_log_ratios(
    component_pairs: np.ndarray,
    scale: float,
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

    if np.any(
        components <= 0.0
    ):
        raise ValueError(
            "component_pairs must contain only strictly positive values"
        )

    value_scale = float(
        scale
    )

    if (
        not np.isfinite(value_scale)
        or value_scale <= 0.0
    ):
        raise ValueError(
            "scale must be finite and strictly positive"
        )

    result = (
        value_scale
        * (
            np.log(
                components[..., 1]
            )
            - np.log(
                components[..., 0]
            )
        )
    )

    if not np.all(
        np.isfinite(result)
    ):
        raise ValueError(
            "computed scaled log ratios must be finite"
        )

    return result.astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_scaled_log_ratios."""
    return [
        {
            "setup": """import numpy as np

component_pairs = np.array(
    [
        [
            [4.0, 1.0],
            [2.0, 3.0],
            [5.0, 0.5],
        ],
        [
            [1.2, 2.4],
            [7.5, 1.5],
            [0.8, 0.8],
        ],
    ],
    dtype=float,
)

scale = 2.5
""",
            "call": "compute_scaled_log_ratios(component_pairs, scale)",
            "gold_call": "_oracle_compute_scaled_log_ratios(component_pairs, scale)",
        },
        {
            "setup": """import numpy as np

component_pairs = np.array(
    [
        [
            [1.0, 1.0],
            [2.5, 2.5],
            [0.125, 0.125],
        ],
        [
            [7.0, 7.0],
            [0.02, 0.02],
            [4.2, 4.2],
        ],
    ],
    dtype=float,
)

scale = 7.25
""",
            "call": "compute_scaled_log_ratios(component_pairs, scale)",
            "gold_call": "_oracle_compute_scaled_log_ratios(component_pairs, scale)",
        },
        {
            "setup": """import numpy as np

component_pairs = np.array(
    [
        [
            [1.0e-8, 2.0e-4],
            [50.0, 0.125],
            [0.75, 8.0],
            [3.0, 1.0e-5],
        ],
        [
            [2.5e-3, 9.0],
            [0.2, 14.0],
            [6.0, 0.03],
            [1.5, 1.5],
        ],
    ],
    dtype=float,
)

scale = 0.875
""",
            "call": "compute_scaled_log_ratios(component_pairs, scale)",
            "gold_call": "_oracle_compute_scaled_log_ratios(component_pairs, scale)",
        },
        {
            "setup": """import numpy as np

component_pairs = np.array(
    [
        [
            [2.2, 0.7],
            [0.8, 4.1],
            [5.5, 2.5],
            [1.3, 1.9],
        ],
        [
            [0.4, 3.0],
            [6.2, 1.1],
            [0.7, 0.2],
            [9.5, 9.5],
        ],
        [
            [1.8, 4.4],
            [2.0, 0.5],
            [7.2, 3.1],
            [0.6, 5.0],
        ],
    ],
    dtype=float,
)

state_order = np.array(
    [2, 0, 1],
    dtype=int,
)

variable_order = np.array(
    [2, 0, 3, 1],
    dtype=int,
)

component_pairs = component_pairs[
    state_order
][:, variable_order, :]

scale = 3.1
""",
            "call": "compute_scaled_log_ratios(component_pairs, scale)",
            "gold_call": "_oracle_compute_scaled_log_ratios(component_pairs, scale)",
        },
    ]
