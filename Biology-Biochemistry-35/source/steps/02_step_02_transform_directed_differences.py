"""
Transform directed differences into positive reduced factors at a supplied thermal scale.

directed_differences is a finite two-dimensional array.



thermal_scale is a finite positive scalar expressed in the same energy units as directed_differences.



For every entry Δ, calculate:



exp(-Δ / thermal_scale)



Preserve the input shape, row order, and column order.

Returns
-------
factors : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def transform_directed_differences(
    directed_differences: np.ndarray,
    thermal_scale: float,
) -> np.ndarray:
    """Transform directed differences into positive reduced factors.

    Parameters
    ----------
    directed_differences
        Finite non-empty two-dimensional array.
    thermal_scale
        Finite strictly positive scalar in the same energy units as
        directed_differences.

    Returns
    -------
    np.ndarray
        Positive floating-point array with the same shape and ordering
        as directed_differences.
    """
    factors = np.empty_like(
        directed_differences,
        dtype=float,
    )
    return factors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_transform_directed_differences(
    directed_differences: np.ndarray,
    thermal_scale: float,
) -> np.ndarray:
    """Reference implementation for transform_directed_differences."""
    import numpy as np

    differences = np.asarray(
        directed_differences,
        dtype=float,
    )

    if (
        differences.ndim != 2
        or differences.shape[0] < 1
        or differences.shape[1] < 1
    ):
        raise ValueError(
            "directed_differences must be a non-empty "
            "two-dimensional array"
        )

    if not np.all(
        np.isfinite(differences)
    ):
        raise ValueError(
            "directed_differences must contain only finite values"
        )

    if isinstance(
        thermal_scale,
        (bool, np.bool_),
    ):
        raise ValueError(
            "thermal_scale must be a real numerical scalar"
        )

    try:
        scale = float(
            thermal_scale
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "thermal_scale must be a real numerical scalar"
        ) from exc

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            "thermal_scale must be finite and strictly positive"
        )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        factors = np.exp(
            -differences / scale
        )

    if (
        not np.all(
            np.isfinite(factors)
        )
        or np.any(
            factors <= 0.0
        )
    ):
        raise ValueError(
            "transformed factors must be finite and "
            "strictly positive"
        )

    return factors.astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for transform_directed_differences."""
    return [
        {
            "setup": """import numpy as np
directed_differences = np.array(
    [
        [1.2, 2.4, -0.6],
        [0.3, -0.9, 1.5],
    ],
    dtype=float,
)
thermal_scale = 0.75
""",
            "call": "transform_directed_differences(directed_differences, thermal_scale)",
            "gold_call": "_oracle_transform_directed_differences(directed_differences, thermal_scale)",
        },
        {
            "setup": """import numpy as np
directed_differences = np.zeros(
    (3, 4),
    dtype=float,
)
thermal_scale = 1.2
""",
            "call": "transform_directed_differences(directed_differences, thermal_scale)",
            "gold_call": "_oracle_transform_directed_differences(directed_differences, thermal_scale)",
        },
        {
            "setup": """import numpy as np
directed_differences = np.array(
    [
        [2.0, 4.0, 6.0],
        [-1.0, 0.5, 3.0],
    ],
    dtype=float,
)
thermal_scale = 0.8
common_scale = 7.5

def run_model():
    baseline = transform_directed_differences(
        directed_differences,
        thermal_scale,
    )
    transformed = transform_directed_differences(
        directed_differences * common_scale,
        thermal_scale * common_scale,
    )
    return float(
        np.max(
            np.abs(
                baseline - transformed
            )
        )
    )

def run_gold():
    baseline = _oracle_transform_directed_differences(
        directed_differences,
        thermal_scale,
    )
    transformed = _oracle_transform_directed_differences(
        directed_differences * common_scale,
        thermal_scale * common_scale,
    )
    return float(
        np.max(
            np.abs(
                baseline - transformed
            )
        )
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
directed_differences = np.array(
    [
        [1.0, 2.0],
    ],
    dtype=float,
)
thermal_scale = 0.0

def run_model():
    try:
        transform_directed_differences(
            directed_differences,
            thermal_scale,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_transform_directed_differences(
            directed_differences,
            thermal_scale,
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
