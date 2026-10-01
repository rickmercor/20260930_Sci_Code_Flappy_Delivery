"""
Evaluate three row-wise observables from positive five-factor states.

rate_factors is a finite positive array of shape (n, 5).



For each row ordered as [q0, q1, q2, q3, q4], calculate:



turnover = q2 q4 / (q2 + q3 + q4)



michaelis = (q2 q4 + q1 q3 + q1 q4) / [q0 (q2 + q3 + q4)]



efficiency = turnover / michaelis



Return one row per input state with columns ordered as:



[turnover, michaelis, efficiency]



Preserve the input row order.

Returns
-------
observables : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_factor_observables(
    rate_factors: np.ndarray,
) -> np.ndarray:
    """Evaluate three observables for positive five-factor states.

    Parameters
    ----------
    rate_factors
        Finite positive array of shape (n, 5), with one five-factor
        state per row.

    Returns
    -------
    np.ndarray
        Floating-point array of shape (n, 3), ordered as
        [turnover, michaelis, efficiency].
    """
    observables = np.empty(
        (
            rate_factors.shape[0],
            3,
        ),
        dtype=float,
    )
    return observables

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_factor_observables(
    rate_factors,
):
    """Reference implementation for evaluate_factor_observables."""
    import numpy as np

    values = np.asarray(
        rate_factors,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != 5
    ):
        raise ValueError(
            "rate_factors must have shape (n, 5) with n >= 1"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "rate_factors must contain only finite values"
        )

    if np.any(
        values <= 0.0
    ):
        raise ValueError(
            "rate_factors must be strictly positive"
        )

    scale = np.max(
        values,
        axis=1,
    )

    scaled = (
        values
        / scale[:, np.newaxis]
    )

    common = (
        scaled[:, 2]
        + scaled[:, 3]
        + scaled[:, 4]
    )

    turnover = (
        scale
        * scaled[:, 2]
        * scaled[:, 4]
        / common
    )

    michaelis = (
        scaled[:, 2] * scaled[:, 4]
        + scaled[:, 1] * scaled[:, 3]
        + scaled[:, 1] * scaled[:, 4]
    ) / (
        scaled[:, 0]
        * common
    )

    efficiency = (
        turnover
        / michaelis
    )

    observables = np.column_stack(
        (
            turnover,
            michaelis,
            efficiency,
        )
    )

    if (
        not np.all(
            np.isfinite(observables)
        )
        or np.any(
            observables <= 0.0
        )
    ):
        raise ValueError(
            "all returned observables must be finite and positive"
        )

    return observables.astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for evaluate_factor_observables."""
    return [
        {
            "setup": """import numpy as np
rate_factors = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 3.0, 5.0, 7.0, 11.0],
    ],
    dtype=float,
)
""",
            "call": "evaluate_factor_observables(rate_factors)",
            "gold_call": "_oracle_evaluate_factor_observables(rate_factors)",
        },
        {
            "setup": """import numpy as np
rate_factors = np.array(
    [
        [2.0e-8, 3.0e-8, 5.0e-8, 7.0e-8, 1.1e-7],
        [1.0e-12, 2.0e-9, 3.0e-7, 4.0e-11, 5.0e-8],
    ],
    dtype=float,
)
""",
            "call": "evaluate_factor_observables(rate_factors)",
            "gold_call": "_oracle_evaluate_factor_observables(rate_factors)",
        },
        {
            "setup": """import numpy as np
rate_factors = np.array(
    [
        [0.7, 1.1, 2.3, 0.9, 4.2],
        [9.0, 0.4, 1.7, 3.1, 2.6],
    ],
    dtype=float,
)
""",
            "call": "evaluate_factor_observables(rate_factors)",
            "gold_call": "_oracle_evaluate_factor_observables(rate_factors)",
        },
        {
            "setup": """import numpy as np
rate_factors = np.array(
    [
        [1.0, 2.0, 3.0, 0.0, 5.0],
    ],
    dtype=float,
)

def run_model():
    try:
        evaluate_factor_observables(
            rate_factors,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_evaluate_factor_observables(
            rate_factors,
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
