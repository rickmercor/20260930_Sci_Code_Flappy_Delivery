"""
Instantiate one deterministic Gaussian trial position from a selected mean, variance, and supplied standard-normal deviate.

Once a Gaussian bridge has been selected, a supplied standard-normal deviate $z$ makes the draw deterministic: $y=\mu+\sqrt{\sigma^2}z$. A strictly positive variance and finite inputs are required.

Returns
-------
float, the proposed bead position as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

from numbers import Real

import numpy as np

def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def generate_trial_position(
    mean: float,
    variance: float,
    normal_draw: float,
) -> float:
    """Instantiate a Gaussian bridge proposal from a fixed normal deviate.

    Parameters
    ----------
    mean : float
        Gaussian proposal mean.
    variance : float
        Positive Gaussian proposal variance.
    normal_draw : float
        Supplied finite standard-normal deviate.

    Returns
    -------
    trial : float
        Proposed bead position as a native Python float.

    Raises
    ------
    ValueError
        If ``mean``, ``variance``, or ``normal_draw`` is not a finite
        real value; if ``variance`` is not strictly positive; or if the
        resulting trial position is non-finite.
    """
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_generate_trial_position(mean: float, variance: float, normal_draw: float) -> float:
    """Reference implementation with function-local dependencies."""
    import math
    import numbers

    import numpy as np

    named_values = (
        ("mean", mean),
        ("variance", variance),
        ("normal_draw", normal_draw),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    mean_f = converted["mean"]
    variance_f = converted["variance"]
    draw_f = converted["normal_draw"]
    if variance_f <= 0.0:
        raise ValueError("variance must be > 0")
    trial = mean_f + math.sqrt(variance_f) * draw_f
    if not math.isfinite(trial):
        raise ValueError("trial position must be finite")
    return float(trial)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid test specifications."""
    return [
        {
            "setup": """mean = 0.07447545834845315\nvariance = 0.09912813911190939\nnormal_draw = 0.7""",
            "call": "generate_trial_position(mean, variance, normal_draw)",
            "gold_call": "_oracle_generate_trial_position(mean, variance, normal_draw)",
        },
        {
            "setup": """mean = -2.5\nvariance = 3.0\nnormal_draw = 0.0""",
            "call": "generate_trial_position(mean, variance, normal_draw)",
            "gold_call": "_oracle_generate_trial_position(mean, variance, normal_draw)",
        },
        {
            "setup": """mean = 1.0\nvariance = 1.0e-24\nnormal_draw = -3.0""",
            "call": "generate_trial_position(mean, variance, normal_draw)",
            "gold_call": "_oracle_generate_trial_position(mean, variance, normal_draw)",
        },
        {
            "setup": """mean = 0.0\nvariance = 0.0\nnormal_draw = 1.0\ndef run_model():\n    try:\n        generate_trial_position(mean, variance, normal_draw)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_generate_trial_position(mean, variance, normal_draw)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """mean = 0.0\nvariance = 1.0\nnormal_draw = float('inf')\ndef run_model():\n    try:\n        generate_trial_position(mean, variance, normal_draw)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_generate_trial_position(mean, variance, normal_draw)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
