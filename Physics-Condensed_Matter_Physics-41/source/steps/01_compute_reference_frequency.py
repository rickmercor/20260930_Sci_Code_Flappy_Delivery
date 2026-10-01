"""
Compute the local harmonic reference frequency from a positive particle mass and positive curvature, returning one native Python float.

Matching the quadratic well near a local minimum to a harmonic reference fixes the positive frequency from the curvature-to-mass ratio. The mathematical result may remain finite even when forming that ratio directly would overflow or underflow in binary64, so evaluate the positive square-root ratio in a scale-safe form rather than relying on one unguarded division.

Returns
-------
float, the positive harmonic reference frequency as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

from numbers import Real

import numpy as np


def compute_reference_frequency(mass: float, curvature: float) -> float:
    """Compute the local harmonic reference frequency.

    Parameters
    ----------
    mass : float
        Positive particle mass.
    curvature : float
        Positive second derivative of the potential at the local minimum.

    Returns
    -------
    omega : float
        Positive native Python float for the harmonic reference frequency.
        The result must remain finite whenever the exact positive
        square-root ratio is representable, even if the direct
        curvature-to-mass quotient is not representable in binary64.

    Raises
    ------
    ValueError
        If ``mass`` or ``curvature`` is not a finite real scalar or is not
        strictly positive, or if the resulting frequency is non-finite or
        non-positive.
    """
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_reference_frequency(mass: float, curvature: float) -> float:
    """Reference implementation with scale-safe positive-ratio evaluation."""
    import math
    import numbers

    import numpy as np

    if isinstance(mass, (bool, np.bool_)) or not isinstance(mass, numbers.Real):
        raise ValueError("mass must be a finite real scalar")
    if isinstance(curvature, (bool, np.bool_)) or not isinstance(
        curvature, numbers.Real
    ):
        raise ValueError("curvature must be a finite real scalar")
    mass_f = float(mass)
    curvature_f = float(curvature)
    if not math.isfinite(mass_f) or mass_f <= 0.0:
        raise ValueError("mass must be finite and > 0")
    if not math.isfinite(curvature_f) or curvature_f <= 0.0:
        raise ValueError("curvature must be finite and > 0")

    ratio = curvature_f / mass_f
    if math.isfinite(ratio) and ratio > 0.0:
        omega = math.sqrt(ratio)
    else:
        # For positive finite inputs, taking square roots before division
        # avoids an intermediate overflow or underflow in curvature / mass.
        omega = math.sqrt(curvature_f) / math.sqrt(mass_f)

    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("reference frequency must be finite and > 0")
    return float(omega)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, scale-separated, and invalid cases."""
    return [
        {
            "setup": """mass = 1.3\ncurvature = 3.2""",
            "call": "compute_reference_frequency(mass, curvature)",
            "gold_call": "_oracle_compute_reference_frequency(mass, curvature)",
        },
        {
            "setup": """mass = 1.0\ncurvature = 1.0""",
            "call": "compute_reference_frequency(mass, curvature)",
            "gold_call": "_oracle_compute_reference_frequency(mass, curvature)",
        },
        {
            "setup": """mass = 1.0e-12\ncurvature = 4.0e-12""",
            "call": "compute_reference_frequency(mass, curvature)",
            "gold_call": "_oracle_compute_reference_frequency(mass, curvature)",
        },
        {
            "setup": """mass = 0.0\ncurvature = 2.0\ndef run_model():\n    try:\n        compute_reference_frequency(mass, curvature)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_reference_frequency(mass, curvature)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """mass = 1.0\ncurvature = float('nan')\ndef run_model():\n    try:\n        compute_reference_frequency(mass, curvature)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_reference_frequency(mass, curvature)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """mass = 1.0e-200\ncurvature = 1.0e200""",
            "call": "compute_reference_frequency(mass, curvature)",
            "gold_call": "_oracle_compute_reference_frequency(mass, curvature)",
        },
    ]
