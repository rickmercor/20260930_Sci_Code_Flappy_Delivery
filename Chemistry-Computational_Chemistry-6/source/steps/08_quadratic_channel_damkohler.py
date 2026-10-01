"""
Return the dimensionless group for a channel that is second order in the donor material. Such a channel has no exchange velocity of its own, so one has to be constructed from the rate constant and a concentration, and the source is specific about which concentration that is. The source fixes a convention here that the natural reading does not; follow the source.

The group of the earlier step compared an exchange velocity with a bulk resistance. A channel whose rate is not proportional to the donor concentration supplies no such velocity directly, and the quantity that plays its part depends on the state of the interface, so a reference state has to be named and quoted with the group.

Returns
-------
float, the dimensionless group for the second-order channel.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quadratic_channel_damkohler(length_m: float, diffusivity_m: float, forward_rate: float, reference_concentration: float) -> float:
    """Return the dimensionless group for a channel that is second order in the donor material. Such a channel has no exchange velocity of its own, so one has to be constructed from the rate constant and a concentration, and the source is specific about which concentration that is. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_m : float
        Thickness of the donor slab. Positive.
    diffusivity_m : float
        Diffusivity in the donor slab. Positive.
    forward_rate : float
        Forward rate constant of the recombination channel. Positive.
    reference_concentration : float
        Donor loading the ratio is evaluated at. Positive.

    Returns
    -------
    float, the dimensionless group for the second-order channel.

    Raises
    ------
    ValueError: if forward_rate or reference_concentration is not positive, or length_m or diffusivity_m is not positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_quadratic_channel_damkohler(length_m: float, diffusivity_m: float, forward_rate: float, reference_concentration: float) -> float:
    if forward_rate <= 0 or reference_concentration <= 0:
        raise ValueError("rate constant and reference concentration must be positive")
    if length_m <= 0 or diffusivity_m <= 0:
        raise ValueError("length and diffusivity must be positive")
    # CONVENTION (paper, sec. 2.7): a channel of order two has no exchange velocity of its own, so the comparison uses the LINEARISED one, dw/dc evaluated at the interface, which is 2 k+ c and not k+ c.
    # The golden solution carries the full argument and the natural wrong answer.
    return 2.0 * forward_rate * reference_concentration * (length_m / diffusivity_m)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "quadratic_channel_damkohler(0.5, 0.5, 3.0, 2.0)",
         "gold_call": "_oracle_quadratic_channel_damkohler(0.5, 0.5, 3.0, 2.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "quadratic_channel_damkohler(1.0, 1.0, 1.0, 1.0)",
         "gold_call": "_oracle_quadratic_channel_damkohler(1.0, 1.0, 1.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "quadratic_channel_damkohler(3.1e-3, 7.4e-11, 5.2e-3, 8.4e2)",
         "gold_call": "_oracle_quadratic_channel_damkohler(3.1e-3, 7.4e-11, 5.2e-3, 8.4e2)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        quadratic_channel_damkohler(0.5, 0.5, 3.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_quadratic_channel_damkohler(0.5, 0.5, 3.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
