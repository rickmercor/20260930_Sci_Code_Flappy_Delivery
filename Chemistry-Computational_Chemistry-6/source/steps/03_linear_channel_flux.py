"""
Return the steady flux crossing a two-slab system whose shared interface carries a single first-order exchange channel, with the concentration held fixed on each outer face. The interface contributes a resistance of its own, set by the exchange velocity. The source fixes a convention here that the natural reading does not; follow the source.

At steady state with no source the same flux crosses both slabs and the interface, so the three elements of the transport circuit combine into one algebraic equation for that flux. The driving term is the difference between the two outer concentrations, which are quoted on the scales of their own sides.

Returns
-------
float, the steady flux crossing the system.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def linear_channel_flux(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float) -> float:
    """Return the steady flux crossing a two-slab system whose shared interface carries a single first-order exchange channel, with the concentration held fixed on each outer face. The interface contributes a resistance of its own, set by the exchange velocity. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_a : float
        Thickness of slab A. Positive.
    diffusivity_a : float
        Diffusivity of the mobile species in slab A. Positive.
    length_b : float
        Thickness of slab B. Positive.
    diffusivity_b : float
        Diffusivity of the mobile species in slab B. Positive.
    partition : float
        Ratio of the forward to the reverse rate constant of the first-order channel. Positive.
    exchange_velocity : float
        Forward rate constant of the first-order channel. Positive.
    c_outer_a : float
        Concentration held on the outer face of slab A.
    c_outer_b : float
        Concentration held on the outer face of slab B.

    Returns
    -------
    float, the steady flux crossing the system.

    Raises
    ------
    ValueError: if exchange_velocity is not positive, or any value it forwards to an earlier step is invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_linear_channel_flux(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float) -> float:
    if exchange_velocity <= 0:
        raise ValueError("exchange velocity must be positive")
    res = _oracle_slab_resistances(length_a, diffusivity_a, length_b,
                                   diffusivity_b, partition)
    # CONVENTION (paper, eq. B.2): three resistances IN SERIES. The interface contributes 1/k+ exactly like a bulk resistance, and the driving term is the difference of the two outer concentrations on the common scale.
    # The golden solution carries the full argument and the natural wrong answer.
    return (c_outer_a - c_outer_b / partition) / (res[0] + res[1]
                                                  + 1.0 / exchange_velocity)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "linear_channel_flux(0.5, 0.5, 0.5, 1.0, 2.0, 100.0, 2.0, 1.0)",
         "gold_call": "_oracle_linear_channel_flux(0.5, 0.5, 0.5, 1.0, 2.0, 100.0, 2.0, 1.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "linear_channel_flux(0.5, 0.5, 0.5, 1.0, 2.0, 1e8, 2.0, 1.0)",
         "gold_call": "_oracle_linear_channel_flux(0.5, 0.5, 0.5, 1.0, 2.0, 1e8, 2.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "linear_channel_flux(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3, 2.7e-6, 8.4e2, 1.1e1)",
         "gold_call": "_oracle_linear_channel_flux(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3, 2.7e-6, 8.4e2, 1.1e1)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        linear_channel_flux(0.5, 0.5, 0.5, 1.0, 2.0, 0.0, 2.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_linear_channel_flux(0.5, 0.5, 0.5, 1.0, 2.0, 0.0, 2.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
