"""
Return the two concentrations the solution takes at the shared interface of the first-order two-slab problem, each reported in the units of the side it sits on. Each profile is linear, so each trace follows from the outer value and the flux the slab carries. The source fixes a convention here that the natural reading does not; follow the source.

Once the steady flux is known, each slab's profile is the straight line joining its outer value to its interfacial value. The two traces are not equal, and their relation is what any algebraic interface condition attempts to prescribe.

Returns
-------
ndarray of shape (2,): the trace on side A, then the trace on side B.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interfacial_traces(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float) -> "np.ndarray":
    """Return the two concentrations the solution takes at the shared interface of the first-order two-slab problem, each reported in the units of the side it sits on. Each profile is linear, so each trace follows from the outer value and the flux the slab carries. The source fixes a convention here that the natural reading does not; follow the source.

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
    ndarray of shape (2,): the trace on side A, then the trace on side B.

    Raises
    ------
    ValueError: if exchange_velocity is not positive, or any value it forwards to an earlier step is invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_interfacial_traces(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float) -> "np.ndarray":
    if exchange_velocity <= 0:
        raise ValueError("exchange velocity must be positive")
    if length_a <= 0 or length_b <= 0:
        raise ValueError("slab lengths must be positive")
    if diffusivity_a <= 0 or diffusivity_b <= 0:
        raise ValueError("diffusivities must be positive")
    flux = _oracle_linear_channel_flux(length_a, diffusivity_a, length_b,
                                       diffusivity_b, partition,
                                       exchange_velocity, c_outer_a, c_outer_b)
    # CONVENTION (paper, eq. B.1): the TRACE on side B is walked back along B's OWN resistance L_B/D_B, NOT along the scaled L_B/(K D_B) that entered the flux.
    # The golden solution carries the full argument and the natural wrong answer.
    return np.array([c_outer_a - flux * (length_a / diffusivity_a),
                     c_outer_b + flux * (length_b / diffusivity_b)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "interfacial_traces(0.5, 0.5, 0.5, 1.0, 2.0, 100.0, 2.0, 1.0)",
         "gold_call": "_oracle_interfacial_traces(0.5, 0.5, 0.5, 1.0, 2.0, 100.0, 2.0, 1.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "interfacial_traces(0.5, 0.5, 0.5, 1.0, 2.0, 1e8, 2.0, 1.0)",
         "gold_call": "_oracle_interfacial_traces(0.5, 0.5, 0.5, 1.0, 2.0, 1e8, 2.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "interfacial_traces(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3, 2.7e-6, 8.4e2, 1.1e1)",
         "gold_call": "_oracle_interfacial_traces(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3, 2.7e-6, 8.4e2, 1.1e1)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        interfacial_traces(0.5, 0.5, 0.5, 1.0, 2.0, -1.0, 2.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_interfacial_traces(0.5, 0.5, 0.5, 1.0, 2.0, -1.0, 2.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
