"""
Return the dimensionless group evaluated separately for each side of an asymmetric two-slab system, each against that side's bulk resistance as the earlier resistance step measures it, together with the single one of them that decides whether the interface may be treated as equilibrated. The source fixes a convention here that the natural reading does not; follow the source.

A system whose two sides transport very differently is not summarised by one comparison made on the pair taken together. The source evaluates the comparison once per side and names which of the two decides; the second side is still measured the way the earlier resistance step measures it.

Returns
-------
ndarray of shape (3,): the group for side A, the group for side B, then the one that decides.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def governing_damkohler(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float) -> "np.ndarray":
    """Return the dimensionless group evaluated separately for each side of an asymmetric two-slab system, each against that side's bulk resistance as the earlier resistance step measures it, together with the single one of them that decides whether the interface may be treated as equilibrated. The source fixes a convention here that the natural reading does not; follow the source.

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

    Returns
    -------
    ndarray of shape (3,): the group for side A, the group for side B, then the one that decides.

    Raises
    ------
    ValueError: if exchange_velocity is not positive, either slab length is not positive, either diffusivity is not positive, or partition is not positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_governing_damkohler(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float) -> "np.ndarray":
    if exchange_velocity <= 0:
        raise ValueError("exchange velocity must be positive")
    if length_a <= 0 or length_b <= 0:
        raise ValueError("slab lengths must be positive")
    if diffusivity_a <= 0 or diffusivity_b <= 0:
        raise ValueError("diffusivities must be positive")
    if partition <= 0:
        raise ValueError("partition must be positive")
    # CONVENTION (paper, sec. 2.7): for an ASYMMETRIC system the comparison is made once PER SIDE, each against that side's own bulk resistance, and the SMALLER of the two governs.
    # The golden solution carries the full argument and the natural wrong answer.
    da_a = exchange_velocity * (length_a / diffusivity_a)
    # CONVENTION: the B side is still measured on the common scale, so its resistance
    # carries the partition exactly as it does in the series sum.
    da_b = exchange_velocity * (length_b / (partition * diffusivity_b))
    return np.array([da_a, da_b, min(da_a, da_b)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "governing_damkohler(0.37, 0.57, 0.53, 0.14, 2.9036144578313254, 0.31)",
         "gold_call": "_oracle_governing_damkohler(0.37, 0.57, 0.53, 0.14, 2.9036144578313254, 0.31)"},   # normal
        {"setup": "import numpy as np",
         "call": "governing_damkohler(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)",
         "gold_call": "_oracle_governing_damkohler(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "governing_damkohler(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3, 2.7e-6)",
         "gold_call": "_oracle_governing_damkohler(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3, 2.7e-6)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        governing_damkohler(0.37, 0.57, 0.53, 0.14, 2.9, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_governing_damkohler(0.37, 0.57, 0.53, 0.14, 2.9, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
