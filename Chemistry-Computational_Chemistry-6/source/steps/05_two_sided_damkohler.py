"""
Return the dimensionless group that decides whether a first-order interface may be treated as locally equilibrated, together with the relative error that treatment makes on the flux. The group compares bulk transport against interfacial exchange. The source fixes a convention here that the natural reading does not; follow the source.

Treating an interface as equilibrated is exact only when its exchange is fast compared with the transport that feeds it. The comparison is a ratio of resistances, and the error the assumption makes on the flux is a closed function of that ratio alone.

Returns
-------
ndarray of shape (2,): the dimensionless group, then the relative error on the flux.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def two_sided_damkohler(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float) -> "np.ndarray":
    """Return the dimensionless group that decides whether a first-order interface may be treated as locally equilibrated, together with the relative error that treatment makes on the flux. The group compares bulk transport against interfacial exchange. The source fixes a convention here that the natural reading does not; follow the source.

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
    ndarray of shape (2,): the dimensionless group, then the relative error on the flux.

    Raises
    ------
    ValueError: if exchange_velocity is not positive, or any value it forwards to an earlier step is invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_two_sided_damkohler(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float) -> "np.ndarray":
    if exchange_velocity <= 0:
        raise ValueError("exchange velocity must be positive")
    res = _oracle_slab_resistances(length_a, diffusivity_a, length_b,
                                   diffusivity_b, partition)
    # CONVENTION (paper, eq. A.2): the group that governs the flux error is built on the TOTAL bulk resistance of BOTH slabs, not on the upstream slab alone.
    # The golden solution carries the full argument and the natural wrong answer.
    da = exchange_velocity * (res[0] + res[1])
    # CONVENTION (paper, eq. A.3): the relative error LTE makes on the flux is
    # 1/(1+Da*), the exact complement of Da*/(1+Da*). Quoting the leading term 1/Da*
    # instead is wrong by O(Da*^-2) and is visible at moderate Da*.
    return np.array([da, 1.0 / (1.0 + da)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "two_sided_damkohler(0.5, 0.5, 0.5, 1.0, 2.0, 100.0)",
         "gold_call": "_oracle_two_sided_damkohler(0.5, 0.5, 0.5, 1.0, 2.0, 100.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "two_sided_damkohler(0.5, 0.5, 0.5, 1.0, 2.0, 1.0)",
         "gold_call": "_oracle_two_sided_damkohler(0.5, 0.5, 0.5, 1.0, 2.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "two_sided_damkohler(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3, 2.7e-6)",
         "gold_call": "_oracle_two_sided_damkohler(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3, 2.7e-6)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        two_sided_damkohler(0.5, 0.5, 0.5, 1.0, 2.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_two_sided_damkohler(0.5, 0.5, 0.5, 1.0, 2.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
