"""
Return the bulk transport resistance contributed by each of the two slabs in a steady one-dimensional permeation problem, in the form the source's interface model adds them in. The two slabs measure concentration on scales that differ by the partition, and the returned pair must already account for that. The source fixes a convention here that the natural reading does not; follow the source.

A slab of thickness L and diffusivity D carrying a steady flux with no source has a linear profile, so it behaves as a resistance in a transport circuit. Two such slabs sharing an interface are placed in one circuit together with the interface itself.

Returns
-------
ndarray of shape (2,): the resistance of slab A, then the resistance of slab B.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def slab_resistances(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float) -> "np.ndarray":
    """Return the bulk transport resistance contributed by each of the two slabs in a steady one-dimensional permeation problem, in the form the source's interface model adds them in. The two slabs measure concentration on scales that differ by the partition, and the returned pair must already account for that. The source fixes a convention here that the natural reading does not; follow the source.

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

    Returns
    -------
    ndarray of shape (2,): the resistance of slab A, then the resistance of slab B.

    Raises
    ------
    ValueError: if either slab length is not positive, either diffusivity is not positive, or partition is not positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_slab_resistances(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float) -> "np.ndarray":
    if length_a <= 0 or length_b <= 0:
        raise ValueError("slab lengths must be positive")
    if diffusivity_a <= 0 or diffusivity_b <= 0:
        raise ValueError("diffusivities must be positive")
    if partition <= 0:
        raise ValueError("partition must be positive")
    # CONVENTION (paper, eq. B.2 second form / eq. A.2): the two bulk resistances
    # are only additive once both are expressed on a COMMON CONCENTRATION SCALE.
    # Side A is measured in its own units; side B must be divided by the partition
    # K = k+/k-, because it is c_B/K that is continuous with c_A at equilibrium.
    # The natural wrong answer is the plain L_B/D_B.
    return np.array([length_a / diffusivity_a,
                     length_b / (partition * diffusivity_b)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "slab_resistances(0.5, 0.5, 0.5, 1.0, 2.0)",
         "gold_call": "_oracle_slab_resistances(0.5, 0.5, 0.5, 1.0, 2.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "slab_resistances(1.0, 1.0, 1.0, 1.0, 1.0)",
         "gold_call": "_oracle_slab_resistances(1.0, 1.0, 1.0, 1.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "slab_resistances(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3)",
         "gold_call": "_oracle_slab_resistances(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 47.3)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        slab_resistances(0.5, 0.5, 0.5, 1.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_slab_resistances(0.5, 0.5, 0.5, 1.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
