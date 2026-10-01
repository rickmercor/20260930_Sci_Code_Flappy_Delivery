"""
Solve the steady two-slab problem whose interface carries the recombination channel of the previous step, returning the channel rate and the two interfacial traces. Atoms live in the first slab and molecules in the second, so the two slabs do not carry the same quantity. The source fixes a convention here that the natural reading does not; follow the source.

Because the channel changes the chemical identity of what is transported, the bookkeeping across the interface is not a simple continuity of flux: what leaves one side and what enters the other are counted in different species. The resulting steady condition is quadratic and has one physical branch.

Returns
-------
ndarray of shape (3,): the channel rate, the trace in the first slab, then the trace in the second.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recombination_interface(length_m: float, diffusivity_m: float, length_s: float, diffusivity_s: float, forward_rate: float, sieverts_constant: float, henry_constant: float, c_outer_m: float, c_outer_s: float) -> "np.ndarray":
    """Solve the steady two-slab problem whose interface carries the recombination channel of the previous step, returning the channel rate and the two interfacial traces. Atoms live in the first slab and molecules in the second, so the two slabs do not carry the same quantity. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_m : float
        Thickness of the donor slab. Positive.
    diffusivity_m : float
        Diffusivity in the donor slab. Positive.
    length_s : float
        Thickness of the receiving slab. Positive.
    diffusivity_s : float
        Diffusivity in the receiving slab. Positive.
    forward_rate : float
        Forward rate constant of the recombination channel. Positive.
    sieverts_constant : float
        Dissociative solubility constant on the donor side. Positive.
    henry_constant : float
        Molecular solubility constant on the receiving side. Positive.
    c_outer_m : float
        Concentration held on the outer face of the donor slab.
    c_outer_s : float
        Concentration held on the outer face of the receiving slab.

    Returns
    -------
    ndarray of shape (3,): the channel rate, the trace in the first slab, then the trace in the second.

    Raises
    ------
    ValueError: if forward_rate is not positive, either slab length is not positive, or either solubility constant is not positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_recombination_interface(length_m: float, diffusivity_m: float, length_s: float, diffusivity_s: float, forward_rate: float, sieverts_constant: float, henry_constant: float, c_outer_m: float, c_outer_s: float) -> "np.ndarray":
    if forward_rate <= 0:
        raise ValueError("forward rate constant must be positive")
    if length_m <= 0 or length_s <= 0:
        raise ValueError("slab lengths must be positive")
    ratio = _oracle_recombination_rate_ratio(sieverts_constant, henry_constant)
    reverse = forward_rate / ratio
    r_m = length_m / diffusivity_m
    r_s = length_s / diffusivity_s
    # CONVENTION (paper, eq. 11 with eq. 7): the two slabs carry DIFFERENT fluxes.
    # The golden solution carries the full argument and the natural wrong answer.
    a = 4.0 * forward_rate * r_m ** 2
    b = -(4.0 * forward_rate * r_m * c_outer_m + reverse * r_s + 1.0)
    c = forward_rate * c_outer_m ** 2 - reverse * c_outer_s
    # CONVENTION: the physical branch is the SMALLER root. The larger one drives the
    # metal trace negative.
    w = (-b - np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return np.array([w, c_outer_m - 2.0 * w * r_m, c_outer_s + w * r_s])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "recombination_interface(0.5, 0.5, 0.5, 1.0, 30.0, 1.0, 0.5, 2.0, 1.0)",
         "gold_call": "_oracle_recombination_interface(0.5, 0.5, 0.5, 1.0, 30.0, 1.0, 0.5, 2.0, 1.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "recombination_interface(0.5, 0.5, 0.5, 1.0, 1e8, 1.0, 0.5, 2.0, 1.0)",
         "gold_call": "_oracle_recombination_interface(0.5, 0.5, 0.5, 1.0, 1e8, 1.0, 0.5, 2.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "recombination_interface(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 5.2e-3, 4.7e-4, 2.9e2, 8.4e2, 1.1e1)",
         "gold_call": "_oracle_recombination_interface(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 5.2e-3, 4.7e-4, 2.9e2, 8.4e2, 1.1e1)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        recombination_interface(0.5, 0.5, 0.5, 1.0, -1.0, 1.0, 0.5, 2.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_recombination_interface(0.5, 0.5, 0.5, 1.0, -1.0, 1.0, 0.5, 2.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
