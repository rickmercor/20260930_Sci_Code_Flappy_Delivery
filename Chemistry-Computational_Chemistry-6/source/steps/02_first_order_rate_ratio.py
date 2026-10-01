"""
Return the ratio of the forward to the reverse rate constant of a first-order exchange channel that moves an atom across an interface without changing its chemical identity, given the solubility of that atom in each of the two materials. Thermodynamic consistency fixes this ratio completely. The source fixes a convention here that the natural reading does not; follow the source.

A reversible channel at an interface cannot have independent forward and reverse rate constants: their ratio is the equilibrium constant of the channel, which follows from the solubilities of the species taking part. Only the magnitude of the pair is a free input.

Returns
-------
float, the ratio of the forward rate constant to the reverse one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def first_order_rate_ratio(solubility_a: float, solubility_b: float) -> float:
    """Return the ratio of the forward to the reverse rate constant of a first-order exchange channel that moves an atom across an interface without changing its chemical identity, given the solubility of that atom in each of the two materials. Thermodynamic consistency fixes this ratio completely. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    solubility_a : float
        Solubility of the dissolved species in slab A. Positive.
    solubility_b : float
        Solubility of the dissolved species in slab B. Positive.

    Returns
    -------
    float, the ratio of the forward rate constant to the reverse one.

    Raises
    ------
    ValueError: if either solubility is not positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_first_order_rate_ratio(solubility_a: float, solubility_b: float) -> float:
    if solubility_a <= 0 or solubility_b <= 0:
        raise ValueError("solubilities must be positive")
    # CONVENTION (paper, eq. 10): detailed balance fixes the ratio of the two rate constants from thermodynamics alone, and it is the RECEIVING side over the DONATING side.
    # The golden solution carries the full argument and the natural wrong answer.
    return solubility_b / solubility_a

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "first_order_rate_ratio(1.0, 2.0)",
         "gold_call": "_oracle_first_order_rate_ratio(1.0, 2.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "first_order_rate_ratio(3.5, 3.5)",
         "gold_call": "_oracle_first_order_rate_ratio(3.5, 3.5)"},   # boundary
        {"setup": "import numpy as np",
         "call": "first_order_rate_ratio(8.2e-7, 4.1e-3)",
         "gold_call": "_oracle_first_order_rate_ratio(8.2e-7, 4.1e-3)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        first_order_rate_ratio(1.0, -2.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_first_order_rate_ratio(1.0, -2.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
