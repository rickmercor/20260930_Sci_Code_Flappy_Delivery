"""
Return the ratio of the forward to the reverse rate constant of a channel in which two atoms dissolved in the first material combine into one molecule that dissolves physically in the second. The two solubility constants describe dissociative dissolution on one side and molecular dissolution on the other. The source fixes a convention here that the natural reading does not; follow the source.

Thermodynamic consistency again fixes the ratio, but the channel is not first order in the donor material, and the order of a channel enters its equilibrium constant. The two constants supplied are not interchangeable with those of a first-order channel.

Returns
-------
float, the ratio of the forward rate constant to the reverse one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recombination_rate_ratio(sieverts_constant: float, henry_constant: float) -> float:
    """Return the ratio of the forward to the reverse rate constant of a channel in which two atoms dissolved in the first material combine into one molecule that dissolves physically in the second. The two solubility constants describe dissociative dissolution on one side and molecular dissolution on the other. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    sieverts_constant : float
        Dissociative solubility constant on the donor side. Positive.
    henry_constant : float
        Molecular solubility constant on the receiving side. Positive.

    Returns
    -------
    float, the ratio of the forward rate constant to the reverse one.

    Raises
    ------
    ValueError: if either solubility constant is not positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_recombination_rate_ratio(sieverts_constant: float, henry_constant: float) -> float:
    if sieverts_constant <= 0 or henry_constant <= 0:
        raise ValueError("solubility constants must be positive")
    # CONVENTION (paper, eq. 12): the channel is SECOND ORDER on the metal side, so
    # detailed balance carries the metal-side constant SQUARED. K_H/K_S is the
    # natural wrong answer: it is what the first-order channel gives, it has the
    # right sign and the right monotonicity, and only the exponent distinguishes it.
    return henry_constant / sieverts_constant ** 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "recombination_rate_ratio(1.0, 0.5)",
         "gold_call": "_oracle_recombination_rate_ratio(1.0, 0.5)"},   # normal
        {"setup": "import numpy as np",
         "call": "recombination_rate_ratio(1.0, 1.0)",
         "gold_call": "_oracle_recombination_rate_ratio(1.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "recombination_rate_ratio(4.7e-4, 2.9e2)",
         "gold_call": "_oracle_recombination_rate_ratio(4.7e-4, 2.9e2)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        recombination_rate_ratio(0.0, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_recombination_rate_ratio(0.0, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
