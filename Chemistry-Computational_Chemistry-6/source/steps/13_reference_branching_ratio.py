"""
Return the branching ratio that indexes the source's map of interface behaviour, formed from the two channel rates evaluated at a supplied reference concentration rather than at a concentration any solve produces. Both carriers are removed at the far face of the second slab, as in the earlier two-channel step. The source fixes a convention here that the natural reading does not; follow the source.

Two channels of different order share one interface, so which of them carries the throughput depends on the loading and the ratio between them is not a property of the interface alone. A ratio used as the axis of a map therefore has to be quoted with the concentration it was formed at, and that concentration is not in general the one the interface settles to.

Returns
-------
float, the branching ratio at the supplied reference concentration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reference_branching_ratio(length_s: float, diff_molecular: float, diff_fluoride: float, recomb_forward: float, recomb_reverse: float, fluor_forward: float, fluor_reverse: float, fluoride_activity: float, reference_concentration: float) -> float:
    """Return the branching ratio that indexes the source's map of interface behaviour, formed from the two channel rates evaluated at a supplied reference concentration rather than at a concentration any solve produces. Both carriers are removed at the far face of the second slab, as in the earlier two-channel step. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_s : float
        Thickness of the receiving slab. Positive.
    diff_molecular : float
        Diffusivity of the molecular carrier in the receiving slab. Positive.
    diff_fluoride : float
        Diffusivity of the fluoride carrier in the receiving slab. Positive.
    recomb_forward : float
        Forward rate constant of the recombination channel. Positive.
    recomb_reverse : float
        Reverse rate constant of the recombination channel.
    fluor_forward : float
        Forward rate constant of the oxidation channel. Positive.
    fluor_reverse : float
        Reverse rate constant of the oxidation channel.
    fluoride_activity : float
        Lumped activity of the oxidising medium. Not negative.
    reference_concentration : float
        Donor loading the ratio is evaluated at. Positive.

    Returns
    -------
    float, the branching ratio at the supplied reference concentration.

    Raises
    ------
    ValueError: if either forward rate constant is not positive, reference_concentration is not positive, or fluoride_activity is negative.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reference_branching_ratio(length_s: float, diff_molecular: float, diff_fluoride: float, recomb_forward: float, recomb_reverse: float, fluor_forward: float, fluor_reverse: float, fluoride_activity: float, reference_concentration: float) -> float:
    if recomb_forward <= 0 or fluor_forward <= 0:
        raise ValueError("forward rate constants must be positive")
    if reference_concentration <= 0:
        raise ValueError("reference concentration must be positive")
    if fluoride_activity < 0:
        raise ValueError("fluoride activity must not be negative")
    # CONVENTION (paper, sec. 2.7): "every dimensionless group quoted in this paper is built at it, the branching ratio included". The group that indexes the failure map is therefore evaluated at the UPSTREAM REFERENCE concentration, not at the interfacial value the solve returns.
    # The golden solution carries the full argument and the natural wrong answer.
    coef_rec = recomb_forward / (1.0 + recomb_reverse * (length_s / diff_molecular))
    coef_flu = (fluor_forward * fluoride_activity
                / (1.0 + fluor_reverse * (length_s / diff_fluoride)))
    w_rec = coef_rec * reference_concentration ** 2
    w_flu = coef_flu * reference_concentration
    # CONVENTION (paper, eq. 15): shares of the ATOMIC budget, so the recombination
    # rate is doubled before the comparison, exactly as at the interface.
    return w_flu / (2.0 * w_rec)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "reference_branching_ratio(0.53, 0.14, 0.035, 1.29, 1.6594617977528092, 0.29, 0.145, 1.97, 1.83)",
         "gold_call": "_oracle_reference_branching_ratio(0.53, 0.14, 0.035, 1.29, 1.6594617977528092, 0.29, 0.145, 1.97, 1.83)"},   # normal
        {"setup": "import numpy as np",
         "call": "reference_branching_ratio(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0)",
         "gold_call": "_oracle_reference_branching_ratio(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "reference_branching_ratio(2.6e-2, 1.9e-9, 4.8e-10, 5.2e-3, 1.7e-2, 9.3e-5, 6.1e-4, 3.6e1, 8.4e2)",
         "gold_call": "_oracle_reference_branching_ratio(2.6e-2, 1.9e-9, 4.8e-10, 5.2e-3, 1.7e-2, 9.3e-5, 6.1e-4, 3.6e1, 8.4e2)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        reference_branching_ratio(0.53, 0.14, 0.035, 1.29, 1.66, 0.29, 0.145, 1.97, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_reference_branching_ratio(0.53, 0.14, 0.035, 1.29, 1.66, 0.29, 0.145, 1.97, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
