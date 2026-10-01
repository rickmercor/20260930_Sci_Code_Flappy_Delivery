"""
Solve the steady problem in which the same interface carries two channels at once: the recombination channel of the earlier step and a first-order channel that oxidises the atom to a second, more soluble carrier. Both carriers are transported in the second slab with different diffusivities, and each is removed at its outer face. The source fixes a convention here that the natural reading does not; follow the source.

Two channels of different order share one interface, and the atomic budget leaving the donor material is shared between them. Because each carrier is removed at the far face of the second slab, each channel is limited by its own carrier's transport and the two do not compete for the same reverse term. The activity of the oxidising medium enters one channel only.

Returns
-------
ndarray of shape (5,): the donor trace, the recombination rate, the oxidation rate, then the two carrier traces in the order molecular, fluoride.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def competing_channel_traces(length_m: float, diffusivity_m: float, length_s: float, diff_molecular: float, diff_fluoride: float, recomb_forward: float, recomb_reverse: float, fluor_forward: float, fluor_reverse: float, fluoride_activity: float, c_outer_m: float) -> "np.ndarray":
    """Solve the steady problem in which the same interface carries two channels at once: the recombination channel of the earlier step and a first-order channel that oxidises the atom to a second, more soluble carrier. Both carriers are transported in the second slab with different diffusivities, and each is removed at its outer face. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_m : float
        Thickness of the donor slab. Positive.
    diffusivity_m : float
        Diffusivity in the donor slab. Positive.
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
    c_outer_m : float
        Concentration held on the outer face of the donor slab.

    Returns
    -------
    ndarray of shape (5,): the donor trace, the recombination rate, the oxidation rate, then the two carrier traces in the order molecular, fluoride.

    Raises
    ------
    ValueError: if either forward rate constant is not positive, or fluoride_activity is negative.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_competing_channel_traces(length_m: float, diffusivity_m: float, length_s: float, diff_molecular: float, diff_fluoride: float, recomb_forward: float, recomb_reverse: float, fluor_forward: float, fluor_reverse: float, fluoride_activity: float, c_outer_m: float) -> "np.ndarray":
    if recomb_forward <= 0 or fluor_forward <= 0:
        raise ValueError("forward rate constants must be positive")
    if fluoride_activity < 0:
        raise ValueError("fluoride activity must not be negative")
    r_m = length_m / diffusivity_m
    r_h2 = length_s / diff_molecular
    r_hf = length_s / diff_fluoride
    # CONVENTION (paper, sec. 4.2): the SWEPT-SALT regime.
    # The golden solution carries the full argument and the natural wrong answer.
    coef_rec = recomb_forward / (1.0 + recomb_reverse * r_h2)
    # CONVENTION (paper, eq. 13): the fluoride activity and the redox potential are
    # LUMPED into one effective activity multiplying the FORWARD constant only. The
    # reverse constant carries no activity.
    coef_flu = fluor_forward * fluoride_activity / (1.0 + fluor_reverse * r_hf)
    # CONVENTION (paper, eq. 14): the atomic flux leaving the metal is 2 w_rec + w_F.
    # The stoichiometric two sits on the RECOMBINATION channel alone, because only
    # that channel consumes two atoms per event; fluorination consumes one.
    a = 2.0 * coef_rec * r_m
    b = 1.0 + coef_flu * r_m
    c_m = (-b + np.sqrt(b * b + 4.0 * a * c_outer_m)) / (2.0 * a)
    w_rec = coef_rec * c_m ** 2
    w_flu = coef_flu * c_m
    return np.array([c_m, w_rec, w_flu, w_rec * r_h2, w_flu * r_hf])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "competing_channel_traces(0.5, 0.5, 0.5, 1.0, 0.25, 3.0, 1.5, 0.8, 0.4, 1.0, 2.0)",
         "gold_call": "_oracle_competing_channel_traces(0.5, 0.5, 0.5, 1.0, 0.25, 3.0, 1.5, 0.8, 0.4, 1.0, 2.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "competing_channel_traces(0.5, 0.5, 0.5, 1.0, 0.25, 3.0, 1.5, 0.8, 0.4, 0.0, 2.0)",
         "gold_call": "_oracle_competing_channel_traces(0.5, 0.5, 0.5, 1.0, 0.25, 3.0, 1.5, 0.8, 0.4, 0.0, 2.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "competing_channel_traces(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 4.8e-10, 5.2e-3, 1.7e-2, 9.3e-5, 6.1e-4, 3.6e1, 8.4e2)",
         "gold_call": "_oracle_competing_channel_traces(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 4.8e-10, 5.2e-3, 1.7e-2, 9.3e-5, 6.1e-4, 3.6e1, 8.4e2)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        competing_channel_traces(0.5, 0.5, 0.5, 1.0, 0.25, 3.0, 1.5, 0.8, 0.4, -1.0, 2.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_competing_channel_traces(0.5, 0.5, 0.5, 1.0, 0.25, 3.0, 1.5, 0.8, 0.4, -1.0, 2.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
