"""
With two isotopes present in the donor material, recombination populates three molecules rather than one. Return the two like-atom and mixed channel rates, the total atomic budget of the heavier isotope leaving the donor, and the share of that budget carried by the mixed molecule measured across the RECOMBINATION channels only, with the oxidation channel excluded from that share. Every channel here is reversible and each carrier's own interfacial concentration is supplied. A single pair of constants parameterises all three recombination channels. The source fixes a convention here that the natural reading does not; follow the source.

The mixed molecule is formed from one atom of each isotope, so its rate depends on both donor concentrations and couples the two isotopes' interface conditions together. Its forward rate constant is not independent of the like-atom ones: the source fixes the relation, and it is not the equality the symmetry of the channels suggests. The oxidation channel pairs no isotopes, since each isotope forms its own product, which is why the mixed share is taken over the recombination channels alone.

Returns
-------
ndarray of shape (4,): the like-atom rate for the light isotope, the mixed rate, the total atomic budget of the heavy isotope, then the mixed molecule's share of the recombination-only budget, w_HT/(w_HT + 2 w_TT).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def isotopologue_fluxes(k_homonuclear_forward: float, k_homonuclear_reverse: float, k_fluorination_forward: float, k_fluorination_reverse: float, c_metal_h: float, c_metal_t: float, c_salt_hh: float, c_salt_ht: float, c_salt_tt: float, c_salt_tf: float, fluoride_activity: float) -> "np.ndarray":
    """With two isotopes present in the donor material, recombination populates three molecules rather than one. Return the two like-atom and mixed channel rates, the total atomic budget of the heavier isotope leaving the donor, and the share of that budget carried by the mixed molecule measured across the RECOMBINATION channels only, with the oxidation channel excluded from that share. Every channel here is reversible and each carrier's own interfacial concentration is supplied. A single pair of constants parameterises all three recombination channels. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    k_homonuclear_forward : float
        Forward rate constant of a like-atom recombination channel. Positive.
    k_homonuclear_reverse : float
        Reverse rate constant shared by the recombination channels. Positive.
    k_fluorination_forward : float
        Forward rate constant of an oxidation channel. Positive.
    k_fluorination_reverse : float
        Reverse rate constant of an oxidation channel. Positive.
    c_metal_h : float
        Interfacial concentration of the light isotope in the donor slab.
    c_metal_t : float
        Interfacial concentration of the heavy isotope in the donor slab.
    c_salt_hh : float
        Interfacial concentration of the light homonuclear molecule.
    c_salt_ht : float
        Interfacial concentration of the mixed molecule.
    c_salt_tt : float
        Interfacial concentration of the heavy homonuclear molecule.
    c_salt_tf : float
        Interfacial concentration of the heavy fluoride.
    fluoride_activity : float
        Lumped activity of the oxidising medium. Not negative.

    Returns
    -------
    ndarray of shape (4,): the like-atom rate for the light isotope, the mixed rate, the total atomic budget of the heavy isotope, then the mixed molecule's share of the recombination-only budget, w_HT/(w_HT + 2 w_TT).

    Raises
    ------
    ValueError: if either homonuclear rate constant is not positive, either fluorination rate constant is not positive, or the recombination channels carry no heavy isotope so the mixed share is undefined.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_isotopologue_fluxes(k_homonuclear_forward: float, k_homonuclear_reverse: float, k_fluorination_forward: float, k_fluorination_reverse: float, c_metal_h: float, c_metal_t: float, c_salt_hh: float, c_salt_ht: float, c_salt_tt: float, c_salt_tf: float, fluoride_activity: float) -> "np.ndarray":
    if k_homonuclear_forward <= 0 or k_homonuclear_reverse <= 0:
        raise ValueError("rate constants must be positive")
    if k_fluorination_forward <= 0 or k_fluorination_reverse <= 0:
        raise ValueError("fluorination rate constants must be positive")
    # CONVENTION (paper, footnote 1 to sec. 2.6): the rate laws carry NO combinatorial prefactor, so the statistical degeneracy of the mixed pair is carried in the value of the FORWARD constant alone: k+_HT = 2 k+_HH = 2 k+_TT, with the reverse constants left EQUAL. That asymmetry is what returns the classical exchange constant K = (c_HT)^2/(c_HH c_TT) = 4 for random pairing.
    # The golden solution carries the full argument and the natural wrong answer.
    w_hh = k_homonuclear_forward * c_metal_h ** 2 - k_homonuclear_reverse * c_salt_hh
    w_ht = (2.0 * k_homonuclear_forward * c_metal_h * c_metal_t
            - k_homonuclear_reverse * c_salt_ht)
    w_tt = k_homonuclear_forward * c_metal_t ** 2 - k_homonuclear_reverse * c_salt_tt
    # Every channel is REVERSIBLE (paper, eqs. 17-21). The fluorination channel has a
    # reverse term exactly as the recombination channels do; dropping it would make the
    # interface a perfect sink for the fluoride, which the transport problem never is.
    w_tf = (k_fluorination_forward * fluoride_activity * c_metal_t
            - k_fluorination_reverse * c_salt_tf)
    # CONVENTION (paper, eq. 23): the stoichiometric two sits only on the HOMONUCLEAR
    # channel. The mixed isotopologue removes ONE tritium atom per event, so it enters
    # the tritium balance with weight one even though it is a recombination channel.
    atomic_t = 2.0 * w_tt + w_ht + w_tf
    # CONVENTION (paper, eq. 26): the mixing fraction is built on the RECOMBINATION
    # channels ALONE. Fluorination pairs no isotopes - each isotope feeds its own
    # fluoride - so w_TF is excluded from the denominator even though it carries
    # tritium across. Including it is the natural wrong answer.
    # With no heavy isotope present there is no recombination budget to take a share of,
    # so the fraction is undefined rather than zero. Say so instead of dividing.
    recomb_budget = w_ht + 2.0 * w_tt
    if recomb_budget == 0.0:
        raise ValueError("mixed share is undefined when the recombination channels carry "
                         "no heavy isotope")
    chi_ht = w_ht / recomb_budget
    return np.array([w_hh, w_ht, atomic_t, chi_ht])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "isotopologue_fluxes(1.29, 1.6594617977528092, 0.29, 0.145, 1.2944441429253861, 0.32361103573134653, 1.1236697638097957, 0.5618348819048979, 0.07022936023811223, 0.8760470442383433, 1.97)",
         "gold_call": "_oracle_isotopologue_fluxes(1.29, 1.6594617977528092, 0.29, 0.145, 1.2944441429253861, 0.32361103573134653, 1.1236697638097957, 0.5618348819048979, 0.07022936023811223, 0.8760470442383433, 1.97)"},   # normal
        {"setup": "import numpy as np",
         "call": "isotopologue_fluxes(1.0, 1.0, 0.5, 0.25, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)",
         "gold_call": "_oracle_isotopologue_fluxes(1.0, 1.0, 0.5, 0.25, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "isotopologue_fluxes(5.2e-3, 1.7e-2, 9.3e-5, 6.1e-4, 8.4e2, 3.1e1, 2.2e0, 4.5e-1, 6.6e-2, 1.3e-1, 3.6e1)",
         "gold_call": "_oracle_isotopologue_fluxes(5.2e-3, 1.7e-2, 9.3e-5, 6.1e-4, 8.4e2, 3.1e1, 2.2e0, 4.5e-1, 6.6e-2, 1.3e-1, 3.6e1)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        isotopologue_fluxes(0.0, 1.0, 0.5, 0.25, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_isotopologue_fluxes(0.0, 1.0, 0.5, 0.25, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
