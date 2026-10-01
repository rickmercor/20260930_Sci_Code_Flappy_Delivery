"""
Run the whole pipeline on one interface and report it. Build the first-order problem from the two solubilities and solve it; build the second-order problem from the two solubility constants and solve it; open both channels together, taking the fluoride diffusivity as a quarter of the molecular one, the oxidation reverse constant as half its forward one, and the heavier isotope at a quarter of the lighter, with every carrier removed at its outer face; then combine the two failure modes into the single indicator the source defines. Report as well the group that decides for this asymmetric pair and the branching ratio at the reference concentration the source builds its map on. The source fixes a convention here that the natural reading does not; follow the source.

The two failure modes of an equilibrated interface are independent of one another: a channel may be too slow, or the throughput may be split between channels no single algebraic law covers. An indicator that reports one interface with one number has to combine them, and the source is explicit that how it combines them is a choice. The departure reported last is measured on the quantity the second-order channel equilibrates, not on a concentration.

Returns
-------
ndarray of shape (11,): the first-order flux, its two interfacial traces, the relative flux error, the second-order dimensionless group, the apparent exponent, the heavy isotope's atomic budget, the failure indicator, the second-order departure, the group that decides for the asymmetric system, then the branching ratio at the reference concentration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interface_regime_report(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, solubility_a: float, solubility_b: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float, sieverts_constant: float, henry_constant: float, recomb_forward: float, fluor_forward: float, fluoride_activity: float) -> "np.ndarray":
    """Run the whole pipeline on one interface and report it. Build the first-order problem from the two solubilities and solve it; build the second-order problem from the two solubility constants and solve it; open both channels together, taking the fluoride diffusivity as a quarter of the molecular one, the oxidation reverse constant as half its forward one, and the heavier isotope at a quarter of the lighter, with every carrier removed at its outer face; then combine the two failure modes into the single indicator the source defines. Report as well the group that decides for this asymmetric pair and the branching ratio at the reference concentration the source builds its map on. The source fixes a convention here that the natural reading does not; follow the source.

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
    solubility_a : float
        Solubility of the dissolved species in slab A. Positive.
    solubility_b : float
        Solubility of the dissolved species in slab B. Positive.
    exchange_velocity : float
        Forward rate constant of the first-order channel. Positive.
    c_outer_a : float
        Concentration held on the outer face of slab A.
    c_outer_b : float
        Concentration held on the outer face of slab B.
    sieverts_constant : float
        Dissociative solubility constant on the donor side. Positive.
    henry_constant : float
        Molecular solubility constant on the receiving side. Positive.
    recomb_forward : float
        Forward rate constant of the recombination channel. Positive.
    fluor_forward : float
        Forward rate constant of the oxidation channel. Positive.
    fluoride_activity : float
        Lumped activity of the oxidising medium. Not negative.

    Returns
    -------
    ndarray of shape (11,): the first-order flux, its two interfacial traces, the relative flux error, the second-order dimensionless group, the apparent exponent, the heavy isotope's atomic budget, the failure indicator, the second-order departure, the group that decides for the asymmetric system, then the branching ratio at the reference concentration.

    Raises
    ------
    ValueError: if any of the three rate constants is not positive, or any value it forwards to an earlier step is invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_interface_regime_report(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, solubility_a: float, solubility_b: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float, sieverts_constant: float, henry_constant: float, recomb_forward: float, fluor_forward: float, fluoride_activity: float) -> "np.ndarray":
    if exchange_velocity <= 0 or recomb_forward <= 0 or fluor_forward <= 0:
        raise ValueError("rate constants must be positive")
    partition = _oracle_first_order_rate_ratio(solubility_a, solubility_b)
    res = _oracle_slab_resistances(length_a, diffusivity_a, length_b,
                                   diffusivity_b, partition)
    flux = _oracle_linear_channel_flux(length_a, diffusivity_a, length_b,
                                       diffusivity_b, partition,
                                       exchange_velocity, c_outer_a, c_outer_b)
    traces = _oracle_interfacial_traces(length_a, diffusivity_a, length_b,
                                        diffusivity_b, partition,
                                        exchange_velocity, c_outer_a, c_outer_b)
    da_pair = _oracle_two_sided_damkohler(length_a, diffusivity_a, length_b,
                                          diffusivity_b, partition,
                                          exchange_velocity)
    ratio2 = _oracle_recombination_rate_ratio(sieverts_constant, henry_constant)
    rec = _oracle_recombination_interface(length_a, diffusivity_a, length_b,
                                          diffusivity_b, recomb_forward,
                                          sieverts_constant, henry_constant,
                                          c_outer_a, c_outer_b)
    da_quad = _oracle_quadratic_channel_damkohler(length_a, diffusivity_a,
                                                  recomb_forward, c_outer_a)
    comp = _oracle_competing_channel_traces(
        length_a, diffusivity_a, length_b, diffusivity_b, diffusivity_b / 4.0,
        recomb_forward, recomb_forward / ratio2, fluor_forward,
        fluor_forward / 2.0, fluoride_activity, c_outer_a)
    be = _oracle_branching_ratio_and_exponent(comp[1], comp[2])
    gov = _oracle_governing_damkohler(length_a, diffusivity_a, length_b,
                                      diffusivity_b, partition, exchange_velocity)
    ref_b = _oracle_reference_branching_ratio(
        length_b, diffusivity_b, diffusivity_b / 4.0, recomb_forward,
        recomb_forward / ratio2, fluor_forward, fluor_forward / 2.0,
        fluoride_activity, c_outer_a)
    # The isotope report is taken in the same swept-salt regime as the competing
    # channels above: every carrier is held at zero on the outer liquid face, so each
    # channel runs forward only and chi_HT is a genuine fraction.
    # The carriers are swept at the OUTER face of the salt, which is not the same place
    # as the interface: with finite transport resistance each carrier sits at its own
    # rate times its own bulk resistance at the interface, exactly as in the two-channel
    # step above. Passing zero here would make the interface a perfect sink and discard
    # every reverse term. Eliminating c = w R against w = k+ P - k- c gives the traces
    # in closed form.
    r_mol = length_b / diffusivity_b
    r_flu = length_b / (diffusivity_b / 4.0)
    krm = recomb_forward / ratio2
    kfm = fluor_forward / 2.0
    c_h = comp[0]
    c_t = 0.25 * c_h
    den_mol = 1.0 + krm * r_mol
    den_flu = 1.0 + kfm * r_flu
    c_hh = recomb_forward * c_h * c_h * r_mol / den_mol
    c_ht = 2.0 * recomb_forward * c_h * c_t * r_mol / den_mol
    c_tt = recomb_forward * c_t * c_t * r_mol / den_mol
    c_tf = fluor_forward * fluoride_activity * c_t * r_flu / den_flu
    iso = _oracle_isotopologue_fluxes(recomb_forward, krm, fluor_forward, kfm,
                                      c_h, c_t, c_hh, c_ht, c_tt, c_tf,
                                      fluoride_activity)
    # CONVENTION (paper, secs. 5.2-5.3): Da and B index the SAME interface, and the one
    # being indicated here carries the two competing channels - not the first-order
    # channel, whose k+ and partition do not exist at this interface.
    # CONVENTION (paper, eq. 25 text and fig. 2 caption): the first term of the indicator
    # IS the relative flux error, "exact for Model 1 by eq. (A.3)", and the caption fixes
    # the group it depends on as the TWO-SIDED one of eq. (A.2). Sec. 2.7's per-side rule
    # diagnoses which side limits equilibration; it is not the map's axis. So the group is
    # built on the total bulk resistance, scaled by this channel's own partition, with the
    # second-order channel linearised at the reference loading.
    da_rec = _oracle_two_sided_damkohler(length_a, diffusivity_a, length_b, diffusivity_b,
                                         ratio2, 2.0 * recomb_forward * c_outer_a)
    # CONVENTION (paper, eq. 25): the indicator takes the LARGER of the two failure modes, not their sum and not their average.
    # The golden solution carries the full argument and the natural wrong answer.
    indicator = max(float(da_rec[1]), min(1.0, ref_b) / (1.0 + ref_b))
    departure = abs(rec[2] / rec[1] ** 2 - ratio2) / ratio2
    # iso[3] is the mixed share, which in the swept limit collapses to
    # c_H/(c_H+c_T) and grades nothing. iso[2], the heavy isotope's total atomic
    # budget, carries both the degeneracy convention and the stoichiometric weights.
    return np.array([flux, traces[0], traces[1], da_pair[1], da_quad,
                     be[1], iso[2], indicator, departure, gov[2], ref_b])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "interface_regime_report(0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 100.0, 2.0, 1.0, 1.7, 2.3, 3.0, 0.8, 1.0)",
         "gold_call": "_oracle_interface_regime_report(0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 100.0, 2.0, 1.0, 1.7, 2.3, 3.0, 0.8, 1.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "interface_regime_report(0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 1e6, 2.0, 1.0, 1.0, 1.0, 50.0, 0.05, 1.0)",
         "gold_call": "_oracle_interface_regime_report(0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 1e6, 2.0, 1.0, 1.0, 1.0, 50.0, 0.05, 1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "interface_regime_report(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 8.2e-7, 4.1e-3, 2.7e-6, 8.4e2, 1.1e1, 4.7e-4, 2.9e2, 5.2e-3, 9.3e-5, 3.6e1)",
         "gold_call": "_oracle_interface_regime_report(3.1e-3, 7.4e-11, 2.6e-2, 1.9e-9, 8.2e-7, 4.1e-3, 2.7e-6, 8.4e2, 1.1e1, 4.7e-4, 2.9e2, 5.2e-3, 9.3e-5, 3.6e1)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        interface_regime_report(0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 0.0, 2.0, 1.0, 1.7, 2.3, 3.0, 0.8, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_interface_regime_report(0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 0.0, 2.0, 1.0, 1.7, 2.3, 3.0, 0.8, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
