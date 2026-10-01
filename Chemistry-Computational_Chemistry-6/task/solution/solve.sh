#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def slab_resistances(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float) -> "np.ndarray":
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

import numpy as np


def first_order_rate_ratio(solubility_a: float, solubility_b: float) -> float:
    if solubility_a <= 0 or solubility_b <= 0:
        raise ValueError("solubilities must be positive")
    # CONVENTION (paper, eq. 10): detailed balance fixes the ratio of the two rate constants from thermodynamics alone, and it is the RECEIVING side over the DONATING side.
    # The golden solution carries the full argument and the natural wrong answer.
    return solubility_b / solubility_a

import numpy as np


def linear_channel_flux(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float) -> float:
    if exchange_velocity <= 0:
        raise ValueError("exchange velocity must be positive")
    res = slab_resistances(length_a, diffusivity_a, length_b,
                                   diffusivity_b, partition)
    # CONVENTION (paper, eq. B.2): three resistances IN SERIES. The interface contributes 1/k+ exactly like a bulk resistance, and the driving term is the difference of the two outer concentrations on the common scale.
    # The golden solution carries the full argument and the natural wrong answer.
    return (c_outer_a - c_outer_b / partition) / (res[0] + res[1]
                                                  + 1.0 / exchange_velocity)

import numpy as np


def interfacial_traces(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float) -> "np.ndarray":
    if exchange_velocity <= 0:
        raise ValueError("exchange velocity must be positive")
    if length_a <= 0 or length_b <= 0:
        raise ValueError("slab lengths must be positive")
    if diffusivity_a <= 0 or diffusivity_b <= 0:
        raise ValueError("diffusivities must be positive")
    flux = linear_channel_flux(length_a, diffusivity_a, length_b,
                                       diffusivity_b, partition,
                                       exchange_velocity, c_outer_a, c_outer_b)
    # CONVENTION (paper, eq. B.1): the TRACE on side B is walked back along B's OWN resistance L_B/D_B, NOT along the scaled L_B/(K D_B) that entered the flux.
    # The golden solution carries the full argument and the natural wrong answer.
    return np.array([c_outer_a - flux * (length_a / diffusivity_a),
                     c_outer_b + flux * (length_b / diffusivity_b)])

import numpy as np


def two_sided_damkohler(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float) -> "np.ndarray":
    if exchange_velocity <= 0:
        raise ValueError("exchange velocity must be positive")
    res = slab_resistances(length_a, diffusivity_a, length_b,
                                   diffusivity_b, partition)
    # CONVENTION (paper, eq. A.2): the group that governs the flux error is built on the TOTAL bulk resistance of BOTH slabs, not on the upstream slab alone.
    # The golden solution carries the full argument and the natural wrong answer.
    da = exchange_velocity * (res[0] + res[1])
    # CONVENTION (paper, eq. A.3): the relative error LTE makes on the flux is
    # 1/(1+Da*), the exact complement of Da*/(1+Da*). Quoting the leading term 1/Da*
    # instead is wrong by O(Da*^-2) and is visible at moderate Da*.
    return np.array([da, 1.0 / (1.0 + da)])

import numpy as np


def recombination_rate_ratio(sieverts_constant: float, henry_constant: float) -> float:
    if sieverts_constant <= 0 or henry_constant <= 0:
        raise ValueError("solubility constants must be positive")
    # CONVENTION (paper, eq. 12): the channel is SECOND ORDER on the metal side, so
    # detailed balance carries the metal-side constant SQUARED. K_H/K_S is the
    # natural wrong answer: it is what the first-order channel gives, it has the
    # right sign and the right monotonicity, and only the exponent distinguishes it.
    return henry_constant / sieverts_constant ** 2

import numpy as np


def recombination_interface(length_m: float, diffusivity_m: float, length_s: float, diffusivity_s: float, forward_rate: float, sieverts_constant: float, henry_constant: float, c_outer_m: float, c_outer_s: float) -> "np.ndarray":
    if forward_rate <= 0:
        raise ValueError("forward rate constant must be positive")
    if length_m <= 0 or length_s <= 0:
        raise ValueError("slab lengths must be positive")
    ratio = recombination_rate_ratio(sieverts_constant, henry_constant)
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

import numpy as np


def quadratic_channel_damkohler(length_m: float, diffusivity_m: float, forward_rate: float, reference_concentration: float) -> float:
    if forward_rate <= 0 or reference_concentration <= 0:
        raise ValueError("rate constant and reference concentration must be positive")
    if length_m <= 0 or diffusivity_m <= 0:
        raise ValueError("length and diffusivity must be positive")
    # CONVENTION (paper, sec. 2.7): a channel of order two has no exchange velocity of its own, so the comparison uses the LINEARISED one, dw/dc evaluated at the interface, which is 2 k+ c and not k+ c.
    # The golden solution carries the full argument and the natural wrong answer.
    return 2.0 * forward_rate * reference_concentration * (length_m / diffusivity_m)

import numpy as np


def competing_channel_traces(length_m: float, diffusivity_m: float, length_s: float, diff_molecular: float, diff_fluoride: float, recomb_forward: float, recomb_reverse: float, fluor_forward: float, fluor_reverse: float, fluoride_activity: float, c_outer_m: float) -> "np.ndarray":
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

import numpy as np


def branching_ratio_and_exponent(recombination_rate: float, fluorination_rate: float) -> "np.ndarray":
    if recombination_rate <= 0:
        raise ValueError("recombination rate must be positive")
    if fluorination_rate < 0:
        raise ValueError("fluorination rate must not be negative")
    # CONVENTION (paper, eq. 15): the branching ratio compares the two channels' shares of the ATOMIC flux, not their raw rates, so the recombination rate is doubled before the comparison.
    # The golden solution carries the full argument and the natural wrong answer.
    branching = fluorination_rate / (2.0 * recombination_rate)
    # CONVENTION (paper, eq. B.7): the apparent interfacial exponent is the
    # logarithmic slope of the ATOMIC FLUX against the interfacial concentration, and
    # in that variable it closes to (2+B)/(1+B), bounded in [1, 2]. The same slope
    # read off the salt-side INVENTORY is a different number whenever the carriers
    # have different diffusivities, which is exactly why the paper gives them
    # different ones.
    return np.array([branching, (2.0 + branching) / (1.0 + branching)])

import numpy as np


def isotopologue_fluxes(k_homonuclear_forward: float, k_homonuclear_reverse: float, k_fluorination_forward: float, k_fluorination_reverse: float, c_metal_h: float, c_metal_t: float, c_salt_hh: float, c_salt_ht: float, c_salt_tt: float, c_salt_tf: float, fluoride_activity: float) -> "np.ndarray":
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

import numpy as np


def governing_damkohler(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float) -> "np.ndarray":
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

import numpy as np


def reference_branching_ratio(length_s: float, diff_molecular: float, diff_fluoride: float, recomb_forward: float, recomb_reverse: float, fluor_forward: float, fluor_reverse: float, fluoride_activity: float, reference_concentration: float) -> float:
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

import numpy as np


def interface_regime_report(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, solubility_a: float, solubility_b: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float, sieverts_constant: float, henry_constant: float, recomb_forward: float, fluor_forward: float, fluoride_activity: float) -> "np.ndarray":
    if exchange_velocity <= 0 or recomb_forward <= 0 or fluor_forward <= 0:
        raise ValueError("rate constants must be positive")
    partition = first_order_rate_ratio(solubility_a, solubility_b)
    res = slab_resistances(length_a, diffusivity_a, length_b,
                                   diffusivity_b, partition)
    flux = linear_channel_flux(length_a, diffusivity_a, length_b,
                                       diffusivity_b, partition,
                                       exchange_velocity, c_outer_a, c_outer_b)
    traces = interfacial_traces(length_a, diffusivity_a, length_b,
                                        diffusivity_b, partition,
                                        exchange_velocity, c_outer_a, c_outer_b)
    da_pair = two_sided_damkohler(length_a, diffusivity_a, length_b,
                                          diffusivity_b, partition,
                                          exchange_velocity)
    ratio2 = recombination_rate_ratio(sieverts_constant, henry_constant)
    rec = recombination_interface(length_a, diffusivity_a, length_b,
                                          diffusivity_b, recomb_forward,
                                          sieverts_constant, henry_constant,
                                          c_outer_a, c_outer_b)
    da_quad = quadratic_channel_damkohler(length_a, diffusivity_a,
                                                  recomb_forward, c_outer_a)
    comp = competing_channel_traces(
        length_a, diffusivity_a, length_b, diffusivity_b, diffusivity_b / 4.0,
        recomb_forward, recomb_forward / ratio2, fluor_forward,
        fluor_forward / 2.0, fluoride_activity, c_outer_a)
    be = branching_ratio_and_exponent(comp[1], comp[2])
    gov = governing_damkohler(length_a, diffusivity_a, length_b,
                                      diffusivity_b, partition, exchange_velocity)
    ref_b = reference_branching_ratio(
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
    iso = isotopologue_fluxes(recomb_forward, krm, fluor_forward, kfm,
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
    da_rec = two_sided_damkohler(length_a, diffusivity_a, length_b, diffusivity_b,
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
SCICODE_GOLD_EOF
