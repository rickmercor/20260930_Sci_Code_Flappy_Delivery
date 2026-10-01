"""
Compose accelerated-frame topology, individual binding, subgroup rebound and bulk rejection into one bound mass fraction. All particle arrays are finite one-dimensional sequences of equal length with at least three entries. The supplied adjacency is a connected simple undirected graph with sorted unique native or NumPy integral IDs, seed IDs are nonempty distinct valid integral IDs and the subgroup size threshold is a positive integral scalar. Build the host boosted potential, descend from the seeds, locate the completed well and true saddle, apply strict individual binding, then process recorded candidate branches in discovery order by local rebinding and all-or-nothing host bulk tests. With unit particle masses, return the number of final mask entries equal to one divided by the completed well size before energetic unbinding. Invalid input or an upstream potential or energy outside the finite native-float range raises ValueError. Representable upstream results remain valid even when a naive intermediate power would overflow.

The denominator is fixed by the graph topology and includes the saddle, while the numerator reflects two energetic levels. Particles first compete individually against the host saddle in the completed well's translated frame. A recorded false-saddle branch is then re-analysed even when some members failed that host pass, and any locally self-bound survivors compete as one bulk object against the host. Discovery order is retained because each bulk decision reads the mask produced by earlier decisions when forming the host velocity reference.

Returns
-------
native Python float, the final bound unit-mass fraction of the completed well
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bound_mass_fraction(position, canonical_velocity, acceleration,
                        raw_potential, adjacency, seed_ids, scale_factor,
                        hubble_rate, omega_m, omega_lambda,
                        turnaround_overdensity, subgroup_min_size):
    """Return the final unit-mass bound fraction of the completed host well.

    Parameters
    ----------
    position : array_like
        Finite one-dimensional positions.
    canonical_velocity : array_like
        Finite one-dimensional canonical velocities.
    acceleration : array_like
        Finite one-dimensional accelerations.
    raw_potential : array_like
        Finite one-dimensional raw potentials.
    adjacency : sequence of sequence of int
        Sorted unique neighbors of a connected simple undirected graph.
    seed_ids : array_like of int
        Nonempty distinct valid native or NumPy integer seed IDs.
    scale_factor : float
        Positive finite scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.
    omega_m : float
        Nonnegative finite matter density parameter.
    omega_lambda : float
        Nonnegative finite dark-energy density parameter.
    turnaround_overdensity : float
        Nonnegative finite turn-around overdensity.
    subgroup_min_size : int
        Positive native or NumPy integral candidate-recording threshold.

    Returns
    -------
    float
        Final unit-mass bound count divided by the pre-unbinding well size.

    Raises
    ------
    ValueError
        If an upstream input or derived-value representability contract fails,
        or no true saddle is found. Upstream ValueError propagates unchanged.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_bound_mass_fraction(position, canonical_velocity, acceleration,
                              raw_potential, adjacency, seed_ids, scale_factor,
                              hubble_rate, omega_m, omega_lambda,
                              turnaround_overdensity, subgroup_min_size):
    boosted = _oracle_boosted_turnaround_potential(
        position, acceleration, raw_potential, seed_ids, scale_factor,
        hubble_rate, omega_m, turnaround_overdensity)
    minimum = _oracle_descend_to_minimum(boosted, adjacency, seed_ids)
    well, saddle, branches = _oracle_locate_host_well(
        boosted, adjacency, minimum, subgroup_min_size)
    mask, escape, physical_potential, _ = _oracle_host_energy_mask(
        position, canonical_velocity, boosted, well, saddle, minimum,
        scale_factor, hubble_rate, omega_m, omega_lambda,
        turnaround_overdensity)
    for branch in branches:
        rebound_ids, surface_ids = _oracle_rebound_subgroup(
            position, canonical_velocity, acceleration, raw_potential, adjacency,
            branch, scale_factor, hubble_rate, omega_m, omega_lambda,
            turnaround_overdensity, subgroup_min_size)
        if rebound_ids:
            mask = _oracle_bulk_binding_mask(
                position, canonical_velocity, physical_potential, mask,
                rebound_ids, surface_ids, escape, minimum, scale_factor,
                hubble_rate)
    return float(sum(mask) / len(well))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    canonical = "position = [10.0, 9.7, 10.4, 9.4, 8.5, 10.8, 11.0, 11.1, 10.9, 11.5, 12.5, 12.7, 13.0, 9.1, 8.9]\nvelocity = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 2.7, 2.7, 2.7, 0.0, -2.0, 0.0, 0.0, -2.0, 0.0]\nacceleration = [-0.3, -0.5, -0.4, -0.2, -0.6, 0.0, -0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\nraw = [-20.0, -19.417625, -18.758, -18.3305, -14.740625, -15.752, -17.7625, -17.183625, -17.618625, -10.840625, -7.890625, -8.627625, -18.5625, -14.438625, -21.063625]\nadjacency = [[1, 2], [0, 3], [0, 5, 10], [1, 4], [3, 13], [2, 6], [5, 7, 8], [6], [6, 9], [8], [2, 11], [10, 12], [11], [4, 14], [13]]\nseeds = [0, 1, 2, 3, 4]\n"
    anchor = "position = [0.0, 1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 4.0]\nvelocity = [0.0, 0.0, 0.0, 0.0, 4.25, 4.25, 4.25, 0.0]\nacceleration = [0.0] * 8\nraw = [0.0, 1.0, 5.0, 6.0, 4.0, 2.0, 3.0, -4.0]\nadjacency = [[1], [0, 2, 4], [1, 3], [2, 7], [1, 5], [4, 6], [5], [3]]\nseeds = np.array([0, 1], dtype=np.int64)\n"
    return [
        {
            "setup": "position = [0.0, 1e-100, 2e-100, 3e-100]\nvelocity = [0.0] * 4\nacceleration = [0.0] * 4\nraw = [0.0, 2e-300, 8e-300, 5e-300]\nadjacency = [[1], [0, 2], [1, 3], [2]]\n",
            "call": "round(bound_mass_fraction(position, velocity, acceleration, raw, adjacency, [0], 1e-200, 1e150, 1.0, 0.0, 4.0, 2), 12)",
            "gold_call": "0.666666666667",
        },
        {
            "setup": "position = [0.0] * 4\nvelocity = [0.0] * 4\nacceleration = [0.0] * 4\nraw = [0.0, 1.0, 2.0, -1.0]\nadjacency = [[1], [0, 2], [1, 3], [2]]\n",
            "call": "round(bound_mass_fraction(position, velocity, acceleration, raw, adjacency, [0], 1.0, 1e200, 0.0, 0.0, 4.55, 2), 12)",
            "gold_call": "0.666666666667",
        },
        {
            "setup": canonical + "def overflow_status(function):\n    try:\n        function(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 1e200, 1.0, 0.0, 4.55, 3)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "overflow_status(bound_mass_fraction)",
            "gold_call": "overflow_status(_oracle_bound_mass_fraction)",
        },
        {
            "setup": anchor + "velocity[4] = 1e200\ndef overflow_status(function):\n    try:\n        function(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 3)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "overflow_status(bound_mass_fraction)",
            "gold_call": "overflow_status(_oracle_bound_mass_fraction)",
        },
        {
            "setup": canonical,
            "call": "round(bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 1.0, 1.0, 0.0, 4.55, np.int64(3)), 12)",
            "gold_call": "round(_oracle_bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 1.0, 1.0, 0.0, 4.55, np.int64(3)), 12)",
        },
        {
            "setup": anchor,
            "call": "round(bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 3), 12)",
            "gold_call": "round(_oracle_bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 3), 12)",
        },
        {
            "setup": anchor + "velocity = [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]\n",
            "call": "round(bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 3), 12)",
            "gold_call": "round(_oracle_bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 3), 12)",
        },
        {
            "setup": anchor,
            "call": "round(bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 4), 12)",
            "gold_call": "round(_oracle_bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 4), 12)",
        },
        {
            "setup": anchor,
            "call": "float(bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 3) - bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 3))",
            "gold_call": "float(_oracle_bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 3) - _oracle_bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 1.0, 0.0, 1.0, 0.0, 0.0, 3))",
        },
        {
            "setup": "def invalid_status(function):\n    try:\n        function([0, 1, 2], [0, 0, 0], [0, 0, 0], [0, 1, 2], [[1], [0, 2], [1]], [0], 1.0, 0.0, 1.0, 0.0, 0.0, 1)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "invalid_status(bound_mass_fraction)",
            "gold_call": "invalid_status(_oracle_bound_mass_fraction)",
        },
        {
            "setup": anchor + "def run_model():\n    try:\n        bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 0.0, 0.0, 1.0, 0.0, 0.0, 3)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_bound_mass_fraction(position, velocity, acceleration, raw, adjacency, seeds, 0.0, 0.0, 1.0, 0.0, 0.0, 3)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": anchor + "def run_model():\n    try:\n        bound_mass_fraction(position, velocity, acceleration, raw, adjacency, [], 1.0, 0.0, 1.0, 0.0, 0.0, 3)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_bound_mass_fraction(position, velocity, acceleration, raw, adjacency, [], 1.0, 0.0, 1.0, 0.0, 0.0, 3)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
