"""
Run the whole analysis end to end and return the requested scalar. Thread the record through every earlier step in order: build the visitation counts and acceptance weights, reduce them to per-ensemble totals, unbias and rescale them, label each trajectory with its highest ensemble, collect the restricted totals at each interface, run the forward recursion for the crossing probabilities, form the normalisers, collapse the bookkeeping into one weight per trajectory on each side, obtain the fraction of reactant-side trajectories terminating at the inner interface, place the interior phase points of every trajectory along the coordinate, accumulate them into the conditional density, and convert that density to a free energy on a scale whose minimum is zero. Return the free energy of the highest bin.

The conditional free energy is a phase-space average restricted to trajectories that most recently visited the reactant state, so it inherits the kinetics of how that state is left and re-entered. Reporting it on a scale whose minimum is zero requires dividing the density by its largest bin, since the free energy is minus the logarithm of the density: dividing by the smallest bin instead would place the maximum at zero and invert the entire profile. The value at the far end of the coordinate is a kinetic barrier and, unlike an ordinary free energy, it moves when the particle masses or the friction coefficient change.

Returns
-------
float, the conditional free energy of the highest histogram bin in units of the thermal energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def conditional_free_energy_barrier(interfaces: np.ndarray, lower_bound: float,
                                    lam_max: np.ndarray, plus_lengths: np.ndarray,
                                    minus_lengths: np.ndarray,
                                    minus_multiplicities: np.ndarray,
                                    ends_at_first_interface: np.ndarray,
                                    n_bins: int) -> float:
    '''Conditional free energy of the highest bin, in units of the thermal energy.

    Parameters
    ----------
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions.
    lower_bound : float
        Outer reactant interface, the lower edge of the histogram.
    lam_max : np.ndarray
        (P,) furthest progress of each barrier-side trajectory.
    plus_lengths : np.ndarray
        (P,) length parameter of each barrier-side trajectory.
    minus_lengths : np.ndarray
        (M,) length parameter of each reactant-side trajectory.
    minus_multiplicities : np.ndarray
        (M,) visitation count of each reactant-side trajectory.
    ends_at_first_interface : np.ndarray
        (M,) boolean flags, true where a reactant-side trajectory ends at the inner interface.
    n_bins : int
        Number of equal-width histogram bins.

    Returns
    -------
    barrier : float
        Conditional free energy of the highest bin.
    
    Raises
    ------
    ValueError
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        lam_max and plus_lengths must have equal length.
        the reactant-side arrays must have equal length.
        n_bins must be a positive integer.
        lower_bound must be finite and below the first interface.
        every barrier-side trajectory must pass the first interface.
        every trajectory must hold at least three phase points.
        every histogram bin must receive weight.
    '''
    return barrier

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_conditional_free_energy_barrier(interfaces: np.ndarray, lower_bound: float,
                                            lam_max: np.ndarray, plus_lengths: np.ndarray,
                                            minus_lengths: np.ndarray,
                                            minus_multiplicities: np.ndarray,
                                            ends_at_first_interface: np.ndarray,
                                            n_bins: int) -> float:
    """Reference implementation. Chains steps 1 to 12 through their oracle functions."""
    interfaces = np.asarray(interfaces, dtype=float)
    lam_max = np.asarray(lam_max, dtype=float)
    plus_lengths = np.asarray(plus_lengths)
    minus_lengths = np.asarray(minus_lengths)
    minus_multiplicities = np.asarray(minus_multiplicities, dtype=float)
    flags = np.asarray(ends_at_first_interface)
    if interfaces.ndim != 1 or interfaces.size < 3:
        raise ValueError("interfaces must be a 1D array with at least three entries")
    if np.any(np.diff(interfaces) <= 0.0):
        raise ValueError("interfaces must be strictly increasing")
    if lam_max.ndim != 1 or lam_max.size != plus_lengths.size:
        raise ValueError("lam_max and plus_lengths must have equal length")
    if minus_lengths.size != minus_multiplicities.size or minus_lengths.size != flags.size:
        raise ValueError("the reactant-side arrays must have equal length")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 1:
        raise ValueError("n_bins must be a positive integer")
    if not np.isfinite(float(lower_bound)) or float(lower_bound) >= float(interfaces[0]):
        raise ValueError("lower_bound must be finite and below the first interface")
    if np.any(lam_max <= interfaces[0]):
        raise ValueError("every barrier-side trajectory must pass the first interface")
    if np.any(plus_lengths < 3) or np.any(minus_lengths < 3):
        raise ValueError("every trajectory must hold at least three phase points")

    n = interfaces.size - 1
    record = _oracle_sampling_record(lam_max, interfaces)
    mu, w = record[0], record[1]
    eta = _oracle_ensemble_path_totals(mu)
    t = _oracle_unbiased_sampling_weights(mu, w)
    index = _oracle_highest_ensemble_index(lam_max, interfaces)
    crossing_totals = np.vstack([
        _oracle_interface_crossing_totals(t, lam_max, interfaces, level)
        for level in range(1, n + 1)])
    probs = _oracle_crossing_probabilities(crossing_totals, eta)
    norm = _oracle_wham_normalisers(eta, probs)
    plus_weights = _oracle_plus_path_weights(t, index, norm)
    minus_weights = _oracle_minus_path_weights(minus_multiplicities)
    fraction = _oracle_end_interface_fraction(minus_weights, flags)

    plus_values, plus_point_weights = [], []
    for j in range(lam_max.size):
        slices = _oracle_path_slice_values(float(lam_max[j]), int(plus_lengths[j]), interfaces)
        plus_values.append(slices)
        plus_point_weights.append(np.full(slices.size, float(plus_weights[j])))
    minus_values, minus_point_weights = [], []
    for j in range(minus_lengths.size):
        slices = _oracle_path_slice_values(float(interfaces[0]), int(minus_lengths[j]), interfaces,
                                           lower_turning_point=float(lower_bound))
        minus_values.append(slices)
        minus_point_weights.append(np.full(slices.size, float(minus_weights[j])))

    density = _oracle_conditional_density_histogram(
        np.concatenate(plus_values), np.concatenate(plus_point_weights),
        np.concatenate(minus_values), np.concatenate(minus_point_weights),
        fraction, float(lower_bound), float(interfaces[-1]), int(n_bins))

    if np.any(density <= 0.0):
        raise ValueError("every histogram bin must receive weight")
    profile = -np.log(density / density.max())
    return float(profile[-1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlower_bound = -1.6\nlam_max = np.array([-0.95, -0.72, -0.40, -0.18, 0.05, 0.20,\n                    0.36, 0.52, 0.70, 0.84, 0.97, 1.12])\nplus_lengths = np.array([11, 7, 10, 6, 9, 12, 8, 11, 7, 10, 6, 9])\nminus_lengths = np.array([8, 5, 7, 9, 6, 8])\nminus_multiplicities = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])\nends_at_first_interface = np.array([True, True, False, True, False, True])\n",
            "call": "conditional_free_energy_barrier(interfaces, lower_bound, lam_max, plus_lengths, minus_lengths, minus_multiplicities, ends_at_first_interface, 8)",
            "gold_call": "_oracle_conditional_free_energy_barrier(interfaces, lower_bound, lam_max, plus_lengths, minus_lengths, minus_multiplicities, ends_at_first_interface, 8)"
        },
        {
            "setup": "import numpy as np\n# boundary: a bounded coordinate, every reactant-side trajectory ending at the inner interface\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlower_bound = -1.6\nlam_max = np.array([-0.95, -0.72, -0.40, -0.18, 0.05, 0.20,\n                    0.36, 0.52, 0.70, 0.84, 0.97, 1.12])\nplus_lengths = np.array([11, 7, 10, 6, 9, 12, 8, 11, 7, 10, 6, 9])\nminus_lengths = np.array([8, 5, 7, 9, 6, 8])\nminus_multiplicities = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])\nends_at_first_interface = np.ones(6, dtype=bool)\n",
            "call": "conditional_free_energy_barrier(interfaces, lower_bound, lam_max, plus_lengths, minus_lengths, minus_multiplicities, ends_at_first_interface, 8)",
            "gold_call": "_oracle_conditional_free_energy_barrier(interfaces, lower_bound, lam_max, plus_lengths, minus_lengths, minus_multiplicities, ends_at_first_interface, 8)"
        },
        {
            "setup": "import numpy as np\n# edge: a coarser histogram over the same record\ninterfaces = np.array([-1.0, -0.4, 0.2, 0.7, 1.0])\nlower_bound = -1.6\nlam_max = np.array([-0.95, -0.72, -0.40, -0.18, 0.05, 0.20,\n                    0.36, 0.52, 0.70, 0.84, 0.97, 1.12])\nplus_lengths = np.array([11, 7, 10, 6, 9, 12, 8, 11, 7, 10, 6, 9])\nminus_lengths = np.array([8, 5, 7, 9, 6, 8])\nminus_multiplicities = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])\nends_at_first_interface = np.array([True, True, False, True, False, True])\n",
            "call": "conditional_free_energy_barrier(interfaces, lower_bound, lam_max, plus_lengths, minus_lengths, minus_multiplicities, ends_at_first_interface, 4)",
            "gold_call": "_oracle_conditional_free_energy_barrier(interfaces, lower_bound, lam_max, plus_lengths, minus_lengths, minus_multiplicities, ends_at_first_interface, 4)"
        },
        {
            "setup": "import numpy as np\n# edge: a three-interface record with a shorter ensemble ladder\ninterfaces = np.array([-1.0, -0.2, 0.4, 1.0])\nlower_bound = -1.5\nlam_max = np.array([-0.5, -0.2, 0.1, 0.4, 0.7, 1.3])\nplus_lengths = np.array([7, 9, 11, 6, 8, 10])\nminus_lengths = np.array([5, 7, 9])\nminus_multiplicities = np.array([2.0, 1.0, 3.0])\nends_at_first_interface = np.array([True, False, True])\n",
            "call": "conditional_free_energy_barrier(interfaces, lower_bound, lam_max, plus_lengths, minus_lengths, minus_multiplicities, ends_at_first_interface, 5)",
            "gold_call": "_oracle_conditional_free_energy_barrier(interfaces, lower_bound, lam_max, plus_lengths, minus_lengths, minus_multiplicities, ends_at_first_interface, 5)"
        },
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 1.0])\nlam_max = np.array([-1.0, 0.5])          # a trajectory sitting on the reactant boundary\nplus_lengths = np.array([7, 9])\nminus_lengths = np.array([5, 7])\nminus_multiplicities = np.array([1.0, 2.0])\nends = np.array([True, False])\ndef run(f):\n    try:\n        f(interfaces, -1.5, lam_max, plus_lengths, minus_lengths,\n          minus_multiplicities, ends, 4); return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(conditional_free_energy_barrier)",
            "gold_call": "run(_oracle_conditional_free_energy_barrier)"
        },
        {
            "setup": "import numpy as np\ninterfaces = np.array([-1.0, -0.4, 0.2, 1.0])\nlam_max = np.array([0.3, 0.5])\nplus_lengths = np.array([7, 9])\nminus_lengths = np.array([5, 7])\nminus_multiplicities = np.array([1.0, 2.0])\nends = np.array([True, False])\ndef run(f):\n    try:\n        f(interfaces, -0.5, lam_max, plus_lengths, minus_lengths,\n          minus_multiplicities, ends, 4); return 0   # lower_bound above the first interface\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(conditional_free_energy_barrier)",
            "gold_call": "run(_oracle_conditional_free_energy_barrier)"
        }
    ]
