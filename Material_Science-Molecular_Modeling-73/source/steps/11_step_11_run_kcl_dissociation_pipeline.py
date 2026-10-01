"""
Chain the sub-problem functions 01-10 end to end on the measured KCl dissociation data set and return the dwell time of the interface that limits the transit.

This step runs the whole measurement end to end on the ion-pair data set. It (i) reduces the sampled path-type populations of every ensemble to local crossing probabilities with sub-problem 01, (ii) enumerates the segment types and fixes their order with sub-problem 02, (iii) wires them into the stochastic matrix with sub-problem 03, (iv) solves the hitting-probability system for the global crossing probability with sub-problem 04, (v) converts the measured three-part segment lengths into non-overlapping accumulated times with sub-problem 05, (vi) obtains the expected occupation of every segment type per passage with sub-problem 06, (vii) groups those occupations into the accumulated-time profile along the order parameter with sub-problem 07, (viii) reconstructs the mean duration of a full excursion above the reactant boundary with sub-problem 08, (ix) turns the flux and the crossing probability into the rate with sub-problem 09, and (x) reduces the profile to the reported dwell time with sub-problem 10.

The two halves of the calculation stay deliberately separate. The profile and its largest entry above the reactant boundary come from the occupation decomposition, while the absolute timescale comes from the flux multiplied by the crossing probability, an estimator that never touches the decomposition. Because the profile sums to the mean duration of one passage, and that duration is the reciprocal of the rate, the two must agree; the returned scalar is a share restored to physical units through the independently obtained rate, so a wiring error in either half shows up in the answer.

Returns
-------
float: the dwell time of the rate-limiting interface above the bound-state  boundary, in picoseconds, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_kcl_dissociation_pipeline(time_step: float = 0.02,
                                  well_scale: float = 1.0) -> float:
    """Run the full dwell-time measurement on the ion-pair data set.

    The sixteen interfaces sit every 2 Angstrom from 4 to 34 Angstrom along the
    interionic distance, the first and last delimiting the bound and the
    dissociated state. The sampled path-type populations and the mean leading,
    middle and trailing pieces of every path type are the measured input and are
    embedded in this step.

    Parameters
    ----------
    time_step : float
        Physical duration of one phase point, in picoseconds
        (time_step > 0).
    well_scale : float
        Multiplier applied to the middle piece of every path type of the
        ensemble centred on 8 Angstrom, which deepens or flattens the
        solvent-separated well (well_scale > 0).

    Returns
    -------
    bottleneck_dwell : float
        Mean time one dissociation event accumulates in segments centred on the
        interface above the bound-state boundary that carries the largest such
        time, in picoseconds, as a native Python float.

    Raises
    ------
    ValueError
        If time_step or well_scale is not positive.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_kcl_dissociation_pipeline(time_step: float = 0.02,
                                          well_scale: float = 1.0) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    for name, value in (("time_step", time_step), ("well_scale", well_scale)):
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")

    n_interfaces = 16

    # -- Sampled path-type populations, ordered (arrive below / depart below,
    #    arrive below / depart above, arrive above / depart below,
    #    arrive above / depart above) for the ensembles centred on
    #    lambda_0 ... lambda_14. The straddling ensemble has no fourth type.
    path_counts = np.array([[3820, 844, 1970, 0],
                            [2538, 1845, 1477, 2421],
                            [2446, 1915, 1412, 2365],
                            [2455, 2061, 1370, 2271],
                            [2432, 2180, 1453, 2333],
                            [2226, 2125, 1536, 2354],
                            [2278, 2305, 1449, 2089],
                            [2216, 2370, 1456, 1969],
                            [2021, 2279, 1522, 2171],
                            [2022, 2395, 1597, 2396],
                            [1916, 2375, 1501, 2363],
                            [2037, 2634, 1498, 2466],
                            [1950, 2624, 1294, 2220],
                            [1900, 2650, 1289, 2297],
                            [1832, 2643, 1401, 2585]], dtype=float)

    # -- Mean leading, middle and trailing piece of every path type, in phase
    #    points, in the same ensemble and type order.
    ensemble_parts = np.array([
        [[0.0, 47.3, 0.0], [0.0, 0.0, 26.8], [24.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[18.4, 11.4, 18.0], [18.7, 8.5, 24.5], [25.2, 9.3, 19.2], [25.7, 12.3, 25.4]],
        [[23.1, 108.5, 22.7], [23.4, 80.8, 29.2], [29.9, 88.6, 23.9], [30.4, 117.5, 30.1]],
        [[27.6, 52.0, 27.2], [27.9, 38.7, 33.7], [34.4, 42.5, 28.4], [34.9, 56.3, 34.6]],
        [[32.0, 45.7, 31.6], [32.3, 34.0, 38.1], [38.8, 37.3, 32.8], [39.3, 49.4, 39.0]],
        [[36.4, 40.1, 36.0], [36.7, 29.9, 42.5], [43.2, 32.8, 37.2], [43.7, 43.4, 43.4]],
        [[41.1, 35.2, 40.7], [41.4, 26.2, 47.2], [47.9, 28.8, 41.9], [48.4, 38.1, 48.1]],
        [[45.9, 30.9, 45.5], [46.2, 23.0, 52.0], [52.7, 25.2, 46.7], [53.2, 33.5, 52.9]],
        [[49.9, 27.1, 49.5], [50.2, 20.2, 56.0], [56.7, 22.2, 50.7], [57.2, 29.4, 56.9]],
        [[54.7, 23.8, 54.3], [55.0, 17.7, 60.8], [61.5, 19.5, 55.5], [62.0, 25.8, 61.7]],
        [[59.5, 20.9, 59.1], [59.8, 15.6, 65.6], [66.3, 17.1, 60.3], [66.8, 22.7, 66.5]],
        [[64.1, 18.4, 63.7], [64.4, 13.7, 70.2], [70.9, 15.0, 64.9], [71.4, 19.9, 71.1]],
        [[68.6, 16.1, 68.2], [68.9, 12.0, 74.7], [75.4, 13.2, 69.4], [75.9, 17.5, 75.6]],
        [[72.9, 14.2, 72.5], [73.2, 10.5, 79.0], [79.7, 11.6, 73.7], [80.2, 15.3, 79.9]],
        [[77.5, 12.4, 77.1], [77.8, 9.3, 83.6], [84.3, 10.2, 78.3], [84.8, 13.5, 84.5]],
    ], dtype=float)
    ensemble_parts[2, :, 1] *= float(well_scale)

    # -- The excursions confined to the bound state are all middle piece.
    reactant_parts = np.array([0.0, 43.7, 0.0])

    # -- Sub-problems 01-03: the chain itself.
    probabilities = _oracle_compute_local_crossing_probabilities(path_counts)
    states = _oracle_build_state_space(n_interfaces)
    transition_matrix = _oracle_build_transition_matrix(states, probabilities)

    # -- Sub-problem 04: probability that a departure commits.
    crossing_probability = _oracle_compute_crossing_probability(transition_matrix, states)

    # -- Sub-problems 05-07: the accumulated-time profile along the coordinate.
    overlap_free_times = _oracle_compute_overlap_free_times(
        states, reactant_parts, ensemble_parts)
    visit_counts = _oracle_compute_visit_counts(transition_matrix, states)
    dwell_times = _oracle_compute_interface_dwell_times(
        states, visit_counts, overlap_free_times)

    # -- Sub-problems 08-09: the independent flux route to the timescale.
    conditional_passage_time = _oracle_compute_conditional_passage_time(
        transition_matrix, states, overlap_free_times, float(reactant_parts[1]))
    rate_constant = _oracle_compute_rate_constant(
        float(reactant_parts[1]), conditional_passage_time,
        crossing_probability, float(time_step))

    # -- Sub-problem 10: the reported dwell time.
    return float(_oracle_compute_bottleneck_dwell(dwell_times, rate_constant))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: the measured configuration (final-answer scenario) ---
        {
            "setup": """import numpy as np
time_step = 0.02
""",
            "call": "run_kcl_dissociation_pipeline(time_step)",
            "gold_call": "_oracle_run_kcl_dissociation_pipeline(time_step)",
        },
        # --- Integration: a finer phase-point spacing rescales the answer ---
        {
            "setup": """import numpy as np
time_step = 0.005
""",
            "call": "run_kcl_dissociation_pipeline(time_step)",
            "gold_call": "_oracle_run_kcl_dissociation_pipeline(time_step)",
        },
        # --- Integration (boundary): a flattened solvent-separated well moves
        #     the bottleneck to another interface ---
        {
            "setup": """import numpy as np
well_scale = 0.25
""",
            "call": "run_kcl_dissociation_pipeline(0.02, well_scale)",
            "gold_call": "_oracle_run_kcl_dissociation_pipeline(0.02, well_scale)",
        },
        # --- Integration (edge): a much deeper well concentrates the time ---
        {
            "setup": """import numpy as np
well_scale = 6.0
""",
            "call": "run_kcl_dissociation_pipeline(0.02, well_scale)",
            "gold_call": "_oracle_run_kcl_dissociation_pipeline(0.02, well_scale)",
        },
        # --- Invalid: a non-positive phase-point duration ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_kcl_dissociation_pipeline(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_kcl_dissociation_pipeline(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative well scaling ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_kcl_dissociation_pipeline(0.02, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_kcl_dissociation_pipeline(0.02, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
