"""
Probability that the planned two-site visit records no copies anywhere.

A monitoring plan fixes when to go, where to go and how much water to take, and

the quantity that decides whether the plan is worth running is the chance that

the whole visit comes back negative although the species is there. Putting the

pieces together means opening the season as early as the standard of evidence

allows, sizing the effort at each site from what that day is expected to yield,

and then evaluating the plan as a whole rather than one site at a time.

Returns
-------
float, the probability that the visit records no copies at either window, between 0 and 1, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def survey_failure_probability(reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                               window_length: float, station_start: float,
                               station_replicates: int, upstream_start: float,
                               capture_volume: float, failure_target: float,
                               copy_target: float, first_day: int, last_day: int) -> float:
    '''Probability that the planned two-site visit records no copies anywhere.

    The plan is built in three moves, all on the model of steps 1 to 8. The visit
    is made on the first whole day between first_day and last_day on which a
    survey of station_replicates replicates at the station window alone meets the
    failure standard failure_target. At the upstream window the number of
    replicates is then set by the planning rule of step 8 for that day, with the
    same capture_volume and the given copy_target. Return the probability that the
    visit, both windows together, records no copies at all.

    Parameters
    ----------
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.
    window_length : float
        Length of every sampled window in km, finite and positive.
    station_start : float
        Downstream edge of the station window in km, as in step 5.
    station_replicates : int
        Number of replicates taken at the station, an integer of at least 1.
    upstream_start : float
        Downstream edge of the upstream window in km, as in step 5; the two
        windows do not overlap.
    capture_volume : float
        Effective assayed volume of one replicate in ml, finite and positive.
    failure_target : float
        The largest acceptable probability of recording no copies at the station
        window alone, finite, greater than 0 and less than 1.
    copy_target : float
        Expected copy yield the upstream window must reach, finite and positive.
    first_day : int
        First whole day of the search, an integer of at least 1.
    last_day : int
        Last whole day of the search, an integer of at least first_day and no
        greater than the season length.

    Returns
    -------
    probability : float
        The probability that the visit records no copies at either window,
        between 0 and 1, as a native Python float.

    Raises
    ------
    ValueError
        If station_replicates is not an integer of at least 1, or if any contract
        of steps 1 to 8 is broken.
    '''
    return probability  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_survey_failure_probability(reach: "np.ndarray", run: "np.ndarray",
                                       shedding_rate: float, window_length: float,
                                       station_start: float, station_replicates: int,
                                       upstream_start: float, capture_volume: float,
                                       failure_target: float, copy_target: float,
                                       first_day: int, last_day: int) -> float:
    if isinstance(station_replicates, bool) or not isinstance(station_replicates, (int, np.integer)) \
            or station_replicates < 1:
        raise ValueError("station_replicates must be an integer of at least 1")
    volume = float(capture_volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("capture_volume must be finite and positive")
    station_volume = float(station_replicates) * volume
    window = _oracle_survey_window_days(station_start, station_volume, window_length,
                                        reach, run, shedding_rate, failure_target,
                                        first_day, last_day)
    opening = float(window[0])
    upstream_replicates = _oracle_planned_replicates(opening, upstream_start, window_length,
                                                     reach, run, shedding_rate, volume,
                                                     copy_target)
    starts = np.array([float(station_start), float(upstream_start)])
    volumes = np.array([station_volume, float(upstream_replicates) * volume])
    return _oracle_nondetection_probability(opening, starts, volumes, window_length,
                                            reach, run, shedding_rate)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = (
        "import numpy as np\n"
        "reach_a = np.array([20.0, 20.0, 2.0, 1.5, 1.2])\n"
        "run_a = np.array([80.0, 1200.0, 6.0, 8.0, 1.8, 30.0])\n"
        "reach_b = np.array([6.0, 45.0, 0.6, 9.0, 0.4])\n"
        "run_b = np.array([45.0, 260.0, 2.5, 2.5, 0.8, 4.0])\n"
        "reach_c = np.array([120.0, 8.0, 5.0, 0.25, 3.0])\n"
        "run_d = np.array([90.0, 9.0e4, 4.0, 11.0, 3.2, 95.0])\n"
    )
    return [
        # --- normal: the shipped plan over the whole season ---
        {
            "setup": setup,
            "call": ("survey_failure_probability(reach_a, run_a, 0.012, 1.0, 0.0, 3, 2.0, "
                     "1.3, 0.05, 6.0, 1, 79)"),
            "gold_call": ("_oracle_survey_failure_probability(reach_a, run_a, 0.012, 1.0, 0.0, "
                          "3, 2.0, 1.3, 0.05, 6.0, 1, 79)"),
            "tol": 1e-7,
        },
        # --- boundary: a short fast reach with a single replicate at each of two touching windows ---
        {
            "setup": setup,
            "call": ("survey_failure_probability(reach_b, run_b, 3.5, 2.0, 0.0, 1, 2.0, "
                     "0.6, 0.3, 4.0, 6, 40)"),
            "gold_call": ("_oracle_survey_failure_probability(reach_b, run_b, 3.5, 2.0, 0.0, "
                          "1, 2.0, 0.6, 0.3, 4.0, 6, 40)"),
            "tol": 1e-7,
        },
        # --- edge: a long slow reach with a small assayed volume and a distant upstream window ---
        {
            "setup": setup,
            "call": ("survey_failure_probability(reach_c, run_d, 0.0004, 4.0, 0.0, 2, 88.0, "
                     "0.05, 0.035, 900.0, 25, 45)"),
            "gold_call": ("_oracle_survey_failure_probability(reach_c, run_d, 0.0004, 4.0, 0.0, "
                          "2, 88.0, 0.05, 0.035, 900.0, 25, 45)"),
            "tol": 1e-7,
        },
    ]
