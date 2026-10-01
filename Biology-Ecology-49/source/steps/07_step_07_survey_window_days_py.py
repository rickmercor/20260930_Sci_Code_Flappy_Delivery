"""
First and last whole day on which a one-site survey meets the failure standard.

Monitoring programmes are written around a standard of evidence: a survey is

worth running only if its chance of missing a species that is present is below

an agreed level. Applied to a migratory run, that standard turns into a calendar

question, because the chance of missing the species falls as the run builds and

rises again as it passes, so only part of the season is worth sampling.

Returns
-------
np.ndarray of shape (2,): the first and the last acceptable whole day, as floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def survey_window_days(site_start: float, assayed_volume: float, window_length: float,
                       reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                       failure_target: float, first_day: int, last_day: int) -> "np.ndarray":
    '''First and last whole day on which a one-site survey meets the failure standard.

    The reach, the field, the assay and the count are those of steps 1 to 6. A
    survey is acceptable on a whole day when its probability of recording no
    copies at the single window starting at site_start is at most
    failure_target. Search the whole days from first_day to last_day inclusive
    and return the first and the last acceptable one.

    Parameters
    ----------
    site_start : float
        Downstream edge of the sampled window in km, as in step 5.
    assayed_volume : float
        Effective assayed volume of the survey's replicates at that window, in ml,
        finite and positive.
    window_length : float
        Length of the sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.
    failure_target : float
        The largest acceptable probability of recording no copies, finite, greater
        than 0 and less than 1.
    first_day : int
        First whole day of the search, an integer of at least 1.
    last_day : int
        Last whole day of the search, an integer of at least first_day and no
        greater than the season length.

    Returns
    -------
    window : np.ndarray
        Shape (2,) of floats, the first and the last acceptable whole day.

    Raises
    ------
    ValueError
        If failure_target is not finite, greater than 0 and less than 1, if
        first_day or last_day is not an integer with 1 <= first_day <= last_day
        and last_day no greater than the season length, if no day in the range is
        acceptable, or if any contract of steps 1, 2, 5 or 6 is broken.
    '''
    return window  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_survey_window_days(site_start: float, assayed_volume: float, window_length: float,
                               reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                               failure_target: float, first_day: int, last_day: int) -> "np.ndarray":
    season = _check_run(run)[0]
    target = float(failure_target)
    if not np.isfinite(target) or target <= 0.0 or target >= 1.0:
        raise ValueError("failure_target must be finite, greater than 0 and less than 1")
    for value in (first_day, last_day):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("first_day and last_day must be integers")
    if first_day < 1 or last_day < first_day or last_day > season:
        raise ValueError("the search range must satisfy 1 <= first_day <= last_day <= season")
    starts = np.array([float(site_start)])
    volumes = np.array([float(assayed_volume)])
    good = []
    for day in range(int(first_day), int(last_day) + 1):
        value = _oracle_nondetection_probability(float(day), starts, volumes, window_length,
                                                 reach, run, shedding_rate)
        if value <= target:
            good.append(float(day))
    if not good:
        raise ValueError("no whole day in the search range meets the failure target")
    return np.array([good[0], good[-1]])

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
        # --- normal: the shipped survey at the station, searched around the opening of its window ---
        {
            "setup": setup,
            "call": ("survey_window_days(0.0, 3.9, 1.0, reach_a, run_a, 0.012, 0.05, 22, 56)"),
            "gold_call": ("_oracle_survey_window_days(0.0, 3.9, 1.0, reach_a, run_a, 0.012, "
                          "0.05, 22, 56)"),
            "tol": 1e-7,
        },
        # --- boundary: a short fast reach with slow migrants, whose window spans most of its season ---
        {
            "setup": setup,
            "call": ("survey_window_days(0.0, 0.6, 6.0, reach_b, run_b, 3.5, 0.05, 8, 45)"),
            "gold_call": ("_oracle_survey_window_days(0.0, 0.6, 6.0, reach_b, run_b, 3.5, "
                          "0.05, 8, 45)"),
            "tol": 1e-7,
        },
        # --- edge: a strict standard far up a long slow reach, where the delay shifts the window late ---
        {
            "setup": setup,
            "call": ("survey_window_days(88.0, 5.0, 4.0, reach_c, run_d, 0.0004, 0.002, 25, 75)"),
            "gold_call": ("_oracle_survey_window_days(88.0, 5.0, 4.0, reach_c, run_d, 0.0004, "
                          "0.002, 25, 75)"),
            "tol": 1e-7,
        },
    ]
