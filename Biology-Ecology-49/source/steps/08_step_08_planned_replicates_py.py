"""
Replicates planned at a site, from the expected copy yield of one replicate.

Field effort at a site is usually sized from a target yield: enough replicate

samples that the assay is expected to recover at least some agreed number of

amplifiable copies, computed from the concentration the site is expected to

carry on the day of the visit. The rule is deliberately coarse, uses only the

expected concentration, and fixes the effort before any survey is run.

Returns
-------
int, the number of replicates, as a native Python int of at least 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def planned_replicates(day: float, site_start: float, window_length: float,
                       reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                       capture_volume: float, copy_target: float) -> int:
    '''Replicates planned at a site, from the expected copy yield of one replicate.

    One replicate assays a volume of capture_volume ml of the water in the window
    starting at site_start, so its expected copy yield is capture_volume times the
    expected concentration of that water, as in step 2. Return the smallest whole
    number of replicates whose expected yield, the number of replicates times the
    expected yield of one, is at least copy_target.

    Parameters
    ----------
    day : float
        Day of the visit, a finite value with 0 < day <= season length.
    site_start : float
        Downstream edge of the sampled window in km, as in step 2.
    window_length : float
        Length of the sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.
    capture_volume : float
        Effective assayed volume of one replicate in ml, finite and positive.
    copy_target : float
        Expected copy yield the site must reach, finite and positive.

    Returns
    -------
    replicates : int
        The number of replicates, as a native Python int of at least 1.

    Raises
    ------
    ValueError
        If capture_volume or copy_target is not finite and positive, if the
        expected concentration of the sampled water is zero, or if any contract of
        steps 1 or 2 is broken.
    '''
    return replicates  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_planned_replicates(day: float, site_start: float, window_length: float,
                               reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                               capture_volume: float, copy_target: float) -> int:
    volume, target = float(capture_volume), float(copy_target)
    if not np.isfinite(volume) or volume <= 0.0 or not np.isfinite(target) or target <= 0.0:
        raise ValueError("capture_volume and copy_target must be finite and positive")
    concentration = _oracle_mean_sampled_concentration(day, site_start, window_length,
                                                       reach, run, shedding_rate)
    if concentration <= 0.0:
        raise ValueError("the expected concentration of the sampled water is zero")
    return int(math.ceil(target / (volume * concentration)))

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
        "run_c = np.array([210.0, 9.0e4, 11.0, 4.0, 3.2, 95.0])\n"
    )
    return [
        # --- normal: the shipped upstream site on the opening day of the survey window ---
        {
            "setup": setup,
            "call": ("planned_replicates(26.0, 2.0, 1.0, reach_a, run_a, 0.012, 1.3, 6.0)"),
            "gold_call": ("_oracle_planned_replicates(26.0, 2.0, 1.0, reach_a, run_a, 0.012, "
                          "1.3, 6.0)"),
        },
        # --- boundary: an early, thin day on the shipped reach, where the rule asks for many replicates ---
        {
            "setup": setup,
            "call": ("planned_replicates(14.0, 0.0, 1.0, reach_a, run_a, 0.012, 1.3, 6.0)"),
            "gold_call": ("_oracle_planned_replicates(14.0, 0.0, 1.0, reach_a, run_a, 0.012, "
                          "1.3, 6.0)"),
        },
        # --- edge: a rich site on a long slow reach with a small assayed volume and a high target ---
        {
            "setup": setup,
            "call": ("planned_replicates(150.0, 88.0, 4.0, reach_c, run_c, 0.0004, 0.05, 900.0)"),
            "gold_call": ("_oracle_planned_replicates(150.0, 88.0, 4.0, reach_c, run_c, 0.0004, "
                          "0.05, 900.0)"),
        },
    ]
