"""
Oldest migration age, in days, that can leave eDNA in the sampled water.

Because migrants move upstream while the eDNA they shed moves downstream, a

sample taken at one place integrates the run over a bounded stretch of past

time rather than over the whole season. The bound is set by the geometry of the

reach and by the two speeds, and it tells a survey designer how much migration

history a single sampling day actually reports on.

Returns
-------
float, the age in days, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sample_memory_horizon(site_start: float, reach: "np.ndarray") -> float:
    '''Oldest migration age, in days, that can leave eDNA in the sampled water.

    Geometry and conventions are those of step 2: x runs upstream from 0 to the
    reach length, water moves towards x = 0 at the flow speed, migrants pass
    x = 0 and move upstream at the ground speed, and eDNA enters the reach only
    where a migrant is. Return the largest age s such that migrants passing x = 0
    exactly s days before the survey can still leave eDNA in the water occupying
    the sampled window at the survey instant; migration older than that cannot.
    Take the run to have been under way long enough that its own start does not
    limit the answer; the answer does not depend on the length of the window.

    Parameters
    ----------
    site_start : float
        Downstream edge of the sampled window in km, finite, at least 0 and less
        than the reach length.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.

    Returns
    -------
    horizon : float
        The age in days, as a native Python float.

    Raises
    ------
    ValueError
        If site_start is not finite, is negative or is not less than the reach
        length, or if reach fails the step 2 contract.
    '''
    return horizon  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sample_memory_horizon(site_start: float, reach: "np.ndarray") -> float:
    length, flow, ground, decay, noise = _check_reach(reach)
    x0 = float(site_start)
    if not np.isfinite(x0) or x0 < 0.0 or x0 >= length:
        raise ValueError("site_start must be finite, at least 0 and less than the reach length")
    return float(length / flow + length / ground - x0 / flow)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = (
        "import numpy as np\n"
        "reach_a = np.array([20.0, 20.0, 2.0, 1.5, 1.2])\n"
        "reach_b = np.array([6.0, 45.0, 0.6, 9.0, 0.4])\n"
        "reach_c = np.array([120.0, 8.0, 5.0, 0.25, 3.0])\n"
    )
    return [
        # --- normal: the shipped reach, the station window at the downstream end ---
        {
            "setup": setup,
            "call": "sample_memory_horizon(0.0, reach_a)",
            "gold_call": "_oracle_sample_memory_horizon(0.0, reach_a)",
        },
        # --- boundary: a short fast reach with slow migrants, sampled at its upstream end ---
        {
            "setup": setup,
            "call": "sample_memory_horizon(5.5, reach_b)",
            "gold_call": "_oracle_sample_memory_horizon(5.5, reach_b)",
        },
        # --- edge: a long slow reach where migrants and water move at comparable speeds ---
        {
            "setup": setup,
            "call": "sample_memory_horizon(34.0, reach_c)",
            "gold_call": "_oracle_sample_memory_horizon(34.0, reach_c)",
        },
    ]
