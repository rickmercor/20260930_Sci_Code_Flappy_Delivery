"""
Weight each past day of the run carries in the survey's log failure probability.

The eDNA a survey could detect was shed by migrants that passed the bottom of

the reach at many different times, each contributing through a different place

and a different lag. Collecting those contributions into a single weight per

past day turns the spatial problem into a one-dimensional record of the run, and

that record is what any statement about the survey's chance of failure rests on.

Returns
-------
np.ndarray of shape (n,): the weight per past day in copies per ml per day per unit count, negative or zero, in the order of backward_times
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sample_weight_kernel(backward_times: "np.ndarray", site_starts: "np.ndarray",
                         assayed_volumes: "np.ndarray", window_length: float,
                         reach: "np.ndarray", shedding_rate: float) -> "np.ndarray":
    '''Weight each past day of the run carries in the survey's log failure probability.

    Geometry, field and assay are those of steps 2 and 4. The survey takes one
    window of length window_length at each site in site_starts, with the effective
    assayed volume of that site's replicates given by the matching entry of
    assayed_volumes; the windows do not overlap. The survey fails when it records
    no copies anywhere.

    Conditional on the whole path of the unit-time migrant count, the log
    probability of failure is the integral over ages s from 0 to the survey day of
    kernel(s) times the count s days before the survey. Return kernel at each
    entry of backward_times, in the order given.

    Parameters
    ----------
    backward_times : np.ndarray
        Shape (n,), ages in days before the survey, finite and not negative.
    site_starts : np.ndarray
        Shape (k,), downstream edges of the windows in km, finite, not negative,
        and each with site_start + window_length inside the reach.
    assayed_volumes : np.ndarray
        Shape (k,), effective assayed volume in ml at each site, finite and
        positive.
    window_length : float
        Length of every sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.

    Returns
    -------
    kernel : np.ndarray
        Shape (n,) of floats, in copies per ml per day per unit count, in the
        order of backward_times. Every entry is negative or zero.

    Raises
    ------
    ValueError
        If backward_times is not a non-empty one-dimensional array of finite
        nonnegative values, if site_starts and assayed_volumes are not
        one-dimensional arrays of the same non-zero length, if any window falls
        outside the reach, if any assayed volume is not finite and positive, if
        window_length or shedding_rate is not finite and positive, or if reach
        fails the step 2 contract.
    '''
    return kernel  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sample_weight_kernel(backward_times: "np.ndarray", site_starts: "np.ndarray",
                                 assayed_volumes: "np.ndarray", window_length: float,
                                 reach: "np.ndarray", shedding_rate: float) -> "np.ndarray":
    length, flow, ground, decay, noise = _check_reach(reach)
    s = np.asarray(backward_times, dtype=float)
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("backward_times must be a non-empty 1-D array of finite nonnegative values")
    starts = np.asarray(site_starts, dtype=float)
    vols = np.asarray(assayed_volumes, dtype=float)
    if starts.ndim != 1 or vols.ndim != 1 or starts.size == 0 or starts.size != vols.size:
        raise ValueError("site_starts and assayed_volumes must be 1-D arrays of the same non-zero length")
    wd, g = float(window_length), float(shedding_rate)
    if not np.isfinite(wd) or wd <= 0.0 or not np.isfinite(g) or g <= 0.0:
        raise ValueError("window_length and shedding_rate must be finite and positive")
    if not np.all(np.isfinite(starts)) or np.any(starts < 0.0) or np.any(starts + wd > length):
        raise ValueError("every sampled window must lie inside the reach")
    if not np.all(np.isfinite(vols)) or np.any(vols <= 0.0):
        raise ValueError("every assayed volume must be finite and positive")

    nodes, weights = np.polynomial.legendre.leggauss(64)
    slope = 1.0 + flow / ground
    out = np.zeros(s.shape)
    for start, vol in zip(starts, vols):
        horizon = _oracle_sample_memory_horizon(float(start), reach)
        lo_x = (start + flow * s) / slope
        hi_x = np.minimum(np.minimum((start + wd + flow * s) / slope, length), ground * s)
        live = (hi_x > lo_x) & (s <= horizon) & (s > 0.0)
        for j in np.nonzero(live)[0]:
            xi_hi = s[j] - lo_x[j] / ground
            xi_lo = s[j] - hi_x[j] / ground
            mid, half = 0.5 * (xi_lo + xi_hi), 0.5 * (xi_hi - xi_lo)
            xi = mid + half * nodes
            r_vals = _oracle_washout_exponent(xi, float(vol), wd, reach)
            out[j] += g * ground * half * float(np.sum(weights * r_vals))
    return out

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
        "s_a = np.array([0.0, 0.5, 2.0, 5.0, 9.0, 10.8, 11.0, 12.0])\n"
        "s_b = np.array([0.0, 1.0, 4.0, 8.0, 10.0, 10.2])\n"
        "s_c = np.linspace(0.0, 40.0, 11)\n"
    )
    return [
        # --- normal: the shipped survey, both windows, across the whole memory horizon ---
        {
            "setup": setup,
            "call": ("sample_weight_kernel(s_a, np.array([0.0, 2.0]), np.array([3.9, 2.6]), "
                     "1.0, reach_a, 0.012)"),
            "gold_call": ("_oracle_sample_weight_kernel(s_a, np.array([0.0, 2.0]), "
                          "np.array([3.9, 2.6]), 1.0, reach_a, 0.012)"),
        },
        # --- boundary: one window filling a short fast reach, sampled past its horizon ---
        {
            "setup": setup,
            "call": ("sample_weight_kernel(s_b, np.array([0.0]), np.array([0.6]), 6.0, "
                     "reach_b, 3.5)"),
            "gold_call": ("_oracle_sample_weight_kernel(s_b, np.array([0.0]), np.array([0.6]), "
                          "6.0, reach_b, 3.5)"),
        },
        # --- edge: three windows on a long slow reach where migrants outrun the water ---
        {
            "setup": setup,
            "call": ("sample_weight_kernel(s_c, np.array([0.0, 30.0, 88.0]), "
                     "np.array([260.0, 40.0, 5.0]), 4.0, reach_c, 0.0004)"),
            "gold_call": ("_oracle_sample_weight_kernel(s_c, np.array([0.0, 30.0, 88.0]), "
                          "np.array([260.0, 40.0, 5.0]), 4.0, reach_c, 0.0004)"),
        },
    ]
