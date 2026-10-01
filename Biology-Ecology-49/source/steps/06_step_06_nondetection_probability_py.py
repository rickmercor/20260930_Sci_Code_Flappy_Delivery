"""
Probability that the survey records no copies at any of its sites.

Because the source of the eDNA is itself random, a survey's chance of failure is

not the failure probability of any one run but an average over the runs that the

season could have produced. Runs with few migrants on the days that matter

inflate that average far beyond what a calculation on the expected run would

suggest, so the averaging has to be carried out over the law of the count rather

than at its mean.

Returns
-------
float, the probability, between 0 and 1, converged in the time step to better than 1e-8 relative, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nondetection_probability(day: float, site_starts: "np.ndarray",
                             assayed_volumes: "np.ndarray", window_length: float,
                             reach: "np.ndarray", run: "np.ndarray",
                             shedding_rate: float) -> float:
    '''Probability that the survey records no copies at any of its sites.

    The reach, the field, the assay and the survey layout are those of steps 2, 4
    and 5, and the unit-time migrant count is the process of step 1, with no eDNA
    in the reach on day 0 and no count outside the season. Return the
    unconditional probability of failure for a survey run on the given day.

    Parameters
    ----------
    day : float
        Survey day, a finite value with 0 < day <= season length.
    site_starts : np.ndarray
        Shape (k,), downstream edges of the windows in km, as in step 5.
    assayed_volumes : np.ndarray
        Shape (k,), effective assayed volume in ml at each site, as in step 5.
    window_length : float
        Length of every sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.

    Returns
    -------
    probability : float
        The probability, between 0 and 1, converged in the time step to better
        than 1e-8 relative, as a native Python float.

    Raises
    ------
    ValueError
        If day is not finite and inside the season, if the step 5 contract on
        site_starts, assayed_volumes, window_length, reach or shedding_rate is
        broken, or if run fails the step 1 contract.
    '''
    return probability  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _simpson_weights(n_intervals, step):
    w = np.ones(n_intervals + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * step / 3.0


def _oracle_nondetection_probability(day: float, site_starts: "np.ndarray",
                                     assayed_volumes: "np.ndarray", window_length: float,
                                     reach: "np.ndarray", run: "np.ndarray",
                                     shedding_rate: float) -> float:
    season, peak, up, down, cv, pull = _check_run(run)
    d = float(day)
    if not np.isfinite(d) or d <= 0.0 or d > season:
        raise ValueError("day must be finite and inside the season")
    steps_per_day = 160
    t_end = min(season, d)
    n_full = 2 * int(np.ceil(steps_per_day * t_end / 2.0))
    step = t_end / n_full
    grid = np.arange(2 * n_full + 1) * (0.5 * step)
    kernel = _oracle_sample_weight_kernel(d - grid, site_starts, assayed_volumes,
                                          window_length, reach, shedding_rate)
    coef = np.zeros((2, grid.size))
    inner = (grid > 0.0) & (grid < season)
    coef[:, inner] = _oracle_migration_rate_coefficients(grid[inner], run)
    drift, diffusion = coef[0], coef[1]
    pull_rate = pull / np.maximum(season - grid, 1e-300)

    def _slope(index, value):
        return pull_rate[index] * value - 0.5 * diffusion[index] * value * value - kernel[index]

    beta = np.zeros(2 * n_full + 1)
    current = 0.0
    for k in range(2 * n_full, 0, -2):
        s1 = _slope(k, current)
        s2 = _slope(k - 1, current - 0.5 * step * s1)
        s3 = _slope(k - 1, current - 0.5 * step * s2)
        s4 = _slope(k - 2, current - step * s3)
        current = current - (step / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
        beta[k - 2] = current
    quad = _simpson_weights(n_full, step)
    return float(np.exp(np.sum(quad * beta[0::2] * drift[0::2])))

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
        # --- normal: the shipped survey on the day both of its sites are sampled ---
        {
            "setup": setup,
            "call": ("nondetection_probability(26.0, np.array([0.0, 2.0]), "
                     "np.array([3.9, 2.6]), 1.0, reach_a, run_a, 0.012)"),
            "gold_call": ("_oracle_nondetection_probability(26.0, np.array([0.0, 2.0]), "
                          "np.array([3.9, 2.6]), 1.0, reach_a, run_a, 0.012)"),
            "tol": 1e-7,
        },
        # --- boundary: a single window filling a short fast reach, on the last day of its season ---
        {
            "setup": setup,
            "call": ("nondetection_probability(45.0, np.array([0.0]), np.array([0.6]), "
                     "6.0, reach_b, run_b, 3.5)"),
            "gold_call": ("_oracle_nondetection_probability(45.0, np.array([0.0]), "
                          "np.array([0.6]), 6.0, reach_b, run_b, 3.5)"),
            "tol": 1e-7,
        },
        # --- edge: three windows and small assayed volumes on a long slow reach, before its peak ---
        {
            "setup": setup,
            "call": ("nondetection_probability(110.0, np.array([0.0, 30.0, 88.0]), "
                     "np.array([2.0, 1.0, 0.5]), 4.0, reach_c, run_c, 0.0004)"),
            "gold_call": ("_oracle_nondetection_probability(110.0, np.array([0.0, 30.0, 88.0]), "
                          "np.array([2.0, 1.0, 0.5]), 4.0, reach_c, run_c, 0.0004)"),
            "tol": 1e-7,
        },
    ]
