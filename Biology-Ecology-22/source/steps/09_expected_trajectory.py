"""
Starting from a founding cohort of n0 females whose lengths follow a normal density with mean mean_len0 and standard deviation sd_len0 (metres), project the expected female length distribution forward through n_years annual censuses with the source's expected projection equation on the midpoint mesh of n_bins equal bins on [z_min, z_max] (step 04): the vital rates of steps 01-03 with their default coefficients, the growth matrix of step 05, the recruit length density of step 06 with mean recruit_mean and standard deviation recruit_sd, the Ricker-Allee target of step 07 with gamma, allee_threshold and carrying_capacity, and the spawning proportion of step 08 with recruits_per_spawner female recruits per spawning female. Each year the density advances by the source's expected kernel: the survivors propagated by the growth matrix plus the recruits produced by the mature females that spawn. Return the array of expected densities (females per metre at the midpoints) with row t the census t, t = 0 to n_years. Call the earlier step functions rather than reimplementing them.

Taking expectations of the source's branching process over a whole cohort gives a deterministic projection of the expected length distribution, whose density dependence enters only through the yearly spawning proportion; the trajectory of expected densities is the backbone of every population-level quantity, including the number of mature females the source's establishment criterion is stated in.

Returns
-------
numpy.ndarray of float64 with shape (n_years + 1, n_bins): the expected density at each census.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def expected_trajectory(n0: float, mean_len0: float, sd_len0: float, n_years: int, n_bins: int,
                        gamma: float, allee_threshold: float, carrying_capacity: float,
                        recruits_per_spawner: int, recruit_mean: float, recruit_sd: float,
                        z_min: float = 0.01, z_max: float = 1.5) -> "numpy.ndarray":
    """Starting from a founding cohort of n0 females whose lengths follow a normal density with mean mean_len0 and standard deviation sd_len0 (metres), project the expected female length distribution forward through n_years annual censuses with the source's expected projection equation on the midpoint mesh of n_bins equal bins on [z_min, z_max] (step 04): the vital rates of steps 01-03 with their default coefficients, the growth matrix of step 05, the recruit length density of step 06 with mean recruit_mean and standard deviation recruit_sd, the Ricker-Allee target of step 07 with gamma, allee_threshold and carrying_capacity, and the spawning proportion of step 08 with recruits_per_spawner female recruits per spawning female. Each year the density advances by the source's expected kernel: the survivors propagated by the growth matrix plus the recruits produced by the mature females that spawn. Return the array of expected densities (females per metre at the midpoints) with row t the census t, t = 0 to n_years. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    n0 : float
        Positive number of founding females.
    mean_len0 : float
        Mean length of the founders in metres.
    sd_len0 : float
        Positive standard deviation of the founder lengths in metres.
    n_years : int
        Number of annual censuses to project, at least 1.
    n_bins : int
        Number of equal midpoint bins on the length domain, at least 1.
    gamma : float
        Positive intrinsic growth rate of the Ricker-Allee model.
    allee_threshold : float
        Allee threshold in kg, with 0 < C < K.
    carrying_capacity : float
        Carrying capacity in kg.
    recruits_per_spawner : int
        Positive number of female recruits per spawning female.
    recruit_mean : float
        Mean recruit length in metres.
    recruit_sd : float
        Positive standard deviation of the recruit length in metres.
    z_min : float
        Lower end of the length domain in metres; default 0.01.
    z_max : float
        Upper end of the length domain in metres; default 1.5.

    Returns
    -------
    trajectory : numpy.ndarray
        Array of shape (n_years + 1, n_bins): row t holds the expected density (females per metre) at the midpoints at census t (float64).

    Raises
    ------
    ValueError
        If any real-valued input is not finite, n0 or sd_len0 is not positive, n_years or recruits_per_spawner is not a positive integer, or the mesh or Ricker-Allee parameters are invalid for the earlier steps.
    """
    return trajectory

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_expected_trajectory(n0: float, mean_len0: float, sd_len0: float, n_years: int, n_bins: int,
                                gamma: float, allee_threshold: float, carrying_capacity: float,
                                recruits_per_spawner: int, recruit_mean: float, recruit_sd: float,
                                z_min: float = 0.01, z_max: float = 1.5) -> "numpy.ndarray":
    """Project the expected female length density n_years censuses forward with Eq 4; row t is census t."""
    if not all(np.isfinite([n0, mean_len0, sd_len0, gamma, allee_threshold, carrying_capacity,
                            recruit_mean, recruit_sd, z_min, z_max])):
        raise ValueError("all real-valued inputs must be finite")
    if n0 <= 0.0 or sd_len0 <= 0.0:
        raise ValueError("n0 and sd_len0 must be positive")
    if int(n_years) != n_years or n_years < 1:
        raise ValueError("n_years must be a positive integer")
    if int(recruits_per_spawner) != recruits_per_spawner or recruits_per_spawner < 1:
        raise ValueError("recruits_per_spawner must be a positive integer")
    mesh = _oracle_midpoint_mesh(z_min, z_max, n_bins)
    z, w = mesh[0], mesh[1]
    weights = _oracle_length_to_weight(z)                             # kg at the midpoints
    survival = _oracle_survival_probability(z)
    maturity = _oracle_maturity_probability(z)
    growth = _oracle_growth_kernel(z, float(w[0]))                     # includes the bin width
    recruit = _oracle_recruit_length_density(z, recruit_mean, recruit_sd)
    k = int(recruits_per_spawner)
    # introduced cohort: n0 females with lengths ~ N(mean_len0, sd_len0), as a density on the mesh
    n = float(n0) * np.exp(-0.5 * ((z - float(mean_len0)) / float(sd_len0)) ** 2) \
        / (float(sd_len0) * np.sqrt(2.0 * np.pi))
    out = np.empty((int(n_years) + 1, z.size))
    out[0] = n
    for t in range(int(n_years)):
        biomass_t = float(np.sum(weights * n * w))                    # b_t
        target = _oracle_ricker_allee_target(biomass_t, gamma, allee_threshold, carrying_capacity)
        a_t = _oracle_spawning_proportion(n, mesh, growth, survival, weights, maturity, recruit, k, target)
        mature_t = float(np.sum(maturity * n * w))                    # m_t
        n = growth @ (survival * n) + k * recruit * (mature_t * a_t)  # Eq 4: survival-growth + recruitment
        out[t + 1] = n
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nw_adult = float(_oracle_length_to_weight(np.array([0.867]))[0])\nn0, mean_len0, sd_len0, n_years, n_bins = 800.0, 0.300, 0.020, 12, 300\ngamma, allee_threshold, carrying_capacity = 1.837, 10.0 * w_adult, 500.0 * w_adult\nrecruits_per_spawner, recruit_mean, recruit_sd = 2, 0.319, 0.040\n",
            "call": "expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd)",
            "gold_call": "_oracle_expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nw_adult = float(_oracle_length_to_weight(np.array([0.867]))[0])\nn0, mean_len0, sd_len0, n_years, n_bins = 300.0, 0.300, 0.020, 15, 300\ngamma, allee_threshold, carrying_capacity = 1.837, 10.0 * w_adult, 500.0 * w_adult\nrecruits_per_spawner, recruit_mean, recruit_sd = 2, 0.319, 0.040\n",
            "call": "expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd)",
            "gold_call": "_oracle_expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nw_adult = float(_oracle_length_to_weight(np.array([0.867]))[0])\nn0, mean_len0, sd_len0, n_years, n_bins = 10.0, 0.867, 0.030, 20, 200\ngamma, allee_threshold, carrying_capacity = 1.837, 10.0 * w_adult, 500.0 * w_adult\nrecruits_per_spawner, recruit_mean, recruit_sd = 2, 0.319, 0.040\n",
            "call": "expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd)",
            "gold_call": "_oracle_expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\ndef _fx_length_to_weight(lengths, intercept=1.02, slope=3.02):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 10.0 ** (float(intercept) + float(slope) * np.log10(zz))\nw_adult = float(_fx_length_to_weight(np.array([0.867]))[0])\nn0, mean_len0, sd_len0, n_years, n_bins = 800.0, 0.300, 0.020, 0, 300\ngamma, allee_threshold, carrying_capacity = 1.837, 10.0 * w_adult, 500.0 * w_adult\nrecruits_per_spawner, recruit_mean, recruit_sd = 2, 0.319, 0.040\ndef run_model():\n    try:\n        expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
