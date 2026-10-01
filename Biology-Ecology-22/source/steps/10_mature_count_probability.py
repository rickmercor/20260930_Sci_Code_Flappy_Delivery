"""
Orchestrator. For the founding cohort and parameters of step 09, return the exact probability that at least threshold mature females are present at census n_years in the source's discrete-individual branching process, computed without simulation. In that process every female present at a census independently follows the source's four outcomes (dies; survives; reproduces; survives and reproduces), survival with the probability of step 02 and reproduction with the maturity probability of step 03 times the spawning proportion of that year, the two events independent; the spawning proportion of every year is the one of the expected trajectory of step 09 (steps 07-08 applied to its rows). A survivor's next-census length is drawn from the growth kernel of step 05 (a survivor whose length falls outside the mesh leaves the population); each of the recruits_per_spawner recruits of a spawning female receives an independent length from the recruit density of step 06; at the final census each female present is mature independently with the probability of step 03. The founders are whole individuals: the expected number in each bin (row 0 of step 09 times the bin width) is converted to individuals by the source's rule for the fractional individual of a bin, and every founder starts a lineage independent of the others. Call the earlier step functions rather than reimplementing them.

The source's model is a branching process of whole individuals, so the number of mature females at a later census has a full probability distribution that the source explores by replicate simulation; with the spawning proportions frozen at their expected-trajectory values (a mean-field simplification for a large founding cohort) and independent lineages, that distribution can be obtained exactly instead of by simulation.

Returns
-------
float in [0, 1], the probability that at least threshold mature females are present at the last census.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mature_count_probability(n0: float, mean_len0: float, sd_len0: float, n_years: int, n_bins: int,
                             gamma: float, allee_threshold: float, carrying_capacity: float,
                             recruits_per_spawner: int, recruit_mean: float, recruit_sd: float,
                             threshold: int, z_min: float = 0.01, z_max: float = 1.5) -> float:
    """Orchestrator. For the founding cohort and parameters of step 09, return the exact probability that at least threshold mature females are present at census n_years in the source's discrete-individual branching process, computed without simulation. In that process every female present at a census independently follows the source's four outcomes (dies; survives; reproduces; survives and reproduces), survival with the probability of step 02 and reproduction with the maturity probability of step 03 times the spawning proportion of that year, the two events independent; the spawning proportion of every year is the one of the expected trajectory of step 09 (steps 07-08 applied to its rows). A survivor's next-census length is drawn from the growth kernel of step 05 (a survivor whose length falls outside the mesh leaves the population); each of the recruits_per_spawner recruits of a spawning female receives an independent length from the recruit density of step 06; at the final census each female present is mature independently with the probability of step 03. The founders are whole individuals: the expected number in each bin (row 0 of step 09 times the bin width) is converted to individuals by the source's rule for the fractional individual of a bin, and every founder starts a lineage independent of the others. Call the earlier step functions rather than reimplementing them.

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
    threshold : int
        Nonnegative integer number of mature females; the probability of at least this many is returned.
    z_min : float
        Lower end of the length domain in metres; default 0.01.
    z_max : float
        Upper end of the length domain in metres; default 1.5.

    Returns
    -------
    probability : float
        Probability that at least threshold mature females are present at census n_years, as a native Python float.

    Raises
    ------
    ValueError
        If threshold is not a nonnegative integer, any real-valued input is not finite, n0 or sd_len0 is not positive, n_years or recruits_per_spawner is not a positive integer, or the mesh or Ricker-Allee parameters are invalid for the earlier steps.
    """
    return probability

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_mature_count_probability(n0: float, mean_len0: float, sd_len0: float, n_years: int, n_bins: int,
                                     gamma: float, allee_threshold: float, carrying_capacity: float,
                                     recruits_per_spawner: int, recruit_mean: float, recruit_sd: float,
                                     threshold: int, z_min: float = 0.01, z_max: float = 1.5) -> float:
    """ORCHESTRATOR: exact probability that at least `threshold` mature females are present at census n_years
    in the source's discrete-individual branching process (Eq 3 outcomes; spawning proportions of the expected
    trajectory, Eq 4; founders = expected numbers per bin with the source's stochastic rounding of the
    fractional individual), by probability generating functions evaluated on the unit circle and inverted
    with the FFT."""
    if int(threshold) != threshold or threshold < 0:
        raise ValueError("threshold must be a nonnegative integer")
    traj = _oracle_expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold,
                                       carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd,
                                       z_min, z_max)
    mesh = _oracle_midpoint_mesh(z_min, z_max, n_bins)
    z, w = mesh[0], mesh[1]
    weights = _oracle_length_to_weight(z)
    survival = _oracle_survival_probability(z)
    maturity = _oracle_maturity_probability(z)
    growth = _oracle_growth_kernel(z, float(w[0]))
    recruit = _oracle_recruit_length_density(z, recruit_mean, recruit_sd)
    k = int(recruits_per_spawner)
    n_years = int(n_years)
    # spawning proportion of each year, frozen at its expected-trajectory value
    a = np.empty(n_years)
    for t in range(n_years):
        target = _oracle_ricker_allee_target(float(np.sum(weights * traj[t] * w)), gamma,
                                             allee_threshold, carrying_capacity)
        a[t] = _oracle_spawning_proportion(traj[t], mesh, growth, survival, weights, maturity, recruit, k, target)
    c = recruit * w                                                   # bin probabilities of a recruit's length
    lost = np.maximum(0.0, 1.0 - growth.sum(axis=0))                  # survivor mass leaving the mesh: no descendants
    # number of evaluation points on the unit circle: a power of two far beyond any count with non-negligible probability
    n_pts = 1 << int(np.ceil(np.log2(max(1024.0, 8.0 * float(np.sum(traj[n_years] * w)) + 256.0))))
    u = np.exp(2j * np.pi * np.arange(n_pts) / n_pts)
    # F_t(u; z): generating function of the number of mature descendants (herself included) at the last census
    # of one female of length z present at census t; at the last census maturity is a Bernoulli event
    F = (1.0 - maturity)[:, None] + maturity[:, None] * u[None, :]
    for t in range(n_years - 1, -1, -1):
        p = maturity * a[t]                                           # probability of reproducing this year
        GF = growth.T @ F + lost[:, None]                             # survivor: average over her next length
        CF = c @ F                                                    # one recruit: average over its length
        # survival and reproduction are independent (Eq 3), the k recruit lengths are iid: the pgf factorises
        F = (1.0 - survival[:, None] + survival[:, None] * GF) * (1.0 - p[:, None] + p[:, None] * CF[None, :] ** k)
    # founders: the expected number in each bin split into whole individuals and a fractional individual that is
    # present with probability equal to the fraction (the source's stochastic rounding); independent lineages
    expected = traj[0] * w
    whole = np.floor(expected)
    frac = expected - whole
    phi = np.ones(n_pts, dtype=complex)
    for j in range(z.size):
        if whole[j] > 0.0 or frac[j] > 0.0:
            phi *= F[j] ** int(whole[j]) * (1.0 - frac[j] + frac[j] * F[j])
    probs = np.fft.fft(phi).real / n_pts                              # P(count = n) for n = 0 .. n_pts - 1
    return float(np.sum(probs[int(threshold):]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nw_adult = float(_oracle_length_to_weight(np.array([0.867]))[0])\nn0, mean_len0, sd_len0, n_years, n_bins = 800.0, 0.300, 0.020, 12, 300\ngamma, allee_threshold, carrying_capacity = 1.837, 10.0 * w_adult, 500.0 * w_adult\nrecruits_per_spawner, recruit_mean, recruit_sd, threshold = 2, 0.319, 0.040, 500\n",
            "call": "mature_count_probability(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd, threshold)",
            "gold_call": "_oracle_mature_count_probability(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd, threshold)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nw_adult = float(_oracle_length_to_weight(np.array([0.867]))[0])\nn0, mean_len0, sd_len0, n_years, n_bins = 300.0, 0.300, 0.020, 15, 300\ngamma, allee_threshold, carrying_capacity = 1.837, 10.0 * w_adult, 500.0 * w_adult\nrecruits_per_spawner, recruit_mean, recruit_sd, threshold = 2, 0.319, 0.040, 600\n",
            "call": "mature_count_probability(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd, threshold)",
            "gold_call": "_oracle_mature_count_probability(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd, threshold)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nw_adult = float(_oracle_length_to_weight(np.array([0.867]))[0])\nn0, mean_len0, sd_len0, n_years, n_bins = 10.0, 0.867, 0.030, 20, 200\ngamma, allee_threshold, carrying_capacity = 1.837, 10.0 * w_adult, 500.0 * w_adult\nrecruits_per_spawner, recruit_mean, recruit_sd, threshold = 2, 0.319, 0.040, 10\n",
            "call": "mature_count_probability(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd, threshold)",
            "gold_call": "_oracle_mature_count_probability(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd, threshold)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\ndef _fx_length_to_weight(lengths, intercept=1.02, slope=3.02):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 10.0 ** (float(intercept) + float(slope) * np.log10(zz))\nw_adult = float(_fx_length_to_weight(np.array([0.867]))[0])\nn0, mean_len0, sd_len0, n_years, n_bins = 800.0, 0.300, 0.020, 12, 300\ngamma, allee_threshold, carrying_capacity = 1.837, 10.0 * w_adult, 500.0 * w_adult\nrecruits_per_spawner, recruit_mean, recruit_sd, threshold = 2, 0.319, 0.040, -1\ndef run_model():\n    try:\n        mature_count_probability(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd, threshold)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_mature_count_probability(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold, carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd, threshold)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
