"""
Return the proportion A_t of the mature females that spawn in the current year according to the source's biomass-balance rule (its Fig. 1, phases 1 and 2), given the expected female length density on the mesh, the mesh of step 04, the growth matrix of step 05, the survival probabilities, weights (kg) and maturity probabilities at the midpoints, the recruit length density of step 06 with recruits_per_spawner female recruits per spawning female, and the target biomass for the next census from step 07. The rule sets the spawning level from how the target compares with the biomass the present females themselves contribute to the next census and with the biomass that recruitment can add; the result lies in [0, 1], with no spawning when the target is already met, universal spawning when recruitment cannot reach it, and 0 whenever no female can spawn (zero maximum recruit biomass, e.g. an empty population).

An integral projection model has no population-level density dependence of its own; the source links the biomass target of a Ricker-Allee model to individual reproduction by letting only the fraction of mature females spawn that is needed to close the gap between the target and what the population would reach without reproduction.

Returns
-------
float in [0, 1], the proportion of mature females that spawn this year.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spawning_proportion(density: "numpy.ndarray", mesh: "numpy.ndarray", growth: "numpy.ndarray",
                        survival: "numpy.ndarray", weights: "numpy.ndarray",
                        maturity: "numpy.ndarray", recruit_density: "numpy.ndarray",
                        recruits_per_spawner: int, target_biomass: float) -> float:
    """Return the proportion A_t of the mature females that spawn in the current year according to the source's biomass-balance rule (its Fig. 1, phases 1 and 2), given the expected female length density on the mesh, the mesh of step 04, the growth matrix of step 05, the survival probabilities, weights (kg) and maturity probabilities at the midpoints, the recruit length density of step 06 with recruits_per_spawner female recruits per spawning female, and the target biomass for the next census from step 07. The rule sets the spawning level from how the target compares with the biomass the present females themselves contribute to the next census and with the biomass that recruitment can add; the result lies in [0, 1], with no spawning when the target is already met, universal spawning when recruitment cannot reach it, and 0 whenever no female can spawn (zero maximum recruit biomass, e.g. an empty population).

    Parameters
    ----------
    density : numpy.ndarray
        One-dimensional array of length n: the expected number of females per metre at each midpoint, nonnegative.
    mesh : numpy.ndarray
        Array of shape (2, n) from step 04: midpoints and bin widths.
    growth : numpy.ndarray
        Array of shape (n, n) from step 05: the midpoint-rule growth matrix.
    survival : numpy.ndarray
        One-dimensional array of length n: annual survival probabilities at the midpoints.
    weights : numpy.ndarray
        One-dimensional array of length n: body weights in kg at the midpoints.
    maturity : numpy.ndarray
        One-dimensional array of length n: maturity probabilities at the midpoints.
    recruit_density : numpy.ndarray
        One-dimensional array of length n: recruit length density at the midpoints (per metre).
    recruits_per_spawner : int
        Positive number of female recruits produced by each spawning female.
    target_biomass : float
        Target biomass for the next census in kg from step 07, nonnegative.

    Returns
    -------
    proportion : float
        The spawning proportion A_t in [0, 1], as a native Python float.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent with the density length, any array contains a non-finite or negative value, a bin width is not positive, recruits_per_spawner is not a positive integer, or target_biomass is negative or not finite.
    """
    return proportion

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spawning_proportion(density: "numpy.ndarray", mesh: "numpy.ndarray", growth: "numpy.ndarray",
                                survival: "numpy.ndarray", weights: "numpy.ndarray",
                                maturity: "numpy.ndarray", recruit_density: "numpy.ndarray",
                                recruits_per_spawner: int, target_biomass: float) -> float:
    """Spawning proportion A_t of the biomass balance b_{t+1} = s_{t+1} + A_t m_t b_R, clamped to [0, 1]."""
    n = np.asarray(density, dtype=np.float64)
    mesh = np.asarray(mesh, dtype=np.float64)
    G = np.asarray(growth, dtype=np.float64)
    arrs = [np.asarray(a, dtype=np.float64) for a in (survival, weights, maturity, recruit_density)]
    if n.ndim != 1 or n.size < 1:
        raise ValueError("density must be a non-empty 1-D array")
    m = n.size
    if mesh.shape != (2, m) or G.shape != (m, m) or any(a.shape != (m,) for a in arrs):
        raise ValueError("mesh, growth and the vital-rate arrays must match the density length")
    if not all(np.all(np.isfinite(a)) for a in (n, mesh, G, *arrs)):
        raise ValueError("all arrays must be finite")
    if np.any(n < 0.0) or np.any(mesh[1] <= 0.0) or np.any(G < 0.0) or any(np.any(a < 0.0) for a in arrs):
        raise ValueError("density, bin widths, growth and vital rates must be nonnegative")
    if int(recruits_per_spawner) != recruits_per_spawner or recruits_per_spawner < 1:
        raise ValueError("recruits_per_spawner must be a positive integer")
    if not np.isfinite(target_biomass) or target_biomass < 0.0:
        raise ValueError("target_biomass must be finite and nonnegative")
    s, W, mat, C1 = arrs
    w = mesh[1]
    # s_{t+1}: the survivors are weighed at the NEXT census, i.e. after growth (integrated over z')
    surviving_biomass = float(np.sum(W * (G @ (s * n)) * w))
    mature_total = float(np.sum(mat * n * w))                          # m_t
    recruit_biomass = int(recruits_per_spawner) * float(np.sum(W * C1 * w))   # b_R = int W(z') k C1(z') dz'
    deficit = float(target_biomass) - surviving_biomass               # recruit deficit, Fig 1 phase 2
    potential = mature_total * recruit_biomass                        # maximum recruit biomass m_t b_R
    if deficit < 0.0 or potential <= 0.0:
        return 0.0                                                    # survivors already exceed the target, or nobody can spawn
    if deficit > potential:
        return 1.0                                                    # even universal spawning cannot bridge it
    return deficit / potential

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndef _fx_midpoint_mesh(z_min, z_max, n_bins):\n    n = int(n_bins)\n    edges = np.linspace(float(z_min), float(z_max), n + 1)\n    mids = 0.5 * (edges[:-1] + edges[1:])\n    widths = np.full(n, (float(z_max) - float(z_min)) / n)\n    return np.vstack([mids, widths])\ndef _fx_length_to_weight(lengths, intercept=1.02, slope=3.02):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 10.0 ** (float(intercept) + float(slope) * np.log10(zz))\ndef _fx_survival_probability(lengths, slope=2.7, exponent=-0.315):\n    zz = np.asarray(lengths, dtype=np.float64)\n    grams = 1000.0 * _fx_length_to_weight(zz)\n    return np.exp(-float(slope) * grams ** float(exponent))\ndef _fx_maturity_probability(lengths, intercept=-7.41, slope=24.98):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 1.0 / (1.0 + np.exp(-(float(intercept) + float(slope) * zz)))\ndef _fx_growth_kernel(midpoints, width, mu0=0.194, mu1=0.841, sd=0.1):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    mean = float(mu0) + float(mu1) * zz[None, :]\n    pdf = np.exp(-0.5 * ((zz[:, None] - mean) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\n    return pdf * float(width)\ndef _fx_recruit_length_density(midpoints, mean=0.319, sd=0.040):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    return np.exp(-0.5 * ((zz - float(mean)) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\ndef _fx_ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity):\n    b, g, C, K = float(biomass), float(gamma), float(allee_threshold), float(carrying_capacity)\n    return b * np.exp(g * (1.0 - b / K) * ((b - C) / K))\nmesh = _fx_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\nweights = _fx_length_to_weight(z)\nsurvival = _fx_survival_probability(z)\nmaturity = _fx_maturity_probability(z)\ngrowth = _fx_growth_kernel(z, float(w[0]))\nrecruit_density = _fx_recruit_length_density(z, 0.319, 0.040)\nw_adult = float(_fx_length_to_weight(np.array([0.867]))[0])\nrecruits_per_spawner = 2\ndensity = 800.0 * np.exp(-0.5 * ((z - 0.300) / 0.020) ** 2) / (0.020 * np.sqrt(2.0 * np.pi))\ntarget_biomass = _fx_ricker_allee_target(float(np.sum(weights * density * w)), 1.837, 10.0 * w_adult, 500.0 * w_adult)\n",
            "call": "spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)",
            "gold_call": "_oracle_spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ndef _fx_midpoint_mesh(z_min, z_max, n_bins):\n    n = int(n_bins)\n    edges = np.linspace(float(z_min), float(z_max), n + 1)\n    mids = 0.5 * (edges[:-1] + edges[1:])\n    widths = np.full(n, (float(z_max) - float(z_min)) / n)\n    return np.vstack([mids, widths])\ndef _fx_length_to_weight(lengths, intercept=1.02, slope=3.02):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 10.0 ** (float(intercept) + float(slope) * np.log10(zz))\ndef _fx_survival_probability(lengths, slope=2.7, exponent=-0.315):\n    zz = np.asarray(lengths, dtype=np.float64)\n    grams = 1000.0 * _fx_length_to_weight(zz)\n    return np.exp(-float(slope) * grams ** float(exponent))\ndef _fx_maturity_probability(lengths, intercept=-7.41, slope=24.98):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 1.0 / (1.0 + np.exp(-(float(intercept) + float(slope) * zz)))\ndef _fx_growth_kernel(midpoints, width, mu0=0.194, mu1=0.841, sd=0.1):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    mean = float(mu0) + float(mu1) * zz[None, :]\n    pdf = np.exp(-0.5 * ((zz[:, None] - mean) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\n    return pdf * float(width)\ndef _fx_recruit_length_density(midpoints, mean=0.319, sd=0.040):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    return np.exp(-0.5 * ((zz - float(mean)) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\ndef _fx_ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity):\n    b, g, C, K = float(biomass), float(gamma), float(allee_threshold), float(carrying_capacity)\n    return b * np.exp(g * (1.0 - b / K) * ((b - C) / K))\nmesh = _fx_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\nweights = _fx_length_to_weight(z)\nsurvival = _fx_survival_probability(z)\nmaturity = _fx_maturity_probability(z)\ngrowth = _fx_growth_kernel(z, float(w[0]))\nrecruit_density = _fx_recruit_length_density(z, 0.319, 0.040)\nw_adult = float(_fx_length_to_weight(np.array([0.867]))[0])\nrecruits_per_spawner = 2\ndensity = 120.0 * np.exp(-0.5 * ((z - 0.867) / 0.030) ** 2) / (0.030 * np.sqrt(2.0 * np.pi))\ntarget_biomass = _fx_ricker_allee_target(float(np.sum(weights * density * w)), 1.837, 10.0 * w_adult, 500.0 * w_adult)\n",
            "call": "spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)",
            "gold_call": "_oracle_spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ndef _fx_midpoint_mesh(z_min, z_max, n_bins):\n    n = int(n_bins)\n    edges = np.linspace(float(z_min), float(z_max), n + 1)\n    mids = 0.5 * (edges[:-1] + edges[1:])\n    widths = np.full(n, (float(z_max) - float(z_min)) / n)\n    return np.vstack([mids, widths])\ndef _fx_length_to_weight(lengths, intercept=1.02, slope=3.02):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 10.0 ** (float(intercept) + float(slope) * np.log10(zz))\ndef _fx_survival_probability(lengths, slope=2.7, exponent=-0.315):\n    zz = np.asarray(lengths, dtype=np.float64)\n    grams = 1000.0 * _fx_length_to_weight(zz)\n    return np.exp(-float(slope) * grams ** float(exponent))\ndef _fx_maturity_probability(lengths, intercept=-7.41, slope=24.98):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 1.0 / (1.0 + np.exp(-(float(intercept) + float(slope) * zz)))\ndef _fx_growth_kernel(midpoints, width, mu0=0.194, mu1=0.841, sd=0.1):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    mean = float(mu0) + float(mu1) * zz[None, :]\n    pdf = np.exp(-0.5 * ((zz[:, None] - mean) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\n    return pdf * float(width)\ndef _fx_recruit_length_density(midpoints, mean=0.319, sd=0.040):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    return np.exp(-0.5 * ((zz - float(mean)) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\ndef _fx_ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity):\n    b, g, C, K = float(biomass), float(gamma), float(allee_threshold), float(carrying_capacity)\n    return b * np.exp(g * (1.0 - b / K) * ((b - C) / K))\nmesh = _fx_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\nweights = _fx_length_to_weight(z)\nsurvival = _fx_survival_probability(z)\nmaturity = _fx_maturity_probability(z)\ngrowth = _fx_growth_kernel(z, float(w[0]))\nrecruit_density = _fx_recruit_length_density(z, 0.319, 0.040)\nw_adult = float(_fx_length_to_weight(np.array([0.867]))[0])\nrecruits_per_spawner = 2\ndensity = 120.0 * np.exp(-0.5 * ((z - 0.867) / 0.030) ** 2) / (0.030 * np.sqrt(2.0 * np.pi)) + 600.0 * np.exp(-0.5 * ((z - 0.300) / 0.020) ** 2) / (0.020 * np.sqrt(2.0 * np.pi))\ntarget_biomass = _fx_ricker_allee_target(float(np.sum(weights * density * w)), 1.837, 10.0 * w_adult, 500.0 * w_adult)\n",
            "call": "spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)",
            "gold_call": "_oracle_spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ndef _fx_midpoint_mesh(z_min, z_max, n_bins):\n    n = int(n_bins)\n    edges = np.linspace(float(z_min), float(z_max), n + 1)\n    mids = 0.5 * (edges[:-1] + edges[1:])\n    widths = np.full(n, (float(z_max) - float(z_min)) / n)\n    return np.vstack([mids, widths])\ndef _fx_length_to_weight(lengths, intercept=1.02, slope=3.02):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 10.0 ** (float(intercept) + float(slope) * np.log10(zz))\ndef _fx_survival_probability(lengths, slope=2.7, exponent=-0.315):\n    zz = np.asarray(lengths, dtype=np.float64)\n    grams = 1000.0 * _fx_length_to_weight(zz)\n    return np.exp(-float(slope) * grams ** float(exponent))\ndef _fx_maturity_probability(lengths, intercept=-7.41, slope=24.98):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 1.0 / (1.0 + np.exp(-(float(intercept) + float(slope) * zz)))\ndef _fx_growth_kernel(midpoints, width, mu0=0.194, mu1=0.841, sd=0.1):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    mean = float(mu0) + float(mu1) * zz[None, :]\n    pdf = np.exp(-0.5 * ((zz[:, None] - mean) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\n    return pdf * float(width)\ndef _fx_recruit_length_density(midpoints, mean=0.319, sd=0.040):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    return np.exp(-0.5 * ((zz - float(mean)) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\ndef _fx_ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity):\n    b, g, C, K = float(biomass), float(gamma), float(allee_threshold), float(carrying_capacity)\n    return b * np.exp(g * (1.0 - b / K) * ((b - C) / K))\nmesh = _fx_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\nweights = _fx_length_to_weight(z)\nsurvival = _fx_survival_probability(z)\nmaturity = _fx_maturity_probability(z)\ngrowth = _fx_growth_kernel(z, float(w[0]))\nrecruit_density = _fx_recruit_length_density(z, 0.319, 0.040)\nw_adult = float(_fx_length_to_weight(np.array([0.867]))[0])\nrecruits_per_spawner = 2\ndensity = np.zeros_like(z)\ntarget_biomass = 0.0\n",
            "call": "spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)",
            "gold_call": "_oracle_spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ndef _fx_midpoint_mesh(z_min, z_max, n_bins):\n    n = int(n_bins)\n    edges = np.linspace(float(z_min), float(z_max), n + 1)\n    mids = 0.5 * (edges[:-1] + edges[1:])\n    widths = np.full(n, (float(z_max) - float(z_min)) / n)\n    return np.vstack([mids, widths])\ndef _fx_length_to_weight(lengths, intercept=1.02, slope=3.02):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 10.0 ** (float(intercept) + float(slope) * np.log10(zz))\ndef _fx_survival_probability(lengths, slope=2.7, exponent=-0.315):\n    zz = np.asarray(lengths, dtype=np.float64)\n    grams = 1000.0 * _fx_length_to_weight(zz)\n    return np.exp(-float(slope) * grams ** float(exponent))\ndef _fx_maturity_probability(lengths, intercept=-7.41, slope=24.98):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 1.0 / (1.0 + np.exp(-(float(intercept) + float(slope) * zz)))\ndef _fx_growth_kernel(midpoints, width, mu0=0.194, mu1=0.841, sd=0.1):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    mean = float(mu0) + float(mu1) * zz[None, :]\n    pdf = np.exp(-0.5 * ((zz[:, None] - mean) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\n    return pdf * float(width)\ndef _fx_recruit_length_density(midpoints, mean=0.319, sd=0.040):\n    zz = np.asarray(midpoints, dtype=np.float64)\n    return np.exp(-0.5 * ((zz - float(mean)) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))\ndef _fx_ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity):\n    b, g, C, K = float(biomass), float(gamma), float(allee_threshold), float(carrying_capacity)\n    return b * np.exp(g * (1.0 - b / K) * ((b - C) / K))\nmesh = _fx_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\nweights = _fx_length_to_weight(z)\nsurvival = _fx_survival_probability(z)\nmaturity = _fx_maturity_probability(z)\ngrowth = _fx_growth_kernel(z, float(w[0]))\nrecruit_density = _fx_recruit_length_density(z, 0.319, 0.040)\nw_adult = float(_fx_length_to_weight(np.array([0.867]))[0])\nrecruits_per_spawner = 2\ndensity = 800.0 * np.exp(-0.5 * ((z - 0.300) / 0.020) ** 2) / (0.020 * np.sqrt(2.0 * np.pi))\ntarget_biomass = _fx_ricker_allee_target(float(np.sum(weights * density * w)), 1.837, 10.0 * w_adult, 500.0 * w_adult)\nrecruits_per_spawner = 0\ndef run_model():\n    try:\n        spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_spawning_proportion(density, mesh, growth, survival, weights, maturity, recruit_density, recruits_per_spawner, target_biomass)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
