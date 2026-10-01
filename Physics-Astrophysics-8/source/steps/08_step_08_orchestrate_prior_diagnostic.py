"""
Run the whole empirical-prior construction for one excitation-energy bin and reduce it to the scalars that measure how the prior sits against the truth.

The pipeline performs the following steps.

1. The composition of the redistribution and resolution operators into the detector response $\mathbf{R}_\gamma = \mathbf{G}_\gamma \mathbf{D}$

2. The background reference $\mathbf{b}_{\mathrm{ref}}$ of every bin from the background run

3. The background-aware Richardson-Lucy iteration from the flat start

4. The selection of the reference iterate $t_{\mathrm{RL}}$ by the change-to-noise rule on Poisson resamples

5. The floored prior centre $\boldsymbol{\mu}_{\mathrm{RL}}$ and the adaptive prior width $\sigma_j$ of every bin

6. The $N$ prior draws from the supplied variates, mapped to the resolution-limited space

7. The mean curve $\bar{\boldsymbol{\eta}}$ and the simultaneous global rank-envelope band $[L_j, U_j]$ of those draws

The construction is judged in the resolution-limited space, where the emitted truth is mapped through the resolution operator alone, $\boldsymbol{\eta}_{\mathrm{true}} = \mathbf{G}_\gamma \boldsymbol{\mu}_{\mathrm{true}}$. In every bin the deviation of the mean curve from the resolution-limited truth is scaled by the half-width of the band, $w_{j,1/2} = \max\{(U_j - L_j)/2,\ \epsilon\}$, giving $S_j = (\bar\eta_j - \eta_{\mathrm{true},j}) / w_{j,1/2}$. The figure reported is the mean over bins of $\lvert S_j \rvert$. The selected iteration and the largest $\lvert S_j \rvert$ over bins are reported beside it.

Returns
-------
np.ndarray, shape (3,). The mean over bins of the absolute scaled deviation of the mean curve from the resolution-limited truth, the selected reference iteration, and the largest absolute scaled deviation over bins. Entry 0 is the final scalar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orchestrate_prior_diagnostic(
    redistribution: np.ndarray,
    resolution: np.ndarray,
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    resampled_counts: np.ndarray,
    emitted_truth: np.ndarray,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    background_prior_shape: float,
    window: int,
    ratio_threshold: float,
    run_length: int,
    max_iterations: int,
    floor: float,
    sigma_min: float,
    sigma_max: float,
    count_scale: float,
    gamma_shape: float,
    mass: float,
    guard: float,
) -> np.ndarray:
    r"""Execute the complete prior construction and return its three scalars.

    Parameters
    ----------
    redistribution : np.ndarray
        Energy-redistribution operator, shape (J, J), every column summing to
        the detection efficiency of that bin.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), column-normalised.
    on_counts : np.ndarray
        Counts of the signal run, shape (J,), non-negative integer values.
    off_counts : np.ndarray
        Counts of the background run, shape (J,), non-negative integer
        values, at least one of them positive.
    resampled_counts : np.ndarray
        Poisson resamples of the signal run, shape (K, J) with K at least 2.
    emitted_truth : np.ndarray
        Emitted spectrum the signal run was generated from, shape (J,),
        non-negative and finite.
    normal_draws : np.ndarray
        Standard-normal variates of the Gaussian prior layer, shape (N, J).
    uniform_draws : np.ndarray
        Uniform variates of the conditional Gamma layer, shape (N, J),
        strictly inside the unit interval.
    background_prior_shape : float
        Shape of the Gamma prior on the background expectation, positive.
    window : int
        Window of the deterministic change, at least 1.
    ratio_threshold : float
        Largest admissible change-to-noise ratio, positive.
    run_length : int
        Number of consecutive qualifying iterations required, at least 1.
    max_iterations : int
        Number of updates run and the fallback selection, at least 1.
    floor : float
        Lowest admissible prior centre in counts, positive.
    sigma_min, sigma_max : float
        Bounds of the log-scale prior width, ``sigma_max`` at least
        ``sigma_min``.
    count_scale : float
        Resolution-limited count level of half activation, positive.
    gamma_shape : float
        Shape of the conditional Gamma distribution, positive.
    mass : float
        Envelope mass level in (0, 1].
    guard : float
        Non-negative constant guarding every denominator and the band
        half-width.

    Returns
    -------
    results : np.ndarray
        Shape (3,). The mean over bins of the absolute scaled deviation of
        the mean curve from the resolution-limited truth, the selected
        reference iteration, and the largest absolute scaled deviation over
        bins. Entry 0 is the final scalar.

    Raises
    ------
    ValueError
        If any upstream contract fails, if ``emitted_truth`` is not of shape
        (J,) or holds a negative or non-finite entry, or if a reported
        scalar is not finite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_orchestrate_prior_diagnostic(
    redistribution: np.ndarray,
    resolution: np.ndarray,
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    resampled_counts: np.ndarray,
    emitted_truth: np.ndarray,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    background_prior_shape: float,
    window: int,
    ratio_threshold: float,
    run_length: int,
    max_iterations: int,
    floor: float,
    sigma_min: float,
    sigma_max: float,
    count_scale: float,
    gamma_shape: float,
    mass: float,
    guard: float,
) -> np.ndarray:
    g = np.asarray(resolution, dtype=float)
    mu_true = np.asarray(emitted_truth, dtype=float)
    eps = float(guard)

    response = _oracle_compose_detector_response(redistribution, resolution)
    n_bins = response.shape[0]
    if mu_true.shape != (n_bins,):
        raise ValueError("emitted_truth must have shape (J,) matching the operators")
    if not np.all(np.isfinite(mu_true)) or np.any(mu_true < 0.0):
        raise ValueError("emitted_truth must be finite and non-negative")

    background_reference = _oracle_build_background_reference(
        off_counts, background_prior_shape
    )
    selected = _oracle_select_reference_iteration(
        on_counts, resampled_counts, response, resolution, background_reference,
        window, ratio_threshold, run_length, max_iterations, guard,
    )
    iterates = _oracle_run_richardson_lucy_iterates(
        on_counts, response, background_reference, selected, guard
    )
    centre_and_width = _oracle_build_prior_centre_and_width(
        iterates[selected], resolution, floor, sigma_min, sigma_max, count_scale
    )
    draws = _oracle_draw_prior_spectra(
        centre_and_width[0], centre_and_width[1], gamma_shape,
        normal_draws, uniform_draws, resolution,
    )
    envelope = _oracle_build_global_rank_envelope(draws, mass)

    # The truth is compared in the resolution-limited space.
    eta_true = g @ mu_true
    half_width = np.maximum(0.5 * (envelope[2] - envelope[1]), eps)
    scaled = (envelope[0] - eta_true) / half_width

    out = np.array([
        float(np.mean(np.abs(scaled))),
        float(selected),
        float(np.max(np.abs(scaled))),
    ], dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("computed diagnostics must be finite")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    def _frozen_setup():
        return """import numpy as np
redistribution = np.array([
    [0.9600, 0.0920, 0.0634, 0.0528, 0.0466, 0.0421, 0.0386, 0.0356],
    [0.0000, 0.8280, 0.0686, 0.0506, 0.0412, 0.0352, 0.0309, 0.0275],
    [0.0000, 0.0000, 0.7480, 0.0646, 0.0511, 0.0427, 0.0368, 0.0323],
    [0.0000, 0.0000, 0.0000, 0.6720, 0.0611, 0.0502, 0.0427, 0.0370],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.6000, 0.0577, 0.0486, 0.0418],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5321, 0.0545, 0.0466],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4679, 0.0513],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4079],
])
resolution = np.array([
    [0.9220, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0780, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0780],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.9220],
])
on_counts = np.array([381, 1541, 460, 157, 293, 39, 0, 0])
off_counts = np.array([48, 154, 51, 22, 51, 15, 1, 0])
resampled_counts = np.array([
    [406, 1578, 429, 153, 292, 35, 0, 0],
    [403, 1568, 417, 147, 302, 38, 0, 0],
    [364, 1606, 456, 166, 298, 22, 0, 0],
    [358, 1573, 453, 172, 290, 35, 0, 0],
])
emitted_truth = np.array([60.0, 1800.0, 400.0, 120.0, 480.0, 6.0, 0.3, 0.1])
normal_draws = np.array([
    [-1.3172, 1.2482, -0.0462, 0.5644, 0.4015, 0.8277, -0.5209, -1.1960],
    [0.2368, -0.2066, 0.6623, -1.0854, -0.1586, 0.0253, 0.3285, 0.9852],
    [1.7075, 0.8812, 0.7809, -0.2051, -0.9937, 1.5846, -0.9145, -0.7679],
    [-0.9799, -1.0106, -1.2303, 1.0254, 0.1357, -1.3587, -2.2404, -0.2426],
    [-0.6562, 1.3738, -0.1800, 1.7620, 2.2083, 0.1282, 0.5415, -0.2978],
    [-1.1052, -0.5546, 0.4613, 1.5446, -0.3304, -0.1789, -0.3454, -0.3855],
    [0.4446, -0.5375, 0.1057, 0.6215, -0.7146, -0.8187, 0.4618, 0.2888],
    [-0.7490, -1.3887, -2.0672, -1.0401, -0.8608, 0.0542, -0.0914, 0.7233],
    [0.1356, -1.7114, -0.2807, 1.2544, -0.0028, -0.2288, 0.0483, 1.6938],
    [0.6963, 0.3910, 1.9035, 0.3795, -0.9446, -0.6751, -0.3260, -0.1255],
    [-1.3818, 0.5132, 0.3204, -0.6988, -0.6736, 0.3571, 0.2233, -0.3653],
    [1.6635, -0.4442, 0.4165, 1.8852, -0.6552, -1.2765, -0.5137, 0.7405],
    [0.0872, 0.7130, 1.0826, 0.4673, -2.1087, -1.9236, 0.2541, -1.0312],
    [-1.1075, 0.8420, -0.3870, 0.1768, -0.0485, -1.0076, -0.4778, 1.6000],
    [0.4420, 0.5495, 0.0880, -0.1985, 1.5141, 1.5562, -0.3511, -1.1659],
    [0.3934, 0.5039, 0.7601, -0.9601, 1.1018, -1.0292, -0.4393, -0.4425],
])
uniform_draws = np.array([
    [0.0888, 0.9355, 0.0318, 0.8306, 0.0715, 0.0664, 0.8101, 0.9222],
    [0.2122, 0.5367, 0.5288, 0.9245, 0.9291, 0.5789, 0.3794, 0.9377],
    [0.1072, 0.6361, 0.0406, 0.5973, 0.2047, 0.2923, 0.4446, 0.3373],
    [0.7535, 0.0263, 0.1654, 0.4910, 0.9082, 0.1059, 0.2709, 0.8614],
    [0.4247, 0.7255, 0.0231, 0.1075, 0.4890, 0.0209, 0.5067, 0.9778],
    [0.3624, 0.5765, 0.7857, 0.5459, 0.3383, 0.1387, 0.2239, 0.4842],
    [0.8331, 0.4785, 0.7814, 0.3562, 0.6291, 0.6273, 0.4827, 0.7815],
    [0.2814, 0.8762, 0.3056, 0.7721, 0.8010, 0.3934, 0.2598, 0.5451],
    [0.1599, 0.9630, 0.5310, 0.0950, 0.0614, 0.6113, 0.1181, 0.2301],
    [0.9018, 0.1109, 0.0210, 0.1611, 0.8519, 0.3587, 0.8923, 0.1646],
    [0.4891, 0.5280, 0.1943, 0.8251, 0.0464, 0.3037, 0.9450, 0.5611],
    [0.7510, 0.7096, 0.5391, 0.4929, 0.1551, 0.2387, 0.2519, 0.5734],
    [0.9566, 0.7836, 0.6671, 0.4128, 0.9785, 0.1016, 0.6567, 0.0204],
    [0.7842, 0.5067, 0.8415, 0.6694, 0.8716, 0.1006, 0.0680, 0.7919],
    [0.2552, 0.0236, 0.8433, 0.1673, 0.7827, 0.7424, 0.9036, 0.2916],
    [0.7761, 0.2803, 0.7159, 0.5054, 0.0913, 0.2354, 0.1605, 0.2408],
])
background_prior_shape = 1.0
window = 10
ratio_threshold = 2.0
run_length = 10
max_iterations = 500
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
gamma_shape = 1.0
mass = 0.75
guard = 1e-12
ARGS = (redistribution, resolution, on_counts, off_counts, resampled_counts, emitted_truth,
        normal_draws, uniform_draws, background_prior_shape, window, ratio_threshold,
        run_length, max_iterations, floor, sigma_min, sigma_max, count_scale, gamma_shape,
        mass, guard)
"""

    def _small_setup():
        # a three-bin instance with three resamples and eight draws
        return """import numpy as np
redistribution = np.array([
    [1.0, 0.3, 0.2],
    [0.0, 0.7, 0.3],
    [0.0, 0.0, 0.5],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([120, 340, 60])
off_counts = np.array([14, 25, 9])
resampled_counts = np.array([
    [131, 322, 55],
    [108, 351, 71],
    [124, 338, 58],
])
emitted_truth = np.array([30.0, 380.0, 90.0])
normal_draws = np.array([
    [0.35, 0.82, 0.33],
    [-1.30, 0.91, 0.45],
    [-0.54, 0.58, 0.36],
    [0.29, 0.03, 0.55],
    [-0.74, -0.16, -0.48],
    [0.60, 0.04, -0.29],
    [-0.78, -0.26, 0.01],
    [-0.28, 1.29, 1.01],
])
uniform_draws = np.array([
    [0.93, 0.71, 0.54],
    [0.29, 0.18, 0.94],
    [0.52, 0.14, 0.62],
    [0.76, 0.61, 0.89],
    [0.07, 0.53, 0.46],
    [0.09, 0.63, 0.83],
    [0.59, 0.27, 0.82],
    [0.51, 0.51, 0.74],
])
background_prior_shape = 1.0
window = 5
ratio_threshold = 1.0
run_length = 4
max_iterations = 200
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
gamma_shape = 1.0
mass = 0.75
guard = 1e-12
ARGS = (redistribution, resolution, on_counts, off_counts, resampled_counts, emitted_truth,
        normal_draws, uniform_draws, background_prior_shape, window, ratio_threshold,
        run_length, max_iterations, floor, sigma_min, sigma_max, count_scale, gamma_shape,
        mass, guard)
"""

    call = "orchestrate_prior_diagnostic(*ARGS)"
    gold = "_oracle_orchestrate_prior_diagnostic(*ARGS)"
    err = ""
    for name, function in (("run_model", "orchestrate_prior_diagnostic"),
                           ("run_gold", "_oracle_orchestrate_prior_diagnostic")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    cases = [
        {
            "setup": _frozen_setup(),
            "call": call,
            "gold_call": gold,
        },
        # the benchmark with a wider band, which retains all but one curve
        {
            "setup": _frozen_setup() + """mass = 0.9
ARGS = ARGS[:18] + (mass, guard)
""",
            "call": call,
            "gold_call": gold,
        },
        # the benchmark with a uniform prior width, the two bounds coinciding
        {
            "setup": _frozen_setup() + """sigma_min = 1.0
sigma_max = 1.0
ARGS = ARGS[:14] + (sigma_min, sigma_max) + ARGS[16:]
""",
            "call": call,
            "gold_call": gold,
        },
        # the benchmark with a fixed late reference, forced by a loose threshold and a large window
        {
            "setup": _frozen_setup() + """window = 60
ratio_threshold = 1e6
run_length = 1
max_iterations = 60
ARGS = ARGS[:9] + (window, ratio_threshold, run_length, max_iterations) + ARGS[13:]
""",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": _small_setup(),
            "call": call,
            "gold_call": gold,
        },
        # boundary: full envelope mass, so the band is the range of all eight draws
        {
            "setup": _small_setup() + """mass = 1.0
ARGS = ARGS[:18] + (mass, guard)
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a diagonal resolution and a diagonal redistribution, so every space coincides
        {
            "setup": _small_setup() + """redistribution = np.eye(3)
resolution = np.eye(3)
ARGS = (redistribution, resolution) + ARGS[2:]
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a threshold that never qualifies inside the cap, so the cap is the reference
        {
            "setup": _small_setup() + """ratio_threshold = 1e-9
max_iterations = 60
ARGS = ARGS[:10] + (ratio_threshold,) + ARGS[11:12] + (max_iterations,) + ARGS[13:]
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a count scale so large that every bin keeps the upper width bound
        {
            "setup": _small_setup() + """count_scale = 1e9
ARGS = ARGS[:16] + (count_scale,) + ARGS[17:]
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a truth of zero in every bin, so the scaled deviation is the mean curve divided by the half-width
        {
            "setup": _small_setup() + """emitted_truth = np.zeros(3)
ARGS = ARGS[:5] + (emitted_truth,) + ARGS[6:]
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a guard larger than the narrowest band half-width, which floors that bin's scale;
        # the same guard also dominates the change-to-noise ratio, so the reference moves to iteration 12
        {
            "setup": _frozen_setup() + """guard = 0.2
ARGS = ARGS[:19] + (guard,)
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: twelve draws over three bins, where two curves with identical sorted ranks straddle
        # the envelope cut and the lower row index counts as the more extreme
        {
            "setup": _small_setup() + """normal_draws = np.array([
    [-0.43, -1.13, 0.67],
    [-1.11, 2.01, 0.92],
    [-0.36, 0.57, 1.61],
    [2.83, -0.92, 1.07],
    [0.52, -0.28, 1.09],
    [0.51, 1.08, -0.53],
    [0.00, 0.39, 0.02],
    [0.02, -0.77, 0.13],
    [0.19, -0.06, -0.01],
    [0.70, -0.66, 0.70],
    [-0.15, 0.06, -0.19],
    [0.90, 0.76, 0.01],
])
uniform_draws = np.array([
    [0.83, 0.14, 0.19],
    [0.71, 0.15, 0.05],
    [0.21, 0.40, 0.69],
    [0.69, 0.23, 0.24],
    [0.08, 0.05, 0.16],
    [0.74, 0.35, 0.36],
    [0.35, 0.26, 0.91],
    [0.91, 0.91, 0.71],
    [0.64, 0.57, 0.41],
    [0.69, 0.53, 0.74],
    [0.72, 0.61, 0.49],
    [0.71, 0.74, 0.70],
])
ARGS = ARGS[:6] + (normal_draws, uniform_draws) + ARGS[8:]
""",
            "call": call,
            "gold_call": gold,
        },
        # the small instance with detection efficiencies of 0.9, 0.8 and 0.7, so the response no
        # longer preserves counts and the sensitivities enter the iteration
        {
            "setup": _small_setup() + """redistribution = np.array([
    [0.90, 0.24, 0.14],
    [0.00, 0.56, 0.21],
    [0.00, 0.00, 0.35],
])
ARGS = (redistribution,) + ARGS[1:]
""",
            "call": call,
            "gold_call": gold,
        },
        # the small instance with a Gamma shape of 2.5 for the emitted intensity, so the conditional
        # layer is no longer exponential
        {
            "setup": _small_setup() + """gamma_shape = 2.5
ARGS = ARGS[:17] + (gamma_shape,) + ARGS[18:]
""",
            "call": call,
            "gold_call": gold,
        },
        # the small instance with a background prior shape of 0.5 and a floor of 0.5 counts
        {
            "setup": _small_setup() + """background_prior_shape = 0.5
floor = 0.5
ARGS = ARGS[:8] + (background_prior_shape,) + ARGS[9:13] + (floor,) + ARGS[14:]
""",
            "call": call,
            "gold_call": gold,
        },
        # the benchmark with a shorter window, a tighter threshold and a shorter run
        {
            "setup": _frozen_setup() + """window = 5
ratio_threshold = 1.0
run_length = 3
ARGS = ARGS[:9] + (window, ratio_threshold, run_length) + ARGS[12:]
""",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": _small_setup() + """emitted_truth = np.array([30.0, -380.0, 90.0])
# invalid: a negative emitted truth
args = ARGS[:5] + (emitted_truth,) + ARGS[6:]
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": _small_setup() + """resampled_counts = np.array([[131, 322, 55]])
# invalid: a single resample, which the reference selection rejects upstream
args = ARGS[:4] + (resampled_counts,) + ARGS[5:]
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": _small_setup() + """emitted_truth = np.array([30.0, np.inf, 90.0])
# invalid: an emitted truth that is not finite
args = ARGS[:5] + (emitted_truth,) + ARGS[6:]
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": _small_setup() + """resolution = resolution.copy()
resolution[1, 1] += 5e-6
# invalid: a resolution column 5e-6 above one, which the response composition rejects upstream
args = ARGS[:1] + (resolution,) + ARGS[2:]
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # the small instance with the counts supplied as float arrays of whole numbers, which hold integer values
        {
            "setup": _small_setup() + """ARGS = ARGS[:2] + (on_counts.astype(float), off_counts.astype(float), resampled_counts.astype(float)) + ARGS[5:]
""",
            "call": call,
            "gold_call": gold,
        },
    ]
    # Both sides receive their own deep copy of every argument, so an in-place
    # implementation on either side cannot contaminate the other's inputs.
    isolate = (
        "\nimport copy as _test_copy\n\n"
        "def _independent_inputs(_function):\n"
        "    def _invoke(*_args, **_kwargs):\n"
        "        _args, _kwargs = _test_copy.deepcopy((_args, _kwargs))\n"
        "        return _function(*_args, **_kwargs)\n"
        "    return _invoke\n\n"
        "orchestrate_prior_diagnostic = _independent_inputs(orchestrate_prior_diagnostic)\n"
        "_oracle_orchestrate_prior_diagnostic = _independent_inputs(_oracle_orchestrate_prior_diagnostic)\n"
    )
    for case in cases:
        case["setup"] += isolate
    return cases
