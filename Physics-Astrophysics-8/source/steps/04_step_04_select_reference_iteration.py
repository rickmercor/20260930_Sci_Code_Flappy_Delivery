"""
Select the Richardson-Lucy iterate that serves as the reference of the empirical prior.

Late Richardson-Lucy iterates amplify Poisson fluctuations into structures that are not physical, while early iterates have not yet located the high- and low-intensity regions. The framework stops the iteration when the deterministic change of the resolution-limited iterate $\boldsymbol{\eta}^{(t)} = \mathbf{G}_\gamma \boldsymbol{\mu}^{(t)}$ over a fixed window of $w$ updates has dropped to the level of its own statistical noise, which it measures by rerunning the same iteration on Poisson resamples of the signal run. Both quantities are relative measures in the resolution-limited space, since that is the space in which the prior is judged.

At iteration $t$ the deterministic change is $\Delta_\eta(t; w) = \lVert \boldsymbol{\eta}^{(t)} - \boldsymbol{\eta}^{(t - w)} \rVert_2 / (\lVert \boldsymbol{\eta}^{(t)} \rVert_2 + \epsilon)$. The noise level $\mathcal{N}_\eta(t)$ is the square root of the summed sample variances, over the $K$ resampled runs with divisor $K - 1$, of the components of the resolution-limited iterate at $t$, divided by the Euclidean norm of the mean over the resampled runs of that iterate, plus the guard $\epsilon$. Each resampled run is the same iteration applied to the resampled counts, with the same background reference and its own flat start. The selected iteration is the earliest $t \ge w$ at which $\Delta_\eta(t; w) / (\mathcal{N}_\eta(t) + \epsilon) \le \tau$ holds for a run of $c$ consecutive iterations starting at $t$. If no run of qualifying iterations ends at or before the iteration cap $T$, the cap itself is the selected iteration.

Returns
-------
int, the selected iteration index between ``window`` and ``max_iterations``. It is ``max_iterations`` when no candidate qualifies, including when ``window + run_length - 1`` exceeds ``max_iterations`` so that no run of iterations fits.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_reference_iteration(
    on_counts: np.ndarray,
    resampled_counts: np.ndarray,
    response: np.ndarray,
    resolution: np.ndarray,
    background_reference: np.ndarray,
    window: int,
    ratio_threshold: float,
    run_length: int,
    max_iterations: int,
    guard: float,
) -> int:
    r"""Apply the change-to-noise stopping rule and return the selected iteration.

    Parameters
    ----------
    on_counts : np.ndarray
        Counts of the signal run, shape (J,), non-negative integer values.
    resampled_counts : np.ndarray
        Poisson resamples of the signal run, shape (K, J) with K at least 2,
        one resample per row, non-negative integer values. Their spread at
        each iteration is a sample variance with divisor K minus one.
    response : np.ndarray
        Composite detector response, shape (J, J), every column summing to
        the detection efficiency of that bin, above zero and at most one to
        within 1e-6.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), non-negative and
        finite, mapping an emitted spectrum to its resolution-limited form.
    background_reference : np.ndarray
        Fixed background expectation of every bin, shape (J,), non-negative
        and finite.
    window : int
        Number of updates between the two iterates whose difference defines
        the deterministic change, at least 1.
    ratio_threshold : float
        Largest admissible value of the change divided by the noise level,
        finite and positive.
    run_length : int
        Number of consecutive iterations, starting at the candidate, over
        which the ratio must stay at or below the threshold, at least 1.
    max_iterations : int
        Number of updates run, at least 1. It is also the fallback when no
        qualifying run of iterations exists below it.
    guard : float
        Non-negative constant added to every denominator, and to every
        predicted count inside the iteration.

    Returns
    -------
    selected : int
        The selected iteration index between ``window`` and
        ``max_iterations``. It is ``max_iterations`` when no candidate
        qualifies, including when ``window + run_length - 1`` exceeds
        ``max_iterations`` so that no run of iterations fits.

    Raises
    ------
    ValueError
        If the arrays are not of the shapes above, are empty or disagree on
        the number of bins, if there are fewer than two resamples, if a count
        is negative, not finite or not an integer value, if an entry of the
        response or of the resolution is negative or not finite, if a column
        of the response sums to zero or exceeds one by more than 1e-6, if
        ``window``,
        ``run_length`` or ``max_iterations`` is not a positive integer, if
        ``ratio_threshold`` is not finite and positive, if ``guard`` is
        negative or not finite, or if the signal run or any resample does
        not hold more counts than the background reference.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_reference_iteration(
    on_counts: np.ndarray,
    resampled_counts: np.ndarray,
    response: np.ndarray,
    resolution: np.ndarray,
    background_reference: np.ndarray,
    window: int,
    ratio_threshold: float,
    run_length: int,
    max_iterations: int,
    guard: float,
) -> int:
    n_on = np.asarray(on_counts, dtype=float)
    res = np.asarray(resampled_counts, dtype=float)
    g = np.asarray(resolution, dtype=float)

    if n_on.ndim != 1 or n_on.size == 0:
        raise ValueError("on_counts must be a non-empty 1D array")
    n_bins = n_on.size
    if res.ndim != 2 or res.shape[1] != n_bins:
        raise ValueError("resampled_counts must have shape (K, J) matching on_counts")
    if res.shape[0] < 2:
        raise ValueError("at least two resamples are needed for a sample variance")
    if not np.all(np.isfinite(res)) or np.any(res < 0.0) or np.any(res != np.round(res)):
        raise ValueError("resampled_counts must hold non-negative integer values")
    if g.ndim != 2 or g.shape != (n_bins, n_bins):
        raise ValueError("resolution must have shape (J, J) matching on_counts")
    if not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("resolution must be finite and non-negative")
    for name, val in (("window", window), ("run_length", run_length),
                      ("max_iterations", max_iterations)):
        if int(val) != val or val < 1:
            raise ValueError(f"{name} must be a positive integer")
    tau = float(ratio_threshold)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("ratio_threshold must be a finite positive number")
    eps = float(guard)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("guard must be a finite non-negative number")

    w, run, t_max = int(window), int(run_length), int(max_iterations)

    # The nominal run, monitored in the resolution-limited space.
    iterates = _oracle_run_richardson_lucy_iterates(
        n_on, response, background_reference, t_max, eps
    )
    eta = iterates @ g.T
    change = np.full(t_max + 1, np.nan)
    for t in range(w, t_max + 1):
        change[t] = np.linalg.norm(eta[t] - eta[t - w]) / (np.linalg.norm(eta[t]) + eps)

    # The resampled runs, whose spread at each iteration is the noise level.
    eta_res = np.empty((res.shape[0], t_max + 1, n_bins), dtype=float)
    for k in range(res.shape[0]):
        eta_res[k] = _oracle_run_richardson_lucy_iterates(
            res[k], response, background_reference, t_max, eps
        ) @ g.T
    mean_eta = eta_res.mean(axis=0)
    var_eta = eta_res.var(axis=0, ddof=1)
    noise = np.sqrt(var_eta.sum(axis=1)) / (np.linalg.norm(mean_eta, axis=1) + eps)
    ratio = change / (noise + eps)

    # Earliest candidate that starts a full run of qualifying iterations.
    for t in range(w, t_max - run + 2):
        block = ratio[t:t + run]
        if np.all(np.isfinite(block)) and np.all(block <= tau):
            return int(t)
    return int(t_max)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    def _benchmark_setup():
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
response = resolution @ redistribution
on_counts = np.array([381, 1541, 460, 157, 293, 39, 0, 0])
resampled_counts = np.array([
    [406, 1578, 429, 153, 292, 35, 0, 0],
    [403, 1568, 417, 147, 302, 38, 0, 0],
    [364, 1606, 456, 166, 298, 22, 0, 0],
    [358, 1573, 453, 172, 290, 35, 0, 0],
])
off_counts = np.array([48, 154, 51, 22, 51, 15, 1, 0])
background_reference = (1.0 + off_counts) / (1.0 / off_counts.mean() + 1.0)
guard = 1e-12
"""

    def _small_setup():
        # a three-bin instance with three resamples and a response that mixes the bins
        return """import numpy as np
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([120, 340, 60])
resampled_counts = np.array([
    [131, 322, 55],
    [108, 351, 71],
    [124, 338, 58],
])
background_reference = np.array([15.0, 22.5, 8.0])
guard = 1e-12
"""

    args = ("on_counts, resampled_counts, response, resolution, background_reference, "
            "window, ratio_threshold, run_length, max_iterations, guard")
    call = f"select_reference_iteration({args})"
    gold = f"_oracle_select_reference_iteration({args})"
    err = ""
    for name, function in (("run_model", "select_reference_iteration"),
                           ("run_gold", "_oracle_select_reference_iteration")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    cases = [
        {
            "setup": _benchmark_setup() + """window = 10
ratio_threshold = 2.0
run_length = 10
max_iterations = 500
""",
            "call": call,
            "gold_call": gold,
        },
        # a tighter threshold selects a later iterate of the same run
        {
            "setup": _benchmark_setup() + """window = 10
ratio_threshold = 0.5
run_length = 10
max_iterations = 500
""",
            "call": call,
            "gold_call": gold,
        },
        # a shorter window lowers the deterministic change and moves the selection earlier
        {
            "setup": _benchmark_setup() + """window = 3
ratio_threshold = 2.0
run_length = 10
max_iterations = 500
""",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": _small_setup() + """window = 5
ratio_threshold = 1.0
run_length = 4
max_iterations = 200
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a run of one iteration, so the first qualifying ratio decides
        {
            "setup": _small_setup() + """window = 5
ratio_threshold = 1.0
run_length = 1
max_iterations = 200
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a window of one update
        {
            "setup": _small_setup() + """window = 1
ratio_threshold = 1.0
run_length = 3
max_iterations = 200
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a threshold so loose that the very first candidate, the window itself, qualifies
        {
            "setup": _small_setup() + """window = 7
ratio_threshold = 1e6
run_length = 5
max_iterations = 60
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: the only run that fits ends exactly at the iteration cap
        {
            "setup": _small_setup() + """window = 7
ratio_threshold = 1e6
run_length = 5
max_iterations = 11
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a threshold so tight that no candidate ever qualifies, so the cap is returned
        {
            "setup": _small_setup() + """window = 5
ratio_threshold = 1e-9
run_length = 3
max_iterations = 80
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: the cap is too small for any run to fit, so the cap is returned untested
        {
            "setup": _small_setup() + """window = 7
ratio_threshold = 1e6
run_length = 5
max_iterations = 10
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: identical resamples give zero noise, so the guard alone keeps the ratio finite
        # and the change never falls to its level inside the cap, which is returned
        {
            "setup": _small_setup() + """resampled_counts = np.array([
    [120, 340, 60],
    [120, 340, 60],
])
window = 5
ratio_threshold = 2.0
run_length = 3
max_iterations = 100
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a diagonal resolution, which makes the monitored space the emitted space
        {
            "setup": _small_setup() + """resolution = np.eye(3)
window = 5
ratio_threshold = 1.0
run_length = 4
max_iterations = 200
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: low counts with resamples that scatter widely, where the background is a sizeable
        # fraction of the counts and has to be folded into every resampled run
        {
            "setup": """import numpy as np
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([12, 34, 6])
resampled_counts = np.array([
    [20, 25, 2],
    [5, 45, 11],
    [15, 30, 9],
    [9, 38, 4],
])
background_reference = np.array([1.5, 2.5, 0.8])
guard = 1e-12
window = 10
ratio_threshold = 0.25
run_length = 3
max_iterations = 60
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a strongly asymmetric resolution, so mapping the iterates through its transpose
        # selects a different iteration
        {
            "setup": _small_setup() + """resolution = np.array([
    [0.80, 0.35, 0.15],
    [0.20, 0.50, 0.35],
    [0.00, 0.15, 0.50],
])
window = 3
ratio_threshold = 3.0
run_length = 3
max_iterations = 60
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: an identity response with identical resamples, so the noise is exactly zero and the
        # iteration converges to machine precision inside the cap, where the guard in the ratio's
        # denominator is what lets the vanishing change qualify
        {
            "setup": """import numpy as np
response = np.eye(3)
resolution = np.eye(3)
on_counts = np.array([80, 30, 40])
resampled_counts = np.array([
    [80, 30, 40],
    [80, 30, 40],
])
background_reference = np.array([2.0, 1.0, 1.0])
guard = 1e-12
window = 5
ratio_threshold = 0.5
run_length = 3
max_iterations = 60
""",
            "call": call,
            "gold_call": gold,
        },
        # a response with detection efficiencies below one, where the sensitivities inside every
        # run, nominal and resampled alike, decide the selected iteration
        {
            "setup": """import numpy as np
response = np.array([
    [0.81, 0.16, 0.07],
    [0.09, 0.56, 0.21],
    [0.00, 0.08, 0.42],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([120, 340, 60])
resampled_counts = np.array([
    [131, 322, 55],
    [108, 351, 71],
    [124, 338, 58],
])
background_reference = np.array([15.0, 22.5, 8.0])
guard = 1e-12
window = 10
ratio_threshold = 2.0
run_length = 5
max_iterations = 200
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: two resamples, the smallest number allowed, so the sample variance has divisor one
        {
            "setup": _small_setup() + """resampled_counts = np.array([
    [131, 322, 55],
    [108, 351, 71],
])
window = 3
ratio_threshold = 1.0
run_length = 3
max_iterations = 200
""",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": """import numpy as np
# invalid: a single resample, which admits no sample variance
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([120, 340, 60])
resampled_counts = np.array([[131, 322, 55]])
background_reference = np.array([15.0, 22.5, 8.0])
args = (on_counts, resampled_counts, response, resolution, background_reference, 3, 1.0, 3, 200, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a window of zero updates, over which no change can be measured
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([120, 340, 60])
resampled_counts = np.array([
    [131, 322, 55],
    [108, 351, 71],
    [124, 338, 58],
])
background_reference = np.array([15.0, 22.5, 8.0])
args = (on_counts, resampled_counts, response, resolution, background_reference, 0, 1.0, 3, 200, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a change-to-noise threshold of zero
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([120, 340, 60])
resampled_counts = np.array([
    [131, 322, 55],
    [108, 351, 71],
    [124, 338, 58],
])
background_reference = np.array([15.0, 22.5, 8.0])
args = (on_counts, resampled_counts, response, resolution, background_reference, 3, 0.0, 3, 200, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a required run of zero iterations, which every candidate would satisfy
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([120, 340, 60])
resampled_counts = np.array([
    [131, 322, 55],
    [108, 351, 71],
    [124, 338, 58],
])
background_reference = np.array([15.0, 22.5, 8.0])
args = (on_counts, resampled_counts, response, resolution, background_reference, 3, 1.0, 0, 200, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a response column 5e-6 above one, outside the stated 1e-6 tolerance
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.100005, 0.60],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([120, 340, 60])
resampled_counts = np.array([
    [131, 322, 55],
    [108, 351, 71],
    [124, 338, 58],
])
background_reference = np.array([15.0, 22.5, 8.0])
args = (on_counts, resampled_counts, response, resolution, background_reference, 3, 1.0, 3, 200, 1e-12)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # counts and resamples supplied as float arrays of whole numbers, which hold integer values
        {
            "setup": """import numpy as np
response = np.array([
    [0.90, 0.20, 0.10],
    [0.10, 0.70, 0.30],
    [0.00, 0.10, 0.60],
])
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.80, 0.15],
    [0.00, 0.10, 0.85],
])
on_counts = np.array([120.0, 340.0, 60.0])
resampled_counts = np.array([
    [131.0, 322.0, 55.0],
    [108.0, 351.0, 71.0],
    [124.0, 338.0, 58.0],
])
background_reference = np.array([15.0, 22.5, 8.0])
window = 3
ratio_threshold = 1.0
run_length = 3
max_iterations = 200
guard = 1e-12
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
        "select_reference_iteration = _independent_inputs(select_reference_iteration)\n"
        "_oracle_select_reference_iteration = _independent_inputs(_oracle_select_reference_iteration)\n"
    )
    for case in cases:
        case["setup"] += isolate
    return cases
