"""
Summarise a set of spectral draws by their mean curve and a simultaneous global rank-envelope band.

Each draw is a whole spectrum over the $\gamma$-energy bins, and the draws are strongly correlated across bins, so a band built bin by bin from marginal quantiles would not be simultaneous. The framework instead ranks whole curves by their joint extremeness. In every bin the draws receive their one-based rank $r$ among the $N$ draws of that bin, the smallest value ranking first and ties receiving the average of the ranks they span, and the rank is folded into the two-sided rank $\min\{r, N + 1 - r\}$, so that a value extreme in either direction ranks low. A curve is then ordered by the extreme rank length rule. Its two-sided ranks are sorted in ascending order and two curves are compared lexicographically, the curve whose first differing sorted rank is the smaller being the more extreme, and, among curves with identical sorted ranks, the curve with the lower row index counting as the more extreme. The band retains the $\lceil m N \rceil$ least extreme curves, $m$ being the envelope mass, and its lower and upper edges in every bin are the minimum and the maximum over the retained curves. The mean over all $N$ draws serves as the central curve.

Returns
-------
np.ndarray, shape (3, J). Row 0 is the mean over all N draws, row 1 the bin-wise minimum over the retained curves and row 2 the bin-wise maximum over the retained curves.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_global_rank_envelope(
    draws: np.ndarray,
    mass: float,
) -> np.ndarray:
    r"""Build the mean curve and the simultaneous rank-envelope band of a set of draws.

    Parameters
    ----------
    draws : np.ndarray
        Spectral draws, shape (N, J) with N at least 2, finite, one draw per
        row and one gamma-energy bin per column.
    mass : float
        Envelope mass level, strictly greater than 0 and at most 1. The
        number of retained curves is the ceiling of ``mass`` times N.

    Returns
    -------
    envelope : np.ndarray
        Shape (3, J). Row 0 is the mean over all N draws, row 1 the
        bin-wise minimum over the retained curves and row 2 the bin-wise
        maximum over the retained curves.

    Raises
    ------
    ValueError
        If ``draws`` is not a two-dimensional array with at least two rows
        and one column, if an entry is not finite, or if ``mass`` is not a
        finite number in the half-open interval (0, 1].
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _midranks(values):
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    sorted_v = v[order]
    ranks = np.empty(v.size, dtype=float)
    start = 0
    while start < v.size:
        stop = start + 1
        while stop < v.size and sorted_v[stop] == sorted_v[start]:
            stop += 1
        # Tied values share the mean of the one-based ranks they span.
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _oracle_build_global_rank_envelope(
    draws: np.ndarray,
    mass: float,
) -> np.ndarray:
    x = np.asarray(draws, dtype=float)
    m = float(mass)

    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("draws must have shape (N, J) with at least two draws")
    if not np.all(np.isfinite(x)):
        raise ValueError("draws must be finite")
    if not np.isfinite(m) or m <= 0.0 or m > 1.0:
        raise ValueError("mass must lie in (0, 1]")

    n_draws, n_bins = x.shape
    two_sided = np.empty((n_draws, n_bins), dtype=float)
    for j in range(n_bins):
        r = _midranks(x[:, j])
        two_sided[:, j] = np.minimum(r, (n_draws + 1) - r)

    # Extreme rank length: sort each curve's two-sided ranks, then order the
    # curves lexicographically, the smallest sorted vector being the most
    # extreme. lexsort takes its primary key last.
    sorted_ranks = np.sort(two_sided, axis=1)
    keys = tuple(sorted_ranks[:, c] for c in range(n_bins - 1, -1, -1))
    order = np.lexsort(keys)

    n_keep = int(np.ceil(m * n_draws))
    n_keep = max(1, min(n_draws, n_keep))
    kept = x[order[n_draws - n_keep:], :]

    centre = x.mean(axis=0)
    return np.vstack([centre, kept.min(axis=0), kept.max(axis=0)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    def _draws_setup():
        # sixteen draws over eight bins with the heavy tails typical of prior draws
        return """import numpy as np
draws = np.array([
    [12.4, 380.5, 61.2, 8.9, 22.1, 3.4, 0.21, 0.031],
    [140.2, 1210.7, 95.4, 40.3, 88.6, 1.1, 0.05, 0.120],
    [58.9, 2650.1, 410.8, 3.2, 130.4, 9.8, 0.33, 0.009],
    [7.7, 155.0, 18.6, 25.7, 411.0, 0.4, 0.02, 0.044],
    [301.6, 990.3, 240.0, 66.1, 9.7, 2.9, 0.48, 0.210],
    [19.5, 415.8, 720.4, 12.5, 57.3, 6.6, 0.11, 0.015],
    [88.0, 70.2, 33.9, 150.8, 202.5, 0.8, 0.09, 0.057],
    [2.3, 3120.9, 12.1, 0.9, 5.4, 12.5, 1.02, 0.180],
    [64.4, 640.6, 155.2, 31.0, 44.8, 4.7, 0.16, 0.026],
    [410.7, 205.4, 86.3, 4.4, 310.2, 0.2, 0.07, 0.003],
    [33.1, 1580.2, 22.7, 19.8, 71.9, 8.1, 0.25, 0.098],
    [5.9, 530.0, 305.5, 92.6, 16.0, 1.7, 0.60, 0.012],
    [150.8, 845.9, 44.0, 2.1, 250.7, 0.6, 0.03, 0.066],
    [26.0, 2210.4, 130.9, 48.2, 3.3, 5.3, 0.14, 0.140],
    [97.3, 1005.1, 66.6, 14.6, 96.4, 2.2, 0.37, 0.021],
    [44.8, 290.7, 505.2, 7.3, 33.5, 0.9, 0.08, 0.075],
])
"""

    call = "build_global_rank_envelope(draws, mass)"
    gold = "_oracle_build_global_rank_envelope(draws, mass)"
    err = ""
    for name, function in (("run_model", "build_global_rank_envelope"),
                           ("run_gold", "_oracle_build_global_rank_envelope")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    cases = [
        {
            "setup": _draws_setup() + "mass = 0.75\n",
            "call": call,
            "gold_call": gold,
        },
        # a higher mass level retains all but one curve
        {
            "setup": _draws_setup() + """mass = 0.9
""",
            "call": call,
            "gold_call": gold,
        },
        # a low mass level retains only the four least extreme curves
        {
            "setup": _draws_setup() + """mass = 0.25
""",
            "call": call,
            "gold_call": gold,
        },
        # curves that tie on their minimum two-sided rank and are separated only by the
        # lexicographic comparison of their later sorted ranks
        {
            "setup": """import numpy as np
draws = np.array([
    [1.0, 5.0, 3.0, 4.0],
    [2.0, 1.0, 5.0, 3.0],
    [3.0, 2.0, 1.0, 5.0],
    [4.0, 3.0, 2.0, 1.0],
    [5.0, 4.0, 4.0, 2.0],
    [2.5, 2.5, 2.5, 2.5],
])
mass = 0.5
""",
            "call": call,
            "gold_call": gold,
        },
        # tied values inside a bin, which receive the mean of the ranks they span
        {
            "setup": """import numpy as np
draws = np.array([
    [1.0, 2.0, 2.0],
    [1.0, 2.0, 5.0],
    [3.0, 2.0, 5.0],
    [3.0, 7.0, 1.0],
    [9.0, 7.0, 1.0],
])
mass = 0.6
""",
            "call": call,
            "gold_call": gold,
        },
        # tied values in both bins, where mid-ranks, first-of-tie ranks and ordinal ranks retain
        # different curves
        {
            "setup": """import numpy as np
draws = np.array([
    [4.0, 2.0],
    [2.0, 3.0],
    [1.0, 1.0],
    [3.0, 4.0],
    [4.0, 1.0],
])
mass = 0.5
""",
            "call": call,
            "gold_call": gold,
        },
        # tie at the cut: rows 6 and 7 share the sorted ranks (1, 2) and only one of them fits
        # into the six retained curves, so the lower row index counts as the more extreme
        {
            "setup": """import numpy as np
draws = np.array([
    [10.0, 10.0],
    [20.0, 20.0],
    [30.0, 30.0],
    [40.0, 40.0],
    [50.0, 50.0],
    [60.0, 60.0],
    [70.0, 80.0],
    [80.0, 70.0],
])
mass = 0.75
""",
            "call": call,
            "gold_call": gold,
        },
        # many draws over few bins, where identical sorted rank vectors are common and two of
        # them straddle the cut
        {
            "setup": """import numpy as np
draws = np.array([
    [15.0, 17.0, 23.0],
    [7.0, 38.0, 39.0],
    [25.0, 4.0, 4.0],
    [17.0, 5.0, 10.0],
    [36.0, 22.0, 20.0],
    [26.0, 30.0, 14.0],
    [39.0, 26.0, 18.0],
    [18.0, 34.0, 22.0],
    [22.0, 35.0, 28.0],
    [19.0, 31.0, 13.0],
    [10.0, 39.0, 16.0],
    [3.0, 25.0, 24.0],
])
mass = 0.75
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: full mass retains every curve, so the band is the bin-wise range of all draws
        {
            "setup": """import numpy as np
draws = np.array([
    [1.0, 5.0, 3.0],
    [2.0, 1.0, 5.0],
    [3.0, 2.0, 1.0],
    [4.0, 3.0, 2.0],
])
mass = 1.0
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a mass level whose product with the number of draws is an integer
        {
            "setup": """import numpy as np
draws = np.array([
    [1.0, 5.0, 3.0],
    [2.0, 1.0, 5.0],
    [3.0, 2.0, 1.0],
    [4.0, 3.0, 2.0],
    [0.5, 4.0, 4.0],
    [6.0, 0.5, 0.5],
    [2.5, 2.5, 2.5],
    [3.5, 3.5, 3.5],
])
mass = 0.5
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a mass level just above a multiple of one over N rounds up to one more curve
        {
            "setup": """import numpy as np
draws = np.array([
    [1.0, 5.0, 3.0],
    [2.0, 1.0, 5.0],
    [3.0, 2.0, 1.0],
    [4.0, 3.0, 2.0],
    [0.5, 4.0, 4.0],
    [6.0, 0.5, 0.5],
    [2.5, 2.5, 2.5],
    [3.5, 3.5, 3.5],
])
mass = 0.51
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: three draws, where the sorted rank vectors alone decide the single dropped curve
        {
            "setup": """import numpy as np
draws = np.array([
    [10.0, 1.0, 4.0],
    [3.0, 6.0, 2.0],
    [5.0, 5.0, 5.0],
])
mass = 0.6
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a single bin, so the envelope is the range of the scalars after both extremes are dropped
        {
            "setup": """import numpy as np
draws = np.array([[3.0], [9.0], [1.0], [4.0], [12.0], [5.5], [0.2]])
mass = 0.7
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a tiny mass level still retains one curve
        {
            "setup": """import numpy as np
draws = np.array([
    [1.0, 5.0, 3.0],
    [2.0, 1.0, 5.0],
    [3.0, 2.0, 1.0],
    [4.0, 3.0, 2.0],
    [2.5, 2.5, 2.5],
])
mass = 1e-3
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: identical draws, which tie in every bin and give a band of zero width
        {
            "setup": """import numpy as np
draws = np.tile(np.array([4.0, 8.0, 15.0]), (5, 1))
mass = 0.6
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: two draws, which always tie in every bin, so the lower row index counts as the more
        # extreme and the single retained curve is the second row
        {
            "setup": """import numpy as np
draws = np.array([
    [3.0, 9.0, 4.0],
    [7.0, 2.0, 8.0],
])
mass = 0.5
""",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": """import numpy as np
# invalid: an envelope mass above one
draws = np.array([
    [1.0, 5.0, 3.0],
    [2.0, 1.0, 5.0],
    [3.0, 2.0, 1.0],
    [4.0, 3.0, 2.0],
])
args = (draws, 1.5)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a single draw, which no band can be built from
draws = np.array([[1.0, 5.0, 3.0]])
args = (draws, 0.75)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a draw holding a value that is not finite
draws = np.array([
    [1.0, 5.0, 3.0],
    [2.0, np.nan, 5.0],
    [3.0, 2.0, 1.0],
    [4.0, 3.0, 2.0],
])
args = (draws, 0.75)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
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
        "build_global_rank_envelope = _independent_inputs(build_global_rank_envelope)\n"
        "_oracle_build_global_rank_envelope = _independent_inputs(_oracle_build_global_rank_envelope)\n"
    )
    for case in cases:
        case["setup"] += isolate
    return cases
