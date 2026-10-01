"""
Turn the selected Richardson-Lucy iterate into the centre and the bin-wise width of the empirical prior.

The prior on the emitted spectrum is centred, bin by bin, on the selected iterate. A bin whose iterate is zero or far below one count would otherwise pin a mean-preserving prior at zero, so the iterate is first floored at a small count level $\epsilon_{\mathrm{RL}}$, and it is the floored spectrum $\boldsymbol{\mu}_{\mathrm{RL}}$ that serves as the centre. The width of the prior is set in the resolution-limited space, by mapping the floored centre through the resolution operator, $\boldsymbol{\eta}_{\mathrm{RL}} = \mathbf{G}_\gamma \boldsymbol{\mu}_{\mathrm{RL}}$, and reading the result in two ways. The relative shape of the resolution-limited reference within the spectrum sets a shape width $\sigma_{\mathrm{shape},j} = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) / (1 + \eta_{\mathrm{RL},j} / \bar\eta_{\mathrm{RL}})$, with $\bar\eta_{\mathrm{RL}}$ the mean of the reference over the bins, which is narrow where the reference is prominent and wide where it is weak. The absolute count level of the same reference decides how much of that shape information is trusted, through a local activation factor $a_j = \eta_{\mathrm{RL},j} / (\eta_{\mathrm{RL},j} + C_{\mathrm{ref}})$ that is one half at the count scale $C_{\mathrm{ref}}$ and vanishes where the reference carries almost no counts. The width of a bin is $\sigma_j = (1 - a_j)\, \sigma_{\max} + a_j\, \sigma_{\mathrm{shape},j}$, so that a bin which is large only relative to its neighbours but small in absolute counts keeps the upper bound.

Returns
-------
np.ndarray, shape (2, J). Row 0 is the floored centre, every entry at least ``floor``. Row 1 is the width of every bin, lying between ``sigma_min`` and ``sigma_max``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_prior_centre_and_width(
    reference_iterate: np.ndarray,
    resolution: np.ndarray,
    floor: float,
    sigma_min: float,
    sigma_max: float,
    count_scale: float,
) -> np.ndarray:
    r"""Floor the selected iterate and derive the adaptive prior width of every bin.

    Parameters
    ----------
    reference_iterate : np.ndarray
        The selected Richardson-Lucy iterate in emitted space, shape (J,),
        non-negative and finite, in counts.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), non-negative and
        finite, every column summing to one to within 1e-6.
    floor : float
        Lowest admissible centre of any bin, in counts, finite and strictly
        positive.
    sigma_min : float
        Lower bound of the log-scale width, finite and strictly positive.
    sigma_max : float
        Upper bound of the log-scale width, finite and at least
        ``sigma_min``.
    count_scale : float
        Resolution-limited count level at which the shape information is
        half activated, finite and strictly positive.

    Returns
    -------
    centre_and_width : np.ndarray
        Shape (2, J). Row 0 is the floored centre, every entry at least
        ``floor``. Row 1 is the width of every bin, lying between
        ``sigma_min`` and ``sigma_max``.

    Raises
    ------
    ValueError
        If the arrays are not of the shapes above, are empty or disagree on
        the number of bins, if an entry is negative or not finite, if a
        column of the resolution does not sum to one to within 1e-6, or if
        ``floor``, ``sigma_min``, ``sigma_max`` or ``count_scale`` violates
        the conditions above.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_prior_centre_and_width(
    reference_iterate: np.ndarray,
    resolution: np.ndarray,
    floor: float,
    sigma_min: float,
    sigma_max: float,
    count_scale: float,
) -> np.ndarray:
    mu_raw = np.asarray(reference_iterate, dtype=float)
    g = np.asarray(resolution, dtype=float)
    floor_v, s_min, s_max, c_ref = map(float, (floor, sigma_min, sigma_max, count_scale))

    if mu_raw.ndim != 1 or mu_raw.size == 0:
        raise ValueError("reference_iterate must be a non-empty 1D array")
    n_bins = mu_raw.size
    if g.ndim != 2 or g.shape != (n_bins, n_bins):
        raise ValueError("resolution must have shape (J, J) matching reference_iterate")
    for name, arr in (("reference_iterate", mu_raw), ("resolution", g)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be non-negative")
    if not np.allclose(g.sum(axis=0), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("every column of resolution must sum to one")
    if not np.isfinite(floor_v) or floor_v <= 0.0:
        raise ValueError("floor must be a finite positive number")
    if not np.isfinite(s_min) or s_min <= 0.0:
        raise ValueError("sigma_min must be a finite positive number")
    if not np.isfinite(s_max) or s_max < s_min:
        raise ValueError("sigma_max must be finite and at least sigma_min")
    if not np.isfinite(c_ref) or c_ref <= 0.0:
        raise ValueError("count_scale must be a finite positive number")

    # The floor is applied first, and the resolution-limited reference is built
    # from the floored centre.
    centre = np.maximum(mu_raw, floor_v)
    eta_ref = g @ centre
    eta_mean = float(np.mean(eta_ref))

    # Shape-dependent schedule, narrow where the reference is prominent.
    sigma_shape = s_min + (s_max - s_min) / (1.0 + eta_ref / eta_mean)
    # Local activation by absolute count level, half activated at count_scale.
    activation = eta_ref / (eta_ref + c_ref)
    sigma = (1.0 - activation) * s_max + activation * sigma_shape
    return np.vstack([centre, sigma])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    def _resolution_setup():
        return """import numpy as np
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
"""

    args = "reference_iterate, resolution, floor, sigma_min, sigma_max, count_scale"
    call = f"build_prior_centre_and_width({args})"
    gold = f"_oracle_build_prior_centre_and_width({args})"
    err = ""
    for name, function in (("run_model", "build_prior_centre_and_width"),
                           ("run_gold", "_oracle_build_prior_centre_and_width")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    cases = [
        # the benchmark iterate, whose two tail bins lie far below the floor
        {
            "setup": _resolution_setup() + """reference_iterate = np.array([42.563007, 1811.339584, 408.571456, 115.808072, 451.533414,
                              9.288457, 0.000013, 0.000001])
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
""",
            "call": call,
            "gold_call": gold,
        },
        # an iterate with no bin below the floor, so the floor changes nothing
        {
            "setup": _resolution_setup() + """reference_iterate = np.array([210.0, 1500.0, 380.0, 120.0, 95.0, 260.0, 40.0, 12.0])
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
""",
            "call": call,
            "gold_call": gold,
        },
        # a four-bin grid with a broader resolution and a different count scale
        {
            "setup": """import numpy as np
resolution = np.array([
    [0.80, 0.15, 0.05, 0.00],
    [0.20, 0.70, 0.15, 0.05],
    [0.00, 0.15, 0.70, 0.20],
    [0.00, 0.00, 0.10, 0.75],
])
reference_iterate = np.array([3.0, 400.0, 25.0, 0.02])
floor = 0.5
sigma_min = 0.3
sigma_max = 5.0
count_scale = 30.0
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: equal width bounds make every bin's width that common value whatever the counts
        {
            "setup": """import numpy as np
resolution = np.eye(5)
reference_iterate = np.array([1.0, 10.0, 100.0, 1000.0, 0.0])
floor = 0.1
sigma_min = 1.0
sigma_max = 1.0
count_scale = 100.0
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a diagonal resolution, so the resolution-limited reference is the centre itself
        # and one bin sits exactly at the count scale
        {
            "setup": """import numpy as np
resolution = np.eye(4)
reference_iterate = np.array([100.0, 300.0, 100.0, 300.0])
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a flat reference, where every bin equals the mean and all widths coincide
        {
            "setup": """import numpy as np
resolution = np.eye(3)
reference_iterate = np.array([50.0, 50.0, 50.0])
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a bin exactly at the floor is left where it is
        {
            "setup": """import numpy as np
resolution = np.eye(3)
reference_iterate = np.array([0.1, 20.0, 0.1])
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: an iterate that is zero everywhere, so the floor sets a flat centre
        {
            "setup": """import numpy as np
resolution = np.eye(4)
reference_iterate = np.zeros(4)
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a single bin, which is its own mean
        {
            "setup": """import numpy as np
resolution = np.array([[1.0]])
reference_iterate = np.array([250.0])
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a very large count scale switches the shape information off almost everywhere
        {
            "setup": """import numpy as np
resolution = np.eye(4)
reference_iterate = np.array([5.0, 900.0, 60.0, 0.3])
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 1e7
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a tiny count scale activates the shape schedule fully in every bin
        {
            "setup": """import numpy as np
resolution = np.eye(4)
reference_iterate = np.array([5.0, 900.0, 60.0, 0.3])
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 1e-6
""",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": """import numpy as np
# invalid: an upper width bound below the lower one
resolution = np.eye(4)
reference_iterate = np.array([5.0, 900.0, 60.0, 0.3])
args = (reference_iterate, resolution, 0.1, 3.0, 1.0, 100.0)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a floor of zero, which cannot lift an empty bin off zero
resolution = np.eye(4)
reference_iterate = np.array([5.0, 900.0, 60.0, 0.0])
args = (reference_iterate, resolution, 0.0, 1.0, 3.0, 100.0)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a negative count in the selected iterate
resolution = np.eye(4)
reference_iterate = np.array([5.0, 900.0, -60.0, 0.3])
args = (reference_iterate, resolution, 0.1, 1.0, 3.0, 100.0)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a count scale of zero, at which the activation is undefined
resolution = np.eye(4)
reference_iterate = np.array([5.0, 900.0, 60.0, 0.3])
args = (reference_iterate, resolution, 0.1, 1.0, 3.0, 0.0)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a resolution column 5e-6 above one, outside the stated 1e-6 tolerance
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.800005, 0.15],
    [0.00, 0.10, 0.85],
])
reference_iterate = np.array([5.0, 900.0, 60.0])
args = (reference_iterate, resolution, 0.1, 1.0, 3.0, 100.0)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # boundary: a resolution column 5e-7 above one, inside the stated tolerance
        {
            "setup": """import numpy as np
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.8000005, 0.15],
    [0.00, 0.10, 0.85],
])
reference_iterate = np.array([5.0, 900.0, 60.0])
floor = 0.1
sigma_min = 1.0
sigma_max = 3.0
count_scale = 100.0
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
        "build_prior_centre_and_width = _independent_inputs(build_prior_centre_and_width)\n"
        "_oracle_build_prior_centre_and_width = _independent_inputs(_oracle_build_prior_centre_and_width)\n"
    )
    for case in cases:
        case["setup"] += isolate
    return cases
