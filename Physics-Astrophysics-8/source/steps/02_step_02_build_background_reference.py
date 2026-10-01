"""
Build the fixed background expectation that the reference iteration folds into its forward model.

The detected spectrum of the signal run holds signal and background counts together, and a separate background run of equal exposure records the background alone. The framework never subtracts the background run from the signal run, since the difference of two Poisson variables is not Poisson and can turn negative. Instead each $\gamma$-energy bin $j$ is given a latent background expectation $b_j$ with an independent $\mathrm{Gamma}(a_0, b_0)$ prior, whose shape $a_0$ is fixed and whose rate $b_0$ is set empirically from the average level of the background run, so that the prior is broad and centred on that average while the individual bins remain governed by their own background counts $n_{\mathrm{off},j}$.

The reference iteration that seeds the empirical prior needs one fixed background expectation per bin rather than a distribution. The framework takes the mean of the posterior that the Gamma prior and the Poisson background counts of that bin imply together.

Returns
-------
np.ndarray, shape (J,). The fixed background expectation of every bin used inside the reference iteration, strictly positive in every bin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_background_reference(
    off_counts: np.ndarray,
    prior_shape: float,
) -> np.ndarray:
    r"""Compute the per-bin background reference from the background run.

    Parameters
    ----------
    off_counts : np.ndarray
        Counts of the background run, shape (J,), non-negative integer
        values, at least one of them positive, recorded with the same
        exposure as the signal run.
    prior_shape : float
        Shape parameter of the Gamma prior on the latent background
        expectation, strictly positive. The rate of that prior is not an
        input. It is fixed by the background run itself.

    Returns
    -------
    background_reference : np.ndarray
        Shape (J,). The fixed background expectation of every bin used inside
        the reference iteration, strictly positive in every bin.

    Raises
    ------
    ValueError
        If ``off_counts`` is not one-dimensional or is empty, if an entry is
        negative, not finite or not an integer value, if every entry is zero,
        or if ``prior_shape`` is not a finite positive number.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_background_reference(
    off_counts: np.ndarray,
    prior_shape: float,
) -> np.ndarray:
    n_off = np.asarray(off_counts, dtype=float)
    a0 = float(prior_shape)

    if n_off.ndim != 1 or n_off.size == 0:
        raise ValueError("off_counts must be a non-empty 1D array")
    if not np.all(np.isfinite(n_off)):
        raise ValueError("off_counts must be finite")
    if np.any(n_off < 0.0) or np.any(n_off != np.round(n_off)):
        raise ValueError("off_counts must hold non-negative integer values")
    if not np.any(n_off > 0.0):
        raise ValueError("the background run must hold at least one count")
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("prior_shape must be a finite positive number")

    # The Gamma rate is set so that the prior mean equals the average OFF count.
    rate = a0 / float(np.mean(n_off))
    # Gamma-Poisson conjugacy: the posterior of one bin is Gamma(a0 + n_off, rate + 1),
    # whose mean is the reference used inside the iteration.
    return (a0 + n_off) / (rate + 1.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "build_background_reference"),
                           ("run_gold", "_oracle_build_background_reference")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    cases = [
        # the benchmark background run, with an empty last bin
        {
            "setup": """import numpy as np
off_counts = np.array([48, 154, 51, 22, 51, 15, 1, 0])
prior_shape = 1.0
""",
            "call": "build_background_reference(off_counts, prior_shape)",
            "gold_call": "_oracle_build_background_reference(off_counts, prior_shape)",
        },
        # a flat background run
        {
            "setup": """import numpy as np
off_counts = np.array([12, 12, 12, 12, 12])
prior_shape = 1.0
""",
            "call": "build_background_reference(off_counts, prior_shape)",
            "gold_call": "_oracle_build_background_reference(off_counts, prior_shape)",
        },
        # a shape parameter other than one, so the prior weight differs from one pseudo-count
        {
            "setup": """import numpy as np
off_counts = np.array([5, 30, 240, 18, 3, 0])
prior_shape = 2.5
""",
            "call": "build_background_reference(off_counts, prior_shape)",
            "gold_call": "_oracle_build_background_reference(off_counts, prior_shape)",
        },
        # boundary: a single bin, whose reference is a weighted mean of its count and the prior mean
        {
            "setup": """import numpy as np
off_counts = np.array([9])
prior_shape = 1.0
""",
            "call": "build_background_reference(off_counts, prior_shape)",
            "gold_call": "_oracle_build_background_reference(off_counts, prior_shape)",
        },
        # boundary: exactly one count in the whole run, so the prior rate is six and every
        # empty bin is raised to one seventh of a count
        {
            "setup": """import numpy as np
off_counts = np.array([0, 0, 1, 0, 0, 0])
prior_shape = 1.0
""",
            "call": "build_background_reference(off_counts, prior_shape)",
            "gold_call": "_oracle_build_background_reference(off_counts, prior_shape)",
        },
        # edge: a very small shape parameter, so the reference is almost the OFF count itself
        {
            "setup": """import numpy as np
off_counts = np.array([40, 0, 7, 19])
prior_shape = 1e-3
""",
            "call": "build_background_reference(off_counts, prior_shape)",
            "gold_call": "_oracle_build_background_reference(off_counts, prior_shape)",
        },
        # edge: a large shape parameter, so the prior mean dominates every bin
        {
            "setup": """import numpy as np
off_counts = np.array([40, 0, 7, 19])
prior_shape = 500.0
""",
            "call": "build_background_reference(off_counts, prior_shape)",
            "gold_call": "_oracle_build_background_reference(off_counts, prior_shape)",
        },
        # edge: high statistics, where the reference tracks the counts to within a percent
        {
            "setup": """import numpy as np
off_counts = np.array([4020, 3987, 4110, 3899, 4055, 3970, 4001])
prior_shape = 1.0
""",
            "call": "build_background_reference(off_counts, prior_shape)",
            "gold_call": "_oracle_build_background_reference(off_counts, prior_shape)",
        },
        {
            "setup": """import numpy as np
# invalid: a background count that is not an integer value
off_counts = np.array([12.0, 30.5, 4.0])
prior_shape = 1.0
args = (off_counts, prior_shape)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a background run with no counts at all, which cannot fix the prior rate
off_counts = np.array([0, 0, 0, 0])
prior_shape = 1.0
args = (off_counts, prior_shape)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a prior shape of zero, which is not a Gamma distribution
off_counts = np.array([12, 30, 4])
prior_shape = 0.0
args = (off_counts, prior_shape)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a negative background count
off_counts = np.array([12, -1, 4])
prior_shape = 1.0
args = (off_counts, prior_shape)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # the benchmark background run supplied as a float array of whole numbers, which holds integer values
        {
            "setup": """import numpy as np
off_counts = np.array([48.0, 154.0, 51.0, 22.0, 51.0, 15.0, 1.0, 0.0])
prior_shape = 1.0
""",
            "call": "build_background_reference(off_counts, prior_shape)",
            "gold_call": "_oracle_build_background_reference(off_counts, prior_shape)",
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
        "build_background_reference = _independent_inputs(build_background_reference)\n"
        "_oracle_build_background_reference = _independent_inputs(_oracle_build_background_reference)\n"
    )
    for case in cases:
        case["setup"] += isolate
    return cases
