"""
Return the probability that a female of the given total length in metres is mature, the logistic function m(z) = 1 / (1 + exp(-(intercept + slope * z))), which increases with length when slope is positive, evaluated element-wise.

Maturity at length is a logistic ogive; with the default coefficients half of the females are mature at 0.297 m, so a subadult cohort introduced at 0.30 m is already about half mature.

Returns
-------
numpy.ndarray of float64, same shape as lengths: probabilities of being mature.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def maturity_probability(lengths: "numpy.ndarray", intercept: float = -7.41,
                         slope: float = 24.98) -> "numpy.ndarray":
    """Return the probability that a female of the given total length in metres is mature, the logistic function m(z) = 1 / (1 + exp(-(intercept + slope * z))), which increases with length when slope is positive, evaluated element-wise.

    Parameters
    ----------
    lengths : numpy.ndarray
        One-dimensional array of finite, nonnegative total lengths in metres.
    intercept : float
        Intercept of the logistic argument; default -7.41.
    slope : float
        Slope of the logistic argument per metre; default 24.98.

    Returns
    -------
    maturity : numpy.ndarray
        Array of the same shape as lengths holding maturity probabilities in (0, 1) (float64).

    Raises
    ------
    ValueError
        If lengths is not a non-empty one-dimensional array of finite nonnegative values, or a coefficient is not finite.
    """
    return maturity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_maturity_probability(lengths: "numpy.ndarray", intercept: float = -7.41,
                                 slope: float = 24.98) -> "numpy.ndarray":
    """Logistic maturity-at-length m(z) = 1 / (1 + exp(-(intercept + slope * z))), increasing in z."""
    z = np.asarray(lengths, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("lengths must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("lengths must be finite and nonnegative")
    if not (np.isfinite(intercept) and np.isfinite(slope)):
        raise ValueError("intercept and slope must be finite")
    # increasing with length: 50% maturity at z = -intercept/slope = 0.2966 m. Table 1 prints the
    # logistic with the argument sign flipped, which would make large fish immature; the authors'
    # code and every result in the paper use the increasing form.
    return 1.0 / (1.0 + np.exp(-(float(intercept) + float(slope) * z)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nlengths = np.array([0.300, 0.867, 1.224])\n",
            "call": "maturity_probability(lengths)",
            "gold_call": "_oracle_maturity_probability(lengths)",
        },
        {
            "setup": "import numpy as np\nlengths = np.linspace(0.01, 1.5, 300)\n",
            "call": "maturity_probability(lengths)",
            "gold_call": "_oracle_maturity_probability(lengths)",
        },
        {
            "setup": "import numpy as np\nlengths = np.array([0.2, 0.3, 0.4])\nintercept, slope = -6.0, 20.0\n",
            "call": "maturity_probability(lengths, intercept, slope)",
            "gold_call": "_oracle_maturity_probability(lengths, intercept, slope)",
        },
        {
            "setup": "import numpy as np\nlengths = np.array([0.3, -0.2, 0.9])\ndef run_model():\n    try:\n        maturity_probability(lengths)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_maturity_probability(lengths)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
