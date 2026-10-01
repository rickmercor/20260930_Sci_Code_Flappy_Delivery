"""
Return the annual survival probability of female fish of the given total lengths in metres, s(z) = exp(-slope * w ** exponent) with w the body weight in grams, obtained from the length-weight relation of step 01 with its default coefficients (kilograms converted to grams), evaluated element-wise.

Natural mortality of fish declines with body size; a power-law (Lorenzen-type) dependence of annual survival on weight in grams is the usual parameterisation and makes small recruits far more vulnerable than adults.

Returns
-------
numpy.ndarray of float64, same shape as lengths: annual survival probabilities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def survival_probability(lengths: "numpy.ndarray", slope: float = 2.7,
                         exponent: float = -0.315) -> "numpy.ndarray":
    """Return the annual survival probability of female fish of the given total lengths in metres, s(z) = exp(-slope * w ** exponent) with w the body weight in grams, obtained from the length-weight relation of step 01 with its default coefficients (kilograms converted to grams), evaluated element-wise.

    Parameters
    ----------
    lengths : numpy.ndarray
        One-dimensional array of finite, strictly positive total lengths in metres.
    slope : float
        Positive coefficient multiplying the weight power; default 2.7.
    exponent : float
        Exponent of the weight in grams; default -0.315.

    Returns
    -------
    survival : numpy.ndarray
        Array of the same shape as lengths holding annual survival probabilities in [0, 1] (float64).

    Raises
    ------
    ValueError
        If lengths is not a non-empty one-dimensional array of finite positive values, slope is not positive and finite, or exponent is not finite.
    """
    return survival

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_survival_probability(lengths: "numpy.ndarray", slope: float = 2.7,
                                 exponent: float = -0.315) -> "numpy.ndarray":
    """Annual survival s(z) = exp(-slope * w^exponent) with w the body weight in grams."""
    z = np.asarray(lengths, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("lengths must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("lengths must be finite and strictly positive")
    if not (np.isfinite(slope) and np.isfinite(exponent)) or slope <= 0.0:
        raise ValueError("slope must be positive and finite, exponent finite")
    # Lorenzen-type mortality is parameterised in GRAMS; the length-weight relation returns kg
    grams = 1000.0 * _oracle_length_to_weight(z)
    return np.exp(-float(slope) * grams ** float(exponent))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nlengths = np.array([0.300, 0.867, 1.224])\n",
            "call": "survival_probability(lengths)",
            "gold_call": "_oracle_survival_probability(lengths)",
        },
        {
            "setup": "import numpy as np\nlengths = np.linspace(0.01, 1.5, 300)\n",
            "call": "survival_probability(lengths)",
            "gold_call": "_oracle_survival_probability(lengths)",
        },
        {
            "setup": "import numpy as np\nlengths = np.array([0.2, 0.6, 1.0])\nslope, exponent = 3.30, -0.261\n",
            "call": "survival_probability(lengths, slope, exponent)",
            "gold_call": "_oracle_survival_probability(lengths, slope, exponent)",
        },
        {
            "setup": "import numpy as np\nlengths = np.array([0.3, -0.2, 0.9])\ndef run_model():\n    try:\n        survival_probability(lengths)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_survival_probability(lengths)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
