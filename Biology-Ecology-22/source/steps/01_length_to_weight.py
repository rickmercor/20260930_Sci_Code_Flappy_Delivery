"""
Return the body weight in kilograms of female fish of the given total lengths in metres from the allometric length-weight relation log10(weight in kg) = intercept + slope * log10(length in m), evaluated element-wise (base-10 logarithms).

Fish weight scales allometrically with length, close to the cube; a log-log linear fit is the standard summary and converts the length distribution that an integral projection model tracks into biomass, which is what density dependence acts on in fisheries models.

Returns
-------
numpy.ndarray of float64, same shape as lengths: body weights in kilograms.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def length_to_weight(lengths: "numpy.ndarray", intercept: float = 1.02,
                     slope: float = 3.02) -> "numpy.ndarray":
    """Return the body weight in kilograms of female fish of the given total lengths in metres from the allometric length-weight relation log10(weight in kg) = intercept + slope * log10(length in m), evaluated element-wise (base-10 logarithms).

    Parameters
    ----------
    lengths : numpy.ndarray
        One-dimensional array of finite, strictly positive total lengths in metres.
    intercept : float
        Intercept of the log10-log10 relation; default 1.02.
    slope : float
        Slope of the log10-log10 relation; default 3.02.

    Returns
    -------
    weights : numpy.ndarray
        Array of the same shape as lengths holding the weights in kilograms (float64).

    Raises
    ------
    ValueError
        If lengths is not a non-empty one-dimensional array of finite positive values, or a coefficient is not finite.
    """
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_length_to_weight(lengths: "numpy.ndarray", intercept: float = 1.02,
                             slope: float = 3.02) -> "numpy.ndarray":
    """Length-weight relation of Table 1: log10(weight in kg) = intercept + slope * log10(length in m)."""
    z = np.asarray(lengths, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("lengths must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("lengths must be finite and strictly positive")
    if not (np.isfinite(intercept) and np.isfinite(slope)):
        raise ValueError("intercept and slope must be finite")
    # log10, not ln: the source's own check "10 adult fish ~ 68 kg" only holds for log10
    return 10.0 ** (float(intercept) + float(slope) * np.log10(z))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nlengths = np.array([0.300, 0.867, 1.224])\n",
            "call": "length_to_weight(lengths)",
            "gold_call": "_oracle_length_to_weight(lengths)",
        },
        {
            "setup": "import numpy as np\nlengths = np.linspace(0.01, 1.5, 300)\n",
            "call": "length_to_weight(lengths)",
            "gold_call": "_oracle_length_to_weight(lengths)",
        },
        {
            "setup": "import numpy as np\nlengths = np.array([0.5, 1.0, 1.5])\nintercept, slope = 0.95, 3.1\n",
            "call": "length_to_weight(lengths, intercept, slope)",
            "gold_call": "_oracle_length_to_weight(lengths, intercept, slope)",
        },
        {
            "setup": "import numpy as np\nlengths = np.array([0.3, 0.0, 0.9])\ndef run_model():\n    try:\n        length_to_weight(lengths)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_length_to_weight(lengths)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
