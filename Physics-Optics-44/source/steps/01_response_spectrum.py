"""
Return the spectral transfer function of the medium's nonlocal response on the given angular-frequency grid. The response in real space is the normalized Gaussian with characteristic length sigma, so that convolving an intensity profile with it is a multiplication on this grid. Raise ValueError if sigma is not strictly positive.

The nonlocal response is what distinguishes this medium from a purely local Kerr one: the index change at a point is an average of the intensity over a neighbourhood whose size is sigma. Because the response is Gaussian and normalized to unit area, its transform is real, positive and equal to one at zero frequency, so the total power entering the nonlinear term is preserved and only its spatial distribution is smoothed.

Returns
-------
ndarray of float64 with the same shape as k: the spectral transfer function.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def response_spectrum(k: 'np.ndarray', sigma: float) -> 'np.ndarray':
    """Return the spectral transfer function of the medium's nonlocal response on the given angular-frequency grid. The response in real space is the normalized Gaussian with characteristic length sigma, so that convolving an intensity profile with it is a multiplication on this grid. Raise ValueError if sigma is not strictly positive.

    Returns
    -------
    ndarray of float64 with the same shape as k: the spectral transfer function.

    Raises
    ------
    ValueError
        If sigma is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_response_spectrum(k: "np.ndarray", sigma: float) -> "np.ndarray":
    k = np.asarray(k, dtype=float)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return np.exp(-0.25 * (k * sigma) ** 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np\nk = np.linspace(-4.0, 4.0, 9)",
                    "call": "response_spectrum(k.copy(), 4.0)",
                    "gold_call": "_oracle_response_spectrum(k, 4.0)"
            },
            {
                    "setup": "import numpy as np\nk = np.array([0.0, 0.5, 2.0])",
                    "call": "response_spectrum(k.copy(), 0.8)",
                    "gold_call": "_oracle_response_spectrum(k, 0.8)"
            },
            {
                    "setup": "import numpy as np\nk = np.array([0.0])",
                    "call": "response_spectrum(k.copy(), 1e-6)",
                    "gold_call": "_oracle_response_spectrum(k, 1e-6)"
            },
            {
                    "setup": "import numpy as np\ndef zero_freq(fn):\n    return float(fn(np.array([0.0]), 3.0)[0])",
                    "call": "zero_freq(response_spectrum)",
                    "gold_call": "zero_freq(_oracle_response_spectrum)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        response_spectrum(np.array([0.0, 1.0]), 0.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_response_spectrum(np.array([0.0, 1.0]), 0.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
