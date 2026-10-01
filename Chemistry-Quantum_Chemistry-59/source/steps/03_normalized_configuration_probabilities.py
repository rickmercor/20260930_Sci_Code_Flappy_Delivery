"""
Convert signed log-amplitudes into stable normalized probabilities.

A neural quantum state may represent each configuration coefficient through a sign and a logarithmic absolute amplitude. Selection probabilities depend on the squared amplitudes, so their logarithms are twice the supplied log-amplitudes. Subtracting the largest log-probability before exponentiation preserves the normalized distribution while preventing overflow and underflow.

Returns
-------
A one-dimensional array containing one normalized nonnegative configuration probability per input entry, in the original order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normalized_configuration_probabilities(
    signs: 'np.ndarray',
    logabs: 'np.ndarray',
) -> 'np.ndarray':
    """Return normalized squared amplitudes without losing dynamic range.

    Returns
    -------
    np.ndarray
        One probability per configuration, in input order.

    Raises
    ------
    ValueError
        If arrays are empty, incompatible, non-finite or contain invalid signs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_normalized_configuration_probabilities(
    signs: 'np.ndarray',
    logabs: 'np.ndarray',
) -> 'np.ndarray':

    s = np.asarray(signs, dtype=float)
    x = np.asarray(logabs, dtype=float)

    if (
        s.ndim != 1
        or x.ndim != 1
        or s.size == 0
        or s.shape != x.shape
    ):
        raise ValueError(
            "signs and logabs must be matching nonempty vectors"
        )
    if np.any(~np.isfinite(s)) or np.any(~np.isfinite(x)):
        raise ValueError("inputs must be finite")
    if np.any(np.abs(s) != 1.0):
        raise ValueError("signs must be exactly -1 or +1")

    log_prob = 2.0 * x
    shifted = np.exp(log_prob - np.max(log_prob))

    return shifted / np.sum(shifted)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ns=np.array([1.,-1.,1.,-1.]); x=np.array([-2.,-.3,-1.1,-.7])",
            "call": "normalized_configuration_probabilities(s.copy(),x.copy())",
            "gold_call": "_oracle_normalized_configuration_probabilities(s,x)",
        },
        {
            "setup": "import numpy as np\ns=np.array([-1.]); x=np.array([700.])",
            "call": "normalized_configuration_probabilities(s.copy(),x.copy())",
            "gold_call": "_oracle_normalized_configuration_probabilities(s,x)",
        },
        {
            "setup": "import numpy as np\ns=np.array([1.,-1.,1.]); x=np.array([1000.,0.,-1000.])",
            "call": "normalized_configuration_probabilities(s.copy(),x.copy())",
            "gold_call": "_oracle_normalized_configuration_probabilities(s,x)",
        },
        {
            "setup": "import numpy as np\ns=np.array([-1.,1.,-1.,1.,1.]); x=np.zeros(5)",
            "call": "normalized_configuration_probabilities(s.copy(),x.copy())",
            "gold_call": "_oracle_normalized_configuration_probabilities(s,x)",
        },
        {
            "setup": "import numpy as np\ns=np.array([1.,0.]); x=np.array([0.,-1.])\ndef check(fn):\n try: fn(s.copy(),x.copy())\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(normalized_configuration_probabilities)",
            "gold_call": "check(_oracle_normalized_configuration_probabilities)",
        },
    ]
