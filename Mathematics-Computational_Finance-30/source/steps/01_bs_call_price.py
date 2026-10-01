"""
Evaluate the European call pricing function that the source states in its Section 2 preliminaries, in exactly the parameterisation given there. The third argument is the dispersion parameter as that section defines it; do not re-scale it. Inputs broadcast against one another. The zero-dispersion and zero-strike limits must return the correct limiting value rather than a division error.

The source works throughout on a normalised driver rather than on a spot process, so its pricing function carries no drift and no discount factor. Section 2 states the exact form and says which quantity the third argument denotes; footnote 2 of that section flags that other texts use a different quantity in the same slot.

Returns
-------
A float64 array of call values broadcast to the common shape of the three inputs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bs_call_price(spot: "float | np.ndarray", strike: "float | np.ndarray", total_var: "float | np.ndarray") -> "np.ndarray":
    """Evaluate the source's European call pricing function in exactly its own parameterisation.

    Parameters
    ----------
    spot : float or numpy.ndarray
        Spot level(s) of the driver; strictly positive.
    strike : float or numpy.ndarray
        Strike(s); non-negative.
    total_var : float or numpy.ndarray
        Dispersion parameter(s) in the source's Section 2 parameterisation; non-negative.

    Returns
    -------
    prices : numpy.ndarray
        float64 array of call values broadcast to the common shape of the three inputs.

    Raises
    ------
    ValueError
        If any dispersion is negative, any spot is not strictly positive, any strike is negative, or any input is not finite.
    """
    return prices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.stats import norm


def _oracle_bs_call_price(spot: "float | np.ndarray", strike: "float | np.ndarray", total_var: "float | np.ndarray") -> "np.ndarray":
    """Paper eq (3). total_var is TOTAL variance v, not a volatility.

    Call(s,k,v) = s N(d+) - k N(d-),  d+- = (-ln(k/s) +- v/2) / sqrt(v)
    """
    sp = np.asarray(spot, dtype=np.float64)
    st = np.asarray(strike, dtype=np.float64)
    tv = np.asarray(total_var, dtype=np.float64)
    if np.any(tv < 0.0):
        raise ValueError("total_var must be non-negative")
    if np.any(st < 0.0) or np.any(sp <= 0.0):
        raise ValueError("spot must be positive and strike non-negative")
    if not (np.all(np.isfinite(sp)) and np.all(np.isfinite(st)) and np.all(np.isfinite(tv))):
        raise ValueError("spot, strike and total_var must all be finite")

    shape = np.broadcast(sp, st, tv).shape
    s, k, v = (np.atleast_1d(np.broadcast_to(x, shape)).astype(np.float64)
               for x in (sp, st, tv))
    out = np.maximum(s - k, 0.0).astype(np.float64)
    live = (v > 0.0) & (k > 0.0) & (s > 0.0)
    if np.any(live):
        sv = np.sqrt(v[live])
        dp = (-np.log(k[live] / s[live]) + 0.5 * sv ** 2) / sv
        dm = (-np.log(k[live] / s[live]) - 0.5 * sv ** 2) / sv
        out[live] = s[live] * norm.cdf(dp) - k[live] * norm.cdf(dm)
    return out.reshape(shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            'setup': 'import numpy as np',
            'call': 'bs_call_price(1.0, 1.0, 0.04)',
            'gold_call': '_oracle_bs_call_price(1.0, 1.0, 0.04)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.7, 0.9, 1.0, 1.1, 1.3])\n',
            'call': 'bs_call_price(1.0, K, 0.075)',
            'gold_call': '_oracle_bs_call_price(1.0, K, 0.075)',
        },
        {
            'setup': 'import numpy as np',
            'call': 'bs_call_price(1.0, 1.2, 0.0)',
            'gold_call': '_oracle_bs_call_price(1.0, 1.2, 0.0)',
        },
        {
            'setup': 'import numpy as np',
            'call': 'bs_call_price(1.0, 0.0, 0.09)',
            'gold_call': '_oracle_bs_call_price(1.0, 0.0, 0.09)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:\n        bs_call_price(1.0, 1.0, -0.1)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_bs_call_price(1.0, 1.0, -0.1)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': 'run_oracle()',
        },
    ]
