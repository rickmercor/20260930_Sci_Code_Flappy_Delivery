"""
Call public Step 2. Require the miscoverage level alpha strictly between zero and one, so the target coverage is 1-alpha. Apply the paper's finite-sample conformal order statistic, clipping its rank to the available scores, to obtain the global multiplicative calibrator. Compute the mean pinball loss at quantile level 1-alpha, the level that this multiplier estimates, and return q_global, pinball, and the unchanged score checksum.

A single calibrated quantile rescales every baseline uncertainty by the same positive number. Because multiplication by one scalar preserves ordering, this route can change interval magnitude but cannot improve rank correlation.

Returns
-------
(q_global: scalar, mean_pinball: scalar, score_checksum: scalar)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_global_conformal(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, alpha: float = 0.5) -> tuple:
    """Return the finite-sample global calibrator, pinball loss, and score checksum.

    Raises:
        ValueError: If alpha or an upstream panel argument is inadmissible.
    """
    return None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _finite_quantile(values, alpha):
    v = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(v) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(v))
    return float(v[rank - 1])


def _oracle_fit_global_conformal(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, alpha: float = 0.5) -> tuple:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    score, error, checksum = _oracle_compute_site_scores(seed, n_cal, n_test, d)
    q = _finite_quantile(score, alpha)
    pinball = float(np.mean(np.where(score >= q, (1.0 - alpha) * (score - q), alpha * (q - score))))
    return q, pinball, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'fit_global_conformal(260427,96,72,8,.5)','gold_call':'_oracle_fit_global_conformal(260427,96,72,8,.5)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'fit_global_conformal(260501,40,24,6,.2)','gold_call':'_oracle_fit_global_conformal(260501,40,24,6,.2)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'fit_global_conformal(261111,120,90,10,.75)','gold_call':'_oracle_fit_global_conformal(261111,120,90,10,.75)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'a=(260427,96,72,8,1.0)','call':'_raises_value_error(fit_global_conformal,*a)','gold_call':'_raises_value_error(_oracle_fit_global_conformal,*a)'},
    ]
