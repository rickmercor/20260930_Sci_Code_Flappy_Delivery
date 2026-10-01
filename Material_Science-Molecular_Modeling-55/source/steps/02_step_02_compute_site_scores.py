"""
Call public Step 1 and compute the source-defined site-specific force calibration score for every calibration environment. Use the Euclidean norm of the force-vector discrepancy and the positive baseline uncertainty exactly as defined in the paper. Return the scores, force-error magnitudes, and the specified index-weighted score checksum.

The paper specializes conformal calibration to atom-centred force uncertainty. This score is the common target used by every downstream calibration method and is therefore load-bearing.

Returns
-------
(score: (n_cal,), force_error: (n_cal,), score_checksum: scalar)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_site_scores(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    """Return atomwise force scores, force errors, and their checksum.

    Raises:
        ValueError: If an upstream panel argument is inadmissible.
    """
    return None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_site_scores(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    panel = _oracle_generate_uncertainty_panel(seed, n_cal, n_test, d)
    x_cal, pred_cal, ref_cal, sigma_cal = panel[:4]
    error = np.linalg.norm(pred_cal - ref_cal, axis=1)
    score = error / sigma_cal
    checksum = float(np.dot(score, 1.0 + (np.arange(n_cal) % 13) / 17.0))
    return score, error, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'compute_site_scores(260427,96,72,8)','gold_call':'_oracle_compute_site_scores(260427,96,72,8)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'compute_site_scores(260501,40,24,6)','gold_call':'_oracle_compute_site_scores(260501,40,24,6)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'compute_site_scores(261111,120,90,10)','gold_call':'_oracle_compute_site_scores(261111,120,90,10)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'a=(260427,96,20,8)','call':'_raises_value_error(compute_site_scores,*a)','gold_call':'_raises_value_error(_oracle_compute_site_scores,*a)'},
    ]
