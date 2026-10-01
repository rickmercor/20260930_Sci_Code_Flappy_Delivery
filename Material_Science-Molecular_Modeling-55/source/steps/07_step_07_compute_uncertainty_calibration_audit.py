"""
This is the final orchestrator. Require an integer seed. Call public Steps 1 through 6 with the stated default path, including the full 4 by 3 by 3 tensor. From the three risk channels compute improvement, median_risk, population risk_spread, worst_risk, leading_singular, transfer_risk, and J exactly as specified. Return J rounded to eight decimals together with q_global, four checksums, the flexible direct-loss objective, and all six risk diagnostics. The protected implementation must make the identical chain using protected Steps 1 through 6 only.

The orchestrator reduces the same complete path tested by integration. The source-derived score, global quantile, conditional partition, smooth calibration, and shifted evaluation all affect the returned J.

Returns
-------
(J: scalar, q_global: scalar, score_checksum: scalar, class_checksum: scalar, flexible_objective: scalar, flexible_checksum: scalar, tensor_checksum: scalar, improvement: scalar, median_risk: scalar, risk_spread: scalar, worst_risk: scalar, leading_singular: scalar, transfer_risk: scalar)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_uncertainty_calibration_audit(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    """Return J and twelve named audit diagnostics from the full default path.

    Raises:
        ValueError: If seed or an upstream argument is inadmissible.
    """
    return None, None, None, None, None, None, None, None, None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_uncertainty_calibration_audit(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    panel = _oracle_generate_uncertainty_panel(int(seed), n_cal, n_test, d)
    score, error, score_checksum = _oracle_compute_site_scores(int(seed), n_cal, n_test, d)
    q_global, pinball, _ = _oracle_fit_global_conformal(int(seed), n_cal, n_test, d)
    q_class, _, _, _, _, _, class_checksum = _oracle_fit_class_conformal(int(seed), n_cal, n_test, d)
    beta, _, _, objective, flex_checksum = _oracle_fit_flexible_quantile(int(seed), n_cal, n_test, d)
    tensor, tensor_checksum = _oracle_evaluate_transfer_tensor(int(seed), n_cal, n_test, d)
    risks = tensor[:, [0, 5, 10]]
    flex = risks[:, 2]
    improvement = float(np.mean(risks[:, 0] - flex))
    median = float(np.median(flex))
    spread = float(np.std(flex, ddof=0))
    worst = float(np.max(flex))
    singular = float(np.linalg.svd(risks - risks.mean(axis=0), compute_uv=False)[0])
    transfer = float(np.mean(flex.reshape(4, 3, 3)[-1]))
    j = float(np.exp(-(1.7 * improvement + 0.8 * median + 0.6 * spread + 0.25 * worst + 0.15 * transfer + 0.04 * singular)) / (1.0 + objective + pinball))
    j = float(np.round(j, 8))
    return (j, q_global, score_checksum, class_checksum, objective, flex_checksum,
            tensor_checksum, improvement, median, spread, worst, singular, transfer)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'compute_uncertainty_calibration_audit(260427,96,72,8)','gold_call':'_oracle_compute_uncertainty_calibration_audit(260427,96,72,8)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'compute_uncertainty_calibration_audit(260501,40,24,6)','gold_call':'_oracle_compute_uncertainty_calibration_audit(260501,40,24,6)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'compute_uncertainty_calibration_audit(261111,120,90,10)','gold_call':'_oracle_compute_uncertainty_calibration_audit(261111,120,90,10)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'a=("bad",96,72,8)','call':'_raises_value_error(compute_uncertainty_calibration_audit,*a)','gold_call':'_raises_value_error(_oracle_compute_uncertainty_calibration_audit,*a)'},
    ]
