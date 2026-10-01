"""
Call public Steps 1 and 2, standardize the calibration descriptors, and fit the specified diagonal Gaussian mixture for exactly 17 EM updates. Require 2 <= n_classes <= min(20,n_cal//4) and the miscoverage level alpha strictly between zero and one. Recompute responsibilities from the final mixture parameters before assigning calibration labels. Apply the paper's class-conditional quantile rule within each component and use the global quantile only for a component with fewer than three members. Return class quantiles, mixture parameters, standardization arrays, and class_checksum.

The paper's discrete conditional construction replaces one global scale by a step function over local-environment classes. The mixture and its fixed iteration count are disclosed deterministic conventions used to instantiate that idea.

Returns
-------
(q_class: (n_classes,), means: (n_classes,d), variances: (n_classes,d), priors: (n_classes,), descriptor_mean: (d,), descriptor_scale: (d,), class_checksum: scalar)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_class_conformal(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, n_classes: int = 7, alpha: float = 0.5) -> tuple:
    """Return class calibrators, mixture parameters, standardization, and checksum.

    Raises:
        ValueError: If alpha, class count, or an upstream panel argument is inadmissible.
    """
    return None, None, None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _finite_quantile(values, alpha):
    v = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(v) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(v))
    return float(v[rank - 1])


def _gmm_partition(x, n_classes, iterations=17):
    n, d = x.shape
    if n_classes < 2 or n_classes > min(20, n // 4):
        raise ValueError("n_classes must be between 2 and min(20, n//4)")
    mu0 = x.mean(axis=0)
    sd0 = x.std(axis=0) + 1e-12
    z = (x - mu0) / sd0
    projection = z @ (np.cos(np.arange(d) * 0.73 + 0.2) / np.sqrt(d))
    order = np.argsort(projection, kind="mergesort")
    starts = ((np.arange(n_classes) + 0.5) * n / n_classes).astype(int)
    means = z[order[np.clip(starts, 0, n - 1)]].copy()
    variances = np.tile(np.var(z, axis=0) + 0.15, (n_classes, 1))
    priors = np.full(n_classes, 1.0 / n_classes)
    for _ in range(iterations):
        logp = np.empty((n, n_classes))
        for k in range(n_classes):
            logp[:, k] = np.log(priors[k] + 1e-15) - 0.5 * np.sum(
                np.log(2.0 * np.pi * variances[k]) + (z - means[k]) ** 2 / variances[k], axis=1
            )
        logp -= logp.max(axis=1, keepdims=True)
        resp = np.exp(logp)
        resp /= resp.sum(axis=1, keepdims=True)
        nk = resp.sum(axis=0) + 1e-9
        priors = nk / nk.sum()
        means = (resp.T @ z) / nk[:, None]
        for k in range(n_classes):
            variances[k] = (resp[:, k, None] * (z - means[k]) ** 2).sum(axis=0) / nk[k] + 0.025
    # Assign labels using the mixture returned after the last M-step.
    for k in range(n_classes):
        logp[:, k] = np.log(priors[k] + 1e-15) - 0.5 * np.sum(
            np.log(2.0 * np.pi * variances[k]) + (z - means[k]) ** 2 / variances[k], axis=1
        )
    labels = np.argmax(logp, axis=1)
    return labels, means, variances, priors, mu0, sd0


def _oracle_fit_class_conformal(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, n_classes: int = 7, alpha: float = 0.5) -> tuple:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    x_cal = _oracle_generate_uncertainty_panel(seed, n_cal, n_test, d)[0]
    score = _oracle_compute_site_scores(seed, n_cal, n_test, d)[0]
    labels, means, variances, priors, mu0, sd0 = _gmm_partition(x_cal, n_classes)
    q_class = np.empty(n_classes)
    global_q = _finite_quantile(score, alpha)
    for k in range(n_classes):
        local = score[labels == k]
        q_class[k] = _finite_quantile(local, alpha) if len(local) >= 3 else global_q
    checksum = float(np.dot(q_class, np.arange(1, n_classes + 1)) + np.dot(priors, np.arange(n_classes) ** 2))
    return q_class, means, variances, priors, mu0, sd0, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'fit_class_conformal(260427,96,72,8,7,.5)','gold_call':'_oracle_fit_class_conformal(260427,96,72,8,7,.5)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'fit_class_conformal(260501,40,24,6,5,.3)','gold_call':'_oracle_fit_class_conformal(260501,40,24,6,5,.3)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'fit_class_conformal(261111,120,90,10,10,.7)','gold_call':'_oracle_fit_class_conformal(261111,120,90,10,10,.7)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'a=(260427,96,72,8,1,.5)','call':'_raises_value_error(fit_class_conformal,*a)','gold_call':'_raises_value_error(_oracle_fit_class_conformal,*a)'},
    ]
