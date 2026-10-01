"""
Call public Steps 1, 3, 4, and 5. Require each supplied tensor axis to contain at least two values. Evaluate every descriptor shift, uncertainty scale, and functional-correction scale in C order. At each cell apply the global scalar, the maximum-responsibility class scale, and the smooth descriptor scale; for each method store risk, coverage, stable-rank Spearman correlation, windowed high-error selection accuracy, and top-k precision. Return the complete fifteen-channel tensor and tensor_checksum.

The source evaluates calibration under local-environment and functional distribution shifts. This benchmark tensor couples that scientific setting to all three fitted calibration strategies rather than treating transfer as a side narrative.

Returns
-------
(transfer_tensor: (len(shifts)*len(uncertainty_scales)*len(error_scales),15), tensor_checksum: scalar)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_transfer_tensor(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, shifts: tuple = (-0.45,-0.15,0.2,0.55), uncertainty_scales: tuple = (0.78,1.0,1.24), correction_scales: tuple = (0.88,1.07,1.31)) -> tuple:
    """Return the complete transfer-risk tensor and its checksum.

    Raises:
        ValueError: If a tensor axis or upstream argument is inadmissible.
    """
    return None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _features(x, mu, sd, width):
    z = (x - mu) / sd
    d = z.shape[1]
    grid = np.arange(d * width, dtype=float).reshape(d, width)
    w = np.sin(0.173 * (grid + 1.0)) + 0.37 * np.cos(0.113 * (grid + 3.0))
    w /= np.sqrt(d)
    b = 0.31 * np.cos(np.arange(width) * 0.47)
    return np.column_stack((np.ones(len(z)), z, np.tanh(z @ w + b)))


def _spearman(a, b):
    ra = np.empty(len(a), dtype=float)
    rb = np.empty(len(b), dtype=float)
    ra[np.argsort(a, kind="mergesort")] = np.arange(len(a), dtype=float)
    rb[np.argsort(b, kind="mergesort")] = np.arange(len(b), dtype=float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(np.dot(ra, rb) / np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))


def _oracle_evaluate_transfer_tensor(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8,
                             shifts: tuple = (-0.45, -0.15, 0.2, 0.55),
                             uncertainty_scales: tuple = (0.78, 1.0, 1.24),
                             correction_scales: tuple = (0.88, 1.07, 1.31)) -> tuple:
    if min(len(shifts), len(uncertainty_scales), len(correction_scales)) < 2:
        raise ValueError("each tensor axis must contain at least two values")
    panel = _oracle_generate_uncertainty_panel(seed, n_cal, n_test, d)
    x_test, pred_test, ref_test, sigma_test = panel[4:]
    source_error_vector = pred_test - ref_test
    q_global = _oracle_fit_global_conformal(seed, n_cal, n_test, d)[0]
    q_class, means, variances, priors, class_mu, class_sd, _ = _oracle_fit_class_conformal(seed, n_cal, n_test, d)
    beta, flex_mu, flex_sd, _, _ = _oracle_fit_flexible_quantile(seed, n_cal, n_test, d)
    direction = np.cos(np.arange(d) * 0.61 + 0.4)
    rows = []
    for shift in shifts:
        x = x_test + shift * direction
        z = (x - class_mu) / class_sd
        logp = np.empty((len(x), len(q_class)))
        for k in range(len(q_class)):
            logp[:, k] = np.log(priors[k] + 1e-15) - 0.5 * np.sum(
                np.log(2.0 * np.pi * variances[k]) + (z - means[k]) ** 2 / variances[k], axis=1
            )
        q_cb = q_class[np.argmax(logp, axis=1)]
        h = _features(x, flex_mu, flex_sd, 13)
        q_flex = np.exp(np.clip(h @ beta, -2.5, 2.5))
        for us in uncertainty_scales:
            sigma = sigma_test * us * (1.0 + 0.045 * shift * shift)
            for es in correction_scales:
                correction_direction = np.column_stack((
                    np.sin(x[:, 0] + x[:, 2]),
                    np.cos(x[:, 1] - x[:, 3]),
                    np.tanh(x[:, 4] + x[:, 5]),
                ))
                correction = (0.08 * es * (1.0 + 0.25 * shift * np.tanh(x[:, 0])))[:, None] * correction_direction
                error = np.linalg.norm(source_error_vector - correction, axis=1)
                predictions = (q_global * sigma, q_cb * sigma, q_flex * sigma)
                metrics = []
                for prediction in predictions:
                    alignment = np.mean(np.abs(prediction - error)) / np.mean(error)
                    coverage = np.mean(error <= prediction)
                    rho = _spearman(prediction, error)
                    hits = []
                    for window_size in (6, 9):
                        for start in range(0, len(error), window_size):
                            stop = min(start + window_size, len(error))
                            if stop - start >= 2:
                                hits.append(float(np.argmax(prediction[start:stop]) == np.argmax(error[start:stop])))
                    window_accuracy = float(np.mean(hits))
                    top_k = min(max(2, int(np.ceil(len(error) / 6))), len(error) - 1)
                    pred_top = np.argsort(-prediction, kind="mergesort")[:top_k]
                    error_top = np.argsort(-error, kind="mergesort")[:top_k]
                    top_k_precision = len(np.intersect1d(pred_top, error_top)) / top_k
                    risk = (0.35 * alignment + 0.20 * abs(coverage - 0.5)
                            + 0.15 * (1.0 - rho) / 2.0
                            + 0.15 * (1.0 - window_accuracy)
                            + 0.15 * (1.0 - top_k_precision))
                    metrics.append((risk, coverage, rho, window_accuracy, top_k_precision))
                rows.append([v for triple in metrics for v in triple])
    tensor = np.asarray(rows, dtype=float)
    checksum = float(np.dot(tensor.ravel(), 1.0 + (np.arange(tensor.size) % 29) / 31.0))
    return tensor, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'evaluate_transfer_tensor()','gold_call':'_oracle_evaluate_transfer_tensor()'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'evaluate_transfer_tensor(260501,40,24,6,(-.2,.3),(.9,1.1),(.8,1.2))','gold_call':'_oracle_evaluate_transfer_tensor(260501,40,24,6,(-.2,.3),(.9,1.1),(.8,1.2))'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'evaluate_transfer_tensor(261111,120,91,10,(-.6,-.1,.4),(1.,1.3),(.7,1.,1.4))','gold_call':'_oracle_evaluate_transfer_tensor(261111,120,91,10,(-.6,-.1,.4),(1.,1.3),(.7,1.,1.4))'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'bad=(0.0,)','call':'_raises_value_error(evaluate_transfer_tensor,260427,96,72,8,bad,(.8,1.0),(.9,1.1))','gold_call':'_raises_value_error(_oracle_evaluate_transfer_tensor,260427,96,72,8,bad,(.8,1.0),(.9,1.1))'},
    ]
