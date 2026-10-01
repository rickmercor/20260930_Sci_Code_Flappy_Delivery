"""
Call public Steps 1 and 2. Require width at least 5, iterations at least 5, and positive ridge. Build the disclosed smooth random-feature basis. Use the weighted log-score regression and fixed IRLS updates only to initialize beta; then perform the stated deterministic subgradient optimization on the physically weighted DIRECT force-space absolute error. Return the best beta, descriptor standardization, its attained direct force-space objective, and flexible_checksum.

The paper replaces a global or piecewise-constant quantile by a smooth function of local descriptors and aligns the calibrated interval directly with force error. This step makes both choices numerically load-bearing.

Returns
-------
(beta: (d+width+1,), descriptor_mean: (d,), descriptor_scale: (d,), direct_objective: scalar, flex_checksum: scalar)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_flexible_quantile(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, width: int = 13, iterations: int = 31, ridge: float = 0.065) -> tuple:
    """Return smooth quantile parameters, standardization, objective, and checksum.

    Raises:
        ValueError: If model settings or an upstream panel argument is inadmissible.
    """
    return None, None, None, None, None

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


def _oracle_fit_flexible_quantile(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8, width: int = 13, iterations: int = 31, ridge: float = 0.065) -> tuple:
    if width < 5 or iterations < 5 or ridge <= 0:
        raise ValueError("width, iterations, and ridge are invalid")
    x_cal = _oracle_generate_uncertainty_panel(seed, n_cal, n_test, d)[0]
    score, error, _ = _oracle_compute_site_scores(seed, n_cal, n_test, d)
    mu = x_cal.mean(axis=0)
    sd = x_cal.std(axis=0) + 1e-12
    h = _features(x_cal, mu, sd, width)
    target = np.log(np.maximum(score, 1e-12))
    physical = 0.25 + error / np.median(error)
    beta = np.linalg.solve(h.T @ (physical[:, None] * h) + ridge * np.eye(h.shape[1]), h.T @ (physical * target))
    # The log-score fit is initialization only, not the optimized objective.
    for _ in range(iterations):
        residual = target - h @ beta
        irls = physical / np.sqrt(residual * residual + 2.5e-5)
        lhs = h.T @ (irls[:, None] * h) + ridge * np.eye(h.shape[1])
        rhs = h.T @ (irls * target)
        beta = np.linalg.solve(lhs, rhs)
    sigma = _oracle_generate_uncertainty_panel(seed, n_cal, n_test, d)[3]

    def direct_loss(coefficients):
        q = np.exp(np.clip(h @ coefficients, -2.5, 2.5))
        residual = sigma * q - error
        return float(np.mean(physical * np.abs(residual)))

    # Deterministic subgradient Adam on the physical force-space L1 objective.
    # Ridge stabilizes the warm start; it does not alter the direct objective.
    moments = np.zeros_like(beta)
    squares = np.zeros_like(beta)
    best = beta.copy()
    best_value = direct_loss(beta)
    for step in range(1, iterations * 10 + 1):
        raw = h @ beta
        clipped = np.clip(raw, -2.5, 2.5)
        predicted = sigma * np.exp(clipped)
        residual = predicted - error
        derivative = (raw > -2.5) & (raw < 2.5)
        gradient = h.T @ (physical * np.sign(residual) * predicted * derivative) / len(error)
        moments = 0.9 * moments + 0.1 * gradient
        squares = 0.999 * squares + 0.001 * gradient * gradient
        corrected_m = moments / (1.0 - 0.9 ** step)
        corrected_v = squares / (1.0 - 0.999 ** step)
        beta = beta - 0.001 * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        value = direct_loss(beta)
        if value < best_value:
            best_value = value
            best = beta.copy()
    beta = best
    objective = best_value
    checksum = float(np.dot(beta, np.sin(np.arange(len(beta)) + 0.7)))
    return beta, mu, sd, objective, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'fit_flexible_quantile(260427,96,72,8,13,31,.065)','gold_call':'_oracle_fit_flexible_quantile(260427,96,72,8,13,31,.065)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'fit_flexible_quantile(260501,40,24,6,5,5,.2)','gold_call':'_oracle_fit_flexible_quantile(260501,40,24,6,5,5,.2)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'fit_flexible_quantile(261111,120,90,10,17,19,.04)','gold_call':'_oracle_fit_flexible_quantile(261111,120,90,10,17,19,.04)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'a=(260427,96,72,8,4,31,.065)','call':'_raises_value_error(fit_flexible_quantile,*a)','gold_call':'_raises_value_error(_oracle_fit_flexible_quantile,*a)'},
    ]
