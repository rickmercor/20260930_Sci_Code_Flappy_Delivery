"""
Generate the deterministic descriptor, force, and baseline-uncertainty panel from the delivered specification. Require integer n_cal, n_test, and d with n_cal at least 40, n_test at least 24, and d at least 6. Preserve every default_rng draw in the stated order. Return x_cal, predicted and reference calibration forces, sigma_cal, x_test, predicted and reference test forces, and sigma_test.

This synthetic panel represents local atomic environments, predicted force vectors, reference forces, and a positive heuristic uncertainty. It is a disclosed test fixture, not a reconstruction of the paper's datasets.

Returns
-------
(x_cal: (n_cal,d), pred_cal: (n_cal,3), ref_cal: (n_cal,3), sigma_cal: (n_cal,), x_test: (n_test,d), pred_test: (n_test,3), ref_test: (n_test,3), sigma_test: (n_test,))
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def generate_uncertainty_panel(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    """Return deterministic calibration and shifted-test force panels.

    Raises:
        ValueError: If a requested panel size or descriptor dimension is inadmissible.
    """
    return None, None, None, None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _validate_panel(n_cal, n_test, d):
    if not isinstance(n_cal, int) or not isinstance(n_test, int) or not isinstance(d, int):
        raise ValueError("sizes must be integers")
    if n_cal < 40 or n_test < 24 or d < 6:
        raise ValueError("n_cal >= 40, n_test >= 24, and d >= 6 are required")


def _oracle_generate_uncertainty_panel(seed: int = 260427, n_cal: int = 96, n_test: int = 72, d: int = 8) -> tuple:
    _validate_panel(n_cal, n_test, d)
    rng = np.random.default_rng(seed)
    w = np.sin((np.arange(d * 3, dtype=float).reshape(d, 3) + 1.0) * 0.371)
    x_cal = rng.normal(0.0, 1.0, (n_cal, d))
    x_test = rng.normal(0.0, 1.0, (n_test, d))
    x_test += 0.18 * np.tanh(x_test[:, [0]]) * np.cos(np.arange(d)[None, :] + 0.5)

    def make_forces(x, z):
        pred = x @ w / np.sqrt(d) + 0.12 * np.column_stack(
            (np.sin(x[:, 0] * x[:, 1]), np.cos(x[:, 2] - x[:, 3]), np.tanh(x[:, 4] + x[:, 5]))
        )
        sigma = 0.075 + 0.019 * np.linalg.norm(x[:, :3], axis=1) + 0.013 * np.abs(np.sin(x[:, 3] + 0.4 * x[:, 5]))
        local = 0.72 + 0.28 * np.exp(0.35 * x[:, 0] - 0.18 * x[:, 2]) + 0.11 * np.abs(x[:, 6 % d])
        direction = z / np.linalg.norm(z, axis=1, keepdims=True)
        magnitude = sigma * local * (0.62 + 0.28 * np.abs(z[:, 0]) + 0.10 * np.abs(z[:, 1]))
        ref = pred + magnitude[:, None] * direction
        return pred, ref, sigma

    pred_cal, ref_cal, sigma_cal = make_forces(x_cal, rng.normal(size=(n_cal, 3)))
    pred_test, ref_test, sigma_test = make_forces(x_test, rng.normal(size=(n_test, 3)))
    return x_cal, pred_cal, ref_cal, sigma_cal, x_test, pred_test, ref_test, sigma_test

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'generate_uncertainty_panel(260427,96,72,8)','gold_call':'_oracle_generate_uncertainty_panel(260427,96,72,8)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'generate_uncertainty_panel(260501,40,24,6)','gold_call':'_oracle_generate_uncertainty_panel(260501,40,24,6)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'np=__import__("numpy")','call':'generate_uncertainty_panel(261111,120,90,10)','gold_call':'_oracle_generate_uncertainty_panel(261111,120,90,10)'},
      {'tol':1e-9,'setup':'import numpy as np\n\ndef _raises_value_error(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n\ndef _flat(value):\n    if isinstance(value, (tuple, list)):\n        out = []\n        for item in value:\n            out.extend(_flat(item))\n        return out\n    return np.asarray(value, dtype=float).ravel().tolist()\n'+'a=(260427,39,72,8)','call':'_raises_value_error(generate_uncertainty_panel,*a)','gold_call':'_raises_value_error(_oracle_generate_uncertainty_panel,*a)'},
    ]
