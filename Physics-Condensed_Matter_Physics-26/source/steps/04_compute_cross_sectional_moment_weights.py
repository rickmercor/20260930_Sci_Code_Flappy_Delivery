"""
Compute the cross-sectional moment-arm weight for every transverse eigenmode, using whichever physical averaging is relevant to that mode's own parity.

The thermal moment that drives bending is built from two different cross-sectional averages, each relevant only to one parity: for an odd-parity (sine) eigenfunction, the relevant weight is its depth-direction first moment, the integral of X*psi(X) dX over [-1/2, 1/2]; for an even-parity (cosine) eigenfunction, the relevant weight is its plain cross-sectional average, the integral of psi(X) dX over [-1/2, 1/2]. Each integral must be worked out directly and completely, by hand or by careful numerical quadrature, for whichever functional form, cosine or sine, matches that row's own parity tag; an integration carried out only partially, or that drops a term arising along the way, will look plausible while being numerically wrong.

Every row of the input eigenbasis must receive exactly one weight value in the output, computed using the integral appropriate to that row's own parity tag. The output must be aligned row-for-row with the input eigenbasis (same order, same length), not grouped or reordered by parity.

Returns
-------
np.ndarray of shape (N,), float: the moment-arm weight for each row of the input eigenbasis, in the same order as the input.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_cross_sectional_moment_weights(modes: np.ndarray) -> np.ndarray:
    """Per-mode cross-sectional moment-arm weight, branching internally by parity.

    Parameters
    ----------
    modes : np.ndarray
        Shape (N, 3), N >= 2, as returned by solve_transverse_eigenbasis (the two parity counts need not be equal):
        column 0 eigenvalue (> 0), column 1 normalization constant (> 0),
        column 2 parity tag (0.0 or 1.0 only).

    Returns
    -------
    weights : np.ndarray
        Array of shape (N,): the moment-arm weight for each row of
        modes, in the same order as modes.

    Raises
    ------
    ValueError
        If modes is not two-dimensional with exactly 3 columns and at
        least 2 rows; if any eigenvalue or normalization constant in modes
        is not finite and > 0; or if any parity tag is not exactly 0.0 or
        1.0.
    """
    return weights  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_cross_sectional_moment_weights(modes: np.ndarray) -> np.ndarray:
    import numpy as np
    from scipy.integrate import quad

    modes = np.asarray(modes, dtype=float)

    if modes.ndim != 2 or modes.shape[1] != 3 or modes.shape[0] < 2:
        raise ValueError("modes must have shape (N, 3) with N >= 2")
    nu_col, c_col, parity_col = modes[:, 0], modes[:, 1], modes[:, 2]
    if not np.all(np.isfinite(nu_col)) or np.any(nu_col <= 0.0):
        raise ValueError("every eigenvalue in modes must be finite and > 0")
    if not np.all(np.isfinite(c_col)) or np.any(c_col <= 0.0):
        raise ValueError("every normalization constant in modes must be finite and > 0")
    if not np.all(np.isin(parity_col, [0.0, 1.0])):
        raise ValueError("every parity tag in modes must be exactly 0.0 or 1.0")

    weights = np.empty(modes.shape[0], dtype=float)
    for i in range(modes.shape[0]):
        nu, C, parity = nu_col[i], c_col[i], parity_col[i]
        if parity == 1.0:
            val, _ = quad(lambda X: X * C * np.sin(nu * X), -0.5, 0.5)
        else:
            val, _ = quad(lambda X: C * np.cos(nu * X), -0.5, 0.5)
        weights[i] = val

    return weights

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark combined eigenbasis (normal scenario) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
modes = np.array([
    [1.10538433, 1.05162999, 0.0],
    [3.52384339, 1.34482657, 1.0],
    [6.49245235, 1.39211673, 0.0],
    [9.56707173, 1.40384712, 1.0],
])
""",
            "call": "compute_cross_sectional_moment_weights(modes.copy())",
            "gold_call": "_oracle_compute_cross_sectional_moment_weights(modes.copy())",
        },
        # --- Valid: a different Biot-number scenario's eigenpairs ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
modes = np.array([
    [1.3, 1.10, 0.0],
    [2.8, 1.30, 1.0],
    [6.6, 1.38, 0.0],
    [8.1, 1.39, 1.0],
])
""",
            "call": "compute_cross_sectional_moment_weights(modes.copy())",
            "gold_call": "_oracle_compute_cross_sectional_moment_weights(modes.copy())",
        },
        # --- Boundary: a single eigenpair of each parity ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
])
""",
            "call": "compute_cross_sectional_moment_weights(modes.copy())",
            "gold_call": "_oracle_compute_cross_sectional_moment_weights(modes.copy())",
        },
        # --- Consistency: closed form matches direct numerical quadrature (independent check) ---
        {
            "setup": """import numpy as np
from scipy.integrate import quad
modes = np.array([
    [1.10538433, 1.05162999, 0.0],
    [3.52384339, 1.34482657, 1.0],
])
def check(fn):
    w = np.asarray(fn(modes.copy()), dtype=float)
    nu_e, C_e = modes[0, 0], modes[0, 1]
    val_e, _ = quad(lambda X: C_e*np.cos(nu_e*X), -0.5, 0.5)
    nu_o, C_o = modes[1, 0], modes[1, 1]
    val_o, _ = quad(lambda X: X*C_o*np.sin(nu_o*X), -0.5, 0.5)
    return int(abs(w[0]-val_e) < 1e-8 and abs(w[1]-val_o) < 1e-8)
""",
            "call": "check(compute_cross_sectional_moment_weights)",
            "gold_call": "check(_oracle_compute_cross_sectional_moment_weights)",
        },
        # --- Invalid: shape mismatch (only 2 columns) ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_cross_sectional_moment_weights(np.array([[1.1, 1.05], [3.6, 1.35]]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_cross_sectional_moment_weights(np.array([[1.1, 1.05], [3.6, 1.35]]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a parity tag that is neither 0.0 nor 1.0 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_cross_sectional_moment_weights(np.array([[1.1, 1.05, 0.0], [3.6, 1.35, 2.0]]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_cross_sectional_moment_weights(np.array([[1.1, 1.05, 0.0], [3.6, 1.35, 2.0]]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
