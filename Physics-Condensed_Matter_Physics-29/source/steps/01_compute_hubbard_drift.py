"""
Compute the hopping drift for the two spin-resolved phase-space matrices.

In the number-conserving fermionic Gaussian phase-space equations, the one-body hopping Hamiltonian produces a commutator-like drift within each spin sector. The interaction contribution is represented separately by diffusion, so this step preserves the source convention and the supplied matrix orientation.

Returns
-------
np.ndarray, the complex hopping drift with shape (2, n_sites, n_sites), with spin-up first and spin-down second
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_hubbard_drift(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    hopping: "np.ndarray",
    hbar: float = 1.0,
) -> "np.ndarray":
    """Compute the spin-resolved hopping drift.

    Parameters
    ----------
    n_up : np.ndarray
        Complex square spin-up phase-space matrix with shape ``(n_sites, n_sites)``.
    n_down : np.ndarray
        Complex square spin-down phase-space matrix with the same shape as ``n_up``.
    hopping : np.ndarray
        Finite square hopping matrix with shape ``(n_sites, n_sites)``.
    hbar : float, default=1.0
        Positive finite reduced Planck constant in the chosen units.

    Returns
    -------
    drift : np.ndarray
        Complex array with shape ``(2, n_sites, n_sites)``; index 0 is spin-up and
        index 1 is spin-down.

    Raises
    ------
    ValueError
        If the matrices are not finite compatible square arrays or ``hbar`` is not
        a positive finite real scalar.
    """
    return drift

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Real

import numpy as np


def _oracle_compute_hubbard_drift(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    hopping: "np.ndarray",
    hbar: float = 1.0,
) -> "np.ndarray":
    """Reference implementation."""
    up = np.asarray(n_up, dtype=complex)
    down = np.asarray(n_down, dtype=complex)
    hop = np.asarray(hopping, dtype=complex)
    if (
        up.ndim != 2
        or up.shape[0] != up.shape[1]
        or up.shape[0] < 1
        or down.shape != up.shape
        or hop.shape != up.shape
        or not np.all(np.isfinite(up))
        or not np.all(np.isfinite(down))
        or not np.all(np.isfinite(hop))
    ):
        raise ValueError("n_up, n_down, and hopping must be finite compatible square arrays")
    if isinstance(hbar, bool) or not isinstance(hbar, Real):
        raise ValueError("hbar must be a positive finite real scalar")
    hbar_value = float(hbar)
    if not math.isfinite(hbar_value) or hbar_value <= 0.0:
        raise ValueError("hbar must be a positive finite real scalar")

    drift_up = (1j / hbar_value) * (hop.T @ up - up @ hop.T)
    drift_down = (1j / hbar_value) * (hop.T @ down - down @ hop.T)
    return np.stack((drift_up, drift_down)).astype(complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic test specifications."""
    return [
        {
            "setup": """import numpy as np
n_up = np.array([
    [0.82 + 0.00j, 0.12 + 0.05j, -0.04 + 0.02j],
    [0.12 - 0.05j, 0.47 + 0.00j, 0.09 - 0.03j],
    [-0.04 - 0.02j, 0.09 + 0.03j, 0.21 + 0.00j],
], dtype=complex)
n_down = np.array([
    [0.18 + 0.00j, -0.07 + 0.02j, 0.05 - 0.01j],
    [-0.07 - 0.02j, 0.53 + 0.00j, -0.11 + 0.04j],
    [0.05 + 0.01j, -0.11 - 0.04j, 0.79 + 0.00j],
], dtype=complex)
hopping = np.array([
    [0.0, 1.0, 0.35],
    [1.0, 0.0, 0.8],
    [0.35, 0.8, 0.0],
], dtype=float)
""",
            "call": "compute_hubbard_drift(n_up.copy(), n_down.copy(), hopping.copy(), 1.0)",
            "gold_call": "_oracle_compute_hubbard_drift(n_up.copy(), n_down.copy(), hopping.copy(), 1.0)",
        },
        {
            "setup": """import numpy as np
n_up = np.array([[0.3 + 0.2j]], dtype=complex)
n_down = np.array([[0.7 - 0.1j]], dtype=complex)
hopping = np.zeros((1, 1), dtype=float)
""",
            "call": "compute_hubbard_drift(n_up.copy(), n_down.copy(), hopping.copy(), 2.0)",
            "gold_call": "_oracle_compute_hubbard_drift(n_up.copy(), n_down.copy(), hopping.copy(), 2.0)",
        },
        {
            "setup": """import numpy as np
n_up = np.array([[0.6 + 0.1j, 0.2 - 0.3j], [-0.1 + 0.05j, 0.4 - 0.2j]], dtype=complex)
n_down = np.array([[0.1 - 0.2j, -0.25 + 0.1j], [0.3 + 0.2j, 0.9 + 0.05j]], dtype=complex)
hopping = np.array([[0.2, 1.1], [-0.4, -0.3]], dtype=float)
""",
            "call": "compute_hubbard_drift(n_up.copy(), n_down.copy(), hopping.copy(), 0.75)",
            "gold_call": "_oracle_compute_hubbard_drift(n_up.copy(), n_down.copy(), hopping.copy(), 0.75)",
        },
        {
            "setup": """import numpy as np
n_up = np.eye(2, dtype=complex)
n_down = np.eye(3, dtype=complex)
hopping = np.eye(2, dtype=float)
def run_model():
    try:
        compute_hubbard_drift(n_up.copy(), n_down.copy(), hopping.copy(), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_hubbard_drift(n_up.copy(), n_down.copy(), hopping.copy(), 1.0)
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
