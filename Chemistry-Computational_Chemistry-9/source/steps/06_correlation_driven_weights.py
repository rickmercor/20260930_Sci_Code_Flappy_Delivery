"""
Implement correlation_driven_weights, which converts the correlation indices of a molecule
into the opposite-spin and same-spin weights of its spin-component-scaled MP2 energy.

Source correction: the paper's fitted parameter labels are transposed in places. The assignment consistent with its reported training-molecule weights has opposite-spin slope 1.38 and same-spin slope 2.89.

Returns
-------
np.ndarray [c_OS, c_SS] of shape (2,), dimensionless
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def correlation_driven_weights(indices: "np.ndarray") -> "np.ndarray":
    '''Opposite-spin and same-spin weights of the two-parameter correlation-driven scheme.

    Parameters
    ----------
    indices : np.ndarray
        Array [I_D, I_ND, I_T] of per-electron dynamic, nondynamic and total correlation
        indices, shape (3,), with I_D >= 0, I_ND >= 0, I_T > 0 and I_T = I_D + I_ND
        (to within 1e-8 relative).

    Returns
    -------
    weights : np.ndarray
        Array [c_OS, c_SS] of shape (2,), dimensionless.

    Raises
    ------
    ValueError
        If indices does not hold three finite values, I_D or I_ND is negative, I_T is not
        positive, or I_T differs from I_D + I_ND.
    '''
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_correlation_driven_weights(indices: "np.ndarray") -> "np.ndarray":
    indices = np.asarray(indices, dtype=float).ravel()
    if indices.size != 3 or not np.all(np.isfinite(indices)):
        raise ValueError("indices must hold three finite values")
    i_d, i_nd, i_t = indices
    if i_d < 0.0 or i_nd < 0.0 or i_t <= 0.0:
        raise ValueError("indices must be non-negative with a positive total")
    if abs(i_t - (i_d + i_nd)) > 1e-8 * i_t:
        raise ValueError("I_T must equal I_D + I_ND")
    return np.array([1.38 * i_d / i_t, 2.89 * i_nd / i_t])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid = """
def run_model():
    try:
        correlation_driven_weights(dc(indices))
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_correlation_driven_weights(dc(indices))
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: indices of a molecule dominated by dynamic correlation ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
indices = np.array([0.0703008228, 0.0152614560, 0.0855622788])
""",
            "call": "correlation_driven_weights(dc(indices))",
            "gold_call": "_oracle_correlation_driven_weights(dc(indices))",
            "tol": 1e-12,
        },
        # --- Typical: indices with a large nondynamic share (same-spin weight above 1) ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
indices = np.array([0.0930434277, 0.0707293388, 0.1637727665])
""",
            "call": "correlation_driven_weights(dc(indices))",
            "gold_call": "_oracle_correlation_driven_weights(dc(indices))",
            "tol": 1e-12,
        },
        # --- Boundary: purely dynamic correlation (I_ND = 0) ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
indices = np.array([0.05, 0.0, 0.05])
""",
            "call": "correlation_driven_weights(dc(indices))",
            "gold_call": "_oracle_correlation_driven_weights(dc(indices))",
            "tol": 1e-12,
        },
        # --- Edge: purely nondynamic correlation (I_D = 0, all occupations one half) ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
indices = np.array([0.0, 0.125, 0.125])
""",
            "call": "correlation_driven_weights(dc(indices))",
            "gold_call": "_oracle_correlation_driven_weights(dc(indices))",
            "tol": 1e-12,
        },
        # --- Invalid: an uncorrelated state (I_T = 0) has no defined weights ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
indices = np.array([0.0, 0.0, 0.0])
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: inconsistent total index ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
indices = np.array([0.07, 0.015, 0.1])
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
