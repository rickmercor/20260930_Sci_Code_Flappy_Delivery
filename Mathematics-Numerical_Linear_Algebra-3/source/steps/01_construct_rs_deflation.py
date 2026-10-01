"""
Transform the supplied operator-learning data into the deterministic basis representation required by the downstream numerical solver.

The source paper uses DeepONet outputs to construct deflation information for parametric linear systems. In the recycling-solution approach, learned solution information is converted into vectors that can be incorporated into the deflation space. The supplied branch and trunk data represent the fixed online information needed for this construction.

Returns
-------
return np.empty((T.shape[0], B.shape[0]), dtype=np.float64)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_rs_deflation(
    B: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Construct the tentative solution-space representation from branch and trunk outputs.

    Parameters
    ----------
    B : np.ndarray
        Two-dimensional branch-output array. The first dimension indexes the
        supplied branch instances/components, and the second dimension indexes
        the shared latent output width. The array must contain finite
        floating-point values.
    T : np.ndarray
        Two-dimensional trunk-output array. The first dimension indexes the
        supplied spatial/evaluation locations, and the second dimension must
        have the same latent output width as ``B``. The array must contain
        finite floating-point values.

    Returns
    -------
    np.ndarray
        A two-dimensional ``float64`` array with shape
        ``(T.shape[0], B.shape[0])``. Rows correspond to the supplied
        evaluation locations and columns correspond to the supplied branch
        components.

    Raises
    ------
    ValueError
        If ``B`` or ``T`` is not two-dimensional.
        If ``B`` or ``T`` has zero rows or zero columns.
        If the latent/output widths of ``B`` and ``T`` are incompatible.
        If either input contains a non-finite value.
    TypeError
        If either argument cannot be interpreted as a numerical NumPy array.
    """
    return np.empty((T.shape[0], B.shape[0]), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_construct_rs_deflation(
    B: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Reference implementation of the deterministic RS construction."""
    B = np.asarray(B, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    if B.ndim != 2 or T.ndim != 2:
        raise ValueError("B and T must be 2D arrays.")

    if B.shape[1] != T.shape[1]:
        raise ValueError("B and T must have the same number of modes.")

    if B.shape[0] < 1 or T.shape[0] < 1:
        raise ValueError("B and T must be non-empty.")

    return T @ B.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

B = np.array([
    [1.15, 1.30, 1.45, 1.60],
    [1.30, 1.60, 1.90, 2.20],
], dtype=np.float64)

T = np.array([
    [1.0, 0.125, 0.015625, 0.001953125],
    [1.0, 0.25, 0.0625, 0.015625],
    [1.0, 0.5, 0.25, 0.125],
    [1.0, 1.0, 1.0, 1.0],
], dtype=np.float64)
""",
            "call": "construct_rs_deflation(B, T)",
            "gold_call": "_oracle_construct_rs_deflation(B, T)",
        },
        {
            "setup": """
import numpy as np

B = np.array([
    [1.0, 2.0],
    [2.0, 4.0],
], dtype=np.float64)

T = np.eye(3, 2, dtype=np.float64)
""",
            "call": "construct_rs_deflation(B, T)",
            "gold_call": "_oracle_construct_rs_deflation(B, T)",
        },
        {
            "setup": """
import numpy as np

B = np.array([[2.0]], dtype=np.float64)
T = np.array([[3.0], [4.0], [5.0]], dtype=np.float64)
""",
            "call": "construct_rs_deflation(B, T)",
            "gold_call": "_oracle_construct_rs_deflation(B, T)",
        },
    ]
