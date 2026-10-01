"""
Evaluate the source direct partition-function estimator for a trajectory ensemble.

The direct stochastic estimator converts trajectories launched from an ordered basis into a single partition-function estimate. Apply the source estimator convention consistently across the starting configurations and their independent samples. The trajectory arrays must not be mutated.

Returns
-------
Return one finite floating-point partition estimate for the supplied trajectory ensemble.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def direct_partition_estimate(
    basis_states: np.ndarray,
    final_states: np.ndarray,
    path_amplitudes: np.ndarray,
) -> float:
    """Evaluate the source direct estimator for a trajectory ensemble.

    Axis 0 indexes starting basis configurations and axis 1 their independent
    trajectories. Return one finite scalar without mutating the trajectory
    arrays.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_direct_partition_estimate(
    basis_states: np.ndarray,
    final_states: np.ndarray,
    path_amplitudes: np.ndarray,
) -> float:
    import numpy as np

    raw_basis = np.asarray(basis_states, dtype=float)
    raw_final = np.asarray(final_states, dtype=float)
    amplitudes = np.asarray(path_amplitudes, dtype=float)
    if raw_basis.ndim != 2 or len(raw_basis) == 0:
        raise ValueError("basis_states must be a nonempty matrix")
    if np.any((raw_basis != 0) & (raw_basis != 1)):
        raise ValueError("basis states must be binary")
    if amplitudes.ndim != 2 or amplitudes.shape[0] != len(raw_basis):
        raise ValueError("path_amplitudes must have shape (dimension, samples)")
    expected_shape = (
        len(raw_basis),
        amplitudes.shape[1],
        raw_basis.shape[1],
    )
    if raw_final.shape != expected_shape:
        raise ValueError("inconsistent trajectory arrays")
    if amplitudes.shape[1] == 0 or np.any(~np.isfinite(amplitudes)):
        raise ValueError("finite nonempty samples required")
    if np.any((raw_final != 0) & (raw_final != 1)):
        raise ValueError("final states must be binary")
    if len({tuple(row) for row in raw_basis.astype(int)}) != len(raw_basis):
        raise ValueError("basis rows must be unique")
    closed = np.all(raw_final == raw_basis[:, None, :], axis=2)
    return float(
        np.sum(np.mean(np.where(closed, amplitudes, 0.0), axis=1))
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nb=np.array([[1,0],[0,1]]); f=np.array([[[1,0],[0,1],[1,0]],[[0,1],[1,0],[0,1]]]); a=np.array([[2.,3.,-1.],[4.,5.,6.]])",
            "call": "direct_partition_estimate(b,f,a)",
            "gold_call": "_oracle_direct_partition_estimate(b,f,a)",
        },
        {
            "setup": "import numpy as np\nb=np.eye(3,dtype=int); f=np.roll(np.repeat(b[:,None,:],2,axis=1),1,axis=2); a=np.ones((3,2))",
            "call": "direct_partition_estimate(b,f,a)",
            "gold_call": "_oracle_direct_partition_estimate(b,f,a)",
        },
        {
            "setup": "import numpy as np\nb=np.array([[1,0,1,0]]); f=b.reshape(1,1,4); a=np.array([[-2.5]])",
            "call": "direct_partition_estimate(b,f,a)",
            "gold_call": "_oracle_direct_partition_estimate(b,f,a)",
        },
    ]
