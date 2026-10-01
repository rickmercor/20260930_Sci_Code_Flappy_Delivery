"""
Find the largest possible RMS response of a real linear density-to-pressure

map. Its columns and rows use the same uniform spatial grid, and its null

constant mode has already been removed. The norm is unweighted spatial RMS,

not a norm weighted by the nonuniform equilibrium density or pressure.

Find the largest possible RMS response of a real linear density-to-pressure

map. Its columns and rows use the same uniform spatial grid, and its null

constant mode has already been removed. The norm is unweighted spatial RMS,

not a norm weighted by the nonuniform equilibrium density or pressure.



Input: response is a nonempty finite real square matrix.

Returns: out, the maximum output RMS divided by input RMS, a nonnegative

float. This is an amplitude ratio, not the squared amplification.

Returns
-------
out, the maximum output RMS divided by input RMS, a nonnegative float. This is an amplitude ratio, not the squared amplification.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def optimal_response(response: np.ndarray) -> float:
    """Return the optimal RMS amplitude ratio.

    response is a nonempty finite real square matrix on a uniform grid.
    Return one nonnegative native Python float, using the RMS amplitude
    convention above. Raise ValueError for an invalid input matrix.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_optimal_response(response: np.ndarray) -> float:
    r=np.asarray(response,dtype=float)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or not len(r) or not np.all(np.isfinite(r)):
        raise ValueError('invalid response matrix')
    out=float(np.linalg.svd(r,compute_uv=False)[0])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    return [
        {'setup': 'import numpy as np\nR=np.array([[1.0, 2.0], [0.0, 1.0]])\n', 'call': 'optimal_response(R)', 'gold_call': '_oracle_optimal_response(R)'},
        {'setup': 'import numpy as np\nR=np.array([[0.0, 0.0], [0.0, 0.0]])\n', 'call': 'optimal_response(R)', 'gold_call': '_oracle_optimal_response(R)'},
        {'setup': 'import numpy as np\nR=np.array([[0.1, -0.2, 0.1], [-0.4, 0.5, -0.1], [0.3, -0.3, 0.0]])\n', 'call': 'optimal_response(R)', 'gold_call': '_oracle_optimal_response(R)'},
        {'setup': 'import numpy as np\nR=np.array([[2.0]])\n', 'call': 'optimal_response(R)', 'gold_call': '_oracle_optimal_response(R)'},
        {'setup': 'import numpy as np\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_status(optimal_response, np.array([[np.nan]]))', 'gold_call': '_status(_oracle_optimal_response, np.array([[np.nan]]))'},
    ]
