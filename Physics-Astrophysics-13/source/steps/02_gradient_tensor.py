"""
Step 02: the gradient tensor of the magnetic field.

Convention (pinned)



The gradient tensor is stored with the derivative index SECOND:



    G[n, i, j] = d B_i / d x_j   evaluated at points[n]



so that the matrix product G[n] @ B[n] is the directional derivative of the

magnetic field along itself. The transposed storage order is a different object

and is not what later steps consume.



The magnetic field comes from the step-01 public function.



Inputs



points : array_like

    Shape (N, 3), N >= 1, all entries finite.



Returns



numpy.ndarray

    Shape (N, 3, 3), dtype float64.

Returns
-------
A numpy.ndarray of shape (N, 3, 3) and dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def gradient_tensor(points):
    """Return the (N, 3, 3) gradient tensor of the magnetic field.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 1, all finite.

    Returns
    -------
    numpy.ndarray
        Shape (N, 3, 3), dtype float64, with the derivative index second.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 1.
    """
    return tensors  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

AL = 0.4
CC = 0.2


def _oracle_gradient_tensor(points):
    """Reference implementation of gradient_tensor (deterministic)."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")

    y = p[:, 1]
    z = p[:, 2]
    n = p.shape[0]
    G = np.zeros((n, 3, 3), dtype=np.float64)
    G[:, 0, 0] = AL
    G[:, 0, 1] = 2.0 * (y - 2.0)
    G[:, 0, 2] = 2.0 * z
    G[:, 1, 0] = -1.0
    G[:, 1, 1] = -AL
    G[:, 1, 2] = CC
    return G

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for gradient_tensor."""
    return [
        {'setup': 'import numpy as np\npts = np.array([[-0.35, 0.9, -0.04], [0.12, 1.4, 0.5],\n                [-0.8, 0.7, -0.8], [0.45, 0.68, 0.8]])\n', 'call': 'gradient_tensor(pts)', 'gold_call': '_oracle_gradient_tensor(pts)'},
        {'setup': 'import numpy as np\npts = np.array([[1.0, 2.0, 0.35]])\n', 'call': 'gradient_tensor(pts)', 'gold_call': '_oracle_gradient_tensor(pts)'},
        {'setup': 'import numpy as np\npts = np.array([[50.0, -20.0, 30.0], [-3.0, 17.0, -11.0]])\n', 'call': 'gradient_tensor(pts)', 'gold_call': '_oracle_gradient_tensor(pts)'},
        {'setup': "import numpy as np\nbad = np.array([[0.0, float('nan'), 0.0]])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n", 'call': '_status(gradient_tensor, bad)', 'gold_call': '_status(_oracle_gradient_tensor, bad)'},
        {'setup': 'import numpy as np\nbad = np.zeros((2, 2, 3))\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(gradient_tensor, bad)', 'gold_call': '_status(_oracle_gradient_tensor, bad)'},
    ]
