"""
Assemble the discrete operators for a uniform axial bar discretised with linear elements: the elastic stiffness operator and the diagonal inertia used by an explicit integrator. Pack both into one array so the pair can be compared exactly.

Axial bar discretised with two-node linear elements of equal size. Explicit time integration of solids uses a diagonal inertia operator so that the update needs no linear solve. See the source for how the inertia is distributed at the two end nodes.

Returns
-------
ndarray of shape (n_elements+2, n_elements+1): the stiffness operator in the first n_elements+1 rows, the diagonal inertia in the last row.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bar_system_matrices(n_elements, length, area, youngs, density):
    """Assemble the discrete operators for a uniform axial bar discretised with linear elements: the elastic stiffness operator and the diagonal inertia used by an explicit integrator. Pack both into one array so the pair can be compared exactly.

    Returns
    -------
    ndarray of shape (n_elements+2, n_elements+1): the stiffness operator in the first n_elements+1 rows, the diagonal inertia in the last row.

    Raises
    ------
    ValueError: if n_elements is below 1, or any of length, area, youngs or density is not positive.
    """
    return np.zeros(1)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bar_system_matrices(n_elements, length, area, youngs, density):
    if n_elements < 1:
        raise ValueError("n_elements must be at least 1")
    if min(length, area, youngs, density) <= 0:
        raise ValueError("length, area, youngs and density must be positive")
    n = int(n_elements) + 1
    h = float(length) / int(n_elements)
    k = float(youngs) * float(area) / h
    K = np.zeros((n, n), dtype=float)
    blk = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for i in range(int(n_elements)):
        K[i:i + 2, i:i + 2] += blk
    m = float(density) * float(area) * h
    Md = np.full(n, m, dtype=float)
    Md[0] *= 0.5                       # lumped mass, half at each end node
    Md[-1] *= 0.5
    out = np.zeros((n + 1, n), dtype=float)
    out[:n, :] = K
    out[n, :] = Md
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "bar_system_matrices(4, 1.0, 2.0e-4, 2.0e11, 7800.0)",
         "gold_call": "_oracle_bar_system_matrices(4, 1.0, 2.0e-4, 2.0e11, 7800.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "bar_system_matrices(1, 0.5, 1.0e-4, 1.0e11, 2700.0)",
         "gold_call": "_oracle_bar_system_matrices(1, 0.5, 1.0e-4, 1.0e11, 2700.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "bar_system_matrices(200, 0.254, 6.45e-4, 211e9, 7847.0)",
         "gold_call": "_oracle_bar_system_matrices(200, 0.254, 6.45e-4, 211e9, 7847.0)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        bar_system_matrices(0, 1.0, 1.0, 1.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_bar_system_matrices(0, 1.0, 1.0, 1.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
