"""
Implements construct_semiseparable_generators, which builds the four deterministic low-rank generator matrices used in the BPS representation

This step constructs the four deterministic generator matrices U, V, W, and S that define the strictly lower and strictly upper semiseparable portions of the matrix. U and V generate the lower-triangular low-rank contribution, while W and S generate the upper-triangular low-rank contribution. Each row is evaluated from the trigonometric formulas supplied in the task and scaled by the square root of the matrix size. These fixed-rank generators allow dense triangular interactions to be represented compactly without explicitly forming the full matrix A.

Returns
-------
Four float np.ndarrays of the semiseparable generators (U, V, W, and S)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_semiseparable_generators(n: int):
    """Constructs the deterministic semiseparable generators, U, V, W, and S.
    Parameters
    ----------
    n: int
      The postiive matrix dimension

    Returns
    -------
    U: np.ndarray
      float64 array of shape (n,2)
    V: np.ndarray
      float64 array of shape (n,2)
    W: np.ndarray
      float64 array of shape (n,2)
    S: np.ndarray
      float64 array of shape (n,2)

    Raises
    ------
    ValueError
        If the requested matrix dimension is not a valid positive integer.
    """
    return U, V, W, S

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_construct_semiseparable_generators(n: int):
    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    i = np.arange(1, n + 1, dtype=np.float64)
    c = np.float64(1.0/np.sqrt(np.float64(n)))
    U = c*np.column_stack((1+0.2*np.sin(0.017*i), 0.7+0.15*np.cos(0.011*i)))
    V = c*np.column_stack((0.9+0.1*np.cos(0.013*i), -0.6+0.12*np.sin(0.019*i)))
    W = c*np.column_stack((0.8+0.18*np.sin(0.023*i), 0.5+0.10*np.cos(0.029*i)))
    S = c*np.column_stack((-0.7+0.11*np.cos(0.031*i), 0.6+0.14*np.sin(0.037*i)))
    return tuple(np.asarray(x,dtype=np.float64) for x in (U,V,W,S))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid test case specifications."""
    return [
        {
            "setup":"""import numpy as np
n=8""",
            "call":"tuple(x.tolist() for x in construct_semiseparable_generators(n))",
            "gold_call":"tuple(x.tolist() for x in _oracle_construct_semiseparable_generators(n))"
        },
        {
            "setup":"""import numpy as np
n=1""",
            "call":"tuple(x.tolist() for x in construct_semiseparable_generators(n))",
            "gold_call":"tuple(x.tolist() for x in _oracle_construct_semiseparable_generators(n))"
        },
        {
            "setup":"""import numpy as np
n=100000""",
            "call":"tuple(x.tolist() for x in construct_semiseparable_generators(n))",
            "gold_call":"tuple(x.tolist() for x in _oracle_construct_semiseparable_generators(n))"
        },
        {
            "setup": """import numpy as np
n = 0
def run_model():
    try:
        construct_semiseparable_generators(n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_construct_semiseparable_generators(n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call":"run_model()",
            "gold_call":"run_oracle()"
        },
        {
            "setup": """import numpy as np
n = "8"
def run_model():
    try:
        construct_semiseparable_generators(n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_construct_semiseparable_generators(n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call":"run_model()",
            "gold_call":"run_oracle()"
        },
    ]
