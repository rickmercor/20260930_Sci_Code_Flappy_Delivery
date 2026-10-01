"""
Implements construct_banded_components, which builds the compact banded part of the deterministic banded-plus-semiseparable (BPS) matrix used by the QR factorization task.

This step constructs the compact banded component B of the banded-plus-semiseparable matrix using a lower bandwidth of 2 and an upper bandwidth of 3. Only six diagonals can contain nonzero values, so the matrix is stored by those diagonals instead of as a dense n-by-n array. The main diagonal follows the task-defined sinusoidal expression, while the two lower and three upper off-diagonals use the fixed values specified in the prompt. This compact representation preserves all information needed by later structured calculations while requiring memory that grows only linearly with n.

Returns
-------
A float64 array of shape (6,n), where the rows correspond to the diagonal offsets (-2, -1, 0, +1, +2, +3). All entries outside of the matrix boundaries are zero
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_banded_component(n: int) -> np.ndarray:
    """Constructs the compact six-diagonal representation of B.
    Parameters
    ----------
    n: int
      The postiive matrix dimension

    Returns
    -------
    bands: np.ndarray
      A float64 array of shape (6,n), where the rows correspond to the diagonal
      offsets (-2, -1, 0, +1, +2, +3). All entries outside of the matrix boundaries
      are zero
      
    Raises
    ------
    ValueError
        If the requested matrix dimension is not a valid positive integer.

    """
    return bands

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_construct_banded_component(n: int) -> np.ndarray:
    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    bands = np.zeros((6, n), dtype=np.float64)
    i = np.arange(1, n + 1, dtype=np.float64)
    bands[2] = 5.0 + 0.2*np.sin(0.01*i)
    if n >= 2:
        bands[1,1:] = 0.6
        bands[3,:-1] = -0.8
    if n >= 3:
        bands[0,2:] = -0.1
        bands[4,:-2] = 0.15
    if n >= 4:
        bands[5,:-3] = -0.05
    return bands

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid test case specifications."""
    return [
        {
            "setup":"""import numpy as np
n=8""",
            "call":"construct_banded_component(n).tolist()",
            "gold_call":"_oracle_construct_banded_component(n).tolist()"},
        {
            "setup":"""import numpy as np
n=3""",
            "call":"construct_banded_component(n).tolist()",
            "gold_call":"_oracle_construct_banded_component(n).tolist()"},
        {
            "setup":"""import numpy as np
n=1""",
            "call":"construct_banded_component(n).tolist()",
            "gold_call":"_oracle_construct_banded_component(n).tolist()"},
        {
            "setup": """import numpy as np
n = 0
def run_model():
    try:
        construct_banded_component(n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_construct_banded_component(n)
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
n = 4.5
def run_model():
    try:
        construct_banded_component(n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_construct_banded_component(n)
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
