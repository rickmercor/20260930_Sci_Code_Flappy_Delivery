"""
Construct the deterministic nonsymmetric banded operator.

A nonsymmetric banded operator provides a controlled setting in which short-recurrence Krylov bases can depart substantially from full orthogonality. The operator is fixed deterministically, so all later sketching comparisons refer to the same linear system.

Returns
-------
np.ndarray, the finite float64 square matrix $K\in\mathbb{R}^{n\times n}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_krylov_operator(n: int = 14) -> np.ndarray:
    r"""Construct the deterministic nonsymmetric banded operator.

    Parameters
    ----------
    n : int, optional
        Positive matrix dimension. The default is 14.

    Returns
    -------
    K : np.ndarray
        Finite float64 square matrix of shape $n\times n$ with $K_{ii}=1+0.05\,i$,
        $K_{i,i+1}=0.8$, $K_{i+1,i}=0.01$, and $K_{i,i+2}=0.5$ (one-based indices).

    Raises
    ------
    ValueError
        If n is not a positive integer.
    """
    return K

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_construct_krylov_operator(n: int = 14) -> np.ndarray:
    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    K = np.zeros((n, n), dtype=np.float64)
    for i1 in range(1, n + 1):
        i = i1 - 1
        K[i, i] = 1.0 + 0.05 * i1
        if i + 1 < n:
            K[i, i + 1] = 0.8
            K[i + 1, i] = 0.01
        if i + 2 < n:
            K[i, i + 2] = 0.5
    return K

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"import numpy as np\nn=14",'call':'construct_krylov_operator(n)','gold_call':'_oracle_construct_krylov_operator(n)'},
        {'setup':"import numpy as np\nn=1",'call':'construct_krylov_operator(n)','gold_call':'_oracle_construct_krylov_operator(n)'},
        {'setup':"import numpy as np\nn=5",'call':'construct_krylov_operator(n)','gold_call':'_oracle_construct_krylov_operator(n)'},
        {'setup':"""import numpy as np
n=0
def run_model():
    try: construct_krylov_operator(n); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_construct_krylov_operator(n); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
n=3.5
def run_model():
    try: construct_krylov_operator(n); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_construct_krylov_operator(n); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
n=-2
def run_model():
    try: construct_krylov_operator(n); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_construct_krylov_operator(n); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
