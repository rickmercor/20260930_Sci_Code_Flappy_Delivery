"""
One coefficient polynomial gives prices and both spot derivatives.

One common polynomial correction supplies prices and both spot derivatives across the strike grid through analytic differentiation.

Returns
-------
tuple Three float arrays (prices, deltas, gammas), each shaped like strikes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def matrix_values(A: np.ndarray, X0: float, strikes: np.ndarray, rho: float, base: tuple) -> tuple:
    """For d=X0-k define P(d)=sum_m sum_j rho**m*A[m,j]*d**j.
    Return base_price+P(d), base_delta+P'(d), base_gamma+P''(d) on all
    strikes. The coefficient array is common to all strikes and derivatives;
    differentiate its monomial polynomial analytically, not by finite
    differences of prices. Equivalent Horner or shift-operator evaluation
    is allowed. Input coefficient conventions are those of coefficient_matrix.

    Parameters
    ----------
    A : np.ndarray
        Finite real 2D array, at least one row and one column.
    X0 : float
        Finite initial asset level.
    strikes : np.ndarray
        Nonempty finite real 1D strike array; preserve order and duplicates.
    rho : float
        Finite correlation with abs(rho)<1.
    base : tuple
        Three finite real 1D arrays (price, delta, gamma), each the same
        shape as strikes, such as returned by normal_call_terms.

    Returns
    -------
    tuple
        Three float arrays (prices, deltas, gammas), each shaped like strikes.

    Raises
    ------
    ValueError
        If any input is nonfinite, A or strikes has the wrong dimensionality
        or is empty, abs(rho)>=1, or base does not contain three arrays with
        the strike shape.
    """
    return ()

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_matrix_values(A: np.ndarray, X0: float, strikes: np.ndarray, rho: float, base: tuple) -> tuple:
    np = __import__('numpy')
    A,strikes=np.asarray(A,dtype=float),np.asarray(strikes,dtype=float)
    if A.ndim!=2 or not all(A.shape) or strikes.ndim!=1 or not strikes.size or not np.all(np.isfinite(A)) or not np.all(np.isfinite(strikes)):
        raise ValueError('invalid array dimensions or values')
    if not np.isfinite(X0) or not np.isfinite(rho) or abs(rho)>=1 or not isinstance(base,(tuple,list)) or len(base)!=3:
        raise ValueError('invalid scalars or base')
    b=[np.asarray(v,dtype=float) for v in base]
    if any(v.shape!=strikes.shape or not np.all(np.isfinite(v)) for v in b):
        raise ValueError('invalid base arrays')
    c=np.power(rho,np.arange(A.shape[0]))@A
    d=X0-strikes
    return tuple(b[j]+np.polynomial.polynomial.polyval(d,np.polynomial.polynomial.polyder(c,j)) for j in range(3))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Normal, boundary, edge and invalid-input cases."""
    return [{'setup': 'import numpy as np\nA=np.array([[1.0, 2.0, 3.0], [-0.5, 1.0, -0.2]]); k=np.array([8.0, 10.0, 12.0]); base=tuple(np.zeros_like(k) for _ in range(3))', 'call': 'np.concatenate(matrix_values(A,10.0,k,-0.6,base))', 'gold_call': 'np.concatenate(_oracle_matrix_values(A,10.0,k,-0.6,base))'}, {'setup': 'import numpy as np\nA=np.array([[2.0]]); k=np.array([0.0, 1.0, 0.0]); base=tuple(np.zeros_like(k) for _ in range(3))', 'call': 'np.concatenate(matrix_values(A,0.0,k,0.0,base))', 'gold_call': 'np.concatenate(_oracle_matrix_values(A,0.0,k,0.0,base))'}, {'setup': 'import numpy as np\nA=np.array([[0.0, 0.0, 1.0, -0.5], [1.0, 0.0, 0.0, 0.2], [0.0, 2.0, 1.0, 0.0]]); k=np.array([1.0, 1.9, 3.0]); base=tuple(np.zeros_like(k) for _ in range(3))', 'call': 'np.concatenate(matrix_values(A,2.0,k,0.5,base))', 'gold_call': 'np.concatenate(_oracle_matrix_values(A,2.0,k,0.5,base))'}, {'setup': 'def run_model_invalid():\n    try:\n        matrix_values(np.ones((1,1)),0.,np.array([0.]),1.,(np.zeros(1),)*3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_matrix_values(np.ones((1,1)),0.,np.array([0.]),1.,(np.zeros(1),)*3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'}]
