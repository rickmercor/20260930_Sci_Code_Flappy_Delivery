"""
*Build the degree-major coefficient tensor of a hidden-variable elimination matrix.*

The two polynomials are $f(x,y)=\sum_{m=0}^{4}\alpha_m(x)y^m$ and

$g(x,y)=\sum_{m=0}^{3}\gamma_m(x)y^m$, where the coefficient arrays store

ascending powers of $x$. Their shifted coefficient rows act on the monomial

vector $(y^6,y^5,y^4,y^3,y^2,y,1)^{\mathsf T}$.

The rows of $B(x)$ produce $q(x)y^2f$, $yf$, $f$, $y^3g$, $y^2g$, $yg$, and

$g$, where $q(x)=(x-u)(x-v)$. The factor $q$ introduces determinant roots

$u$ and $v$. An invertible left factor $L$ preserves the right null space

and gives $\det(LB)=\det(L)\det(B)$. The contract allows $|\det L|=1$;

the main instance has $\det L=1$.

Returns
-------
A NumPy array of shape $(k+1,7,7)$, where $k$ is `determinant_degree`, ordered by degree, row, and column.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_resultant_tensor(
    alpha: np.ndarray,
    gamma: np.ndarray,
    u: float,
    v: float,
    row_mixing: np.ndarray,
    determinant_degree: int = 12,
) -> np.ndarray:
    r"""Build the padded coefficient tensor of the elimination matrix.
    
    Parameters
    ----------
    alpha : np.ndarray
        Finite real coefficients of shape $(5,3)$. Entry $\alpha_{m\ell}$
        multiplies $x^\ell y^m$ in $f(x,y)$; both degree axes are ascending.
    gamma : np.ndarray
        Finite real coefficients of shape $(4,2)$. Entry $\gamma_{m\ell}$
        multiplies $x^\ell y^m$ in $g(x,y)$; both degree axes are ascending.
    u : float
        Finite first root of the row multiplier $q(x)=(x-u)(x-v)$.
    v : float
        Finite second root, separated from $u$ by more than $10^{-12}$.
    row_mixing : np.ndarray
        Finite real matrix $L$ of shape $(7,7)$ applied on the left.
        Require $\bigl||\det L|-1\bigr|\le10^{-9}$.
    determinant_degree : int, optional
        Degree bound $k\ge1$, default $12$. Allocates $k+1$ coefficient
        slices and must accommodate every row polynomial.
    
    Returns
    -------
    coefficient_tensor : np.ndarray
        Real array of shape $(k+1,7,7)$ such that
        $M(x)=\sum_{n=0}^{k}M_nx^n$. Slice $n$ is $M_n$ after row
        scaling and multiplication by $L$. Unused high-degree slices are zero.
    
    Raises
    ------
    ValueError
        If coefficient or row-mixing shapes are invalid, any data are nonfinite,
        $|u-v|\le10^{-12}$, $k$ is not a positive integer, the row factor
        fails its determinant-magnitude tolerance, or the allocated degree
        axis is too short for the row polynomials.
    
    Notes
    -----
    Construct the seven-row Sylvester stack for the quartic and cubic,
    scale its first row by $q(x)$, and apply $L$ on the left. The monomial
    order is $(y^6,y^5,y^4,y^3,y^2,y,1)$. Do not mutate input arrays.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _validated_problem_data(alpha, gamma, row_mixing, u, v, determinant_degree):
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    row_mixing = np.asarray(row_mixing, dtype=float)
    if alpha.shape != (5, 3) or gamma.shape != (4, 2) or row_mixing.shape != (7, 7):
        raise ValueError("alpha, gamma, and row_mixing have invalid shapes")
    if (
        not np.all(np.isfinite(alpha))
        or not np.all(np.isfinite(gamma))
        or not np.all(np.isfinite(row_mixing))
    ):
        raise ValueError("all coefficient data must be finite")
    if (
        not np.isfinite(u)
        or not np.isfinite(v)
        or np.isclose(u, v, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("u and v must be finite and distinct")
    if not isinstance(determinant_degree, (int, np.integer)) or determinant_degree < 1:
        raise ValueError("determinant_degree must be a positive integer")
    if abs(abs(np.linalg.det(row_mixing)) - 1.0) > 1e-9:
        raise ValueError("row_mixing must have unit determinant magnitude")
    return alpha, gamma, row_mixing, float(u), float(v), int(determinant_degree)


def _oracle_construct_resultant_tensor(
    alpha: np.ndarray,
    gamma: np.ndarray,
    u: float,
    v: float,
    row_mixing: np.ndarray,
    determinant_degree: int = 12,
) -> np.ndarray:
    """Reference construction of the padded dense elimination tensor."""
    alpha, gamma, row_mixing, u, v, determinant_degree = _validated_problem_data(
        alpha, gamma, row_mixing, u, v, determinant_degree
    )
    base = np.zeros((determinant_degree + 1, 7, 7), dtype=float)
    q = np.array([u * v, -(u + v), 1.0])
    for column, y_degree in enumerate((4, 3, 2, 1, 0)):
        scaled = np.polynomial.polynomial.polymul(q, alpha[y_degree])
        if scaled.size > determinant_degree + 1:
            raise ValueError("determinant_degree is too small for the data")
        base[: scaled.size, 0, column] = scaled
        base[: alpha[y_degree].size, 1, column + 1] = alpha[y_degree]
        base[: alpha[y_degree].size, 2, column + 2] = alpha[y_degree]
    for shift in range(4):
        for column, y_degree in enumerate((3, 2, 1, 0)):
            base[: gamma[y_degree].size, 3 + shift, shift + column] = gamma[y_degree]
    return np.einsum("ij,djk->dik", row_mixing, base)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Scientific cases with independent mutable inputs on both sides."""
    return [{'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v, determinant_degree = 0.6, 2.5, 12\n',
      'call': 'construct_resultant_tensor(alpha.copy(), gamma.copy(), u, v, row_mixing.copy(), '
              'determinant_degree)',
      'gold_call': '_oracle_construct_resultant_tensor(alpha.copy(), gamma.copy(), u, v, '
                   'row_mixing.copy(), determinant_degree)'},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v, determinant_degree = 0.60000001, 0.60000003, 12\n',
      'call': 'construct_resultant_tensor(alpha.copy(), gamma.copy(), u, v, row_mixing.copy(), '
              'determinant_degree)',
      'gold_call': '_oracle_construct_resultant_tensor(alpha.copy(), gamma.copy(), u, v, '
                   'row_mixing.copy(), determinant_degree)'},
     {'setup': 'import numpy as np\n'
               'alpha = np.array([[1.0, -0.5, 0.25], [-1.5, 1.0, 0.5], [2.0, 0.5, -1.0], [-1.0, 0.25, '
               '0.5], [1.0, 0.0, 0.0]])\n'
               'gamma = np.array([[-2.0, 0.5], [1.0, -1.5], [0.5, 0.75], [1.0, 0.25]])\n'
               'row_mixing = np.array([[1, 0, 0, 0, 0, 0, -1], [0, 1, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, '
               '0, 0], [-2, 0, 0, 1, -2, 0, 2], [0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, '
               '0, 0, 0, 0, 1]], dtype=float)\n'
               'u = v = 0.6\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        construct_resultant_tensor(alpha.copy(), gamma.copy(), u, v, '
               'row_mixing.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_construct_resultant_tensor(alpha.copy(), gamma.copy(), u, v, '
               'row_mixing.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
