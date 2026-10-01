"""
Recover the second variable at each candidate and discard spurious roots.

At a rank-six zero of the determinant, a nonzero row of signed cofactors

provides a right null vector. For a genuine common root it is proportional

to $(y^6,y^5,y^4,y^3,y^2,y,1)^{\mathsf T}$, so the ratio of adjacent

entries recovers $y$. The factor $q(x)$ can introduce determinant roots

that fail the original equations. Each recovered pair is tested using

$\widehat r=\max\{|f(x,y)|,|g(x,y)|\}/\max\{1,\sqrt{x^2+y^2}\}$.

Returns
-------
A real NumPy array of shape $(V,3)$, where $V$ is the number of valid solutions, with columns $x$, $y$, and residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recover_and_filter_solutions(
    coefficient_tensor: np.ndarray,
    candidates: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    deletion_row: int = 0,
    anchor_column: int = 6,
    residual_threshold: float = 1e-3,
) -> np.ndarray:
    r"""Recover $y$ from a cofactor null vector and retain valid pairs.
    
    Parameters
    ----------
    coefficient_tensor : np.ndarray
        Finite coefficient tensor of shape $(K,7,7)$ representing $M(x)$
        in ascending degree order.
    candidates : np.ndarray
        Finite real vector of shape $(C,)$ containing hidden-variable
        candidates $x$ to test against the original polynomial equations.
    alpha : np.ndarray
        Real array of shape $(5,3)$; entry $\alpha_{m\ell}$ is the
        coefficient of $x^\ell y^m$ in the original quartic $f(x,y)$.
    gamma : np.ndarray
        Real array of shape $(4,2)$; entry $\gamma_{m\ell}$ is the
        coefficient of $x^\ell y^m$ in the original cubic $g(x,y)$.
    deletion_row : int, optional
        Row $d$ used for signed cofactors, default $0$, with $0\le d<7$.
    anchor_column : int, optional
        Denominator column $a$, default $6$, with $1\le a<7$.
        Recover $y=C_{d,a-1}/C_{d,a}$ in descending monomial order.
    residual_threshold : float, optional
        Finite positive threshold $\eta$, default $10^{-3}$.
        Retain a recovered pair only when its normalized residual is
        strictly less than $\eta$.
    
    Returns
    -------
    solutions : np.ndarray
        Nonempty real array of shape $(V,3)$, with each row
        $(x,y,\widehat r)$ containing a surviving pair and its normalized
        original-equation residual. Sort rows by increasing $x$, with
        $y$ and then residual used to break ties.
    
    Raises
    ------
    ValueError
        If tensor, candidate, or equation-coefficient shapes are invalid;
        indices are outside their stated ranges; the threshold is nonfinite
        or nonpositive; tensor or candidate entries are nonfinite; or
        no candidate yields an accepted recovered pair.
    
    Notes
    -----
    Use signed cofactors with monomial order $(y^6,y^5,y^4,y^3,y^2,y,1)$.
    Skip a candidate if its anchor magnitude is at most
    $100\epsilon\max\{1,\max_j|C_{d,j}|\}$, or if the recovered ratio
    has imaginary magnitude above $10^{-8}$. These individual rejections
    are not errors when other candidates survive.
    Evaluate the original equations using the real part of the accepted ratio.
    The normalized residual is
    $\widehat r=\max\{|f(x,y)|,|g(x,y)|\}/\max\{1,\sqrt{x^2+y^2}\}$.
    Do not mutate input arrays.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _evaluate_polynomial_matrix(
    coefficient_tensor: np.ndarray, x_value: float
) -> np.ndarray:
    powers = float(x_value) ** np.arange(coefficient_tensor.shape[0])
    return np.tensordot(powers, coefficient_tensor, axes=(0, 0))


def _cofactor_null_vector(matrix: np.ndarray, deletion_row: int) -> np.ndarray:
    cofactors = []
    for column in range(matrix.shape[1]):
        minor = np.delete(np.delete(matrix, deletion_row, axis=0), column, axis=1)
        determinant = _oracle_compute_determinant_samples(minor[None, :, :])[0]  # noqa: F821
        cofactors.append((-1) ** (deletion_row + column) * determinant)
    return np.asarray(cofactors)


def _oracle_recover_and_filter_solutions(
    coefficient_tensor: np.ndarray,
    candidates: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    deletion_row: int = 0,
    anchor_column: int = 6,
    residual_threshold: float = 1e-3,
) -> np.ndarray:
    """Reference cofactor recovery and original-system residual filter."""
    tensor = np.asarray(coefficient_tensor)
    candidates = np.asarray(candidates, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if tensor.ndim != 3 or tensor.shape[1:] != (7, 7) or candidates.ndim != 1:
        raise ValueError("tensor or candidate shape is invalid")
    if alpha.shape != (5, 3) or gamma.shape != (4, 2):
        raise ValueError("equation coefficient shapes are invalid")
    if deletion_row not in range(7) or anchor_column not in range(1, 7):
        raise ValueError("deletion_row or anchor_column is invalid")
    if not np.isfinite(residual_threshold) or residual_threshold <= 0:
        raise ValueError("residual_threshold must be positive and finite")
    if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(candidates)):
        raise ValueError("inputs must be finite")

    polyval = np.polynomial.polynomial.polyval
    records = []
    for x_value in candidates:
        matrix = _evaluate_polynomial_matrix(tensor, x_value)
        null_vector = _cofactor_null_vector(matrix, deletion_row)
        scale = max(1.0, float(np.max(np.abs(null_vector))))
        if abs(null_vector[anchor_column]) <= 100 * np.finfo(float).eps * scale:
            continue
        y_value = null_vector[anchor_column - 1] / null_vector[anchor_column]
        if abs(y_value.imag) > 1e-8:
            continue
        y_value = float(y_value.real)
        f_value = sum(polyval(x_value, alpha[m]) * y_value**m for m in range(5))
        g_value = sum(polyval(x_value, gamma[m]) * y_value**m for m in range(4))
        denominator = max(1.0, float(np.hypot(x_value, y_value)))
        residual = float(max(abs(f_value), abs(g_value)) / denominator)
        if residual < residual_threshold:
            records.append((float(x_value), y_value, residual))
    if not records:
        raise ValueError("no candidate satisfies the residual gate")
    records.sort(key=lambda row: (row[0], row[1], row[2]))
    return np.asarray(records)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Scientific cases with independent mutable inputs on both sides."""
    return [{'setup': 'import numpy as np\n'
               'alpha = np.array([[1.0, -0.5, 0.25], [-1.5, 1.0, 0.5], [2.0, 0.5, -1.0], [-1.0, 0.25, '
               '0.5], [1.0, 0.0, 0.0]])\n'
               'gamma = np.array([[-2.0, 0.5], [1.0, -1.5], [0.5, 0.75], [1.0, 0.25]])\n'
               'row_mixing = np.array([[1, 0, 0, 0, 0, 0, -1], [0, 1, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, '
               '0, 0], [-2, 0, 0, 1, -2, 0, 2], [0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, '
               '0, 0, 0, 0, 1]], dtype=float)\n'
               'coefficient_tensor = _oracle_construct_resultant_tensor(alpha.copy(), gamma.copy(), '
               '0.6, 2.5, row_mixing.copy(), 12)\n'
               'candidates = np.array([-5.220620395748996, 0.6, 1.630324484134974, 1.7867722886247168, '
               '2.346011578311881, 2.500000000015429])',
      'call': 'recover_and_filter_solutions(coefficient_tensor.copy(), candidates.copy(), '
              'alpha.copy(), gamma.copy())',
      'gold_call': '_oracle_recover_and_filter_solutions(coefficient_tensor.copy(), candidates.copy(), '
                   'alpha.copy(), gamma.copy())'},
     {'setup': 'import numpy as np\n'
               'alpha = np.array([[1.0, -0.5, 0.25], [-1.5, 1.0, 0.5], [2.0, 0.5, -1.0], [-1.0, 0.25, '
               '0.5], [1.0, 0.0, 0.0]])\n'
               'gamma = np.array([[-2.0, 0.5], [1.0, -1.5], [0.5, 0.75], [1.0, 0.25]])\n'
               'row_mixing = np.array([[1, 0, 0, 0, 0, 0, -1], [0, 1, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, '
               '0, 0], [-2, 0, 0, 1, -2, 0, 2], [0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, '
               '0, 0, 0, 0, 1]], dtype=float)\n'
               'coefficient_tensor = _oracle_construct_resultant_tensor(alpha.copy(), gamma.copy(), '
               '0.6, 2.5, row_mixing.copy(), 12)\n'
               'candidates = np.array([-5.220620395748996, 0.6, 1.630324484134974, 1.7867722886247168, '
               '2.346011578311881, 2.500000000015429])\n'
               'residual_threshold = 0.05',
      'call': 'recover_and_filter_solutions(coefficient_tensor.copy(), candidates.copy(), '
              'alpha.copy(), gamma.copy(), residual_threshold=residual_threshold)',
      'gold_call': '_oracle_recover_and_filter_solutions(coefficient_tensor.copy(), candidates.copy(), '
                   'alpha.copy(), gamma.copy(), residual_threshold=residual_threshold)'},
     {'setup': 'import numpy as np\n'
               'alpha = np.array([[1.0, -0.5, 0.25], [-1.5, 1.0, 0.5], [2.0, 0.5, -1.0], [-1.0, 0.25, '
               '0.5], [1.0, 0.0, 0.0]])\n'
               'gamma = np.array([[-2.0, 0.5], [1.0, -1.5], [0.5, 0.75], [1.0, 0.25]])\n'
               'row_mixing = np.array([[1, 0, 0, 0, 0, 0, -1], [0, 1, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, '
               '0, 0], [-2, 0, 0, 1, -2, 0, 2], [0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, '
               '0, 0, 0, 0, 1]], dtype=float)\n'
               'coefficient_tensor = np.zeros((13, 7, 7))\n'
               'candidates = np.array([0.0])\n'
               'anchor_column = 0\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        recover_and_filter_solutions(coefficient_tensor.copy(), candidates.copy(), '
               'alpha.copy(), gamma.copy(), anchor_column=anchor_column)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_recover_and_filter_solutions(coefficient_tensor.copy(), '
               'candidates.copy(), alpha.copy(), gamma.copy(), anchor_column=anchor_column)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
