"""
Run the complete elimination-and-recovery pipeline and report one scalar.

For determinant degree $k$, the orchestrator evaluates ordinary matrix

derivatives through order three at

$S=\max\{2,\lceil(k+1)/4\rceil\}$ unit-circle nodes. It forms determinant

derivative samples, recovers the ascending coefficients, and extracts nearly

real roots in $[0,3]$. Cofactor recovery and the original-equation residual

gate determine the retained set. The reported scalar is the largest

$\kappa(x)=\|(1,x,\ldots,x^k)\|_2/|D'(x)|$ over that set.

Returns
-------
One finite float: the maximum retained condition number for roots in $[0,3]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_hidden_root_condition(
    alpha: np.ndarray,
    gamma: np.ndarray,
    u: float,
    v: float,
    row_mixing: np.ndarray,
    determinant_degree: int = 12,
    imaginary_tolerance: float = 1e-8,
    residual_threshold: float = 1e-3,
) -> float:
    r"""Return the largest condition number among the retained roots.
    
    Parameters
    ----------
    alpha : np.ndarray
        Finite real array of shape $(5,3)$ containing the coefficients
        of $f(x,y)$ in ascending $y$ degree and then ascending $x$ degree.
    gamma : np.ndarray
        Finite real array of shape $(4,2)$ containing the coefficients
        of $g(x,y)$ with the same degree conventions.
    u : float
        Finite first root of $q(x)=(x-u)(x-v)$.
    v : float
        Finite second root, separated from $u$ by more than $10^{-12}$.
    row_mixing : np.ndarray
        Finite real matrix $L$ of shape $(7,7)$ satisfying
        $\bigl||\det L|-1\bigr|\le10^{-9}$.
    determinant_degree : int, optional
        Positive degree bound $k$, default $12$, large enough for the
        row polynomials and the determinant. The coefficient vector has
        length $k+1$, including its padding.
    imaginary_tolerance : float, optional
        Finite positive root-reality tolerance, default $10^{-8}$.
    residual_threshold : float, optional
        Finite positive original-equation residual threshold, default $10^{-3}$.
        Residual acceptance uses a strict inequality.
    
    Returns
    -------
    condition_number : float
        Finite maximum of $\kappa(x)=\|(1,x,\ldots,x^k)\|_2/|D'(x)|$
        over roots in $[0,3]$ that survive reality and residual filtering.
        The polynomial $D$ includes the supplied row scaling and mixing.
    
    Raises
    ------
    ValueError
        If construction inputs or either tolerance violate the stated
        contracts, any intermediate stage rejects the data, no candidate
        survives filtering, a retained root is numerically multiple, or
        the resulting maximum is nonfinite. A retained root is numerically
        multiple when $|D'(x)|\le100\epsilon\max\{1,\max_n|c_n|\}$,
        with $\epsilon$ the double-precision machine epsilon.
    
    Notes
    -----
    Apply the earlier stages in order. Use ordinary derivative channels
    $r=0,1,2,3$ at $S=\max\{2,\lceil(k+1)/4\rceil\}$ Fourier points,
    then recover coefficients by confluent interpolation. Sampling matrices
    may be singular. Restrict candidates to the closed interval $[0,3]$
    before cofactor recovery and residual filtering. Do not mutate inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_hidden_root_condition(
    alpha: np.ndarray,
    gamma: np.ndarray,
    u: float,
    v: float,
    row_mixing: np.ndarray,
    determinant_degree: int = 12,
    imaginary_tolerance: float = 1e-8,
    residual_threshold: float = 1e-3,
) -> float:
    """Reference end-to-end solver composed from the seven earlier oracles."""
    if not np.isfinite(imaginary_tolerance) or imaginary_tolerance <= 0:
        raise ValueError("imaginary_tolerance must be positive and finite")
    if not np.isfinite(residual_threshold) or residual_threshold <= 0:
        raise ValueError("residual_threshold must be positive and finite")
    tensor = _oracle_construct_resultant_tensor(  # noqa: F821
        alpha, gamma, u, v, row_mixing, determinant_degree
    )
    sample_count = max(2, (int(determinant_degree) + 4) // 4)
    points = _oracle_generate_unit_circle_samples(sample_count - 1)  # noqa: F821
    matrices = _oracle_evaluate_resultant_fft(tensor, points, derivative_order=3)  # noqa: F821
    samples = _oracle_compute_determinant_samples(matrices)  # noqa: F821
    coefficients = _oracle_recover_determinant_coefficients(samples, tensor)  # noqa: F821
    candidates = _oracle_extract_hidden_candidates(coefficients, imaginary_tolerance, interval=(0.0,3.0))  # noqa: F821
    solutions = _oracle_recover_and_filter_solutions(  # noqa: F821
        tensor,
        candidates,
        alpha,
        gamma,
        residual_threshold=residual_threshold,
    )
    derivative_coefficients = np.arange(1, coefficients.size) * coefficients[1:]
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    conditions = []
    for x_value in solutions[:, 0]:
        derivative = np.polynomial.polynomial.polyval(x_value, derivative_coefficients)
        if abs(derivative) <= 100 * np.finfo(float).eps * scale:
            raise ValueError("a retained root is numerically multiple")
        powers = x_value ** np.arange(coefficients.size)
        conditions.append(float(np.linalg.norm(powers) / abs(derivative)))
    result = max(conditions)
    if not np.isfinite(result):
        raise ValueError("the condition number is not finite")
    return result

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
               'u, v = 0.6, 2.5\n',
      'call': 'compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, row_mixing.copy())',
      'gold_call': '_oracle_compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, '
                   'row_mixing.copy())'},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 1.4, -0.8\n',
      'call': 'compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, row_mixing.copy())',
      'gold_call': '_oracle_compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, '
                   'row_mixing.copy())'},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'row_mixing = np.eye(7)\n'
               'u, v = 0.6, 2.5\n',
      'call': 'compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, row_mixing.copy())',
      'gold_call': '_oracle_compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, '
                   'row_mixing.copy())'},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'alpha = '
               'np.array([[0.75,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'u, v = 0.6, 2.5\n',
      'call': 'compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, row_mixing.copy())',
      'gold_call': '_oracle_compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, '
                   'row_mixing.copy())'},
     {'setup': 'import numpy as np\n'
               'alpha = np.array([[1.0, -0.5, 0.25], [-1.5, 1.0, 0.5], [2.0, 0.5, -1.0], [-1.0, 0.25, '
               '0.5], [1.0, 0.0, 0.0]])\n'
               'gamma = np.array([[-2.0, 0.5], [1.0, -1.5], [0.5, 0.75], [1.0, 0.25]])\n'
               'row_mixing = np.array([[1, 0, 0, 0, 0, 0, -1], [0, 1, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, '
               '0, 0], [-2, 0, 0, 1, -2, 0, 2], [0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0], [0, 0, '
               '0, 0, 0, 0, 1]], dtype=float)\n'
               'u, v = (0.6, 2.5)\n'
               'residual_threshold = 0.0\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, '
               'row_mixing.copy(), residual_threshold=residual_threshold)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, '
               'row_mixing.copy(), residual_threshold=residual_threshold)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'u=1.;v=2.5',
      'call': 'compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, row_mixing.copy())',
      'gold_call': '_oracle_compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, '
                   'row_mixing.copy())',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'row_mixing[0]*=-1',
      'call': 'compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, row_mixing.copy())',
      'gold_call': '_oracle_compute_hidden_root_condition(alpha.copy(), gamma.copy(), u, v, '
                   'row_mixing.copy())',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n',
      'call': 'compute_hidden_root_condition(alpha.copy(), gamma.copy(), -1.0, 2.5, row_mixing.copy())',
      'gold_call': '_oracle_compute_hidden_root_condition(alpha.copy(), gamma.copy(), -1.0, 2.5, '
                   'row_mixing.copy())',
      'tol': 1e-06}]
