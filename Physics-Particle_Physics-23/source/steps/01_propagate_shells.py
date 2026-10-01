"""
Propagate a regular coupled radial basis through successive constant spherical shells.

The s-wave radial equation is the ell=0 specialization of Eq. (2.1) in the PDF. The spherical-well example in Fig. 2 motivates these constructed shells; matrix ordering represents a physical sequence of interactions.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def propagate_shells(p: float, gaps: np.ndarray, wells: np.ndarray, widths: np.ndarray, depth: float) -> np.ndarray:
    """Propagate a regular coupled radial basis through successive constant spherical shells.

    p : float
        Reference momentum in GeV, p >= 0.
    gaps : ndarray, shape (N,)
        Nonnegative threshold offsets 2*mu*Delta_i in GeV^2, nondecreasing, gaps[0]=0.
    wells : ndarray, shape (J,N,N)
        Real symmetric attraction matrices W_j in GeV^2; shell potential
        2*mu*V_j = diag(gaps)-depth*W_j. Repulsive eigenvalues are permitted.
    widths : ndarray, shape (J,)
        Positive successive shell widths in GeV^-1, ordered from the origin.
    depth : float
        Dimensionless interaction multiplier.
    Returns
    -------
    ndarray, shape (2*N,N), real
        Top N rows are U(R) in GeV^-1; bottom N rows are U'(R), dimensionless,
        at R=sum(widths), with U(0)=0 and U'(0)=I. The radial equation is
        -U''+(2*mu*V-p^2*I)U=0; column order is the origin basis order.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Negative momentum, unordered thresholds, nonpositive widths, inconsistent shell counts or nonsymmetric wells.
        Symmetry/Hermiticity is checked with rtol=1e-10 and atol=1e-12.
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

def _finite_array(value, name, ndim, real=False):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in 'iufc' or (real and np.iscomplexobj(array)):
            raise ValueError(name + ' must contain real numeric values' if real else name + ' must be numeric')
        array = np.asarray(array, dtype=float if real else complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a finite numeric array') from exc
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(name + ' has invalid rank or nonfinite entries')
    return array


def _finite_scalar(value, name):
    return float(_finite_array(value, name, 0, real=True))


def _thresholds(gaps, strict=False):
    gaps = _finite_array(gaps, 'gaps', 1, real=True)
    if gaps.size == 0 or gaps[0] != 0 or np.any(gaps < 0):
        raise ValueError('gaps must be nonempty, nonnegative and start at zero')
    if np.any(np.diff(gaps) <= 0 if strict else np.diff(gaps) < 0):
        raise ValueError('gaps must be strictly increasing' if strict else 'gaps must be nondecreasing')
    return gaps


def _shell_data(gaps, wells, widths, strict=False):
    gaps = _thresholds(gaps, strict)
    wells = _finite_array(wells, 'wells', 3, real=True)
    widths = _finite_array(widths, 'widths', 1, real=True)
    n = len(gaps)
    if widths.size == 0 or wells.shape != (len(widths), n, n) or np.any(widths <= 0):
        raise ValueError('wells and positive widths must describe the same nonempty shell sequence')
    if not np.allclose(wells, wells.transpose(0, 2, 1), rtol=1e-10, atol=1e-12):
        raise ValueError('wells must be real symmetric')
    return gaps, wells, widths


def _open_momenta(p_open):
    p_open = _finite_array(p_open, 'p_open', 1, real=True)
    if p_open.size == 0 or np.any(p_open <= 0):
        raise ValueError('p_open must be nonempty and positive')
    return p_open


def _matched_data(p_open, matched):
    p_open = _open_momenta(p_open)
    matched = _finite_array(matched, 'matched', 2)
    m = len(p_open)
    if matched.shape[1] != m or matched.shape[0] < 2*m:
        raise ValueError('matched must have shape (N+M,M) with N>=M>=1')
    sigma, s0 = matched[:-m], matched[-m:]
    if not np.allclose(s0.conj().T @ s0, np.eye(m), rtol=1e-8, atol=1e-8):
        raise ValueError('S_0 must be unitary')
    return p_open, sigma, s0


def _hermitian_matrix(value, name, n):
    matrix = _finite_array(value, name, 2)
    if matrix.shape != (n, n):
        raise ValueError(name + ' has incompatible shape')
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError(name + ' must be Hermitian')
    return matrix


def _short_matrix(value, n):
    matrix = _finite_array(value, 'short_amplitude', 2)
    if matrix.shape != (n, n):
        raise ValueError('short_amplitude has incompatible shape')
    gamma = (matrix - matrix.conj().T) / (2j)
    if np.linalg.eigvalsh(gamma)[0] < -1e-12 * max(1., np.linalg.norm(gamma, ord=2)):
        raise ValueError('short_amplitude must have positive semidefinite absorptive part')
    return matrix


def _dressed_data(p_open, dressed):
    p_open = _open_momenta(p_open)
    dressed = _finite_array(dressed, 'dressed', 2)
    m = len(p_open)
    if dressed.shape[1] != m or dressed.shape[0] < m:
        raise ValueError('dressed must have shape (N,M) with N>=M>=1')
    return p_open, dressed


def _checked_solve(matrix, rhs, name):
    try:
        result = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(name + ' must be nonsingular') from exc
    return _checked_result(result, name)


def _checked_result(result, name):
    if not np.all(np.isfinite(result)):
        raise ValueError(name + ' produced a nonfinite result')
    return result


def _oracle_propagate_shells(p: float, gaps: np.ndarray, wells: np.ndarray, widths: np.ndarray, depth: float) -> np.ndarray:
    (gaps, wells, widths) = _shell_data(gaps, wells, widths)
    p = _finite_scalar(p, 'p')
    depth = _finite_scalar(depth, 'depth')
    if p < 0 or p > np.sqrt(np.finfo(float).max):
        raise ValueError('p must be nonnegative with finite squared momentum')
    n = len(gaps)
    eye = np.eye(n)
    zero = np.zeros((n, n))
    y = np.vstack((zero, eye))
    for (well, width) in zip(wells, widths):
        w = np.diag(gaps) - depth * well - p ** 2 * eye
        y = expm(width * np.block([[zero, eye], [w, zero]])) @ y
    return _checked_result(y, 'shell propagation')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific differential cases."""
    return [{'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.37\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[0.0]]], dtype=float)\n'
               'widths = np.array([0.9], dtype=float)\n'
               'depth = 1.0\n',
      'call': 'propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_oracle_propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.0\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[0.0]]], dtype=float)\n'
               'widths = np.array([1.2], dtype=float)\n'
               'depth = 1.0\n',
      'call': 'propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_oracle_propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.4\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[3.0]]], dtype=float)\n'
               'widths = np.array([0.8], dtype=float)\n'
               'depth = 0.7\n',
      'call': 'propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_oracle_propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.2\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[-2.0]]], dtype=float)\n'
               'widths = np.array([1.1], dtype=float)\n'
               'depth = 1.0\n',
      'call': 'propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_oracle_propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.1\n'
               'gaps = np.array([0.0, 0.25], dtype=float)\n'
               'wells = np.array([[[2.0, 0.0], [0.0, 1.0]]], dtype=float)\n'
               'widths = np.array([0.6], dtype=float)\n'
               'depth = 0.8\n',
      'call': 'propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_oracle_propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.3\n'
               'gaps = np.array([0.0, 0.0], dtype=float)\n'
               'wells = np.array([[[2.0, 0.0], [0.0, 2.0]]], dtype=float)\n'
               'widths = np.array([1.0], dtype=float)\n'
               'depth = 1.0\n',
      'call': 'propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_oracle_propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.003\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'wells = np.array([[[2.9, -1.3, 0.55], [-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]], [[1.4, 0.6, -0.45], '
               '[0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]]], dtype=float)\n'
               'widths = np.array([0.43, 0.57], dtype=float)\n'
               'depth = 1.04\n',
      'call': 'propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_oracle_propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.007\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'wells = np.array([[[1.4, 0.6, -0.45], [0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]], [[2.9, -1.3, 0.55], '
               '[-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]]], dtype=float)\n'
               'widths = np.array([0.57, 0.43], dtype=float)\n'
               'depth = 0.97\n',
      'call': 'propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_oracle_propagate_shells(*_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.37\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[0.0]]], dtype=float)\n'
               'widths = np.array([0.9], dtype=float)\n'
               'depth = 1.0\n'
               '\n'
               'p = -0.1\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_exception_code(_oracle_propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.37\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[0.0]]], dtype=float)\n'
               'widths = np.array([0.9], dtype=float)\n'
               'depth = 1.0\n'
               '\n'
               'wells[0,0,0] = np.nan\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_exception_code(_oracle_propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.37\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[0.0]]], dtype=float)\n'
               'widths = np.array([0.9], dtype=float)\n'
               'depth = 1.0\n'
               '\n'
               'widths = np.array([0.5,0.4])\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_exception_code(_oracle_propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.37\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[0.0]]], dtype=float)\n'
               'widths = np.array([0.9], dtype=float)\n'
               'depth = 1.0\n'
               '\n'
               'widths[0] = 0.0\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_exception_code(_oracle_propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.003\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'wells = np.array([[[2.9, -1.3, 0.55], [-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]], [[1.4, 0.6, -0.45], '
               '[0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]]], dtype=float)\n'
               'widths = np.array([0.43, 0.57], dtype=float)\n'
               'depth = 1.04\n'
               '\n'
               'gaps = gaps[[0,2,1]]\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_exception_code(_oracle_propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'p = 0.003\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'wells = np.array([[[2.9, -1.3, 0.55], [-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]], [[1.4, 0.6, -0.45], '
               '[0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]]], dtype=float)\n'
               'widths = np.array([0.43, 0.57], dtype=float)\n'
               'depth = 1.04\n'
               '\n'
               'wells[0,0,1] += 0.1\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))',
      'gold_call': '_exception_code(_oracle_propagate_shells, *_copy_input((p, gaps, wells, widths, depth)))'}]
