"""
Locate a zero-energy pole with decaying closed channels and constant neutral asymptotics.

The multichannel effective-range pole condition is discussed in Sec. 4.1, Eq. (4.25). For a compact potential its zero-energy boundary condition can be imposed at the outer radius. The bracket selects a unique pole.

Returns
-------
return value
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def threshold_pole(gaps: np.ndarray, wells: np.ndarray, widths: np.ndarray, bracket: tuple) -> float:
    """Locate a zero-energy pole with decaying closed channels and constant neutral asymptotics.

    gaps : ndarray, shape (N,)
        Strictly increasing threshold offsets in GeV^2, gaps[0]=0; other entries positive.
    wells : ndarray, shape (J,N,N)
        Real symmetric attraction matrices in GeV^2 as in propagate_shells.
    widths : ndarray, shape (J,)
        Positive shell widths in GeV^-1.
    bracket : tuple of two floats
        Positive ordered dimensionless depths, bracketing one simple pole.
    Returns
    -------
    float
        Dimensionless depth where a nonzero regular zero-energy solution is
        constant in channel 0 and decays in the other channels outside R.
        Resolve the root to absolute accuracy 1e-11. Use propagate_shells.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Non-strict threshold ordering, invalid shell data, or a bracket without two positive ordered endpoints and a sign change (an endpoint root is allowed).
        Symmetry/Hermiticity is checked with rtol=1e-10 and atol=1e-12.
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return value

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


def _oracle_threshold_pole(gaps: np.ndarray, wells: np.ndarray, widths: np.ndarray, bracket: tuple) -> float:
    (gaps, wells, widths) = _shell_data(gaps, wells, widths, strict=True)
    bracket = _finite_array(bracket, 'bracket', 1, real=True)
    if bracket.shape != (2,) or not 0 < bracket[0] < bracket[1]:
        raise ValueError('bracket must contain two positive ordered depths')
    n = len(gaps)

    def _det(depth):
        y = _oracle_propagate_shells(0.0, gaps, wells, widths, depth)
        return float(_checked_result(np.linalg.det(y[n:] + np.sqrt(gaps)[:, None] * y[:n]), 'pole determinant'))
    (left, right) = (_det(bracket[0]), _det(bracket[1]))
    if left != 0 and right != 0 and (np.signbit(left) == np.signbit(right)):
        raise ValueError('bracket must enclose a pole')
    try:
        return float(brentq(_det, *bracket, xtol=5e-15, rtol=1e-14))
    except RuntimeError as exc:
        raise ValueError('pole search failed to converge within the bracket') from exc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific differential cases."""
    return [{'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[1.0]]], dtype=float)\n'
               'widths = np.array([1.0], dtype=float)\n'
               'bracket = (2.0, 3.0)\n',
      'call': 'threshold_pole(*_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_oracle_threshold_pole(*_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[1.0]]], dtype=float)\n'
               'widths = np.array([1.0], dtype=float)\n'
               'bracket = (21.0, 23.0)\n',
      'call': 'threshold_pole(*_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_oracle_threshold_pole(*_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[1.0]], [[1.0]]], dtype=float)\n'
               'widths = np.array([0.3, 0.7], dtype=float)\n'
               'bracket = (2.0, 3.0)\n',
      'call': 'threshold_pole(*_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_oracle_threshold_pole(*_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0, 0.2], dtype=float)\n'
               'wells = np.array([[[1.0, 0.0], [0.0, 0.1]]], dtype=float)\n'
               'widths = np.array([1.0], dtype=float)\n'
               'bracket = (2.0, 3.0)\n',
      'call': 'threshold_pole(*_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_oracle_threshold_pole(*_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'wells = np.array([[[2.9, -1.3, 0.55], [-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]], [[1.4, 0.6, -0.45], '
               '[0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]]], dtype=float)\n'
               'widths = np.array([0.43, 0.57], dtype=float)\n'
               'bracket = (0.5, 0.8)\n',
      'call': 'threshold_pole(*_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_oracle_threshold_pole(*_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'wells = np.array([[[2.9, -1.3, 0.55], [-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]], [[1.4, 0.6, -0.45], '
               '[0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]]], dtype=float)\n'
               'widths = np.array([0.43, 0.57], dtype=float)\n'
               'bracket = (0.9, 1.2)\n',
      'call': 'threshold_pole(*_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_oracle_threshold_pole(*_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'wells = np.array([[[2.9, -1.3, 0.55], [-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]], [[1.4, 0.6, -0.45], '
               '[0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]]], dtype=float)\n'
               'widths = np.array([0.43, 0.57], dtype=float)\n'
               'bracket = (1.6, 2.0)\n',
      'call': 'threshold_pole(*_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_oracle_threshold_pole(*_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0, 3.025e-05, 6.4e-05], dtype=float)\n'
               'wells = np.array([[[2.9, 0.55, -1.3], [0.55, 1.5, -0.8], [-1.3, -0.8, 4.2]], [[1.4, -0.45, 0.6], '
               '[-0.45, 3.2, 1.1], [0.6, 1.1, 2.7]]], dtype=float)\n'
               'widths = np.array([0.43, 0.57], dtype=float)\n'
               'bracket = (0.9, 1.2)\n',
      'call': 'threshold_pole(*_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_oracle_threshold_pole(*_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[1.0]]], dtype=float)\n'
               'widths = np.array([1.0], dtype=float)\n'
               'bracket = (2.0, 3.0)\n'
               '\n'
               'bracket = (3.0,2.0)\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(threshold_pole, *_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_exception_code(_oracle_threshold_pole, *_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[1.0]]], dtype=float)\n'
               'widths = np.array([1.0], dtype=float)\n'
               'bracket = (2.0, 3.0)\n'
               '\n'
               'gaps[0] = np.inf\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(threshold_pole, *_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_exception_code(_oracle_threshold_pole, *_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[1.0]]], dtype=float)\n'
               'widths = np.array([1.0], dtype=float)\n'
               'bracket = (2.0, 3.0)\n'
               '\n'
               'wells = np.zeros((1,2,2))\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(threshold_pole, *_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_exception_code(_oracle_threshold_pole, *_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0], dtype=float)\n'
               'wells = np.array([[[1.0]]], dtype=float)\n'
               'widths = np.array([1.0], dtype=float)\n'
               'bracket = (2.0, 3.0)\n'
               '\n'
               'bracket = (0.1,0.2)\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(threshold_pole, *_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_exception_code(_oracle_threshold_pole, *_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'wells = np.array([[[2.9, -1.3, 0.55], [-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]], [[1.4, 0.6, -0.45], '
               '[0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]]], dtype=float)\n'
               'widths = np.array([0.43, 0.57], dtype=float)\n'
               'bracket = (0.9, 1.2)\n'
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
      'call': '_exception_code(threshold_pole, *_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_exception_code(_oracle_threshold_pole, *_copy_input((gaps, wells, widths, bracket)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'gaps = np.array([0.0, 1.6e-05, 0.00012099999999999999], dtype=float)\n'
               'wells = np.array([[[2.9, -1.3, 0.55], [-1.3, 4.2, -0.8], [0.55, -0.8, 1.5]], [[1.4, 0.6, -0.45], '
               '[0.6, 2.7, 1.1], [-0.45, 1.1, 3.2]]], dtype=float)\n'
               'widths = np.array([0.43, 0.57], dtype=float)\n'
               'bracket = (0.9, 1.2)\n'
               '\n'
               'gaps[2] = gaps[1]\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(threshold_pole, *_copy_input((gaps, wells, widths, bracket)))',
      'gold_call': '_exception_code(_oracle_threshold_pole, *_copy_input((gaps, wells, widths, bracket)))'}]
