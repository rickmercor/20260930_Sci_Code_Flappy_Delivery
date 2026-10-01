"""
Form the branching fraction after averaging incident-channel reaction rates over a momentum distribution.

This is the declared task-specific halo extension. Partial-wave cross sections in Eqs. (2.13), (4.30) imply the incident velocity weighting of event rates. A branching fraction is defined from the averaged rates.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def halo_branching(nodes: np.ndarray, weights: np.ndarray, losses: np.ndarray, conversions: np.ndarray, p_ref: float) -> np.ndarray:
    """Form the branching fraction after averaging incident-channel reaction rates over a momentum distribution.

    nodes : ndarray, shape (Q,)
        Positive channel-0 momenta p in GeV.
    weights : ndarray, shape (Q,)
        Nonnegative quadrature weights times momentum density, with positive sum.
    losses : ndarray, shape (Q,)
        Nonnegative dimensionless inclusive neutral-channel flux losses L[0,0].
    conversions : ndarray, shape (Q,)
        Nonnegative dimensionless sums of off-diagonal neutral-incident transition strengths.
    p_ref : float
        Positive reference momentum in GeV, used to scale rate coefficients.
    Returns
    -------
    ndarray, real, length 3
        Entries [a,c,B], all dimensionless and in this order. a and c are
        mu*p_ref/pi times the averaged annihilation and conversion rate
        coefficients. B is their branching fraction a/(a+c). The weights
        describe one combined distribution and a+c is positive.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Unequal or empty vector shapes, nonpositive momenta, negative weights/strengths, zero total weight or zero total averaged reaction rate.
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


def _oracle_halo_branching(nodes: np.ndarray, weights: np.ndarray, losses: np.ndarray, conversions: np.ndarray, p_ref: float) -> np.ndarray:
    nodes = _finite_array(nodes, 'nodes', 1, real=True)
    weights = _finite_array(weights, 'weights', 1, real=True)
    losses = _finite_array(losses, 'losses', 1, real=True)
    conversions = _finite_array(conversions, 'conversions', 1, real=True)
    p_ref = _finite_scalar(p_ref, 'p_ref')
    if nodes.size == 0 or any((x.shape != nodes.shape for x in (weights, losses, conversions))):
        raise ValueError('halo arrays must have the same nonempty shape')
    if np.any(nodes <= 0) or p_ref <= 0 or any((np.any(x < 0) for x in (weights, losses, conversions))):
        raise ValueError('momenta must be positive and weights and reaction strengths nonnegative')
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise ValueError('quadrature weights must have a positive finite sum')
    measure = weights / np.sum(weights)
    a = float(np.dot(measure * p_ref / nodes, losses))
    c = float(np.dot(measure * p_ref / nodes, conversions))
    if not np.isfinite(a + c) or a + c <= 0:
        raise ValueError('total averaged reaction rate must be positive and finite')
    return np.array([a, c, a / (a + c)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific differential cases."""
    return [{'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.02], dtype=float)\n'
               'weights = np.array([1.0], dtype=float)\n'
               'losses = np.array([0.2], dtype=float)\n'
               'conversions = np.array([0.3], dtype=float)\n'
               'p_ref = 0.001\n',
      'call': 'halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_oracle_halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.01, 0.03], dtype=float)\n'
               'weights = np.array([0.2, 0.8], dtype=float)\n'
               'losses = np.array([0.4, 0.7], dtype=float)\n'
               'conversions = np.array([0.0, 0.0], dtype=float)\n'
               'p_ref = 0.001\n',
      'call': 'halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_oracle_halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.01, 0.03], dtype=float)\n'
               'weights = np.array([0.2, 0.8], dtype=float)\n'
               'losses = np.array([0.0, 0.0], dtype=float)\n'
               'conversions = np.array([0.4, 0.7], dtype=float)\n'
               'p_ref = 0.001\n',
      'call': 'halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_oracle_halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.001, 0.02], dtype=float)\n'
               'weights = np.array([1.0, 1.0], dtype=float)\n'
               'losses = np.array([0.2, 0.8], dtype=float)\n'
               'conversions = np.array([0.9, 0.05], dtype=float)\n'
               'p_ref = 0.001\n',
      'call': 'halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_oracle_halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.003, 0.006, 0.009], dtype=float)\n'
               'weights = np.array([4.0, 2.0, 9.0], dtype=float)\n'
               'losses = np.array([0.6, 0.2, 0.1], dtype=float)\n'
               'conversions = np.array([0.1, 0.3, 0.7], dtype=float)\n'
               'p_ref = 0.001\n',
      'call': 'halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_oracle_halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.001, 0.004, 0.008], dtype=float)\n'
               'weights = np.array([1.0, 0.0, 2.0], dtype=float)\n'
               'losses = np.array([0.4, 100.0, 0.8], dtype=float)\n'
               'conversions = np.array([0.2, 900.0, 0.1], dtype=float)\n'
               'p_ref = 0.001\n',
      'call': 'halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_oracle_halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.001, 0.0041, 0.0111], dtype=float)\n'
               'weights = np.array([0.7, 0.2, 0.1], dtype=float)\n'
               'losses = np.array([0.8, 0.3, 0.2], dtype=float)\n'
               'conversions = np.array([0.0, 0.1, 0.5], dtype=float)\n'
               'p_ref = 0.001\n',
      'call': 'halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_oracle_halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.002, 0.007], dtype=float)\n'
               'weights = np.array([0.3, 0.7], dtype=float)\n'
               'losses = np.array([0.15, 0.42], dtype=float)\n'
               'conversions = np.array([0.35, 0.18], dtype=float)\n'
               'p_ref = 0.009\n',
      'call': 'halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_oracle_halo_branching(*_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.02], dtype=float)\n'
               'weights = np.array([1.0], dtype=float)\n'
               'losses = np.array([0.2], dtype=float)\n'
               'conversions = np.array([0.3], dtype=float)\n'
               'p_ref = 0.001\n'
               '\n'
               'nodes[0] = 0.0\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_exception_code(_oracle_halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.02], dtype=float)\n'
               'weights = np.array([1.0], dtype=float)\n'
               'losses = np.array([0.2], dtype=float)\n'
               'conversions = np.array([0.3], dtype=float)\n'
               'p_ref = 0.001\n'
               '\n'
               'weights[0] = np.nan\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_exception_code(_oracle_halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.02], dtype=float)\n'
               'weights = np.array([1.0], dtype=float)\n'
               'losses = np.array([0.2], dtype=float)\n'
               'conversions = np.array([0.3], dtype=float)\n'
               'p_ref = 0.001\n'
               '\n'
               'losses = np.array([0.1,0.2])\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_exception_code(_oracle_halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.02], dtype=float)\n'
               'weights = np.array([1.0], dtype=float)\n'
               'losses = np.array([0.2], dtype=float)\n'
               'conversions = np.array([0.3], dtype=float)\n'
               'p_ref = 0.001\n'
               '\n'
               'conversions[0] = -0.1\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_exception_code(_oracle_halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.02], dtype=float)\n'
               'weights = np.array([1.0], dtype=float)\n'
               'losses = np.array([0.2], dtype=float)\n'
               'conversions = np.array([0.3], dtype=float)\n'
               'p_ref = 0.001\n'
               '\n'
               'weights[:] = 0.0\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_exception_code(_oracle_halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.02], dtype=float)\n'
               'weights = np.array([1.0], dtype=float)\n'
               'losses = np.array([0.2], dtype=float)\n'
               'conversions = np.array([0.3], dtype=float)\n'
               'p_ref = 0.001\n'
               '\n'
               'losses[:] = 0.0\n'
               'conversions[:] = 0.0\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_exception_code(_oracle_halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))'},
     {'setup': 'from copy import deepcopy as _copy_input\n'
               'import numpy as np\n'
               'nodes = np.array([0.02], dtype=float)\n'
               'weights = np.array([1.0], dtype=float)\n'
               'losses = np.array([0.2], dtype=float)\n'
               'conversions = np.array([0.3], dtype=float)\n'
               'p_ref = 0.001\n'
               '\n'
               'p_ref = -0.001\n'
               '\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))',
      'gold_call': '_exception_code(_oracle_halo_branching, *_copy_input((nodes, weights, losses, conversions, p_ref)))'}]
