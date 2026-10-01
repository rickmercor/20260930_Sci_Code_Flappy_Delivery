"""
Construct a finite-Fock squeezing or beam-mixer Gaussian unitary and optionally apply it to amplitude columns.

Use `a[n-1,n]=sqrt(n)` and the projected squeezing and mixing generators in

the signature. A full unitary matrix and its action on amplitude columns

describe the same operation. Spectral, matrix-exponential and equivalent

number-sector evaluations are all valid.

Returns
-------
Return `(d,d)` for S or `(d*d,d*d)` for B, or the same shape as supplied amplitude columns for U@columns. Every cell is a matrix element or amplitude.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gaussian_gate(d: int, kind: str, parameters: object, columns: object = None) -> object:
    """Return a finite-Fock Gaussian unitary as a complex ndarray.

    d is an integer in [2, 40], excluding bool. kind is 'squeeze' or 'mix'.
    parameters is a length-2 real numeric sequence, not complex or string data.
    For squeeze it is (r, phi), with 0 <= r <= 1 and any finite phase.
    For mix it is (theta, phi), with 0 <= theta <= pi/2 and any finite phase.
    S = exp((conj(z)*a*a-z*a.H*a.H)/2), z=r*exp(1j*phi).
    B = exp(theta*(exp(1j*phi)*a.H tensor a-exp(-1j*phi)*a tensor a.H)).
    a[n-1,n]=sqrt(n) on 0,...,d-1. The tensor order is |i,j> -> i*d+j.
    Return shape (d,d) for S or (d*d,d*d) for B. Generators are projected
    before exponentiation. Alternatively columns is a finite numeric real or
    complex matrix (dimension,rank), rank>=1, with squared Frobenius norm at
    most 1+1e-10, and the return is U@columns with that same shape. The two
    forms are equivalent. Each cell must agree to relative 1e-9 or absolute
    1e-12. Raise ValueError for any violation or nonfinite input.
    Returns
    -------
    Return `(d,d)` for S or `(d*d,d*d)` for B, or the same shape as supplied amplitude columns for U@columns. Every cell is a matrix element or amplitude.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Finite-Fock Gaussian optics: projected generators define the retained model.

These conventional gates support, but do not constitute, the adaptive source.
"""
import numpy as np
from functools import lru_cache
from scipy.linalg import expm
def _r6_real(value, shape=None):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or (shape is not None and raw.shape != shape):
            raise ValueError("real numeric kind and declared shape required")
        out = raw.astype(float)
        if not np.all(np.isfinite(out)):
            raise ValueError("finite data required")
        return out
    except (TypeError, OverflowError) as exc:
        raise ValueError("real numeric data required") from exc
def _r6_integer(value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("integer required, excluding bool")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError("integer outside domain")
    return value
def _r6_scalar(value, low=None, high=None):
    value = float(_r6_real(value, ()))
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError("scalar outside domain")
    return value
@lru_cache(maxsize=39)
def _r6_bs_blocks(d):
    blocks = []
    for total in range(2*d-1):
        pairs = [(i, total-i) for i in range(d) if 0 <= total-i < d]
        indices = np.array([i*d+j for i, j in pairs])
        generator = np.zeros((len(pairs), len(pairs)), complex)
        for col, (i, j) in enumerate(pairs):
            if i+1 < d and j > 0:
                generator[col+1, col] = np.sqrt((i+1)*j)
                generator[col, col+1] = -np.sqrt((i+1)*j)
        values, vectors = np.linalg.eigh(1j*generator)
        blocks.append((indices, np.array([i for i, _ in pairs]), values, vectors))
    return blocks
def _oracle_gaussian_gate(d: int, kind: str, parameters: object, columns: object = None) -> object:
    d = _r6_integer(d, 2, 40)
    parameters = _r6_real(parameters, (2,))
    if not isinstance(kind, str) or kind not in ("squeeze", "mix"):
        raise ValueError("unknown Gaussian gate")
    if columns is not None:
        raw = np.asarray(columns)
        dimension = d if kind == "squeeze" else d*d
        if raw.dtype.kind not in "iufc" or raw.ndim != 2 or raw.shape[0] != dimension or raw.shape[1] < 1:
            raise ValueError("numeric amplitude columns with gate dimension required")
        columns = raw.astype(complex)
        if not np.all(np.isfinite(columns)) or np.vdot(columns, columns).real > 1+1e-10:
            raise ValueError("finite amplitude columns with total weight at most one required")
    magnitude, phase = parameters
    if kind == "squeeze":
        _r6_scalar(magnitude, 0, 1)
        a = np.diag(np.sqrt(np.arange(1, d)), 1)
        z = magnitude*np.exp(1j*phase)
        unitary = expm((z.conjugate()*a@a-z*a.T@a.T)/2)
        return unitary if columns is None else unitary@columns
    _r6_scalar(magnitude, 0, np.pi/2)
    if magnitude == 0:
        return np.eye(d*d, dtype=complex) if columns is None else columns.copy()
    if columns is not None:
        # Apply the generator directly. Spectral reconstruction can introduce
        # a spurious component even when the input is exactly in its kernel;
        # subsequent rare-event conditioning would magnify that component.
        steps = max(1, int(np.ceil(2*d*magnitude)))
        h = magnitude/steps
        roots = np.sqrt(np.arange(1, d))
        weights = roots[:, None, None]*roots[None, :, None]
        forward = np.exp(1j*phase)*weights
        backward = np.exp(-1j*phase)*weights
        result = columns.reshape(d, d, -1).copy()
        for _ in range(steps):
            term = result.copy()
            moved = result.copy()
            for order in range(1, 25):
                derivative = np.zeros_like(term)
                derivative[1:, :-1] += forward*term[:-1, 1:]
                derivative[:-1, 1:] -= backward*term[1:, :-1]
                term = (h/order)*derivative
                moved += term
                if not np.any(term):
                    break
            result = moved
        return result.reshape(columns.shape)
    result = np.zeros((d*d, d*d), complex)
    for ids, ns, values, vectors in _r6_bs_blocks(d):
        phases = np.exp(1j*phase)**ns
        block = (vectors*np.exp(-1j*magnitude*values))@vectors.conj().T
        result[np.ix_(ids, ids)] = phases[:, None]*block*phases.conj()[None, :]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'pass\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 5, 'squeeze', [0.32, -0.7]))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 5, 'squeeze', [0.32, -0.7]))"},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'pass\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 4, 'mix', [0.0, 0.8]))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 4, 'mix', [0.0, 0.8]))"},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'pass\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 6, 'mix', [0.81, 1.3]))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 6, 'mix', [0.81, 1.3]))"},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'pass\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 3, 'squeeze', [1.0, 2.1]))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 3, 'squeeze', [1.0, 2.1]))"},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'w = np.zeros((25, 2), complex)\n'
           'w[[2, 7, 11], 0] = [0.2, 0.4j, -0.3]\n'
           'w[[3, 8, 13], 1] = [0.1j, -0.2, 0.5]\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 5, 'mix', [0.63, -0.9], w))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 5, 'mix', [0.63, -0.9], w))"},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'w = np.array([[0.1, 0.2j], [0.3, 0.0], [-0.2j, 0.4], [0.0, 0.1]])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 4, 'squeeze', [0.23, 0.72], w))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 4, 'squeeze', [0.23, 0.72], w))"},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'pass\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 4, 'mix', [0.3, 1e+308]))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 4, 'mix', [0.3, 1e+308]))"},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'import numpy as np\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, np.int64(2), 'squeeze', [1.0, -1e+300]))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, np.int64(2), 'squeeze', [1.0, "
               '-1e+300]))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'w = np.zeros((1600, 1), complex)\n'
           'w[2, 0] = np.sqrt(1 + 9e-11)\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 40, 'mix', [np.pi / 2, 0.4], w))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 40, 'mix', [np.pi / 2, 0.4], w))"},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'import numpy as np\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 3, 'squeeze', [0, 0], np.zeros((3, 2))))",
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 3, 'squeeze', [0, 0], np.zeros((3, "
               '2))))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'import numpy as np\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': "_case_encode(_case_run(gaussian_gate, 3, 'squeeze', np.array([1, 0]), np.eye(3)[:, "
          ':1].tolist()))',
  'gold_call': "_case_encode(_case_run(_oracle_gaussian_gate, 3, 'squeeze', np.array([1, 0]), "
               'np.eye(3)[:, :1].tolist()))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n'
           'pass',
  'call': "_case_value_error(lambda: gaussian_gate(5,'squeeze',[0.32, -0.7],np.zeros((2,3,4,5))))",
  'gold_call': "_case_value_error(lambda: _oracle_gaussian_gate(5,'squeeze',[0.32, "
               '-0.7],np.zeros((2,3,4,5))))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n'
           'pass',
  'call': "_case_value_error(lambda: gaussian_gate(np.nan,'squeeze',[0.32, -0.7],None))",
  'gold_call': "_case_value_error(lambda: _oracle_gaussian_gate(np.nan,'squeeze',[0.32, "
               '-0.7],None))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n'
           'pass',
  'call': "_case_value_error(lambda: gaussian_gate(5,'squeeze',np.array([-1e-06, -0.7]),None))",
  'gold_call': "_case_value_error(lambda: _oracle_gaussian_gate(5,'squeeze',np.array([-1e-06, "
               '-0.7]),None))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n'
           'pass',
  'call': "_case_value_error(lambda: gaussian_gate('0.5','squeeze',[0.32, -0.7],None))",
  'gold_call': "_case_value_error(lambda: _oracle_gaussian_gate('0.5','squeeze',[0.32, "
               '-0.7],None))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n'
           'pass',
  'call': "_case_value_error(lambda: gaussian_gate(5,'squeeze',[0.32, -0.7],np.full((5,1),2.)))",
  'gold_call': "_case_value_error(lambda: _oracle_gaussian_gate(5,'squeeze',[0.32, "
               '-0.7],np.full((5,1),2.)))'}]
