"""
Prepare the squeezed three-mode source, herald the first count, and apply both memory losses to the retained pair.

Squeeze initial vacua 0,1,2, then apply preparation mixers to pairs (0,1),

(1,2),(0,2) in that temporal order. Destructively measure original mode 0

and apply independent memory attenuation to original modes 1 and 2. Retain

the correlated joint state, relabeling those modes 0 and 1. The supplied

count sequence labels distinct first records; no successful-record

renormalization is performed.

Returns
-------
Return complex shape `(len(counts),d*d,d*d)` in first-count order. Every density slice has its own physical probability in its trace. Zero-probability records remain full zero slices, with no padding or omitted cells.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def retained_states(d: int, squeezers: object, mixers: object, counts: object, efficiencies: object) -> object:
    """Return the first-record ensemble after both retained-mode memories.

    d is integer in [2,40], excluding bool. squeezers and mixers are finite
    real numeric (3,2) arrays, not complex/string/bool data. Each squeezer row
    is (r,phase), 0<=r<=1. Rows act on initial vacuum modes 0,1,2. Each mixer
    row is (theta,phase), 0<=theta<=pi/2, in temporal order (0,1),(1,2),(0,2).
    All phases are unrestricted finite radians. counts contains one or two
    DISTINCT nonnegative integer first records, excluding booleans, in return
    order. efficiencies is a finite real numeric length-3 array in [0,1]:
    first detector, memory on original mode 1, memory on original mode 2.
    Apply the given preparation, destructively measure original mode 0, then
    apply the two local attenuation channels. Return complex array
    (len(counts),d*d,d*d), unnormalized, with retained order |i,j> -> i*d+j
    for original modes 1,2. Zero-probability records remain zero rows. Each
    density cell agrees to relative 1e-9 or absolute 1e-12. Raise ValueError
    for invalid kinds, shapes, nonfinite entries, duplicate counts or any
    stated discrete/continuous domain violation.
    Returns
    -------
    Return complex shape `(len(counts),d*d,d*d)` in first-count order. Every density slice has its own physical probability in its trace. Zero-probability records remain full zero slices, with no padding or omitted cells.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Correlated quantum memory selected by a first optical measurement."""
import numpy as np
def _r7_numeric(value, shape=None, real=False):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in ('iuf' if real else 'iufc'):
            raise ValueError('numeric kind required')
        if shape is not None and raw.shape != shape:
            raise ValueError('shape mismatch')
        if not np.all(np.isfinite(raw)):
            raise ValueError('finite numeric values required')
        result = raw.astype(float if real else complex)
        if not np.all(np.isfinite(result)):
            raise ValueError('finite double-precision data required')
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('numeric data required') from exc
def _r7_int(value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('integer required, excluding bool')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError('integer outside domain')
    return value
def _r7_counts(value, length=None, distinct=False):
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1 or not 1 <= len(raw) <= 2:
        raise ValueError('one or two counts required')
    if length is not None and len(raw) != length:
        raise ValueError('count length mismatch')
    result = [_r7_int(v, 0) for v in raw]
    if distinct and len(set(result)) != len(result):
        raise ValueError('distinct first records required')
    return result
def _oracle_retained_states(d: int, squeezers: object, mixers: object, counts: object, efficiencies: object) -> object:
    d = _r7_int(d, 2, 40)
    squeezers = _r7_numeric(squeezers, (3, 2), True)
    mixers = _r7_numeric(mixers, (3, 2), True)
    efficiencies = _r7_numeric(efficiencies, (3,), True)
    counts = _r7_counts(counts, distinct=True)
    if np.any(squeezers[:, 0] < 0) or np.any(squeezers[:, 0] > 1):
        raise ValueError('squeezing magnitude outside [0,1]')
    if np.any(mixers[:, 0] < 0) or np.any(mixers[:, 0] > np.pi/2):
        raise ValueError('mixing angle outside [0,pi/2]')
    if np.any(efficiencies < 0) or np.any(efficiencies > 1):
        raise ValueError('efficiency outside [0,1]')
    vacua = [_oracle_gaussian_gate(d, 'squeeze', row)[:, 0] for row in squeezers]
    psi = np.einsum('i,j,k->ijk', *vacua)
    for pair, row in zip(((0, 1), (1, 2), (0, 2)), mixers):
        permutation = list(pair)+[i for i in range(3) if i not in pair]
        columns = psi.transpose(permutation).reshape(d*d, d)
        moved = _oracle_gaussian_gate(d, 'mix', row, columns)
        psi = moved.reshape(d, d, d).transpose(np.argsort(permutation))
    rows = []
    for count in counts:
        rho = _oracle_herald(psi.reshape(d**3, 1), count, efficiencies[0], 0, True, 3)
        rho = _oracle_attenuate(rho, efficiencies[1], 0, 2)
        rows.append(_oracle_attenuate(rho, efficiencies[2], 1, 2))
    return np.array(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           's = np.array([[0.22, 0.1], [0.31, -0.4], [0.19, 0.8]])\n'
           'b = np.array([[0.42, 0.6], [0.27, -0.2], [0.38, 0.9]])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(retained_states, 4, s, b, [1, 2], [0.87, 0.93, 0.81]))',
  'gold_call': '_case_encode(_case_run(_oracle_retained_states, 4, s, b, [1, 2], [0.87, 0.93, '
               '0.81]))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           's = np.zeros((3, 2))\n'
           'b = np.zeros((3, 2))\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(retained_states, 3, s, b, [0, 1], [1.0, 1.0, 1.0]))',
  'gold_call': '_case_encode(_case_run(_oracle_retained_states, 3, s, b, [0, 1], [1.0, 1.0, '
               '1.0]))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           's = np.array([[0.0, 0.8], [1.0, -0.3], [0.25, 0.2]])\n'
           'b = np.array([[0.0, 0.7], [0.7, 0.4], [np.pi / 2, -0.8]])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(retained_states, 3, s, b, [2], [0.0, 0.74, 0.0]))',
  'gold_call': '_case_encode(_case_run(_oracle_retained_states, 3, s, b, [2], [0.0, 0.74, 0.0]))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           's = np.array([[0.3, 1e+300], [0.2, -1e+300], [0.4, 1e+308]])\n'
           'b = np.array([[np.pi / 2, 0.4], [0.0, 0.8], [0.3, -0.7]])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(retained_states, np.int64(3), s, b, [0, 1], [0.0, 1.0, 0.0]))',
  'gold_call': '_case_encode(_case_run(_oracle_retained_states, np.int64(3), s, b, [0, 1], [0.0, '
               '1.0, 0.0]))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: '
          'retained_states(3,np.zeros((2,3,4,5)),np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_retained_states(3,np.zeros((2,3,4,5)),np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: '
          'retained_states(3,np.full(np.shape(np.array([[.2,.1],[.3,.5],[.1,-.7]])),np.nan),np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_retained_states(3,np.full(np.shape(np.array([[.2,.1],[.3,.5],[.1,-.7]])),np.nan),np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n'
           '_case_bad=np.array(np.array([[.2,.1],[.3,.5],[.1,-.7]]),float)\n'
           '_case_bad[0,0]=-1e-6\n',
  'call': '_case_value_error(lambda: '
          'retained_states(3,_case_bad,np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_retained_states(3,_case_bad,np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: '
          "retained_states(3,'0.5',np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))",
  'gold_call': '_case_value_error(lambda: '
               "_oracle_retained_states(3,'0.5',np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))"},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: '
          'retained_states(41,np.array([[.2,.1],[.3,.5],[.1,-.7]]),np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_retained_states(41,np.array([[.2,.1],[.3,.5],[.1,-.7]]),np.array([[.3,.4],[.2,-.8],[.5,.6]]),[1,2],[.8,.9,.7]))'}]
