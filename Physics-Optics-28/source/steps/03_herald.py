"""
Condition on an inefficient photon-number-resolving count and trace out the detected mode while preserving unnormalized branch weight.

Destructive number detection after vacuum loss has diagonal effect

$E_k[n,n]=comb(n,k)*eta**k*(1-eta)**(n-k)$ for n>=k, otherwise zero.

Take the partial trace of this effect on detected_mode only. For amplitude

columns W the physical input is W W.H. For a density input it is C(H).

Both forms preserve the same record mass and surviving quantum coherences.

For density input, $C(H)$ means hermitize, clip negative eigenvalues to zero,

and rescale the positive part to $m=max(0,Re trace(H))$, with zero returned

when $m=0$.

Returns
-------
Return `(d**(modes-1),d**(modes-1))`, in the surviving modes' original order. The trace includes the incoming branch probability. Impossible counts give the zero matrix, not an exception, even at zero efficiency.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def herald(state: object, count: int, efficiency: float, detected_mode: int = 0, factorized: bool = False, modes: int = 2) -> object:
    """Return the unnormalized retained density matrix after a PNR record.

    modes is integer 2 or 3, detected_mode is integer in [0,modes-1], and
    count is a nonnegative integer, all excluding booleans. Local dimension
    d is in [2,40] and tensor order is lexicographic. factorized must be bool
    or numpy.bool_. If False, state is a numeric physical (d**modes,d**modes)
    matrix interpreted as shared C(H). If True, state is a finite real/complex
    numeric array (d**modes,rank), rank>=1, representing state@state.H with
    squared Frobenius norm <=1+1e-10. No particular factor representation is
    required. efficiency is finite real numeric in [0,1], not complex/string/
    bool. Detection has diagonal effect comb(n,count)*efficiency**count*
    (1-efficiency)**(n-count) for n>=count and zero otherwise. Trace out only
    detected_mode, keeping the other modes in their original order and all
    their coherences. Return shape (d**(modes-1),d**(modes-1)), not a normalized
    state. Counts >=d and zero-probability outcomes return zero. Cells agree to
    relative 1e-9 or absolute 1e-12. Raise ValueError for every kind, shape,
    finite-data, physicality or discrete/continuous domain violation.
    Returns
    -------
    Return `(d**(modes-1),d**(modes-1))`, in the surviving modes' original order. The trace includes the incoming branch probability. Impossible counts give the zero matrix, not an exception, even at zero efficiency.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Destructive number-resolved measurement with a surviving subsystem."""
import math
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
def _r7_scalar(value, low=None, high=None):
    value = float(_r7_numeric(value, (), True))
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError('scalar outside domain')
    return value
def _r7_bool(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean required')
    return bool(value)
def _r7_dimension(size, modes):
    for d in range(2, 41):
        if d**modes == size:
            return d
    raise ValueError('dimension is not a supported tensor power')
def _r7_state_parts(value, size=None):
    rho = _r7_numeric(value)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError('square state required')
    if size is not None and rho.shape != (size, size):
        raise ValueError('state dimension mismatch')
    if np.max(np.abs(rho-rho.conj().T)) > 1e-10:
        raise ValueError('Hermitian state required')
    h = (rho+rho.conj().T)/2
    trace = math.fsum(float(value) for value in h.diagonal().real)
    if trace < -1e-10 or trace > 1+1e-10:
        raise ValueError('state trace outside domain')
    scale = float(np.max(np.abs(h)))
    if scale == 0:
        return np.zeros_like(h), 0.
    try:
        values, vectors = np.linalg.eigh(h.real/scale+1j*(h.imag/scale))
    except np.linalg.LinAlgError as exc:
        raise ValueError('physical state required') from exc
    if values.min()*scale < -1e-10:
        raise ValueError('positive state required')
    mass = max(0., trace)
    if mass == 0:
        return np.zeros_like(h), 0.
    positive = np.maximum(values, 0.)
    root = vectors*np.sqrt(positive/positive.sum())
    return root, mass
def _r7_factor_parts(value, size):
    w = _r7_numeric(value)
    if w.ndim != 2 or w.shape[0] != size or w.shape[1] < 1:
        raise ValueError('amplitude-column shape mismatch')
    scale = float(np.max(np.abs(w)))
    if scale == 0:
        return w, 0.
    if scale > np.sqrt(1+1e-10):
        raise ValueError('amplitude mass exceeds one')
    scaled = w.real/scale+1j*(w.imag/scale)
    norm = float(np.sqrt(np.vdot(scaled, scaled).real))
    amplitude = scale*norm
    if amplitude**2 > 1+1e-10:
        raise ValueError('amplitude mass exceeds one')
    return scaled/norm, amplitude
def _oracle_herald(state: object, count: int, efficiency: float, detected_mode: int = 0, factorized: bool = False, modes: int = 2) -> object:
    modes = _r7_int(modes, 2, 3)
    detected_mode = _r7_int(detected_mode, 0, modes-1)
    count = _r7_int(count, 0)
    efficiency = _r7_scalar(efficiency, 0, 1)
    factorized = _r7_bool(factorized)
    raw = _r7_numeric(state)
    if raw.ndim != 2:
        raise ValueError('matrix or amplitude columns required')
    d = _r7_dimension(raw.shape[0], modes)
    if factorized:
        root, amplitude = _r7_factor_parts(raw, d**modes)
        mass = amplitude**2
    else:
        root, mass = _r7_state_parts(raw, d**modes)
    size = d**(modes-1)
    if count >= d or mass == 0:
        return np.zeros((size, size), complex)
    shaped = root.reshape((d,)*modes+(root.shape[1],))
    order = [i for i in range(modes) if i != detected_mode]+[detected_mode, modes]
    retained = shaped.transpose(order).reshape(size, d, -1)
    result = np.zeros((size, size), complex)
    for n in range(count, d):
        weight = math.comb(n, count)*efficiency**count*(1-efficiency)**(n-count)
        result += weight*(retained[:, n]@retained[:, n].conj().T)
    return mass*result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'w = np.array([[0.1, 0.2j], [0.2j, 0.0], [-0.1, 0.3], [0.2, 0.1j]])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(herald, w, 1, 0.84, 0, True, 2))',
  'gold_call': '_case_encode(_case_run(_oracle_herald, w, 1, 0.84, 0, True, 2))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'rho = np.diag([0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(herald, rho, 2, 1.0, 1, False, 3))',
  'gold_call': '_case_encode(_case_run(_oracle_herald, rho, 2, 1.0, 1, False, 3))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'w = np.zeros((27, 2), complex)\n'
           'w[[1, 6, 17], 0] = [0.2, 0.3j, -0.1]\n'
           'w[[2, 10, 23], 1] = [0.1j, 0.2, 0.1]\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(herald, w, 1, 0.67, 2, True, 3))',
  'gold_call': '_case_encode(_case_run(_oracle_herald, w, 1, 0.67, 2, True, 3))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'w = np.zeros((27, 2), complex)\n'
           'w[[1, 4, 5, 13, 16], 0] = [0.2, 0.3j, 0.1, 0.2j, -0.1]\n'
           'w[[2, 8, 17], 1] = [0.1j, 0.2, -0.2j]\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(herald, w, 1, 0.73, 1, np.bool_(True), 3))',
  'gold_call': '_case_encode(_case_run(_oracle_herald, w, 1, 0.73, 1, np.bool_(True), 3))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'w = np.zeros((27, 2), complex)\n'
           'w[[1, 4, 5, 13, 16], 0] = [0.2, 0.3j, 0.1, 0.2j, -0.1]\n'
           'w[[2, 8, 17], 1] = [0.1j, 0.2, -0.2j]\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(herald, w @ w.conj().T, 1, 0.73, 1, False, 3))',
  'gold_call': '_case_encode(_case_run(_oracle_herald, w @ w.conj().T, 1, 0.73, 1, False, 3))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'w = np.zeros((27, 1), complex)\n'
           'w[[1, 4, 5, 13, 16], 0] = [0.2, 0.3j, 0.1, 0.2j, -0.1]\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(herald, w, 0, 0.0, 2, True, 3))',
  'gold_call': '_case_encode(_case_run(_oracle_herald, w, 0, 0.0, 2, True, 3))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           's = np.diag([0.2, 0.1, 0.3, 0.0])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(herald, s, 10 ** 100, 1.0))',
  'gold_call': '_case_encode(_case_run(_oracle_herald, s, 10 ** 100, 1.0))'},
 {'setup': 'import copy\n'
           'def _case_run(fn, *args, **kwargs):\n'
           '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
           'w = np.diag([0.2, 0.1, 0.3, 0.0])\n'
           'def _case_encode(value):\n'
           '    value = np.asarray(value)\n'
           '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
           'np.all(np.isfinite(value)):\n'
           '        return np.array([np.nan])\n'
           '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
           "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
           '    return np.concatenate((shape, value.real.astype(float).ravel(), '
           'value.imag.astype(float).ravel()))\n',
  'call': '_case_encode(_case_run(herald, w, 0, 0.8, 0, True, 2))',
  'gold_call': '_case_encode(_case_run(_oracle_herald, w, 0, 0.8, 0, True, 2))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: herald(np.zeros((2,3,4,5)),1,.8,0,False,2))',
  'gold_call': '_case_value_error(lambda: _oracle_herald(np.zeros((2,3,4,5)),1,.8,0,False,2))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: '
          'herald(np.full(np.shape(np.diag([.2,.1,.3,.0])),np.nan),1,.8,0,False,2))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_herald(np.full(np.shape(np.diag([.2,.1,.3,.0])),np.nan),1,.8,0,False,2))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: herald(np.diag([.2,.1,.3,.0]),1,-1e-06,0,False,2))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_herald(np.diag([.2,.1,.3,.0]),1,-1e-06,0,False,2))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': "_case_value_error(lambda: herald('0.5',1,.8,0,False,2))",
  'gold_call': "_case_value_error(lambda: _oracle_herald('0.5',1,.8,0,False,2))"},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: herald(np.diag([-.1,-.2,.4,.4]),1,.8,0,False,2))',
  'gold_call': '_case_value_error(lambda: '
               '_oracle_herald(np.diag([-.1,-.2,.4,.4]),1,.8,0,False,2))'}]
