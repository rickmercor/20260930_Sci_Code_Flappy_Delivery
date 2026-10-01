"""
Apply one count-selected two-mode mixer-squeezers-mixer Gaussian controller while preserving branch mass.

The selected controller acts on the two retained modes as

$G=B(theta_b,phase_b) @ (S(r0,phi0) tensor S(r1,phi1)) @ B(theta_a,phase_a)$.

The two-mode state is already correlated before this operation. Neither

retained mode is replaced by a fresh vacuum. No measurement or loss occurs

inside this step. Both squeezing magnitudes vanish in the passive limit.

For density input, first form $C(H)$: hermitize, clip negative eigenvalues to

zero, and rescale the positive part to $m=max(0,Re trace(H))$, returning zero

when $m=0$.

Returns
-------
For density input return `G C(H) G.H` of shape `(d*d,d*d)`. For amplitude columns return G@state with the original shape, rank and physical amplitude. An exact zero remains zero without normalizing by zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def feedforward_state(state: object, controls: object, factorized: bool = False) -> object:
    """Apply one two-mode Gaussian controller, preserving physical branch mass.

    controls is a finite real numeric length-8 array in the order
    (theta_a,phase_a,r0,phi0,r1,phi1,theta_b,phase_b). Angles are in [0,pi/2],
    both squeezing magnitudes in [0,0.5], and all phases are unrestricted.
    Complex, string and boolean controls are invalid. The operation is
    G=B_b (S0 tensor S1) B_a with the shared finite-Fock gate conventions.
    factorized is bool or numpy.bool_. If False, state is a physical numeric
    (d*d,d*d) matrix, d in [2,40], interpreted as shared C(H), and the result
    is G C(H) G.H of that shape. If True, state is finite numeric real/complex
    amplitude columns (d*d,rank), rank>=1, squared Frobenius norm<=1+1e-10,
    and the result is G@state with the same shape. No factorization convention
    is imposed. The modes retain their order and there is no detector or
    additional loss in this operation. Exact zero stays zero. Preserve the
    input mass, including accepted roundoff and small positive mass, not unit
    probability. Every returned cell agrees to relative 1e-9 or absolute
    1e-12. Raise ValueError for all invalid kinds, shapes, finite-data,
    physicality, flag or control-domain conditions.
    Returns
    -------
    For density input return `G C(H) G.H` of shape `(d*d,d*d)`. For amplitude columns return G@state with the original shape, rank and physical amplitude. An exact zero remains zero without normalizing by zero.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Outcome-selected Gaussian processing of a retained two-mode state."""
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
def _oracle_feedforward_state(state: object, controls: object, factorized: bool = False) -> object:
    x = _r7_numeric(controls, (8,), True)
    factorized = _r7_bool(factorized)
    for i in (0, 6):
        _r7_scalar(x[i], 0, np.pi/2)
    for i in (2, 4):
        _r7_scalar(x[i], 0, .5)
    raw = _r7_numeric(state)
    if raw.ndim != 2:
        raise ValueError('matrix or amplitude columns required')
    d = _r7_dimension(raw.shape[0], 2)
    if factorized:
        root, amplitude = _r7_factor_parts(raw, d*d)
        mass = amplitude**2
    else:
        root, mass = _r7_state_parts(raw, d*d)
        amplitude = np.sqrt(mass)
    if amplitude == 0:
        return np.zeros_like(raw)
    w = _oracle_gaussian_gate(d, 'mix', x[:2], root)
    tensor = w.reshape(d, d, -1)
    s0 = _oracle_gaussian_gate(d, 'squeeze', x[2:4])
    s1 = _oracle_gaussian_gate(d, 'squeeze', x[4:6])
    tensor = np.einsum('ai,bj,ijr->abr', s0, s1, tensor, optimize=True)
    w = _oracle_gaussian_gate(d, 'mix', x[6:8], tensor.reshape(d*d, -1))
    return amplitude*w if factorized else mass*(w@w.conj().T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'v = np.array([0.2, 0.3j, -0.1, 0.4])\n'
               'rho = np.outer(v, v.conj())\n'
               'x = [0.36, 0.7, 0.23, -0.8, 0.19, 0.2, 0.51, -0.3]\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(feedforward_state, rho, x))',
      'gold_call': '_case_encode(_case_run(_oracle_feedforward_state, rho, x))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'rho = np.diag([5e-11, 1e-10, 0.0, 0.0])\n'
               'x = np.zeros(8)\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Resolve the contract's small positive signal at the bundled comparison scale.\n"
               '    return np.concatenate((shape, (1000.0 * value.real.astype(float)).ravel(), (1000.0 '
               '* value.imag.astype(float)).ravel()))\n',
      'call': '_case_encode(_case_run(feedforward_state, rho, x))',
      'gold_call': '_case_encode(_case_run(_oracle_feedforward_state, rho, x))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'w = np.zeros((9, 2), complex)\n'
               'w[[1, 5, 8], 0] = [0.2, 0.1j, -0.3]\n'
               'w[[2, 4], 1] = [0.2j, 0.1]\n'
               'x = [np.pi / 2, -0.2, 0.0, 0.7, 0.5, 0.1, 0.0, -0.6]\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(feedforward_state, w, x, True))',
      'gold_call': '_case_encode(_case_run(_oracle_feedforward_state, w, x, True))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'w = np.zeros((9, 3), complex)\n'
               'w[[1, 4, 5], 0] = [0.2, 0.3j, 0.1]\n'
               'w[[2, 7], 2] = [0.1j, 0.2]\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(feedforward_state, w, [0.4, 0.2, 0.5, 0.7, 0.5, -0.6, np.pi / 2, '
              '0.8], True))',
      'gold_call': '_case_encode(_case_run(_oracle_feedforward_state, w, [0.4, 0.2, 0.5, 0.7, 0.5, '
                   '-0.6, np.pi / 2, 0.8], True))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               's = np.diag([9e-11, 1 - 9e-11, 0.0, 0.0])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(feedforward_state, s, [0.2, 0.4, 0.0, 0.5, 0.0, -0.7, 0.3, '
              '0.6]))',
      'gold_call': '_case_encode(_case_run(_oracle_feedforward_state, s, [0.2, 0.4, 0.0, 0.5, 0.0, '
                   '-0.7, 0.3, 0.6]))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'w = np.array([[0.0], [1e-200], [2e-200j], [0.0]])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(feedforward_state, w, [0.2, 0.4, 0.1, 0.5, 0.2, -0.7, 0.3, 0.6], '
              'True))',
      'gold_call': '_case_encode(_case_run(_oracle_feedforward_state, w, [0.2, 0.4, 0.1, 0.5, 0.2, '
                   '-0.7, 0.3, 0.6], True))'},
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
      'call': '_case_encode(_case_run(feedforward_state, w, [0.3, 0.4, 0.0, 0.5, 0.0, -0.7, 0.2, 0.6], '
              'True))',
      'gold_call': '_case_encode(_case_run(_oracle_feedforward_state, w, [0.3, 0.4, 0.0, 0.5, 0.0, '
                   '-0.7, 0.2, 0.6], True))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: '
              'feedforward_state(np.diag([.2,.1,.3,.0]),np.zeros((2,3,4,5)),False))',
      'gold_call': '_case_value_error(lambda: '
                   '_oracle_feedforward_state(np.diag([.2,.1,.3,.0]),np.zeros((2,3,4,5)),False))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: '
              'feedforward_state(np.diag([.2,.1,.3,.0]),np.full(np.shape([.3,.4,.2,.5,.1,-.6,.7,.8]),np.nan),False))',
      'gold_call': '_case_value_error(lambda: '
                   '_oracle_feedforward_state(np.diag([.2,.1,.3,.0]),np.full(np.shape([.3,.4,.2,.5,.1,-.6,.7,.8]),np.nan),False))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '_case_bad=np.array([.3,.4,.2,.5,.1,-.6,.7,.8],float)\n'
               '_case_bad[0]=-1e-6\n',
      'call': '_case_value_error(lambda: feedforward_state(np.diag([.2,.1,.3,.0]),_case_bad,False))',
      'gold_call': '_case_value_error(lambda: '
                   '_oracle_feedforward_state(np.diag([.2,.1,.3,.0]),_case_bad,False))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': "_case_value_error(lambda: feedforward_state(np.diag([.2,.1,.3,.0]),'0.5',False))",
      'gold_call': '_case_value_error(lambda: '
                   "_oracle_feedforward_state(np.diag([.2,.1,.3,.0]),'0.5',False))"},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: '
              'feedforward_state(np.diag([-.1,-.2,.4,.4]),[.3,.4,.2,.5,.1,-.6,.7,.8],False))',
      'gold_call': '_case_value_error(lambda: '
                   '_oracle_feedforward_state(np.diag([-.1,-.2,.4,.4]),[.3,.4,.2,.5,.1,-.6,.7,.8],False))'},
     {'setup': 'import numpy as np\n'
               'def _case_zero_input(fn):\n'
               '    controls = np.array([0.36, 0.7, 0.23, -0.8, 0.19, 0.2, 0.51, -0.3])\n'
               '    value = np.asarray(fn(np.zeros((4, 4), complex), controls.copy(), False))\n'
               '    return bool(value.shape == (4, 4) and value.dtype.kind in "iufc"\n'
               '                and np.all(np.isfinite(value))\n'
               '                and np.all(np.abs(value) <= 1e-12))\n',
      'call': '_case_zero_input(feedforward_state)',
      'gold_call': '_case_zero_input(_oracle_feedforward_state)'},
     {'setup': 'import numpy as np\n'
               'def _case_zero_input(fn):\n'
               '    controls = np.array([0.36, 0.7, 0.23, -0.8, 0.19, 0.2, 0.51, -0.3])\n'
               '    value = np.asarray(fn(np.zeros((4, 2), complex), controls.copy(), True))\n'
               '    return bool(value.shape == (4, 2) and value.dtype.kind in "iufc"\n'
               '                and np.all(np.isfinite(value))\n'
               '                and np.all(np.abs(value) <= 1e-12))\n',
      'call': '_case_zero_input(feedforward_state)',
      'gold_call': '_case_zero_input(_oracle_feedforward_state)'}]
