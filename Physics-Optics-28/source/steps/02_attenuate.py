"""
Apply quantum-limited vacuum attenuation to one selected mode of a one- or two-mode density matrix.

For intensity transmission eta, use

$K_l[n-l,n]=sqrt(comb(n,l)*(1-eta)**l*eta**(n-l))$. The source amplitude

parameter is `sqrt(eta)`. Apply `sum_l K_l C(H) K_l.H` on the selected mode,

with identity on every other mode. Unobserved loss does not measure all

surviving occupations. This prompt-defined map is the generic vacuum

beam-splitter Stinespring channel, not a paper-derived input. At eta=0 the

selected mode becomes vacuum, retaining the other mode's reduced state;

eta=1 is the identity channel on C(H).

Here $C(H)$ is explicit in this step: hermitize the supplied density, clip

negative eigenvalues to zero, then rescale the positive part to

$m=max(0,Re trace(H))$; return zero when $m=0$.

Returns
-------
Return the same square density shape as the input, preserving its physical trace. All coherences and both tensor axes are included in the output budget.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def attenuate(rho: object, eta: float, mode: int = 0, modes: int = 1) -> object:
    """Return the complex density matrix after single-mode vacuum attenuation.

    rho is a finite real/complex numeric square matrix of size d**modes, where
    d is in [2,40]. modes is integer 1 or 2 and mode is an integer in
    [0,modes-1], excluding booleans. Tensor order is lexicographic in mode
    occupations. eta is a finite real numeric scalar in [0,1], not complex,
    boolean or string data. rho obeys the shared physicality convention:
    max(abs(rho-rho.H))<=1e-10, trace real part in [-1e-10,1+1e-10], and
    Hermitian-part minimum eigenvalue >=-1e-10. Interpret it as C(H), the
    disclosed trace-preserving PSD canonicalization, before propagation.
    K_l[n-l,n]=sqrt(comb(n,l)*(1-eta)**l*eta**(n-l)); act with K_l only on
    mode, sum K_l C(H) K_l.H, and retain the original record mass, not unit
    probability. Return the same shape, including coherences. All cells must
    agree to relative 1e-9 or absolute 1e-12. Raise ValueError for any invalid
    kind, shape, discrete argument, nonfinite value or physical/domain failure.
    Returns
    -------
    Return the same square density shape as the input, preserving its physical trace. All coherences and both tensor axes are included in the output budget.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Coherent vacuum attenuation on one mode of a retained subsystem."""
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
def _r7_target(value, d):
    target = _r7_numeric(value, (d,))
    if abs(np.vdot(target, target).real-1) > 1e-10:
        raise ValueError('normalized target required')
    return target
def _oracle_attenuate(rho: object, eta: float, mode: int = 0, modes: int = 1) -> object:
    modes = _r7_int(modes, 1, 2)
    mode = _r7_int(mode, 0, modes-1)
    eta = _r7_scalar(eta, 0, 1)
    raw = _r7_numeric(rho)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError('square state required')
    d = _r7_dimension(raw.shape[0], modes)
    root, mass = _r7_state_parts(raw)
    if mass == 0:
        return np.zeros_like(raw)
    state = (root@root.conj().T).reshape((d,)*(2*modes))
    permutation = [mode, modes+mode]+[i for i in range(2*modes) if i not in (mode, modes+mode)]
    arranged = state.transpose(permutation)
    result = np.zeros_like(arranged)
    for ell in range(d):
        coefficients = np.array([math.sqrt(math.comb(i+ell, ell)*(1-eta)**ell*eta**i) for i in range(d-ell)])
        shape = (d-ell, d-ell)+(1,)*(2*modes-2)
        result[:d-ell, :d-ell] += arranged[ell:, ell:]*(coefficients[:, None]*coefficients[None, :]).reshape(shape)
    return mass*result.transpose(np.argsort(permutation)).reshape(raw.shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'rho = np.array([[0.4, 0.1j], [-0.1j, 0.3]])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(attenuate, rho, 0.73))',
      'gold_call': '_case_encode(_case_run(_oracle_attenuate, rho, 0.73))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'rho = np.diag([0.2, 0.1, 0.3, 0.0])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(attenuate, rho, 0.0, 1, 2))',
      'gold_call': '_case_encode(_case_run(_oracle_attenuate, rho, 0.0, 1, 2))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'v = np.array([0.1, 0.2j, 0.3, -0.1j])\n'
               'rho = np.outer(v, v.conj())\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(attenuate, rho, 1.0, 0, 2))',
      'gold_call': '_case_encode(_case_run(_oracle_attenuate, rho, 1.0, 0, 2))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'v = np.array([0.1, 0.2j, 0.3, -0.1j])\n'
               's = np.outer(v, v.conj())\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(attenuate, s, 0.62, 1, 2))',
      'gold_call': '_case_encode(_case_run(_oracle_attenuate, s, 0.62, 1, 2))'},
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
               "    # Resolve the contract's small positive signal at the bundled comparison scale.\n"
               '    return np.concatenate((shape, (1000.0 * value.real.astype(float)).ravel(), (1000.0 '
               '* value.imag.astype(float)).ravel()))\n',
      'call': '_case_encode(_case_run(attenuate, s, 1.0, 0, 2))',
      'gold_call': '_case_encode(_case_run(_oracle_attenuate, s, 1.0, 0, 2))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               's = np.diag([0.0, 2e-280, 0.0, 0.0])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(attenuate, s, 0.6, 1, 2))',
      'gold_call': '_case_encode(_case_run(_oracle_attenuate, s, 0.6, 1, 2))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: attenuate(np.zeros((2,3,4,5)),.7,0,2))',
      'gold_call': '_case_value_error(lambda: _oracle_attenuate(np.zeros((2,3,4,5)),.7,0,2))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: '
              'attenuate(np.full(np.shape(np.diag([.2,.1,.3,.0])),np.nan),.7,0,2))',
      'gold_call': '_case_value_error(lambda: '
                   '_oracle_attenuate(np.full(np.shape(np.diag([.2,.1,.3,.0])),np.nan),.7,0,2))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: attenuate(np.diag([.2,.1,.3,.0]),-1e-06,0,2))',
      'gold_call': '_case_value_error(lambda: _oracle_attenuate(np.diag([.2,.1,.3,.0]),-1e-06,0,2))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': "_case_value_error(lambda: attenuate('0.5',.7,0,2))",
      'gold_call': "_case_value_error(lambda: _oracle_attenuate('0.5',.7,0,2))"},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: attenuate(np.diag([-.1,-.2,.4,.4]),.7,0,2))',
      'gold_call': '_case_value_error(lambda: _oracle_attenuate(np.diag([-.1,-.2,.4,.4]),.7,0,2))'}]
