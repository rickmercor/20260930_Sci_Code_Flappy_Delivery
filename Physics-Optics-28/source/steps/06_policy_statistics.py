"""
Compute each labeled record's joint success probability and target overlap after its controller, second herald, and final loss.

Each labeled first record selects its own G. After G, measure retained mode 1

with the paired SECOND count and detector efficiency, then apply final signal

loss to retained mode 0. The final state sigma_k has joint incident-attempt

probability $P_k=trace(sigma_k)$ and overlap `M_k=target.H@sigma_k@target`.

The density and arbitrary amplitude-factor representations are equivalent.

Distinct first records form a classical ensemble, not a coherent superposition.

Each density slice is interpreted with $C(H)$: hermitize it, clip negative

eigenvalues to zero, and rescale the positive part to

$m=max(0,Re trace(H))$, using zero when $m=0$.

Returns
-------
Return real shape `(b,2)`, row k exactly `(P_k,M_k)`. Both cells are zero for a zero-success branch. All 2*b cells are specified, with no unused padding.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def policy_statistics(states: object, target: object, controls: object, counts: object, efficiencies: object, factorized: bool = False) -> object:
    """Return the (P_k,M_k) array for one or two labeled first records.

    factorized is bool or numpy.bool_. With False, states is finite numeric
    real/complex shape (b,d*d,d*d), b=1 or 2, d in [2,40], containing physical
    matrices interpreted individually by shared C(H). Their total real trace
    is <=1+1e-10. With True, states has shape (b,d*d,rank), rank>=1, and is an
    ensemble of amplitude factors with total squared norm<=1+1e-10. Zero
    columns may pad different factor ranks. No factor convention is required.
    target is a finite numeric real/complex length-d vector with squared norm
    within 1e-10 of one. controls is a real numeric (b,8) array of controllers
    with feedforward_state domains. counts has b nonnegative integers,
    excluding booleans, specifying SECOND counts paired in the same order as
    states. Repeated second counts are allowed. efficiencies is a finite real
    numeric length-2 array in [0,1]: second detector, final signal transmission.
    Apply each G, destructively measure retained mode 1, then attenuate mode 0.
    Row k is (trace(sigma_k),target.H@sigma_k@target), including first-record
    mass. Both columns are real and zero when that record has zero success.
    Never average normalized branches equally. Cells agree to relative 1e-9
    or absolute 1e-12. Raise ValueError for every kind, shape, finite-data,
    physicality, normalization, total-mass, flag or stated domain violation.
    Returns
    -------
    Return real shape `(b,2)`, row k exactly `(P_k,M_k)`. Both cells are zero for a zero-success branch. All 2*b cells are specified, with no unused padding.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Physical success and target overlap for a record-indexed instrument."""
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
def _r7_policy_inputs(states, target, controls, counts, efficiencies, factorized):
    factorized = _r7_bool(factorized)
    raw = _r7_numeric(states)
    if raw.ndim != 3 or raw.shape[0] not in (1, 2):
        raise ValueError('one or two branch matrices required')
    d = _r7_dimension(raw.shape[1], 2)
    target = _r7_target(target, d)
    controls = _r7_numeric(controls, (len(raw), 8), True)
    if np.any(controls[:, [0, 6]] < 0) or np.any(controls[:, [0, 6]] > np.pi/2):
        raise ValueError('mixing angle outside domain')
    if np.any(controls[:, [2, 4]] < 0) or np.any(controls[:, [2, 4]] > .5):
        raise ValueError('inline squeezing outside domain')
    counts = _r7_counts(counts, len(raw))
    efficiencies = _r7_numeric(efficiencies, (2,), True)
    if np.any(efficiencies < 0) or np.any(efficiencies > 1):
        raise ValueError('efficiency outside domain')
    roots, masses = [], []
    for state in raw:
        if factorized:
            root, amplitude = _r7_factor_parts(state, d*d)
            mass = amplitude**2
        else:
            root, mass = _r7_state_parts(state, d*d)
        roots.append(root)
        masses.append(mass)
    if sum(masses) > 1+1e-10:
        raise ValueError('total physical mass exceeds one')
    return d, roots, masses, target, controls, counts, efficiencies
def _oracle_policy_statistics(states: object, target: object, controls: object, counts: object, efficiencies: object, factorized: bool = False) -> object:
    if states is None:
        raise ValueError('states are required')
    d, roots, masses, target, controls, counts, efficiencies = _r7_policy_inputs(states, target, controls, counts, efficiencies, factorized)
    result = np.zeros((len(roots), 2))
    for k, (root, mass) in enumerate(zip(roots, masses)):
        if mass == 0:
            continue
        moved = _oracle_feedforward_state(root, controls[k], True)
        conditional = _oracle_herald(moved, counts[k], efficiencies[0], 1, True, 2)
        signal = _oracle_attenuate(conditional, efficiencies[1])
        result[k] = mass*np.array([np.trace(signal).real, np.vdot(target, signal@target).real])
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'v = np.array([0.2, 0.3j, -0.1, 0.4])\n'
               's = np.array([np.outer(v, v.conj()), np.diag([0.1, 0.0, 0.08, 0.02])])\n'
               't = np.array([1.0, 1j]) / np.sqrt(2)\n'
               'x = np.array([[0.2, 0.7, 0.13, -0.4, 0.21, 0.5, 0.6, -0.2], [0.7, -0.2, 0.3, 0.6, 0.1, '
               '0.8, 0.4, 0.9]])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(policy_statistics, s, t, x, [0, 1], [0.81, 0.92]))',
      'gold_call': '_case_encode(_case_run(_oracle_policy_statistics, s, t, x, [0, 1], [0.81, 0.92]))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               's = np.zeros((1, 4, 4))\n'
               't = np.array([1.0, 0.0])\n'
               'x = np.zeros((1, 8))\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(policy_statistics, s, t, x, [0], [0.0, 0.0]))',
      'gold_call': '_case_encode(_case_run(_oracle_policy_statistics, s, t, x, [0], [0.0, 0.0]))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'w = np.zeros((2, 9, 2), complex)\n'
               'w[0, [1, 5], 0] = [0.3, 0.2j]\n'
               'w[1, [2, 7], 1] = [0.1j, 0.4]\n'
               't = np.array([0.5, 1j, 0.5]) / np.sqrt(1.5)\n'
               'x = np.array([[0.7, 0.2, 0.5, 0.8, 0.2, -0.3, 0.4, 0.9], [0.5, -0.9, 0.1, 0.4, 0.4, '
               '0.5, 0.8, 0.1]])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(policy_statistics, w, t, x, [1, 1], [1.0, 0.73], True))',
      'gold_call': '_case_encode(_case_run(_oracle_policy_statistics, w, t, x, [1, 1], [1.0, 0.73], '
                   'True))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'v = np.array([0.2, 0.3, 0.1j, -0.2])\n'
               's = np.array([np.outer(v, v.conj())])\n'
               't = np.array([1.0, 1.0]) / np.sqrt(2)\n'
               'x = np.array([[0.3, 0.7, 0.0, 0.5, 0.0, -0.4, 0.6, 0.1]])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(policy_statistics, s, t, x, [0], [0.9, 0.8]))',
      'gold_call': '_case_encode(_case_run(_oracle_policy_statistics, s, t, x, [0], [0.9, 0.8]))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'w = np.zeros((2, 4, 2), complex)\n'
               'w[0, [1, 2], 0] = [0.3, 0.2j]\n'
               'w[1, [0, 3], 1] = [0.2, 0.4j]\n'
               't = np.array([1.0, 1j]) / np.sqrt(2)\n'
               'x = np.array([[0.2, 0.5, 0.1, 0.7, 0.2, 0.3, 0.4, 0.9], [0.3, -0.5, 0.2, 0.6, 0.3, '
               '0.4, 0.7, -0.2]])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(policy_statistics, w, t, x, [1, 0], [0.9, 0.8], True))',
      'gold_call': '_case_encode(_case_run(_oracle_policy_statistics, w, t, x, [1, 0], [0.9, 0.8], '
                   'True))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'w = np.zeros((2, 4, 2), complex)\n'
               'w[0, [1, 2], 0] = [0.3, 0.2j]\n'
               'w[1, [0, 3], 1] = [0.2, 0.4j]\n'
               't = np.array([1.0, 1j]) / np.sqrt(2)\n'
               'x = np.array([[0.2, 0.5, 0.1, 0.7, 0.2, 0.3, 0.4, 0.9], [0.3, -0.5, 0.2, 0.6, 0.3, '
               '0.4, 0.7, -0.2]])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(policy_statistics, w[::-1], t, x[::-1], [0, 1], [0.9, 0.8], '
              'True))',
      'gold_call': '_case_encode(_case_run(_oracle_policy_statistics, w[::-1], t, x[::-1], [0, 1], '
                   '[0.9, 0.8], True))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               's = np.array([np.diag([-9e-11, 1 + 9e-11, 0.0, 0.0])])\n'
               't = np.array([1.0, 0.0])\n'
               'x = np.zeros((1, 8))\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(policy_statistics, s, t, x, [1], [1.0, 1.0]))',
      'gold_call': '_case_encode(_case_run(_oracle_policy_statistics, s, t, x, [1], [1.0, 1.0]))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               's = np.array([np.diag([-5e-11, 0.0, 1e-10, 0.0])])\n'
               't = np.array([0.0, 1.0])\n'
               'x = np.zeros((1, 8))\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Resolve the contract's small positive signal at the bundled comparison scale.\n"
               '    return np.concatenate((shape, (1000.0 * value.real.astype(float)).ravel(), (1000.0 '
               '* value.imag.astype(float)).ravel()))\n',
      'call': '_case_encode(_case_run(policy_statistics, s, t, x, [0], [1.0, 1.0]))',
      'gold_call': '_case_encode(_case_run(_oracle_policy_statistics, s, t, x, [0], [1.0, 1.0]))'},
     {'setup': 'import copy\n'
               'def _case_run(fn, *args, **kwargs):\n'
               '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n'
               'w = np.array([np.diag([0.2, 0.1, 0.3, 0.0])])\n'
               't = np.array([1.0, 1j]) / np.sqrt(2)\n'
               'x = np.array([[0.3, 0.4, 0.0, 0.5, 0.0, -0.7, 0.2, 0.6]])\n'
               'def _case_encode(value):\n'
               '    value = np.asarray(value)\n'
               '    if value.dtype.kind not in "iufc" or value.ndim > 4 or not '
               'np.all(np.isfinite(value)):\n'
               '        return np.array([np.nan])\n'
               '    shape = np.array((value.ndim, *value.shape, *((0,) * (4 - value.ndim))), float)\n'
               "    # Studio's 1e-9 numeric comparison applies to each original component.\n"
               '    return np.concatenate((shape, value.real.astype(float).ravel(), '
               'value.imag.astype(float).ravel()))\n',
      'call': '_case_encode(_case_run(policy_statistics, w, t, x, [0], [0.8, 0.9], True))',
      'gold_call': '_case_encode(_case_run(_oracle_policy_statistics, w, t, x, [0], [0.8, 0.9], '
                   'True))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: '
              'policy_statistics(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],np.zeros((2,3,4,5)),[1],[.8,.9],False))',
      'gold_call': '_case_value_error(lambda: '
                   '_oracle_policy_statistics(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],np.zeros((2,3,4,5)),[1],[.8,.9],False))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: '
              'policy_statistics(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],np.full(np.shape([[.3,.4,.2,.5,.1,-.6,.7,.8]]),np.nan),[1],[.8,.9],False))',
      'gold_call': '_case_value_error(lambda: '
                   '_oracle_policy_statistics(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],np.full(np.shape([[.3,.4,.2,.5,.1,-.6,.7,.8]]),np.nan),[1],[.8,.9],False))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '_case_bad=np.array([[.3,.4,.2,.5,.1,-.6,.7,.8]],float)\n'
               '_case_bad[0,0]=-1e-6\n',
      'call': '_case_value_error(lambda: '
              'policy_statistics(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],_case_bad,[1],[.8,.9],False))',
      'gold_call': '_case_value_error(lambda: '
                   '_oracle_policy_statistics(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],_case_bad,[1],[.8,.9],False))'},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: '
              "policy_statistics(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],'0.5',[1],[.8,.9],False))",
      'gold_call': '_case_value_error(lambda: '
                   "_oracle_policy_statistics(np.array([np.diag([.2,.1,.3,.0])]),[1.,0.],'0.5',[1],[.8,.9],False))"},
     {'setup': 'def _case_value_error(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_value_error(lambda: '
              'policy_statistics(np.array([np.diag([-.1,-.2,.4,.4])]),[1.,0.],[[.3,.4,.2,.5,.1,-.6,.7,.8]],[1],[.8,.9],False))',
      'gold_call': '_case_value_error(lambda: '
                   '_oracle_policy_statistics(np.array([np.diag([-.1,-.2,.4,.4])]),[1.,0.],[[.3,.4,.2,.5,.1,-.6,.7,.8]],[1],[.8,.9],False))'}]
