"""
Construct the prescribed unit-trace second-beam polarization density S=P/Tr(P).

For $u=(\cos\beta,\sin\beta)$, $\beta=\mathrm{azimuth}$, and $A=x/(1-x)+(1-x)/x$, the projected physical epsilon-zero kernel is $P=2C_A[A I_2+2x(1-x)uu^\top]$, with $C_A=3$. Return $S=P/\operatorname{tr}P$, not the spin-averaged scalar. The scalar $x$ is in $[0.1,0.9]$ and $\beta$ is in $[-\pi,\pi]$. Reject invalid or nonfinite inputs.

Returns
-------
Real symmetric positive unit-trace (2,2) density in x,y order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gluon_beam_density(x: float, azimuth: float) -> 'np.ndarray':
    """Construct the prescribed unit-trace beam-polarization density.

    Parameters
    ----------
    x : float
        Momentum fraction in the closed interval ``[0.1, 0.9]``.
    azimuth : float
        Physical-plane angle beta in radians in ``[-pi, pi]``.

    Returns
    -------
    density : numpy.ndarray
        Real symmetric positive unit-trace array of shape ``(2, 2)`` in
        ``x, y`` order.

    Raises
    ------
    ValueError
        If an input is boolean, string, object-valued, complex (including
        zero-imaginary complex), nonscalar, nonfinite, or outside its interval.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _joint_has_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_joint_has_bool(v) for v in value)
    return isinstance(value, np.ndarray) and value.dtype.kind == 'b'
def _joint_array(value, shape, complex_ok=False):
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid numeric array') from exc
    if (_joint_has_bool(value) or raw.dtype.kind not in ('iufc' if complex_ok else 'iuf') or raw.shape != shape):
        raise ValueError('invalid numeric type or shape')
    out = np.asarray(raw, dtype=complex if complex_ok else float)
    if not np.all(np.isfinite(out)):
        raise ValueError('nonfinite input')
    return out
def _joint_scalar(value, lo=None, hi=None):
    out = float(_joint_array(value, ()))
    if (lo is not None and out < lo) or (hi is not None and out > hi):
        raise ValueError('scalar outside public domain')
    return out
def _oracle_gluon_beam_density(x: float, azimuth: float) -> 'np.ndarray':
    try:
        x = _joint_scalar(x, .1, .9)
        azimuth = _joint_scalar(azimuth, -np.pi, np.pi)
    except ValueError as exc:
        raise ValueError('invalid beam-density input') from exc
    u = np.array([np.cos(azimuth), np.sin(azimuth)])
    p = 6*((x/(1-x)+(1-x)/x)*np.eye(2) + 2*x*(1-x)*np.outer(u, u))
    return p/np.trace(p)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(gluon_beam_density(0.1, -3.141592653589793), (2, 2))',
      'gold_call': '_checked_numeric(_oracle_gluon_beam_density(0.1, -3.141592653589793), (2, 2))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(gluon_beam_density(0.9, 3.141592653589793), (2, 2))',
      'gold_call': '_checked_numeric(_oracle_gluon_beam_density(0.9, 3.141592653589793), (2, 2))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(gluon_beam_density(0.37, 0.7), (2, 2))',
      'gold_call': '_checked_numeric(_oracle_gluon_beam_density(0.37, 0.7), (2, 2))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(gluon_beam_density(0.5, 0.0), (2, 2))',
      'gold_call': '_checked_numeric(_oracle_gluon_beam_density(0.5, 0.0), (2, 2))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(gluon_beam_density(0.24, -1.2), (2, 2))',
      'gold_call': '_checked_numeric(_oracle_gluon_beam_density(0.24, -1.2), (2, 2))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(gluon_beam_density(0.68, 2.1), (2, 2))',
      'gold_call': '_checked_numeric(_oracle_gluon_beam_density(0.68, 2.1), (2, 2))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, True,.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, True,.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, 1+0j,.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, 1+0j,.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, "0.3",.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, "0.3",.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, np.array(.3,dtype=object),.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, np.array(.3,dtype=object),.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, np.nan,.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, np.nan,.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, np.inf,.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, np.inf,.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, -np.inf,.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, -np.inf,.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, [.3],.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, [.3],.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, .099,.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, .099,.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, .901,.2)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, .901,.2)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, .3,True)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, .3,True)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, .3,np.nan)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, .3,np.nan)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, .3,np.inf)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, .3,np.inf)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, .3,-np.inf)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, .3,-np.inf)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(gluon_beam_density, .3,4.)',
      'gold_call': 'rejects_value_error(_oracle_gluon_beam_density, .3,4.)'}]
