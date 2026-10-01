"""
Extract the two common Laurent prefactors for the physical traceless matching term.

For $z\in[0.1,0.9]$ and $\mathrm{scale\_ratio}=\mu/Q_0\in[0.2,5]$, return the common simple-pole and finite Laurent prefactors. In $SU(3)$, $a_{-1}=(1-z)/(18\pi^2z)$ and $a_0=a_{-1}[4+6\ln(\mathrm{scale\_ratio})]$. Do not include the percentage, Hermitian conjugate or hard contraction.

Returns
-------
Real (2,) array: simple-pole then finite prefactor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def finite_glauber_prefactors(z: float, scale_ratio: float) -> 'np.ndarray':
    """Extract the common simple-pole and finite Laurent prefactors.

    Parameters
    ----------
    z : float
        Real momentum fraction in ``[0.1, 0.9]``.
    scale_ratio : float
        Positive ratio ``mu/Q0`` in ``[0.2, 5]``.

    Returns
    -------
    prefactors : numpy.ndarray
        Real array of shape ``(2,)`` ordered as ``[a_minus1, a_zero]``.

    Raises
    ------
    ValueError
        If either input is boolean, string, object-valued, complex (including
        zero-imaginary complex), nonscalar, nonfinite, or outside its interval,
        or if the result is not representable as finite float64 values.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _r11_has_bool(value):
    if isinstance(value,(bool,np.bool_)):
        return True
    if isinstance(value,(tuple,list)):
        return any(_r11_has_bool(x) for x in value)
    return isinstance(value,np.ndarray) and value.dtype.kind=='b'
def _r11_array(value,shape=None,complex_ok=False):
    try:
        a=np.asarray(value)
    except (TypeError,ValueError) as exc:
        raise ValueError('invalid numeric input') from exc
    if _r11_has_bool(value) or a.dtype.kind not in ('iufc' if complex_ok else 'iuf'):
        raise ValueError('invalid numeric type')
    if shape is not None and a.shape!=shape:
        raise ValueError('invalid shape')
    a=a.astype(complex if complex_ok else float)
    if not np.all(np.isfinite(a)):
        raise ValueError('nonfinite input')
    return a
def _r11_scalar(value,lo,hi):
    x=float(_r11_array(value,()))
    if not lo<=x<=hi:
        raise ValueError('outside public interval')
    return x
def _r11_angles(eta,phi,jet_eta,jet_phi):
    e=_r11_array(eta); p=_r11_array(phi)
    if e.ndim>2 or p.ndim>2 or not e.size or not p.size:
        raise ValueError('radiation input must be nonempty of dimension at most two')
    try:
        e,p=np.broadcast_arrays(e,p)
    except ValueError as exc:
        raise ValueError('radiation inputs must broadcast') from exc
    if np.any(abs(e)>.7) or np.any(abs(p)>2*np.pi):
        raise ValueError('radiation input outside domain')
    y=_r11_scalar(jet_eta,-2.5,2.5)
    if abs(y)<1:
        raise ValueError('hard direction outside separated domain')
    a=_r11_scalar(jet_phi,-2*np.pi,2*np.pi)
    return e,p,y,a
def _r11_st(t):
    return t-np.trace(t,axis1=-2,axis2=-1)[...,None,None]*np.eye(2)/2
def _r11_transverse(phi,jet_phi):
    u=np.stack([np.cos(phi),np.sin(phi)],axis=-1)
    v=np.array([np.cos(jet_phi),np.sin(jet_phi)])
    uu=np.einsum('...a,...b->...ab',u,u)
    cross=np.einsum('...a,b->...ab',u,v)+np.einsum('a,...b->...ab',v,u)
    return uu,cross
def _r11_fraction_float(value):
    try:
        x=float(value)
    except OverflowError as exc:
        raise ValueError('output not finite float64') from exc
    if not np.isfinite(x):
        raise ValueError('output not finite float64')
    return x
def _oracle_finite_glauber_prefactors(z: float,
                                      scale_ratio: float) -> 'np.ndarray':
    try:
        z=_r11_scalar(z,.1,.9)
        ratio=_r11_scalar(scale_ratio,.2,5.)
    except ValueError as exc:
        raise ValueError('invalid prefactor input') from exc
    pole=(1-z)/(18*np.pi**2*z)
    return np.array([pole,pole*(4+6*np.log(ratio))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'C = np.array([[[1 + 0j, 0.3j, 0.4 + 0j], [0.2 + 0.1j, -0 - 0.5j, 0.1 + 0j]], [[0.3 + 0j, 0.4 + '
               '0.2j, -0 - 0.2j], [-0 - 0.4j, 0.1 + 0j, 0.6 + 0.2j]]], dtype=complex)\n'
               'K = (np.arange(8.0).reshape(2, 2, 2) - 3) / 11 + 1j * np.arange(8.0).reshape(2, 2, 2) / 19\n'
               'T = (np.arange(24.0).reshape(2, 3, 2, 2) - 9) / 17\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(finite_glauber_prefactors(0.32, 1.0), (2,))',
      'gold_call': '_checked_numeric(_oracle_finite_glauber_prefactors(0.32, 1.0), (2,))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'C = np.array([[[1 + 0j, 0.3j, 0.4 + 0j], [0.2 + 0.1j, -0 - 0.5j, 0.1 + 0j]], [[0.3 + 0j, 0.4 + '
               '0.2j, -0 - 0.2j], [-0 - 0.4j, 0.1 + 0j, 0.6 + 0.2j]]], dtype=complex)\n'
               'K = (np.arange(8.0).reshape(2, 2, 2) - 3) / 11 + 1j * np.arange(8.0).reshape(2, 2, 2) / 19\n'
               'T = (np.arange(24.0).reshape(2, 3, 2, 2) - 9) / 17\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(finite_glauber_prefactors(0.1, 0.2), (2,))',
      'gold_call': '_checked_numeric(_oracle_finite_glauber_prefactors(0.1, 0.2), (2,))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'C = np.array([[[1 + 0j, 0.3j, 0.4 + 0j], [0.2 + 0.1j, -0 - 0.5j, 0.1 + 0j]], [[0.3 + 0j, 0.4 + '
               '0.2j, -0 - 0.2j], [-0 - 0.4j, 0.1 + 0j, 0.6 + 0.2j]]], dtype=complex)\n'
               'K = (np.arange(8.0).reshape(2, 2, 2) - 3) / 11 + 1j * np.arange(8.0).reshape(2, 2, 2) / 19\n'
               'T = (np.arange(24.0).reshape(2, 3, 2, 2) - 9) / 17\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(finite_glauber_prefactors(0.9, 5.0), (2,))',
      'gold_call': '_checked_numeric(_oracle_finite_glauber_prefactors(0.9, 5.0), (2,))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'C = np.array([[[1 + 0j, 0.3j, 0.4 + 0j], [0.2 + 0.1j, -0 - 0.5j, 0.1 + 0j]], [[0.3 + 0j, 0.4 + '
               '0.2j, -0 - 0.2j], [-0 - 0.4j, 0.1 + 0j, 0.6 + 0.2j]]], dtype=complex)\n'
               'K = (np.arange(8.0).reshape(2, 2, 2) - 3) / 11 + 1j * np.arange(8.0).reshape(2, 2, 2) / 19\n'
               'T = (np.arange(24.0).reshape(2, 3, 2, 2) - 9) / 17\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(finite_glauber_prefactors(0.57, 0.63), (2,))',
      'gold_call': '_checked_numeric(_oracle_finite_glauber_prefactors(0.57, 0.63), (2,))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'C = np.array([[[1 + 0j, 0.3j, 0.4 + 0j], [0.2 + 0.1j, -0 - 0.5j, 0.1 + 0j]], [[0.3 + 0j, 0.4 + '
               '0.2j, -0 - 0.2j], [-0 - 0.4j, 0.1 + 0j, 0.6 + 0.2j]]], dtype=complex)\n'
               'K = (np.arange(8.0).reshape(2, 2, 2) - 3) / 11 + 1j * np.arange(8.0).reshape(2, 2, 2) / 19\n'
               'T = (np.arange(24.0).reshape(2, 3, 2, 2) - 9) / 17\n'
               '\n'
               'def _checked_numeric(value, shape):\n'
               '    result = np.asarray(value)\n'
               "    if result.shape != shape or result.dtype.kind not in 'iufc':\n"
               "        raise ValueError('unexpected numerical result type or shape')\n"
               '    if not np.all(np.isfinite(result)):\n'
               "        raise ValueError('nonfinite result')\n"
               '    # Compare real and imaginary components without complex-magnitude overflow.\n'
               '    return np.stack((result.real, result.imag), axis=0)\n',
      'call': '_checked_numeric(finite_glauber_prefactors(0.23, 2.1), (2,))',
      'gold_call': '_checked_numeric(_oracle_finite_glauber_prefactors(0.23, 2.1), (2,))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, .099,1.)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, .099,1.)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, .901,1.)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, .901,1.)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, .3,.199)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, .3,.199)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, .3,5.01)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, .3,5.01)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, np.full(np.asarray(0.32).shape,True),1.0)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, '
                   'np.full(np.asarray(0.32).shape,True),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, np.full(np.asarray(0.32).shape,np.nan),1.0)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, '
                   'np.full(np.asarray(0.32).shape,np.nan),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, np.full(np.asarray(0.32).shape,np.inf),1.0)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, '
                   'np.full(np.asarray(0.32).shape,np.inf),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, np.full(np.asarray(0.32).shape,-np.inf),1.0)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, '
                   'np.full(np.asarray(0.32).shape,-np.inf),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, np.asarray(0.32).astype(str),1.0)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, np.asarray(0.32).astype(str),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, np.asarray(0.32,dtype=object),1.0)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, np.asarray(0.32,dtype=object),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, np.zeros((1,1,1,1)),1.0)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, np.zeros((1,1,1,1)),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, np.asarray(0.32,dtype=complex),1.0)',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, np.asarray(0.32,dtype=complex),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, 0.32,np.full(np.asarray(1.0).shape,True))',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, '
                   '0.32,np.full(np.asarray(1.0).shape,True))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, 0.32,np.full(np.asarray(1.0).shape,np.nan))',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, '
                   '0.32,np.full(np.asarray(1.0).shape,np.nan))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, 0.32,np.full(np.asarray(1.0).shape,np.inf))',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, '
                   '0.32,np.full(np.asarray(1.0).shape,np.inf))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, 0.32,np.full(np.asarray(1.0).shape,-np.inf))',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, '
                   '0.32,np.full(np.asarray(1.0).shape,-np.inf))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, 0.32,np.asarray(1.0).astype(str))',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, 0.32,np.asarray(1.0).astype(str))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, 0.32,np.asarray(1.0,dtype=object))',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, 0.32,np.asarray(1.0,dtype=object))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, 0.32,np.zeros((1,1,1,1)))',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, 0.32,np.zeros((1,1,1,1)))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(finite_glauber_prefactors, 0.32,np.asarray(1.0,dtype=complex))',
      'gold_call': 'rejects_value_error(_oracle_finite_glauber_prefactors, 0.32,np.asarray(1.0,dtype=complex))'}]
