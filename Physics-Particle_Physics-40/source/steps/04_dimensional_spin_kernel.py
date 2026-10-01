"""
Evaluate the two dimensional-expansion coefficients of the physical traceless source tensor.

For radiation $(\eta,\phi)$, hard direction $(y,a)=(\mathrm{jet\_eta},\mathrm{jet\_phi})$, and physical-plane fraction $r$, set $C=\cosh(\eta-y)$, $u=(\cos\phi,\sin\phi)$, $v=(\cos a,\sin a)$, and $F=\cosh^2\eta/[C-r\cos(\phi-a)]$. The source-projected coefficients are $B_0=F[-2Cr^2uu^\top+r(uv^\top+vu^\top)]$ and $B_1=4F\sinh(\eta-y)r^2uu^\top$. Return $B_k-\operatorname{tr}(B_k)I_2/2$ for $k=0,1$ in that order, broadcasting over eta and phi. Inputs obey the public bounds in the function contract.

Returns
-------
Real array with shape S+(2,2,2), coefficient axis then x,y ket/bra.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dimensional_spin_kernel(eta: 'float | ArrayLike',
                            phi: 'float | ArrayLike',
                            jet_eta: float,
                            jet_phi: float,
                            radius: float) -> 'np.ndarray':
    """Evaluate the physical traceless source tensor.

    Parameters
    ----------
    eta : float or array-like
        Nonempty real scalar, vector, or matrix convertible with ``np.asarray``
        and lying in ``[-0.7, 0.7]``.
    phi : float or array-like
        Real scalar, vector, or matrix convertible with ``np.asarray`` and
        lying in ``[-2*pi, 2*pi]``, broadcastable with ``eta``. Arrays may
        have at most two dimensions.
    jet_eta : float
        Hard-direction pseudorapidity satisfying ``1 <= abs(jet_eta) <= 2.5``.
    jet_phi : float
        Hard-direction azimuth in radians in ``[-2*pi, 2*pi]``.
    radius : float
        Physical-plane fraction ``r`` in ``[0, 1]``.

    Returns
    -------
    kernel : numpy.ndarray
        Real array of shape ``S + (2, 2, 2)`` with coefficient axis followed
        by ``x, y`` ket/bra axes. Scalar ``eta`` and ``phi`` give ``(2, 2, 2)``.

    Raises
    ------
    ValueError
        If inputs have invalid types or shapes, are complex or nonfinite, if
        ``eta`` and ``phi`` are empty or not broadcastable, if any value is
        outside its public interval, or if radiation and hard directions are
        not separated enough for a finite float64 result.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
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
def _oracle_dimensional_spin_kernel(eta: 'float | ArrayLike',
                                    phi: 'float | ArrayLike',
                                    jet_eta: float,
                                    jet_phi: float,
                                    radius: float) -> 'np.ndarray':
    try:
        e,p,y,a=_r11_angles(eta,phi,jet_eta,jet_phi)
        r=_r11_scalar(radius,0.,1.)
    except ValueError as exc:
        raise ValueError('invalid dimensional-kernel input') from exc
    c=np.cosh(e-y); q=np.cos(p-a)
    uu,cross=_r11_transverse(p,a)
    factor=(np.cosh(e)**2/(c-q*r))[...,None,None]
    b0=factor*(-2*c[...,None,None]*r*r*uu+r*cross)
    b1=factor*4*np.sinh(e-y)[...,None,None]*r*r*uu
    return np.stack([_r11_st(b0),_r11_st(b1)],axis=-3)

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
      'call': '_checked_numeric(dimensional_spin_kernel(0.0, 0.3, 1.3, 0.2, 1.0), (2, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_dimensional_spin_kernel(0.0, 0.3, 1.3, 0.2, 1.0), (2, 2, 2))',
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
      'call': '_checked_numeric(dimensional_spin_kernel([-0.7, 0.2, 0.7], 0.4, -1.0, -0.8, 0.3), (3, 2, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_dimensional_spin_kernel([-0.7, 0.2, 0.7], 0.4, -1.0, -0.8, 0.3), '
                   '(3, 2, 2, 2))',
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
      'call': '_checked_numeric(dimensional_spin_kernel([[0.2], [-0.3]], [0.1, 2.0], 2.5, 6.283185307179586, '
              '0.81), (2, 2, 2, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_dimensional_spin_kernel([[0.2], [-0.3]], [0.1, 2.0], 2.5, '
                   '6.283185307179586, 0.81), (2, 2, 2, 2, 2))',
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
      'call': '_checked_numeric(dimensional_spin_kernel(-0.7, -6.283185307179586, -1.0, -0.4, 0.0), (2, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_dimensional_spin_kernel(-0.7, -6.283185307179586, -1.0, -0.4, 0.0), '
                   '(2, 2, 2))',
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
      'call': '_checked_numeric(dimensional_spin_kernel(0.7, 0.0, 1.0, 0.0, 0.999999), (2, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_dimensional_spin_kernel(0.7, 0.0, 1.0, 0.0, 0.999999), (2, 2, 2))',
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
      'call': 'rejects_value_error(dimensional_spin_kernel, [],.2,1.3,.2,.4)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, [],.2,1.3,.2,.4)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, np.zeros((1,1,1)),.2,1.3,.2,.4)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, np.zeros((1,1,1)),.2,1.3,.2,.4)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, [.1,.2],[.1,.2,.3],1.3,.2,.4)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, [.1,.2],[.1,.2,.3],1.3,.2,.4)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, .701,.2,1.3,.2,.4)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, .701,.2,1.3,.2,.4)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.,7.,1.3,.2,.4)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.,7.,1.3,.2,.4)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.,.2,.9,.2,.4)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.,.2,.9,.2,.4)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.,.2,2.6,.2,.4)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.,.2,2.6,.2,.4)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.,.2,1.3,7.,.4)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.,.2,1.3,7.,.4)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.,.2,1.3,.2,-.01)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.,.2,1.3,.2,-.01)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.,.2,1.3,.2,1.01)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.,.2,1.3,.2,1.01)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, np.full(np.asarray(0.0).shape,True),0.3,1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   'np.full(np.asarray(0.0).shape,True),0.3,1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              'np.full(np.asarray(0.0).shape,np.nan),0.3,1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   'np.full(np.asarray(0.0).shape,np.nan),0.3,1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              'np.full(np.asarray(0.0).shape,np.inf),0.3,1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   'np.full(np.asarray(0.0).shape,np.inf),0.3,1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              'np.full(np.asarray(0.0).shape,-np.inf),0.3,1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   'np.full(np.asarray(0.0).shape,-np.inf),0.3,1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, np.asarray(0.0).astype(str),0.3,1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   'np.asarray(0.0).astype(str),0.3,1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, np.asarray(0.0,dtype=object),0.3,1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   'np.asarray(0.0,dtype=object),0.3,1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, np.zeros((1,1,1,1)),0.3,1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, np.zeros((1,1,1,1)),0.3,1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, np.asarray(0.0,dtype=complex),0.3,1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   'np.asarray(0.0,dtype=complex),0.3,1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,np.full(np.asarray(0.3).shape,True),1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,np.full(np.asarray(0.3).shape,True),1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,np.full(np.asarray(0.3).shape,np.nan),1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,np.full(np.asarray(0.3).shape,np.nan),1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,np.full(np.asarray(0.3).shape,np.inf),1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,np.full(np.asarray(0.3).shape,np.inf),1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,np.full(np.asarray(0.3).shape,-np.inf),1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,np.full(np.asarray(0.3).shape,-np.inf),1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,np.asarray(0.3).astype(str),1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,np.asarray(0.3).astype(str),1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,np.asarray(0.3,dtype=object),1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,np.asarray(0.3,dtype=object),1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,np.zeros((1,1,1,1)),1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.0,np.zeros((1,1,1,1)),1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,np.asarray(0.3,dtype=complex),1.3,0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,np.asarray(0.3,dtype=complex),1.3,0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,np.full(np.asarray(1.3).shape,True),0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,np.full(np.asarray(1.3).shape,True),0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,0.3,np.full(np.asarray(1.3).shape,np.nan),0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,np.full(np.asarray(1.3).shape,np.nan),0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,0.3,np.full(np.asarray(1.3).shape,np.inf),0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,np.full(np.asarray(1.3).shape,np.inf),0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,0.3,np.full(np.asarray(1.3).shape,-np.inf),0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,np.full(np.asarray(1.3).shape,-np.inf),0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,np.asarray(1.3).astype(str),0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,np.asarray(1.3).astype(str),0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,np.asarray(1.3,dtype=object),0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,np.asarray(1.3,dtype=object),0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,np.zeros((1,1,1,1)),0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.0,0.3,np.zeros((1,1,1,1)),0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,np.asarray(1.3,dtype=complex),0.2,1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,np.asarray(1.3,dtype=complex),0.2,1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,np.full(np.asarray(0.2).shape,True),1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,np.full(np.asarray(0.2).shape,True),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,0.3,1.3,np.full(np.asarray(0.2).shape,np.nan),1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,np.full(np.asarray(0.2).shape,np.nan),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,0.3,1.3,np.full(np.asarray(0.2).shape,np.inf),1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,np.full(np.asarray(0.2).shape,np.inf),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,0.3,1.3,np.full(np.asarray(0.2).shape,-np.inf),1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,np.full(np.asarray(0.2).shape,-np.inf),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,np.asarray(0.2).astype(str),1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,np.asarray(0.2).astype(str),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,np.asarray(0.2,dtype=object),1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,np.asarray(0.2,dtype=object),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,np.zeros((1,1,1,1)),1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.0,0.3,1.3,np.zeros((1,1,1,1)),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,np.asarray(0.2,dtype=complex),1.0)',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,np.asarray(0.2,dtype=complex),1.0)'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,0.2,np.full(np.asarray(1.0).shape,True))',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,0.2,np.full(np.asarray(1.0).shape,True))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,0.3,1.3,0.2,np.full(np.asarray(1.0).shape,np.nan))',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,0.2,np.full(np.asarray(1.0).shape,np.nan))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,0.3,1.3,0.2,np.full(np.asarray(1.0).shape,np.inf))',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,0.2,np.full(np.asarray(1.0).shape,np.inf))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, '
              '0.0,0.3,1.3,0.2,np.full(np.asarray(1.0).shape,-np.inf))',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,0.2,np.full(np.asarray(1.0).shape,-np.inf))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,0.2,np.asarray(1.0).astype(str))',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,0.2,np.asarray(1.0).astype(str))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,0.2,np.asarray(1.0,dtype=object))',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,0.2,np.asarray(1.0,dtype=object))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,0.2,np.zeros((1,1,1,1)))',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, 0.0,0.3,1.3,0.2,np.zeros((1,1,1,1)))'},
     {'setup': 'import numpy as np\n'
               'def rejects_value_error(fn,*args,**kwargs):\n'
               '    try: fn(*args,**kwargs)\n'
               '    except ValueError: return 1.0\n'
               '    return 0.0\n'
               'C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), '
               '(0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)\n'
               'K=(np.arange(8.).reshape(2,2,2)-3)/11+1j*np.arange(8.).reshape(2,2,2)/19\n'
               'T=(np.arange(24.).reshape(2,3,2,2)-9)/17\n',
      'call': 'rejects_value_error(dimensional_spin_kernel, 0.0,0.3,1.3,0.2,np.asarray(1.0,dtype=complex))',
      'gold_call': 'rejects_value_error(_oracle_dimensional_spin_kernel, '
                   '0.0,0.3,1.3,0.2,np.asarray(1.0,dtype=complex))'}]
