"""
Integrate the three Laurent tensor contributions over the angular veto.

For each attachment $j$, let $B_0(\eta,\phi,j,r)$ and $B_1(\eta,\phi,j,r)$ be the two traceless matrices defined in step 04, and set $w(\eta)=1/[4\pi\cosh^2\eta]$. Over the one continuous $\eta$-$\phi$ rectangle, return $T_0=\int_{\rm rect}wB_0(r=1)\,d\eta\,d\phi$, $T_1=\int_{\rm rect}wB_1(r=1)\,d\eta\,d\phi$, and $T_2=\int_{\rm rect}w\int_0^1[-2r/(1-r^2)][B_0(r)-B_0(1)]\,dr\,d\eta\,d\phi$. The last integral is the continued-orientation finite coefficient. Use tensor-product Gauss-Legendre quadrature with 72 nodes on each rectangle axis and 48 nodes on $r\in[0,1]$; the open nodes avoid the removable endpoint singularity. Do not include the common Laurent prefactors or hard contraction.

Returns
-------
Real (3,3,2,2) array: attachment, coefficient channel, x,y ket/bra.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def finite_veto_tensors(jet_eta: 'ArrayLike',
                        jet_phi: 'ArrayLike',
                        gap: 'ArrayLike',
                        azimuth_window: 'ArrayLike') -> 'np.ndarray':
    """Integrate the three Laurent tensor contributions over one veto.

    Parameters
    ----------
    jet_eta : array-like
        Real finite values convertible with ``np.asarray`` to shape ``(3,)``
        in attachment ``3, 4, 5`` order, with each entry in the
        dimensional-kernel hard-direction domain.
    jet_phi : array-like
        Real finite values convertible with ``np.asarray`` to shape ``(3,)``
        in the same order and azimuth domain.
    gap : array-like
        Strictly increasing real endpoints convertible with ``np.asarray``
        to shape ``(2,)`` in ``[-0.7, 0.7]``.
    azimuth_window : array-like
        Strictly increasing real unwrapped endpoints convertible with
        ``np.asarray`` to shape ``(2,)`` in ``[-2*pi, 2*pi]`` with width in
        ``(0, 2*pi]``.

    Returns
    -------
    tensors : numpy.ndarray
        Real array of shape ``(3, 3, 2, 2)`` with attachment, ``T0/T1/T2``,
        and ``x, y`` ket/bra axes. Use 72-point Gauss-Legendre quadrature on
        each rectangle axis and 48-point Gauss-Legendre quadrature on the
        radial interval; the open radial nodes avoid the removable endpoint
        singularity.

    Raises
    ------
    ValueError
        If an input has a wrong type or shape, is complex or nonfinite, if a
        hard direction is outside the step-04 domain, or if either interval is
        non-increasing, outside its bounds, or has invalid azimuthal width.
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
def _oracle_finite_veto_tensors(jet_eta: 'ArrayLike',
                                jet_phi: 'ArrayLike',
                                gap: 'ArrayLike',
                                azimuth_window: 'ArrayLike') -> 'np.ndarray':
    from scipy.special import roots_legendre

    jets_eta = _r11_array(jet_eta, (3,))
    jets_phi = _r11_array(jet_phi, (3,))
    gap = _r11_array(gap, (2,))
    azimuth_window = _r11_array(azimuth_window, (2,))
    if np.any(abs(gap) > 0.7) or gap[0] >= gap[1]:
        raise ValueError("gap must be increasing in [-0.7,0.7]")
    if np.any(abs(azimuth_window) > 2 * np.pi) or not 0 < azimuth_window[1] - azimuth_window[0] <= 2 * np.pi:
        raise ValueError("azimuth endpoints must lie in [-2pi,2pi], width in (0,2pi]")
    for one_eta, one_phi in zip(jets_eta, jets_phi):
        _r11_angles(0.0, 0.0, one_eta, one_phi)
    nodes, weights = roots_legendre(72)
    eta = gap[0] + (nodes + 1) * (gap[1] - gap[0]) / 2
    phi = azimuth_window[0] + (nodes + 1) * (azimuth_window[1] - azimuth_window[0]) / 2
    eta, phi = np.meshgrid(eta, phi, indexing="ij")
    measure = np.outer(weights, weights) * (gap[1] - gap[0]) * (azimuth_window[1] - azimuth_window[0]) / (16 * np.pi * np.cosh(eta) ** 2)
    first = []
    second = []
    third = []
    radial_nodes, radial_weights = roots_legendre(48)
    radial_nodes = (radial_nodes + 1) / 2
    radial_weights = radial_weights / 2
    for one_eta, one_phi in zip(jets_eta, jets_phi):
        kernel = _oracle_dimensional_spin_kernel(eta, phi, one_eta, one_phi, 1.0)
        first.append(np.einsum("ij,ijab->ab", measure, kernel[:, :, 0]))
        second.append(np.einsum("ij,ijab->ab", measure, kernel[:, :, 1]))
        endpoint = kernel[:, :, 0]
        evanescent = np.zeros_like(endpoint)
        for radius, weight in zip(radial_nodes, radial_weights):
            radial = _oracle_dimensional_spin_kernel(eta, phi, one_eta, one_phi, radius)[:, :, 0]
            evanescent += weight * (-2 * radius) * (radial - endpoint) / (1 - radius * radius)
        third.append(np.einsum("ij,ijab->ab", measure, evanescent))
    return np.stack((np.array(first), np.array(second), np.array(third)), axis=1)

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
      'call': '_checked_numeric(finite_veto_tensors([1.55, -1.05, 1.0], [0.15, 2.3, -1.45], [-0.45, 0.65], '
              '[-0.9, 2.5]), (3, 3, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_finite_veto_tensors([1.55, -1.05, 1.0], [0.15, 2.3, -1.45], [-0.45, '
                   '0.65], [-0.9, 2.5]), (3, 3, 2, 2))',
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
      'call': '_checked_numeric(finite_veto_tensors([2.5, 1.1, -1.4], [-1.7, 2.3, 4.8], [-0.3, 0.5], [0.2, '
              '4.1]), (3, 3, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_finite_veto_tensors([2.5, 1.1, -1.4], [-1.7, 2.3, 4.8], [-0.3, '
                   '0.5], [0.2, 4.1]), (3, 3, 2, 2))',
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
      'call': '_checked_numeric(finite_veto_tensors([-1.4, -2.1, 1.2], [0.6, -0.8, 2.7], [-0.1, 0.7], [-4.2, '
              '-0.2]), (3, 3, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_finite_veto_tensors([-1.4, -2.1, 1.2], [0.6, -0.8, 2.7], [-0.1, '
                   '0.7], [-4.2, -0.2]), (3, 3, 2, 2))',
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
      'call': '_checked_numeric(finite_veto_tensors([1.7, -1.2, 2.2], [1.3, -0.4, -2.0], [-0.6, -0.59], [0.2, '
              '0.23]), (3, 3, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_finite_veto_tensors([1.7, -1.2, 2.2], [1.3, -0.4, -2.0], [-0.6, '
                   '-0.59], [0.2, 0.23]), (3, 3, 2, 2))',
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
      'call': '_checked_numeric(finite_veto_tensors([1.0, -2.5, 1.8], [-6.283185307179586, 0.0, '
              '6.283185307179586], [-0.7, 0.7], [-2.3, 3.1]), (3, 3, 2, 2))',
      'gold_call': '_checked_numeric(_oracle_finite_veto_tensors([1.0, -2.5, 1.8], [-6.283185307179586, 0.0, '
                   '6.283185307179586], [-0.7, 0.7], [-2.3, 3.1]), (3, 3, 2, 2))',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'def _case_accuracy(value, reference):\n'
               '    try:\n'
               '        actual=np.asarray(value); expected=np.asarray(reference)\n'
               "        if actual.shape!=expected.shape or actual.dtype.kind not in 'iuf': return False\n"
               '        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)): return 0.0\n'
               '        budget=np.maximum(1e-7,2*np.abs(np.spacing(expected.astype(np.float64))))\n'
               '        return float(np.all(np.abs(actual-expected)<=budget))\n'
               '    except (TypeError,ValueError,OverflowError): return 0.0\n'
               'def rejects_value_error(fn,*args):\n'
               '    try: fn(*args)\n'
               '    except ValueError: return 1.0\n'
               '    except Exception: return 0.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(finite_veto_tensors,[1.3],[.2,3.],[-.4,.6],[-1.,1.])',
      'gold_call': 'rejects_value_error(_oracle_finite_veto_tensors,[1.3],[.2,3.],[-.4,.6],[-1.,1.])'},
     {'setup': 'import numpy as np\n'
               'def _case_accuracy(value, reference):\n'
               '    try:\n'
               '        actual=np.asarray(value); expected=np.asarray(reference)\n'
               "        if actual.shape!=expected.shape or actual.dtype.kind not in 'iuf': return False\n"
               '        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)): return 0.0\n'
               '        budget=np.maximum(1e-7,2*np.abs(np.spacing(expected.astype(np.float64))))\n'
               '        return float(np.all(np.abs(actual-expected)<=budget))\n'
               '    except (TypeError,ValueError,OverflowError): return 0.0\n'
               'def rejects_value_error(fn,*args):\n'
               '    try: fn(*args)\n'
               '    except ValueError: return 1.0\n'
               '    except Exception: return 0.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(finite_veto_tensors,[.9,-1.3],[.2,3.],[-.4,.6],[-1.,1.])',
      'gold_call': 'rejects_value_error(_oracle_finite_veto_tensors,[.9,-1.3],[.2,3.],[-.4,.6],[-1.,1.])'},
     {'setup': 'import numpy as np\n'
               'def _case_accuracy(value, reference):\n'
               '    try:\n'
               '        actual=np.asarray(value); expected=np.asarray(reference)\n'
               "        if actual.shape!=expected.shape or actual.dtype.kind not in 'iuf': return False\n"
               '        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)): return 0.0\n'
               '        budget=np.maximum(1e-7,2*np.abs(np.spacing(expected.astype(np.float64))))\n'
               '        return float(np.all(np.abs(actual-expected)<=budget))\n'
               '    except (TypeError,ValueError,OverflowError): return 0.0\n'
               'def rejects_value_error(fn,*args):\n'
               '    try: fn(*args)\n'
               '    except ValueError: return 1.0\n'
               '    except Exception: return 0.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(finite_veto_tensors,[1.3,-1.3],[.2,7.],[-.4,.6],[-1.,1.])',
      'gold_call': 'rejects_value_error(_oracle_finite_veto_tensors,[1.3,-1.3],[.2,7.],[-.4,.6],[-1.,1.])'},
     {'setup': 'import numpy as np\n'
               'def _case_accuracy(value, reference):\n'
               '    try:\n'
               '        actual=np.asarray(value); expected=np.asarray(reference)\n'
               "        if actual.shape!=expected.shape or actual.dtype.kind not in 'iuf': return False\n"
               '        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)): return 0.0\n'
               '        budget=np.maximum(1e-7,2*np.abs(np.spacing(expected.astype(np.float64))))\n'
               '        return float(np.all(np.abs(actual-expected)<=budget))\n'
               '    except (TypeError,ValueError,OverflowError): return 0.0\n'
               'def rejects_value_error(fn,*args):\n'
               '    try: fn(*args)\n'
               '    except ValueError: return 1.0\n'
               '    except Exception: return 0.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(finite_veto_tensors,[1.3,-1.3],[.2,3.],[.4,.4],[-1.,1.])',
      'gold_call': 'rejects_value_error(_oracle_finite_veto_tensors,[1.3,-1.3],[.2,3.],[.4,.4],[-1.,1.])'},
     {'setup': 'import numpy as np\n'
               'def _case_accuracy(value, reference):\n'
               '    try:\n'
               '        actual=np.asarray(value); expected=np.asarray(reference)\n'
               "        if actual.shape!=expected.shape or actual.dtype.kind not in 'iuf': return False\n"
               '        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)): return 0.0\n'
               '        budget=np.maximum(1e-7,2*np.abs(np.spacing(expected.astype(np.float64))))\n'
               '        return float(np.all(np.abs(actual-expected)<=budget))\n'
               '    except (TypeError,ValueError,OverflowError): return 0.0\n'
               'def rejects_value_error(fn,*args):\n'
               '    try: fn(*args)\n'
               '    except ValueError: return 1.0\n'
               '    except Exception: return 0.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(finite_veto_tensors,[1.3,-1.3],[.2,3.],[-.4,.6],[1.,1.])',
      'gold_call': 'rejects_value_error(_oracle_finite_veto_tensors,[1.3,-1.3],[.2,3.],[-.4,.6],[1.,1.])'},
     {'setup': 'import numpy as np\n'
               'def _case_accuracy(value, reference):\n'
               '    try:\n'
               '        actual=np.asarray(value); expected=np.asarray(reference)\n'
               "        if actual.shape!=expected.shape or actual.dtype.kind not in 'iuf': return False\n"
               '        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)): return 0.0\n'
               '        budget=np.maximum(1e-7,2*np.abs(np.spacing(expected.astype(np.float64))))\n'
               '        return float(np.all(np.abs(actual-expected)<=budget))\n'
               '    except (TypeError,ValueError,OverflowError): return 0.0\n'
               'def rejects_value_error(fn,*args):\n'
               '    try: fn(*args)\n'
               '    except ValueError: return 1.0\n'
               '    except Exception: return 0.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(finite_veto_tensors,np.full((2,),True),[.2,3.],[-.4,.6],[-1.,1.])',
      'gold_call': 'rejects_value_error(_oracle_finite_veto_tensors,np.full((2,),True),[.2,3.],[-.4,.6],[-1.,1.])'},
     {'setup': 'import numpy as np\n'
               'def _case_accuracy(value, reference):\n'
               '    try:\n'
               '        actual=np.asarray(value); expected=np.asarray(reference)\n'
               "        if actual.shape!=expected.shape or actual.dtype.kind not in 'iuf': return False\n"
               '        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)): return 0.0\n'
               '        budget=np.maximum(1e-7,2*np.abs(np.spacing(expected.astype(np.float64))))\n'
               '        return float(np.all(np.abs(actual-expected)<=budget))\n'
               '    except (TypeError,ValueError,OverflowError): return 0.0\n'
               'def rejects_value_error(fn,*args):\n'
               '    try: fn(*args)\n'
               '    except ValueError: return 1.0\n'
               '    except Exception: return 0.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(finite_veto_tensors,np.full((2,),np.nan),[.2,3.],[-.4,.6],[-1.,1.])',
      'gold_call': 'rejects_value_error(_oracle_finite_veto_tensors,np.full((2,),np.nan),[.2,3.],[-.4,.6],[-1.,1.])'},
     {'setup': 'import numpy as np\n'
               'def _case_accuracy(value, reference):\n'
               '    try:\n'
               '        actual=np.asarray(value); expected=np.asarray(reference)\n'
               "        if actual.shape!=expected.shape or actual.dtype.kind not in 'iuf': return False\n"
               '        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)): return 0.0\n'
               '        budget=np.maximum(1e-7,2*np.abs(np.spacing(expected.astype(np.float64))))\n'
               '        return float(np.all(np.abs(actual-expected)<=budget))\n'
               '    except (TypeError,ValueError,OverflowError): return 0.0\n'
               'def rejects_value_error(fn,*args):\n'
               '    try: fn(*args)\n'
               '    except ValueError: return 1.0\n'
               '    except Exception: return 0.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(finite_veto_tensors,[1.3,-1.3],np.asarray([.2,3.],dtype=complex),[-.4,.6],[-1.,1.])',
      'gold_call': 'rejects_value_error(_oracle_finite_veto_tensors,[1.3,-1.3],np.asarray([.2,3.],dtype=complex),[-.4,.6],[-1.,1.])'},
     {'setup': 'import numpy as np\n'
               'def _case_accuracy(value, reference):\n'
               '    try:\n'
               '        actual=np.asarray(value); expected=np.asarray(reference)\n'
               "        if actual.shape!=expected.shape or actual.dtype.kind not in 'iuf': return False\n"
               '        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)): return 0.0\n'
               '        budget=np.maximum(1e-7,2*np.abs(np.spacing(expected.astype(np.float64))))\n'
               '        return float(np.all(np.abs(actual-expected)<=budget))\n'
               '    except (TypeError,ValueError,OverflowError): return 0.0\n'
               'def rejects_value_error(fn,*args):\n'
               '    try: fn(*args)\n'
               '    except ValueError: return 1.0\n'
               '    except Exception: return 0.0\n'
               '    return 0.0\n',
      'call': 'rejects_value_error(finite_veto_tensors,[1.3,-1.3],[.2,3.],np.asarray([-.4,.6],dtype=object),[-1.,1.])',
      'gold_call': 'rejects_value_error(_oracle_finite_veto_tensors,[1.3,-1.3],[.2,3.],np.asarray([-.4,.6],dtype=object),[-1.,1.])'}]
