"""
Compute the converted conductivity tensor and its first two ordinary derivatives from the field-prescribed duality Jacobian jet.

The duality map converts a homogeneous isotropic reference medium into the physical anisotropic medium. Its constitutive response is expressed in fixed Cartesian coordinates, and its Jacobian can vary in scale, shear and orientation. The homogeneous reference conductivity is fixed. The Jacobian jet is supplied by differentiate_duality_jacobian for the field-prescribed device; other smooth orientation-preserving local maps use the same conversion law.

Returns
-------
np.ndarray of shape (3,e,2,2): Cartesian converted conductivity, first ordinary $\eta$ derivative, second ordinary $\eta$ derivative; no factorial scaling.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def differentiate_duality_tensor(
    jacobian_jet: "np.ndarray",
    background_conductivity: float,
) -> "np.ndarray":
    r"""Return the Cartesian conductivity jet of the duality-converted medium.

    Use the virtual-to-physical map convention of compute_duality_jacobian
    and the constitutive conversion underlying realize_duality_laminate.
    The reference medium is homogeneous and isotropic. All derivatives are
    ordinary $\eta$ derivatives at zero, without factorial scaling.

    Parameters
    ----------
    jacobian_jet : np.ndarray
        Finite (3,e,2,2) array, $e\ge 1$: Jacobian, first derivative and second
        derivative. Last axes index physical output and virtual input
        coordinates. Each baseline Jacobian has positive determinant;
        derivative matrices need not be symmetric or invertible.
    background_conductivity : float
        Finite positive conductivity of the fixed reference medium.

    Returns
    -------
    np.ndarray
        Shape (3,e,2,2): converted conductivity and its first two ordinary
        derivatives in fixed Cartesian axes. All matrices are symmetric;
        only the baseline matrices are necessarily positive definite.

    Raises
    ------
    ValueError
        If the array shape is invalid, any entry is not finite, any
        baseline Jacobian determinant is nonpositive, or the background
        conductivity is not a finite positive scalar.
    """
    return tensor_jet

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_differentiate_duality_tensor(
    jacobian_jet: "np.ndarray",
    background_conductivity: float,
) -> "np.ndarray":
    """Differentiate the conductivity push-forward and its determinant."""
    j = np.asarray(jacobian_jet, dtype=float)
    if (j.ndim != 4 or j.shape[0] != 3 or j.shape[1] == 0
            or j.shape[2:] != (2,2) or not np.all(np.isfinite(j))):
        raise ValueError('invalid Jacobian jet')
    if (not np.isscalar(background_conductivity)
            or not np.isfinite(background_conductivity) or background_conductivity <= 0):
        raise ValueError('background must be finite and positive')
    # A fixed elementwise rescaling cancels from the two-dimensional law.
    scale = np.max(np.abs(j[0]),axis=(1,2))
    if np.any(scale == 0):
        raise ValueError('baseline Jacobian must preserve orientation')
    j = j/scale[None,:,None,None]
    a,b,c = j
    d = a[:,0,0]*a[:,1,1]-a[:,0,1]*a[:,1,0]
    if np.any(d <= 0):
        raise ValueError('baseline Jacobian must preserve orientation')
    d1 = (b[:,0,0]*a[:,1,1]+a[:,0,0]*b[:,1,1]
          -b[:,0,1]*a[:,1,0]-a[:,0,1]*b[:,1,0])
    d2 = (c[:,0,0]*a[:,1,1]+2*b[:,0,0]*b[:,1,1]+a[:,0,0]*c[:,1,1]
          -c[:,0,1]*a[:,1,0]-2*b[:,0,1]*b[:,1,0]-a[:,0,1]*c[:,1,0])
    def _product(left,right):
        return left @ right.transpose(0,2,1)
    h = _product(a,a)
    h1 = _product(b,a)+_product(a,b)
    h2 = _product(c,a)+2*_product(b,b)+_product(a,c)
    result = np.empty_like(j)
    result[0] = background_conductivity*h/d[:,None,None]
    result[1] = (background_conductivity*h1-d1[:,None,None]*result[0])/d[:,None,None]
    result[2] = (background_conductivity*h2-d2[:,None,None]*result[0]
                 -2*d1[:,None,None]*result[1])/d[:,None,None]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Explicit normal, boundary, edge and invalid-input configurations."""
    return [{'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J=J[:,:1].copy()\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J=np.zeros((3,1,2,2)); J[0,0]=np.eye(2)\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J[1:]=0.\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J[1]=0.\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J[1]=.3*J[0]; J[2]=-.2*J[0]\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J=J*1e3\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J=J*1e-3\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'B=.07\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J=np.zeros((3,1,2,2)); J[0,0]=[[2.,0.],[0.,.5]]; J[1,0]=[[0.,-.2],[.8,0.]]; '
               'J[2,0]=-.16*J[0,0]\n',
      'call': '_values(differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_tensor(J.copy(),B),J.shape[1])',
      'tol': 2e-09},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J=J[:,:,0,:]\n',
      'call': '_status(lambda: differentiate_duality_tensor(J.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_tensor(J.copy(),B))'},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J=J[:,:0]\n',
      'call': '_status(lambda: differentiate_duality_tensor(J.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_tensor(J.copy(),B))'},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J[0,0]=0.\n',
      'call': '_status(lambda: differentiate_duality_tensor(J.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_tensor(J.copy(),B))'},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J[0,0,:,0]*=-1\n',
      'call': '_status(lambda: differentiate_duality_tensor(J.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_tensor(J.copy(),B))'},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'J[1,0,0,0]=np.nan\n',
      'call': '_status(lambda: differentiate_duality_tensor(J.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_tensor(J.copy(),B))'},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'B=0.\n',
      'call': '_status(lambda: differentiate_duality_tensor(J.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_tensor(J.copy(),B))'},
     {'setup': 'import numpy as np\n'
               'J=np.array([[[[1.,.2],[-.1,.7]],[[.3,-.8],[1.1,.2]]],\n'
               '            [[[.2,-.1],[.4,.3]],[[-.2,.5],[.1,.4]]],\n'
               '            [[[-.3,.6],[.1,-.2]],[[.7,-.3],[.2,-.1]]]])\n'
               'B=1.3\n'
               'def _values(value,e):\n'
               '    value=np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'B=np.inf\n',
      'call': '_status(lambda: differentiate_duality_tensor(J.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_tensor(J.copy(),B))'},
     {'setup': 'import numpy as np\n'
               'K = np.array([[.1,2.4,1.,.7],[.1,-.28,0.,.2],[.1,-.16,0.,-.3]])\n'
               'G = np.array([[[1.,.3],[.4,1.2],[1.,0.],[-.7,.6]],\n'
               '              [[.2,-.1],[-.3,.2],[0.,0.],[.1,.4]],\n'
               '              [[-.1,.5],[.6,-.2],[0.,0.],[-.3,.1]]])\n'
               'B = 1.\n'
               'def _values(value,e):\n'
               '    value = np.asarray(value)\n'
               '    assert value.shape == (3,e,2,2)\n'
               '    assert np.all(np.isfinite(value))\n'
               '    return value\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n',
      'call': 'differentiate_duality_tensor(differentiate_duality_jacobian(K.copy(),G.copy(),B),B)',
      'gold_call': '_oracle_differentiate_duality_tensor(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),B)',
      'tol': 2e-08}]
