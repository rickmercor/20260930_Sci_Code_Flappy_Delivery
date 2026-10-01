"""
Compute the local field-prescribed duality Jacobian and its first two ordinary derivatives on the transported isotropic design.

The physical-geometric duality determines an orientation-preserving virtual-to-physical map from the design's local temperature gradient and scalar conductivity. Both the gradient magnitude and direction can change along the material-and-shape perturbation. The homogeneous reference conductivity and reference gradient $(1,0)$ remain fixed. This step extends the field-prescribed conversion to ordinary derivatives.

Returns
-------
np.ndarray of shape (3,e,2,2): virtual-to-physical Jacobian, first ordinary $\eta$ derivative, second ordinary $\eta$ derivative; no factorial scaling.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def differentiate_duality_jacobian(
    conductivity_jet: "np.ndarray",
    gradient_jet: "np.ndarray",
    background_conductivity: float,
) -> "np.ndarray":
    r"""Return the field-prescribed virtual-to-physical Jacobian jet.

    Use the conversion and map direction of compute_duality_jacobian, with
    the fixed reference gradient $G_0=(1,0)$. The physical gradient derivatives
    already include movement of the interpolation basis and temperature.
    Derivatives are ordinary $\eta$ derivatives at zero, without factorial
    scaling. Cartesian axes remain fixed throughout the perturbation.

    Parameters
    ----------
    conductivity_jet : np.ndarray
        Finite (3,e) array, $e\ge 1$: local isotropic conductivity and its
        first and second derivatives. Baseline conductivities are positive.
    gradient_jet : np.ndarray
        Finite (3,e,2) array: physical generating temperature gradient and
        its first and second derivatives. Baseline norms exceed $10^{-14}$.
    background_conductivity : float
        Finite positive conductivity of the fixed reference medium.

    Returns
    -------
    np.ndarray
        Shape (3,e,2,2): Jacobian, first derivative, second derivative.
        Last axes index physical output coordinates and virtual input
        coordinates, respectively. The baseline determinant is positive.
        Include the field-prescribed scale as well as the local frame.

    Raises
    ------
    ValueError
        If shapes are invalid, entries are not finite, baseline material
        values or background are not positive, or any baseline generating
        gradient has norm at most $10^{-14}$.
    """
    return jacobian_jet

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_differentiate_duality_jacobian(
    conductivity_jet: "np.ndarray",
    gradient_jet: "np.ndarray",
    background_conductivity: float,
) -> "np.ndarray":
    """Differentiate the scaled, stretched field-aligned coordinate map."""
    k = np.asarray(conductivity_jet, dtype=float)
    g = np.asarray(gradient_jet, dtype=float)
    if (k.ndim != 2 or k.shape[0] != 3 or k.shape[1] == 0
            or g.shape != (3,k.shape[1],2) or not np.all(np.isfinite(k))
            or not np.all(np.isfinite(g)) or np.any(k[0] <= 0)):
        raise ValueError('invalid material or physical-gradient jets')
    if (not np.isscalar(background_conductivity)
            or not np.isfinite(background_conductivity) or background_conductivity <= 0):
        raise ValueError('background must be finite and positive')
    s = np.sum(g[0]*g[0],axis=1)
    if np.any(s <= 1e-28):
        raise ValueError('generating gradient is zero')
    s1 = 2*np.sum(g[0]*g[1],axis=1)
    s2 = 2*np.sum(g[1]*g[1]+g[0]*g[2],axis=1)
    stretch = background_conductivity/k[0]
    stretch1 = -stretch*k[1]/k[0]
    stretch2 = stretch*(2*(k[1]/k[0])**2-k[2]/k[0])
    perp = np.stack([-g[:,:,1],g[:,:,0]],axis=2)
    f = np.empty((3,k.shape[1],2,2))
    f[:,:,:,0] = g
    f[0,:,:,1] = stretch[:,None]*perp[0]
    f[1,:,:,1] = stretch1[:,None]*perp[0]+stretch[:,None]*perp[1]
    f[2,:,:,1] = (stretch2[:,None]*perp[0]+2*stretch1[:,None]*perp[1]
                  +stretch[:,None]*perp[2])
    result = np.empty_like(f)
    result[0] = f[0]/s[:,None,None]
    result[1] = (f[1]-s1[:,None,None]*result[0])/s[:,None,None]
    result[2] = (f[2]-s2[:,None,None]*result[0]
                 -2*s1[:,None,None]*result[1])/s[:,None,None]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Explicit normal, boundary, edge and invalid-input configurations."""
    return [{'setup': 'import numpy as np\n'
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
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'K=K[:,:1].copy(); G=G[:,:1].copy()\n',
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'K[0]=1.; K[1:]=0.\n',
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'K[0]=1.\n',
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'K[1:]=0.\n',
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'G[1:]=0.\n',
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'G[1]=.7*G[0]; G[2]=-.2*G[0]\n',
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'G[1]=0.; K[1]=0.\n',
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'K*=1.7; B=1.7\n',
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'G[:,:,0]*=-1.\n',
      'call': '_values(differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'gold_call': '_values(_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B),K.shape[1])',
      'tol': 2e-08},
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
               '    return 0\n'
               'K[0,0]=0.\n',
      'call': '_status(lambda: differentiate_duality_jacobian(K.copy(),G.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B))'},
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
               '    return 0\n'
               'G[0,0]=0.\n',
      'call': '_status(lambda: differentiate_duality_jacobian(K.copy(),G.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B))'},
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
               '    return 0\n'
               'G=G[:2]\n',
      'call': '_status(lambda: differentiate_duality_jacobian(K.copy(),G.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B))'},
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
               '    return 0\n'
               'G[2,0,0]=np.nan\n',
      'call': '_status(lambda: differentiate_duality_jacobian(K.copy(),G.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B))'},
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
               '    return 0\n'
               'B=0.\n',
      'call': '_status(lambda: differentiate_duality_jacobian(K.copy(),G.copy(),B))',
      'gold_call': '_status(lambda: _oracle_differentiate_duality_jacobian(K.copy(),G.copy(),B))'},
     {'setup': 'import numpy as np\n'
               'K=np.array([[2.],[0.],[0.]])\n'
               'G=np.array([[[1.,2.]],[[.3,.6]],[[.4,.8]]])\n',
      'call': 'differentiate_duality_jacobian(K.copy(),G.copy(),1.)',
      'gold_call': '_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),1.)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'K=np.array([[.8],[0.],[0.]])\n'
               'G=np.array([[[1.,0.]],[[0.,.7]],[[-.49,0.]]])\n',
      'call': 'differentiate_duality_jacobian(K.copy(),G.copy(),1.3)',
      'gold_call': '_oracle_differentiate_duality_jacobian(K.copy(),G.copy(),1.3)',
      'tol': 1e-10}]
