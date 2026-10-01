"""
Recover the entropy-consistent positive directional pair and its energy estimate.

At fixed net flux, directional entropy fixes the product of the two directional variables. Their ratio supplies an inferred energy, including the zero estimate of a kinetically blocked reaction.

Returns
-------
A finite ndarray of shape (C,P,N,3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def directional_state(net: "np.ndarray", weights: "np.ndarray",
                      factors: "np.ndarray", rt: float) -> "np.ndarray":
    """
    net has shape (C,P,N); weights is positive finite (C,N) and factors
    positive finite (P,N). They are the original weights and independent
    condition factors used in phenotype inference. rt is positive and
    finite in kJ/mol. Return (C,P,N,3), trailing entries forward flux,
    reverse flux and inferred energy in kJ/mol. Both directional fluxes
    are positive, even at net zero. Invalid aligned shapes, nonfinite
    values or nonpositive weights, factors or rt raise ValueError. Valid
    inputs are in a range where directional outputs are representable.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_directional_state(net: "np.ndarray", weights: "np.ndarray",
                      factors: "np.ndarray", rt: float) -> "np.ndarray":
    import numpy as np
    n, g, f = map(lambda x: np.asarray(x, dtype=float), (net, weights, factors))
    if (n.ndim != 3 or g.shape != (n.shape[0], n.shape[2])
            or f.shape != n.shape[1:] or np.any(g <= 0) or np.any(f <= 0)
            or any(not np.isfinite(x).all() for x in (n, g, f))
            or not np.isfinite(rt) or rt <= 0):
        raise ValueError('Positive aligned weights and finite net fluxes required')
    geometric = g[:, None, :] * f[None, :, :] / np.e
    angle = np.arcsinh(n/(2.*geometric))
    forward = np.exp(np.log(geometric)+angle)
    reverse = np.exp(np.log(geometric)-angle)
    return np.stack([forward, reverse, -2.*rt*angle], axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'n=np.array([[[2.,-3.],[1.,5.]]]); g=np.array([[2.,4.]]); '
               'f=np.array([[1.,1.],[.2,3.]])',
      'call': 'directional_state(n,g,f,2.5)',
      'gold_call': '_oracle_directional_state(n,g,f,2.5)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'n=np.zeros((1,1,1)); g=np.array([[3.]]); f=np.ones((1,1))',
      'call': 'directional_state(n,g,f,1.)',
      'gold_call': '_oracle_directional_state(n,g,f,1.)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'n=np.array([[[1e5,-1e5]]]); g=np.array([[.01,.02]]); f=np.ones((1,2))',
      'call': 'directional_state(n,g,f,2.5)',
      'gold_call': '_oracle_directional_state(n,g,f,2.5)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'n=np.zeros((1,1,1)); g=np.array([[0.]]); f=np.ones((1,1))\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(directional_state, (n,g,f,1.,))',
      'gold_call': '_raises_value_error(_oracle_directional_state, (n,g,f,1.,))',
      'tol': 0.0}]
