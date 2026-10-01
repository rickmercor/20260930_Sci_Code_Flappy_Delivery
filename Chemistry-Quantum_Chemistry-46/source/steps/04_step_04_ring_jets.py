"""
Differentiate the stabilizing Riccati branch on a rectangular multivariate grid.

The Riccati linearization is a Sylvester operator. Its two factors need not commute with the higher-order matrix coefficients. All lower-order partitions contribute to a mixed response.

Returns
-------
return t
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ring_jets(a, b, t0):
    """Differentiate the stabilizing Riccati branch on a rectangular multivariate grid.

    Parameters
    ----------
    a, b : real arrays, shape (*grid,n,n)
        Symmetric matrix Taylor coefficients on identical grids.
    t0 : real symmetric array, shape (n,n)
        Stabilizing base solution. a[zero]+b[zero]@t0 has positive real spectrum.
    
    Returns
    -------
    t : real array, shape (*grid,n,n)
        t[zero]=t0. Every retained coefficient of B+A@T+T@A+T@B@T is zero.
        Use the stable analytic continuation, including mixed matrix coefficients
        and degenerate positive base frequencies. No eigenvector gauge is an output.
    A matrix jet has shape (*grid,n,n), with one to three leading axes.
    Each grid length is between 1 and 4. For multi-index alpha, J[alpha] is
    the coefficient of x**alpha, i.e. the ordinary derivative divided by
    prod_j factorial(alpha_j). Retain the entire rectangular grid, including
    mixed coefficients up to its total degree. Products are ordered Taylor
    convolutions without binomial factors. Inputs are finite real array-like
    objects. Return float64-compatible arrays and do not mutate inputs.
    NumPy and SciPy are available; import dependencies inside the function.
    Numerical tests use absolute and relative tolerances of 2e-8.
    """
    return t

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ring_jets(a, b, t0):
    import numpy as np
    from scipy.linalg import solve_sylvester
    a, b, t0 = np.asarray(a, float), np.asarray(b, float), np.asarray(t0, float)
    shape = a.shape[:-2]
    zero = (0,)*len(shape)
    t = np.zeros_like(a)
    t[zero] = t0
    left, right = a[zero]+t0@b[zero], a[zero]+b[zero]@t0
    for alpha in sorted(np.ndindex(shape), key=sum)[1:]:
        residual = b[alpha].copy()
        for beta in np.ndindex(tuple(k+1 for k in alpha)):
            remainder = tuple(x-y for x,y in zip(alpha,beta))
            residual += a[beta]@t[remainder]+t[beta]@a[remainder]
            for gamma in np.ndindex(tuple(k+1 for k in remainder)):
                delta = tuple(x-y for x,y in zip(remainder,gamma))
                residual += t[beta]@b[gamma]@t[delta]
        t[alpha] = solve_sylvester(left,right,-residual)
    return t

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'call': 'ring_jets(a,b,t0)',
      'gold_call': '_oracle_ring_jets(a,b,t0)',
      'name': 'sixth_mixed',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n"
               "a,b=cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],model['orders'])\n"
               'omega,t0=stable_ring(a[0,0],b[0,0])\n',
      'tol': 2e-08},
     {'call': 'ring_jets(a,b,t0)',
      'gold_call': '_oracle_ring_jets(a,b,t0)',
      'name': 'third_one_axis',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "model['orders']=[0,3]\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n"
               "a,b=cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],model['orders'])\n"
               'omega,t0=stable_ring(a[0,0],b[0,0])\n',
      'tol': 2e-08},
     {'call': 'ring_jets(a,b,t0)',
      'gold_call': '_oracle_ring_jets(a,b,t0)',
      'name': 'both_off',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "model['couplings']=[0.,0.]\n"
               "model['orders']=[2,2]\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n"
               "a,b=cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],model['orders'])\n"
               'omega,t0=stable_ring(a[0,0],b[0,0])\n',
      'tol': 2e-08},
     {'call': 'ring_jets(a,b,t0)',
      'gold_call': '_oracle_ring_jets(a,b,t0)',
      'name': 'explicit_mixed',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               'a=np.zeros((3,3,2,2));b=np.zeros_like(a)\n'
               'a[0,0]=[[1.1,.13],[.13,.8]];b[0,0]=[[.1,.04],[.04,.08]]\n'
               'a[1,1]=[[.07,-.03],[-.03,.1]];b[2,1]=[[.02,.01],[.01,-.04]]\n'
               'a[0,1]=[[.1,.04],[.04,-.05]]\n'
               'omega,t0=stable_ring(a[0,0],b[0,0])\n',
      'tol': 2e-08},
     {'name': 'three_mode_fractional_benchmark',
      'setup': "model = {'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37, -0.24], "
               "[-0.46, 0.71, 0.53], [0.35, 0.58, -0.67], [0.62, -0.29, 0.41]], 'frequencies': [0.53, "
               "0.88, 0.67], 'couplings': [0.37, 0.46, 0.41], 'orders': [2, 2, 2], 'renyi': 1.5}\n"
               '\n'
               'import numpy as np\n'
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n"
               "a,b=cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],model['orders'])\n"
               "zero=(0,)*len(model['orders'])\n"
               'omega,t0=stable_ring(a[zero],b[zero])',
      'call': 'ring_jets(a,b,t0)',
      'gold_call': '_oracle_ring_jets(a,b,t0)',
      'tol': 2e-08},
     {'name': 'coupled_degenerate_mixed_response',
      'setup': 'import numpy as np\n'
               'o=np.array([[.8,-.6,0.],[.6,.8,0.],[0.,0.,1.]])\n'
               'r=np.array([.2,-.35,.1]);w=np.array([.7,.7,.9])\n'
               'a=np.zeros((3,2,3,3,3));b=np.zeros_like(a)\n'
               'a[0,0,0]=o@np.diag(w*np.cosh(2*r))@o.T\n'
               'b[0,0,0]=o@np.diag(w*np.sinh(2*r))@o.T\n'
               'a[1,0,0]=[[.04,.07,-.03],[.07,-.02,.01],[-.03,.01,.05]]\n'
               'b[0,1,0]=[[.02,-.03,.04],[-.03,.06,.02],[.04,.02,-.01]]\n'
               'b[1,0,1]=[[.01,.02,0],[.02,-.04,.03],[0,.03,.02]]\n'
               'a[0,1,1]=[[.03,-.02,.01],[-.02,.04,0],[.01,0,-.02]]\n'
               't0=-o@np.diag(np.tanh(r))@o.T\n',
      'call': 'ring_jets(a,b,t0)',
      'gold_call': '_oracle_ring_jets(a,b,t0)',
      'tol': 2e-08},
     {'name': 'one_axis_ring',
      'setup': "model = {'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37, -0.24], "
               "[-0.46, 0.71, 0.53], [0.35, 0.58, -0.67], [0.62, -0.29, 0.41]], 'frequencies': [0.53, "
               "0.88, 0.67], 'couplings': [0.37, 0.46, 0.41], 'orders': [2, 2, 2], 'renyi': 1.5}\n"
               '\n'
               'import numpy as np\n'
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n"
               "a,b=cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],model['orders'])\n"
               "zero=(0,)*len(model['orders'])\n"
               'omega,t0=stable_ring(a[zero],b[zero])\n'
               'a=a[:,0,0]\n'
               'b=b[:,0,0]',
      'call': 'ring_jets(a,b,t0)',
      'gold_call': '_oracle_ring_jets(a,b,t0)',
      'tol': 2e-08}]
