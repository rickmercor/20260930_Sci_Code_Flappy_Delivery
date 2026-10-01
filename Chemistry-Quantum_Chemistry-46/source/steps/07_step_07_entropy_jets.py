"""
Evaluate the photon Renyi entropy series at index 2 or 3/2.

Renyi-2 entropy depends only on a determinant. Renyi-3/2 entropy depends on the symplectic spectrum through fractional matrix powers. Repeated mixed symplectic eigenvalues are regular, whereas the generic fractional-index response at a pure mode is not analytic.

Returns
-------
return entropy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def entropy_jets(covariance, renyi=2.0):
    """Evaluate the photon Renyi entropy series at index 2 or 3/2.

    Parameters
    ----------
    covariance : real array, shape (*grid,2,m,m)
        Symmetric q and p Taylor coefficients; no q-p block. Both base blocks
        must be positive definite. Their product need not be symmetric.
    renyi : float, default 2.0
        Supported values are exactly 2.0 and 1.5. For 1.5, every base symplectic
        eigenvalue nu=sqrt(eig(Vq@Vp)) must be strictly greater than .5+1e-12.
        The 1e-12 gap resolves roundoff at the pure-state boundary. Repeated
        eigenvalues above this bound are allowed. A generic jet touching a pure
        mode is excluded from the 1.5 branch because it need not be analytic.
    
    Returns
    -------
    entropy : real array, shape (*grid)
        Taylor coefficients of S_alpha=log(Tr rho**alpha)/(1-alpha).
        At alpha=2, use .5*logdet(2*Vq)+.5*logdet(2*Vp), including its constant.
        At alpha=1.5, S=2*sum_j log((nu_j+.5)**1.5-(nu_j-.5)**1.5).
        Use the analytic continuation from the basepoint and natural logarithms.
        The input matrices and their derivatives need not commute; do not discard
        derivatives of spectral projectors. Earlier inverse_jets is available.
    
    Raises
    ------
    ValueError
        If renyi is unsupported, either base covariance block is not positive
        definite, or the 1.5 branch has any base nu<=.5+1e-12. Check these before
        attempting a singular inverse. For alpha=2 a constant vacuum is valid.
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
    return entropy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_entropy_jets(covariance, renyi=2.0):
    import numpy as np
    from scipy.linalg import sqrtm, solve_sylvester
    covariance = np.asarray(covariance, float)
    shape = covariance.shape[:-3]
    zero = (0,)*len(shape)
    indices = sorted(np.ndindex(shape),key=sum)
    m = covariance.shape[-1]
    if renyi not in (1.5,2.0):
        raise ValueError('Supported Renyi indices are 1.5 and 2.0.')
    q, p = covariance[...,0,:,:], covariance[...,1,:,:]
    if min(np.linalg.eigvalsh(q[zero]).min(),np.linalg.eigvalsh(p[zero]).min()) <= 0:
        raise ValueError('Base quadrature blocks must be positive definite.')

    def product(x,y):
        out = np.zeros_like(x)
        for alpha in indices:
            for beta in np.ndindex(tuple(k+1 for k in alpha)):
                rest = tuple(a-b for a,b in zip(alpha,beta))
                out[alpha] += x[beta]@y[rest]
        return out

    def square_root(x):
        out = np.zeros_like(x)
        out[zero] = np.real_if_close(sqrtm(x[zero]))
        for alpha in indices[1:]:
            residual = x[alpha].copy()
            for beta in np.ndindex(tuple(k+1 for k in alpha)):
                if beta == zero or beta == alpha:
                    continue
                rest = tuple(a-b for a,b in zip(alpha,beta))
                residual -= out[beta]@out[rest]
            out[alpha] = solve_sylvester(out[zero],out[zero],residual)
        return out

    def logdet_series(x):
        sign, value = np.linalg.slogdet(x[zero])
        if sign <= 0:
            raise ValueError('Nonpositive base determinant.')
        inverse = _oracle_inverse_jets(x)
        out = np.zeros(shape)
        out[zero] = value
        for alpha in indices[1:]:
            axis = next(i for i,k in enumerate(alpha) if k)
            previous = list(alpha)
            previous[axis] -= 1
            value = 0.
            for beta in np.ndindex(tuple(k+1 for k in previous)):
                rest = tuple(a-b for a,b in zip(alpha,beta))
                value += rest[axis]*np.trace(inverse[beta]@x[rest])
            out[alpha] = value/alpha[axis]
        return out

    if renyi == 2.0:
        return .5*(logdet_series(2*q)+logdet_series(2*p))
    base_q = np.real_if_close(sqrtm(q[zero]))
    squared_symplectic = np.linalg.eigvalsh(base_q@p[zero]@base_q)
    if squared_symplectic.min() <= (.5+1e-12)**2:
        raise ValueError('Renyi 3/2 jets require a base symplectic gap above 1e-12.')
    nu = square_root(product(q,p))
    plus, minus = nu.copy(), nu.copy()
    plus[zero] += .5*np.eye(m)
    minus[zero] -= .5*np.eye(m)
    kernel = product(plus,square_root(plus))-product(minus,square_root(minus))
    return 2*logdet_series(kernel)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'call': 'entropy_jets(cov)',
      'gold_call': '_oracle_entropy_jets(cov)',
      'name': 'full_entropy',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n"
               "a,b=cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],model['orders'])\n"
               'omega,t0=stable_ring(a[0,0],b[0,0])\n'
               't=ring_jets(a,b,t0)\n'
               "cov=photon_covariance_jets(t,len(model['gaps']))\n",
      'tol': 2e-08},
     {'call': 'entropy_jets(cov)',
      'gold_call': '_oracle_entropy_jets(cov)',
      'name': 'vacuum',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               'cov=np.zeros((4,4,2,2,2));cov[0,0,0]=.5*np.eye(2);cov[0,0,1]=.5*np.eye(2)\n',
      'tol': 2e-08},
     {'call': 'entropy_jets(cov)',
      'gold_call': '_oracle_entropy_jets(cov)',
      'name': 'coupled_quadratures',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               'cov=np.zeros((3,3,2,2,2));cov[0,0,0]=[[.7,.12],[.12,.8]];cov[0,0,1]=[[.6,-.07],[-.07,.9]];cov[1,0,0]=[[.2,.08],[.08,.1]];cov[0,1,1]=[[.1,.04],[.04,.3]];cov[1,1,0]=[[.02,-.05],[-.05,.01]]\n',
      'tol': 2e-08},
     {'call': 'expect_value_error(entropy_jets,cov)',
      'gold_call': 'expect_value_error(_oracle_entropy_jets,cov)',
      'name': 'negative_determinant',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               'def expect_value_error(fn, *args):\n'
               '    try:\n'
               '        fn(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'cov=np.zeros((1,1,2,1,1));cov[0,0,:,0,0]=[-.5,.5]\n',
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
               'omega,t0=stable_ring(a[zero],b[zero])\n'
               't=ring_jets(a,b,t0)\n'
               "cov=photon_covariance_jets(t,len(model['gaps']))\n",
      'call': 'entropy_jets(cov,1.5)',
      'gold_call': '_oracle_entropy_jets(cov,1.5)',
      'tol': 2e-08},
     {'name': 'repeated_symplectic_eigenvalues',
      'setup': 'import numpy as np\n'
               'cov=np.zeros((3, 3, 3, 2, 3, 3))\n'
               'cov[0,0,0,0]=[[0.65, 0.06, -0.02], [0.06, 0.9, 0.04], [-0.02, 0.04, 0.75]]\n'
               'cov[0,0,0,1]=[[0.9917662707051367, -0.06745306682253528, 0.03004459744933886], '
               '[-0.06745306682253528, 0.717388402822204, -0.040059463265785154], [0.030044597449338864, '
               '-0.040059463265785154, 0.8562710273061577]]\n'
               'cov[0,0,1,0]=[[0.03, -0.02, 0.04], [-0.02, 0.05, 0.02], [0.04, 0.02, -0.01]]\n'
               'cov[0,0,1,1]=[[0.02, 0.01, -0.01], [0.01, -0.03, 0.02], [-0.01, 0.02, 0.04]]\n'
               'cov[0,1,0,1]=[[-0.05, 0.02, 0.01], [0.02, 0.07, -0.03], [0.01, -0.03, 0.02]]\n'
               'cov[0,1,1,1]=[[0.003, 0.002, -0.006], [0.002, -0.004, 0.001], [-0.006, 0.001, 0.008]]\n'
               'cov[1,0,0,0]=[[0.08, 0.03, -0.02], [0.03, -0.04, 0.01], [-0.02, 0.01, 0.05]]\n'
               'cov[1,1,0,0]=[[0.01, -0.007, 0.002], [-0.007, 0.012, -0.003], [0.002, -0.003, -0.01]]\n'
               'cov[2,1,1,0]=[[0.002, -0.001, 0.0004], [-0.001, 0.003, -0.0005], [0.0004, -0.0005, '
               '-0.002]]\n'
               'cov[2,2,2,1]=[[0.0003, -0.0002, 0.0001], [-0.0002, -0.0001, 0.0002], [0.0001, 0.0002, '
               '0.0004]]\n',
      'call': 'entropy_jets(cov,1.5)',
      'gold_call': '_oracle_entropy_jets(cov,1.5)',
      'tol': 2e-08},
     {'name': 'nonorthogonal_canonical_coordinates',
      'setup': 'import numpy as np\n'
               'cov=np.zeros((3, 3, 3, 2, 3, 3))\n'
               'cov[0,0,0,0]=[[1.1768000000000003, 0.18925000000000003, -0.01639999999999999], '
               '[0.18925000000000003, 0.6024750000000001, 0.16345], [-0.016400000000000012, 0.16345, '
               '0.9096000000000002]]\n'
               'cov[0,0,0,1]=[[0.5755142890322957, -0.19300532085140204, 0.04505843671206166], '
               '[-0.19300532085140198, 1.1814523967708763, -0.21578021274644096], [0.045058436712061674, '
               '-0.21578021274644096, 0.7431928695420884]]\n'
               'cov[0,0,1,0]=[[0.030999999999999996, -0.005850000000000002, 0.06580000000000001], '
               '[-0.005849999999999998, 0.03657500000000001, 0.014950000000000001], '
               '[0.06580000000000001, 0.014950000000000003, -0.0030000000000000005]]\n'
               'cov[0,0,1,1]=[[0.013014186390809765, 0.004051648207492364, -0.00857836247446637], '
               '[0.004051648207492362, -0.04971421075317179, 0.032177058151084126], '
               '[-0.00857836247446637, 0.03217705815108412, 0.023639924289274937]]\n'
               'cov[0,1,0,1]=[[-0.029522872509885504, 0.030162103408856656, -0.0014840426528738221], '
               '[0.030162103408856656, 0.09613912782743954, -0.045786997994790204], '
               '[-0.0014840426528738213, -0.045786997994790204, 0.027950563145368337]]\n'
               'cov[0,1,1,1]=[[0.002450253930773411, 0.0011406832705534006, -0.004580573827327073], '
               '[0.0011406832705534004, -0.006973482505950038, 0.0033529356646239757], '
               '[-0.004580573827327073, 0.0033529356646239757, 0.005002005209797418]]\n'
               'cov[1,0,0,0]=[[0.15450000000000005, 0.01965, -0.020700000000000003], [0.01965, '
               '-0.022075, 0.01915], [-0.020700000000000003, 0.019150000000000004, '
               '0.05690000000000001]]\n'
               'cov[1,1,0,0]=[[0.013240000000000002, -0.0046700000000000005, 0.004440000000000001], '
               '[-0.004670000000000001, 0.006735, -0.0048200000000000005], [0.00444, '
               '-0.0048200000000000005, -0.01156]]\n'
               'cov[2,1,1,0]=[[0.0028760000000000005, -0.000427, 0.0009180000000000002], [-0.000427, '
               '0.0017550000000000003, -0.000844], [0.0009180000000000001, -0.000844, '
               '-0.0023120000000000003]]\n'
               'cov[2,2,2,1]=[[0.00016117951312756505, -0.00024971046269747565, 9.514251981784447e-05], '
               '[-0.0002497104626974757, -4.146848822173496e-05, 0.0001986169674481363], '
               '[9.514251981784448e-05, 0.0001986169674481363, 0.00028695676617754534]]\n',
      'call': 'entropy_jets(cov,1.5)',
      'gold_call': '_oracle_entropy_jets(cov,1.5)',
      'tol': 2e-08},
     {'name': 'fractional_rectangular_orders',
      'setup': 'import numpy as np\n'
               'cov=np.zeros((3,3,3,2,3,3))\n'
               'cov[0,0,0,0]=np.diag([.7,.9,1.1]);cov[0,0,0,1]=np.diag([.64/.7,.64/.9,.64/1.1])\n'
               'cov[1,0,0,0]=[[.07,.05,-.03],[.05,-.04,.02],[-.03,.02,.06]]\n'
               'cov[0,1,0,1]=[[.02,-.03,.04],[-.03,.05,.01],[.04,.01,-.02]]\n'
               'cov[0,0,1,0]=[[.03,.01,.04],[.01,.05,-.02],[.04,-.02,-.01]]\n'
               'cov[1,1,0,1]=[[.01,-.02,.01],[-.02,.03,.02],[.01,.02,-.01]]\n'
               'cov[0,1,1,0]=[[.03,.02,-.01],[.02,-.01,.04],[-.01,.04,.02]]\n'
               '\n'
               'cov=cov[:1,:3,:2]',
      'call': 'entropy_jets(cov,1.5)',
      'gold_call': '_oracle_entropy_jets(cov,1.5)',
      'tol': 2e-08},
     {'name': 'zero_determinant',
      'setup': 'import numpy as np\n'
               'def expect_value_error(fn,*args):\n'
               '    try:\n'
               '        fn(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'cov=np.zeros((1,2,2,2));cov[0,0]=[[0,0],[0,.5]];cov[0,1]=.5*np.eye(2)',
      'call': 'expect_value_error(entropy_jets,cov,2.0)',
      'gold_call': 'expect_value_error(_oracle_entropy_jets,cov,2.0)',
      'tol': 2e-08},
     {'name': 'positive_determinant_indefinite',
      'setup': 'import numpy as np\n'
               'def expect_value_error(fn,*args):\n'
               '    try:\n'
               '        fn(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'cov=np.zeros((1,2,2,2));cov[0,0]=-np.eye(2);cov[0,1]=np.eye(2)',
      'call': 'expect_value_error(entropy_jets,cov,2.0)',
      'gold_call': 'expect_value_error(_oracle_entropy_jets,cov,2.0)',
      'tol': 2e-08},
     {'name': 'fractional_pure_mode',
      'setup': 'import numpy as np\n'
               'def expect_value_error(fn,*args):\n'
               '    try:\n'
               '        fn(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'cov=np.zeros((1,2,2,2));cov[0,0]=cov[0,1]=.5*np.eye(2)',
      'call': 'expect_value_error(entropy_jets,cov,1.5)',
      'gold_call': 'expect_value_error(_oracle_entropy_jets,cov,1.5)',
      'tol': 2e-08},
     {'name': 'unsupported_renyi',
      'setup': 'import numpy as np\n'
               'def expect_value_error(fn,*args):\n'
               '    try:\n'
               '        fn(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'cov=np.zeros((1,2,2,2));cov[0,0]=cov[0,1]=np.eye(2)',
      'call': 'expect_value_error(entropy_jets,cov,1.1)',
      'gold_call': 'expect_value_error(_oracle_entropy_jets,cov,1.1)',
      'tol': 2e-08}]
