"""
Reduce the normalized ring vacuum to its photon quadrature covariance.

Tracing out electronic transition bosons leaves a mixed photonic Gaussian state. Form the full covariance before taking photon principal submatrices.

Returns
-------
return covariance
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def photon_covariance_jets(t, electronic_count):
    """Reduce the normalized ring vacuum to its photon quadrature covariance.

    Parameters
    ----------
    t : real array, shape (*grid,n,n)
        Symmetric ring-amplitude coefficients; all eigenvalues of t[zero] are
        strictly between -1 and 1.
    electronic_count : integer
        1 <= electronic_count < n. The remaining m coordinates are photons.
    
    Returns
    -------
    covariance : real array, shape (*grid,2,m,m)
        Block 0 is the photon principal submatrix of Vq=.5*(I+T)@inv(I-T).
        Block 1 is the photon principal submatrix of Vp=.5*(I-T)@inv(I+T).
        The inverses act on all n coordinates before restriction to photons.
        q=(c+c_dagger)/sqrt(2), p=(c-c_dagger)/(i*sqrt(2)); vacuum variance .5.
        The real branch has zero q-p covariance. Earlier inverse_jets is available.
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
    return covariance

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_photon_covariance_jets(t, electronic_count):
    import numpy as np
    t = np.asarray(t, float)
    shape = t.shape[:-2]
    zero = (0,)*len(shape)
    minus, plus = -t.copy(), t.copy()
    minus[zero] += np.eye(t.shape[-1])
    plus[zero] += np.eye(t.shape[-1])
    iminus, iplus = _oracle_inverse_jets(minus), _oracle_inverse_jets(plus)
    ne = int(electronic_count)
    nm = t.shape[-1]-ne
    covariance = np.zeros(shape+(2,nm,nm))
    for alpha in np.ndindex(shape):
        for beta in np.ndindex(tuple(k+1 for k in alpha)):
            rest = tuple(x-y for x,y in zip(alpha,beta))
            covariance[alpha][0] += .5*(plus[beta]@iminus[rest])[ne:,ne:]
            covariance[alpha][1] += .5*(minus[beta]@iplus[rest])[ne:,ne:]
    return covariance

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'call': 'photon_covariance_jets(t,4)',
      'gold_call': '_oracle_photon_covariance_jets(t,4)',
      'name': 'two_photons',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n"
               "a,b=cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],model['orders'])\n"
               'omega,t0=stable_ring(a[0,0],b[0,0])\n'
               't=ring_jets(a,b,t0)\n',
      'tol': 2e-08},
     {'call': 'photon_covariance_jets(t,5)',
      'gold_call': '_oracle_photon_covariance_jets(t,5)',
      'name': 'one_photon',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               "ea,eb=transition_blocks(model['gaps'],model['factors'])\n"
               "a,b=cavity_jets(ea,eb,model['dipoles'],model['frequencies'],model['couplings'],model['orders'])\n"
               'omega,t0=stable_ring(a[0,0],b[0,0])\n'
               't=ring_jets(a,b,t0)\n',
      'tol': 2e-08},
     {'call': 'photon_covariance_jets(t,2)',
      'gold_call': '_oracle_photon_covariance_jets(t,2)',
      'name': 'vacuum_integer',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               't=np.zeros((3,2,4,4),dtype=int)\n',
      'tol': 2e-08},
     {'call': 'photon_covariance_jets(t,1)',
      'gold_call': '_oracle_photon_covariance_jets(t,1)',
      'name': 'noncommuting_series',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n"
               't=np.zeros((3,3,3,3));t[0,0]=[[.1,.08,-.03],[.08,-.12,.05],[-.03,.05,.2]];t[1,0]=[[.02,.1,0],[.1,.03,.07],[0,.07,-.04]];t[0,1]=[[.05,0,.09],[0,-.02,.08],[.09,.08,.04]]\n',
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
               't=ring_jets(a,b,t0)',
      'call': 'photon_covariance_jets(t,4)',
      'gold_call': '_oracle_photon_covariance_jets(t,4)',
      'tol': 2e-08}]
